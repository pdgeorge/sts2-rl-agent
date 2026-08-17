"""A policy is a versioned config file, not a constellation of module constants.

Until 2026-08-16 the live agent's tunables were module constants --
``EvalWeights`` defaults, ``ROOM_MIN_HP_FRACTION``, ``QUALITY_BAR_SCALE``,
``SKIP_THRESHOLD``, ``CARD_REWARD_LARGE_DECK_SIZE`` -- and every sweep
monkey-patched globals inside worker processes. That is how a sweep once ran
400 runs with its baseline arm doing the exact opposite of its name
(`PHASE_TWO.md` section 3.1), and it is why no journal could ever say which
weights produced it.

The module constants still exist -- they are the shipped defaults and old
scripts read them -- but the decision path reads from the ACTIVE POLICY, and
the active policy comes from a JSON file under ``policies/`` whose keys are
validated: an unknown key or a missing one fails loudly at load, because a
config that silently disagrees with the code is the same failure the audit
scripts exist to catch, one layer down.

Rules this module exists to keep:

1. **Nothing patches a global.** Sweeps build a ``PolicyConfig`` and pass it
   (``run_agent(..., policy=...)`` for the live path,
   ``set_active_policy(...)`` when they must drive the module functions
   directly). ``apply_active_policy`` is the ONLY sanctioned writer of the
   legacy module constants, and it exists so those legacy readers and the
   config cannot drift apart.

2. **Every run says which policy played it.** ``policy_version`` and the git
   sha at load time travel with the journal's run_start and with every eval
   summary, so a number can never again be attributed by timestamp-archaeology
   against ``git log``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from sts2_env.search.evaluate import EvalWeights

logger = logging.getLogger(__name__)

#: `policies/` sits at the repository root, next to `output/` and `scripts/`.
POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"

DEFAULT_POLICY_NAME = "v001"

_EVAL_WEIGHT_FIELDS = tuple(f.name for f in fields(EvalWeights))


def git_sha(repo: Path | None = None) -> str:
    """The short sha the code was checked out at, or 'unknown'.

    Never raises: a policy must load in a packaged checkout, a copied tree and
    a CI container alike, and 'unknown' is an honest stamp where a sha is not
    available. A version without a sha is still better than a number nobody
    can tie to code at all.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo) if repo else str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- stamped metadata must not break play
        pass
    return "unknown"


@dataclass(frozen=True)
class PolicyConfig:
    """One complete, validated set of decision parameters with its identity."""

    policy_version: str
    source_path: str
    git_sha: str
    eval_weights: EvalWeights
    room_min_hp_fraction: dict[str, float]
    quality_bar_scale: float
    skip_threshold: float
    large_deck_size: int
    scaling_bonus: float
    block_need_bonus: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str = "<memory>") -> "PolicyConfig":
        _validate(data, source_path)
        weights = EvalWeights(**data["eval_weights"])
        return cls(
            policy_version=str(data["policy_version"]),
            source_path=source_path,
            git_sha=git_sha(),
            eval_weights=weights,
            room_min_hp_fraction={str(k): float(v) for k, v in
                                  data["room_min_hp_fraction"].items()},
            quality_bar_scale=float(data["quality_bar_scale"]),
            skip_threshold=float(data["skip_threshold"]),
            large_deck_size=int(data["large_deck_size"]),
            scaling_bonus=float(data["scaling_bonus"]),
            block_need_bonus=float(data["block_need_bonus"]),
        )

    @classmethod
    def load(cls, name_or_path: str | Path | None = None) -> "PolicyConfig":
        """Load by version name (resolved under policies/) or direct path.

        ``None`` loads the default policy, so a caller that does not care yet
        still gets a versioned, stamped object rather than a pile of globals.
        """
        if name_or_path is None:
            name_or_path = DEFAULT_POLICY_NAME
        path = Path(name_or_path)
        if not path.is_file():
            path = POLICY_DIR / f"{name_or_path}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no policy at {name_or_path!r} (also tried {path})")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data, source_path=str(path))

    def with_weights(self, **changes: float) -> "PolicyConfig":
        """A copy with eval-weight overrides, for sweeps and experiments.

        The sanctioned alternative to monkey-patching EvalWeights fields: the
        result carries the same version and identity, and callers pass it
        through the normal channels, so what was measured is always what was
        written down.
        """
        current = {f.name: getattr(self.eval_weights, f.name)
                   for f in fields(self.eval_weights)}
        unknown = set(changes) - set(current)
        if unknown:
            raise ValueError(f"unknown eval weight(s): {sorted(unknown)}")
        current.update(changes)
        return PolicyConfig(
            policy_version=self.policy_version,
            source_path=self.source_path,
            git_sha=self.git_sha,
            eval_weights=EvalWeights(**current),
            room_min_hp_fraction=dict(self.room_min_hp_fraction),
            quality_bar_scale=self.quality_bar_scale,
            skip_threshold=self.skip_threshold,
            large_deck_size=self.large_deck_size,
            scaling_bonus=self.scaling_bonus,
            block_need_bonus=self.block_need_bonus,
        )


#: Scalars and their home modules' legacy constant names. `apply_active_policy`
#: writes these so legacy readers see the active values; the table is the
#: single definition of the mapping, so a new tunable added here gets its
#: legacy reader for free -- and a rename fails the validation below instead of
#: silently tuning nothing.
_SCALAR_TARGETS = {
    "quality_bar_scale": ("sts2_env.bridge.agent_runner", "QUALITY_BAR_SCALE"),
    "skip_threshold": ("sts2_env.bridge.card_quality", "SKIP_THRESHOLD"),
    "large_deck_size": ("sts2_env.bridge.agent_runner", "CARD_REWARD_LARGE_DECK_SIZE"),
    "scaling_bonus": ("sts2_env.bridge.card_quality", "SCALING_BONUS"),
    "block_need_bonus": ("sts2_env.bridge.card_quality", "BLOCK_NEED_BONUS"),
}


def _validate(data: dict[str, Any], source_path: str) -> None:
    required = {
        "policy_version", "eval_weights", "room_min_hp_fraction",
        "quality_bar_scale", "skip_threshold", "large_deck_size",
        "scaling_bonus", "block_need_bonus",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"{source_path}: policy is missing {sorted(missing)}")
    unknown = set(data) - required
    if unknown:
        raise ValueError(f"{source_path}: policy has unknown keys {sorted(unknown)} "
                         f"-- a config the code will quietly ignore is worse than a crash")
    wf = data["eval_weights"]
    missing_w = set(_EVAL_WEIGHT_FIELDS) - set(wf)
    unknown_w = set(wf) - set(_EVAL_WEIGHT_FIELDS)
    if missing_w or unknown_w:
        raise ValueError(f"{source_path}: eval_weights missing {sorted(missing_w)} "
                         f"unknown {sorted(unknown_w)}")
    rooms = data["room_min_hp_fraction"]
    required_rooms = {"boss", "elite", "monster", "unknown", "event"}
    missing_r = required_rooms - set(rooms)
    if missing_r:
        raise ValueError(f"{source_path}: room_min_hp_fraction missing {sorted(missing_r)}")


_active: PolicyConfig | None = None


def active_policy() -> PolicyConfig:
    """The policy the live decision path is running, loading the default lazily."""
    global _active
    if _active is None:
        _active = PolicyConfig.load()
    return _active


def set_active_policy(policy: PolicyConfig) -> None:
    """Make `policy` the one the module-level decision readers see.

    `run_agent` calls this once per session; sweeps that drive the module
    functions directly call it instead of patching constants. Applying it also
    writes the legacy module constants, because six different readers still
    read them and a config that does not reach its own readers is a stamp, not
    a policy.
    """
    global _active
    _active = policy
    apply_active_policy(policy)


def apply_active_policy(policy: PolicyConfig) -> None:
    """Write the policy's values into the legacy constants its readers use."""
    from sts2_env.bridge import agent_runner, card_quality  # noqa: PLC0415

    agent_runner.ROOM_MIN_HP_FRACTION = dict(policy.room_min_hp_fraction)
    for attr, (module_name, const) in _SCALAR_TARGETS.items():
        module = agent_runner if module_name.endswith("agent_runner") else card_quality
        setattr(module, const, getattr(policy, attr))
    logger.info("policy %s applied (from %s, git %s)",
                policy.policy_version, policy.source_path, policy.git_sha)
