"""Turn a live card-reward state into something the battery can play.

The evaluator works on real `CardInstance` objects because it plays actual
fights. The bridge sends JSON. This is the seam, and it is deliberately small and
separately testable, because a silent mistranslation here would mean the agent
carefully evaluates a deck it does not have.

WHAT CAN GO WRONG, AND WHAT HAPPENS INSTEAD

* **A card name that does not resolve.** New content, or a naming change. The card
  is dropped from the reconstructed deck and counted, and the caller can refuse
  to use a reconstruction that lost too much. Evaluating a deck missing a third
  of its cards is worse than not evaluating at all, because it looks like it
  worked.
* **No deck in the state at all.** Older mod builds send `deck_size` without
  `deck`. Returns None rather than an empty deck, which would score as if the
  player owned nothing and rank every candidate identically.
* **Upgrade state.** Honoured when present. A deck read as all-unupgraded would
  systematically understate itself, and upgrades measurably matter -- the same
  five cards upgraded cut elite HP cost from 61.1 to 35.5.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

MIN_RESOLVED_FRACTION = 0.8
"""Refuse a reconstruction that lost more than a fifth of the deck.

Below this the evaluation is measuring a deck the player does not have, and a
confidently wrong ranking is worse than falling back to whatever chose before.
"""


def _canonical(name: object) -> str:
    name = getattr(name, "name", name)
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _card_id(raw: Any):
    """Resolve a bridge card name to a CardId, or None."""
    from sts2_env.core.enums import CardId

    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("id") or raw.get("name") or raw.get("card_id")
        if raw is None:
            return None

    text = str(getattr(raw, "name", raw))
    if text in CardId.__members__:
        return CardId[text]

    wanted = _canonical(text)
    for member in CardId:
        if _canonical(member.name) == wanted:
            return member
    return None


def _is_upgraded(entry: Any) -> bool:
    if isinstance(entry, Mapping):
        return bool(entry.get("upgraded") or entry.get("is_upgraded"))
    return False


def build_card(entry: Any):
    """One bridge card entry -> a CardInstance, or None if unresolvable."""
    from sts2_env.cards.factory import create_card

    card_id = _card_id(entry)
    if card_id is None:
        return None
    try:
        return create_card(card_id, upgraded=_is_upgraded(entry))
    except Exception:  # noqa: BLE001 -- an unconstructable card is a dropped card
        logger.debug("Could not construct %s", card_id)
        return None


@dataclass(frozen=True)
class RewardContext:
    """Everything the evaluator needs, reconstructed from a live state."""

    deck: list
    candidates: list
    candidate_indexes: list[int]
    floor: int
    max_hp: int
    resolved_fraction: float
    can_skip: bool

    @property
    def usable(self) -> bool:
        return (
            bool(self.candidates)
            and self.resolved_fraction >= MIN_RESOLVED_FRACTION
        )


def _read_deck_entries(state: Mapping[str, Any]) -> list | None:
    run_state = state.get("run_state")
    if isinstance(run_state, Mapping):
        deck = run_state.get("deck")
        if isinstance(deck, list):
            return deck
    deck = state.get("deck")
    return deck if isinstance(deck, list) else None


def _read_int(state: Mapping[str, Any], *names: str, default: int = 0) -> int:
    for container in (state, state.get("run_state"), state.get("player")):
        if not isinstance(container, Mapping):
            continue
        for name in names:
            value = container.get(name)
            if isinstance(value, (int, float)):
                return int(value)
    return default


def reward_context(state: Mapping[str, Any]) -> RewardContext | None:
    """Reconstruct a card reward, or None when it cannot be trusted."""
    entries = _read_deck_entries(state)
    if entries is None:
        logger.warning(
            "Card reward state has no `deck`; cannot evaluate candidates. "
            "An older mod build sends deck_size only."
        )
        return None

    deck = [c for c in (build_card(e) for e in entries) if c is not None]
    resolved = len(deck) / len(entries) if entries else 0.0
    if resolved < 1.0:
        logger.warning(
            "Reconstructed %d of %d deck cards (%.0f%%); unresolved names are "
            "dropped", len(deck), len(entries), resolved * 100,
        )

    offered = state.get("cards")
    if not isinstance(offered, list) or not offered:
        return None

    candidates, indexes = [], []
    for index, entry in enumerate(offered):
        card = build_card(entry)
        if card is not None:
            candidates.append(card)
            indexes.append(index)
        else:
            logger.warning("Offered card %r did not resolve; not a candidate", entry)

    return RewardContext(
        deck=deck,
        candidates=candidates,
        candidate_indexes=indexes,
        floor=_read_int(state, "total_floor", "floor", default=1),
        max_hp=_read_int(state, "max_hp", default=80) or 80,
        resolved_fraction=resolved,
        can_skip=bool(state.get("can_skip", False)),
    )


def choose_card_index(state: Mapping[str, Any], pilot, **kwargs) -> int | None:
    """Measured card-reward choice for a live state.

    Returns the index into the state's own `cards` list, None to skip, and
    raises nothing -- a caller on the live path should fall back rather than
    crash a run over a ranking.
    """
    from sts2_env.evaluation.card_choice import rank_candidates

    context = reward_context(state)
    if context is None or not context.usable:
        return None

    ranked = rank_candidates(
        context.deck,
        context.candidates,
        pilot,
        floor=context.floor,
        max_hp=context.max_hp,
        include_skip=context.can_skip,
        **kwargs,
    )
    if not ranked:
        return None

    # Log what was chosen and what it beat. "index 2" is unreadable after the
    # fact: diagnosing a draft needs the card names, the scores, and the margin,
    # because a 0.002 margin means the measurement did not actually separate them
    # and the pick is arbitrary.
    # Every pick carries how often real players took the same card, because a
    # score alone cannot tell you the drafter has wandered somewhere no human
    # goes. Free -- the scraped table already holds it -- and it catches a whole
    # class of silent regression within one evening of live play.
    from sts2_env.evaluation.card_priors import card_stats
    from sts2_env.evaluation.deck_metrics import describe as describe_deck

    def _annotate(s) -> str:
        if s.card is None:
            return f"SKIP={s.score:+.2f}"
        stats = card_stats(s.card, decision="card_reward", floor=context.floor)
        taken = f" [humans {stats['taken_pct']}%]" if stats and "taken_pct" in stats else ""
        return f"{s.label}={s.score:+.2f}{taken}"

    logger.info("draft: %s", "  ".join(_annotate(s) for s in ranked))
    logger.info("deck:  %s", describe_deck(context.deck))
    if len(ranked) >= 2:
        gap = ranked[0].score - ranked[1].score
        if gap < 0.15:
            logger.info(
                "draft: margin %.4f -- too close to call, pick is effectively "
                "arbitrary", gap,
            )

    winner = ranked[0].card
    if winner is None:
        return None
    for position, card in enumerate(context.candidates):
        if card is winner:
            return context.candidate_indexes[position]
    return None


def choose_rest_option(state: Mapping[str, Any], pilot, **kwargs) -> int | None:
    """Measured rest-site choice: the index of heal or smith, or None to fall back.

    Only decides heal-versus-smith. Which card to smith comes on a separate
    card-select screen and is still handled by the old heuristic.
    """
    from sts2_env.evaluation.rest_choice import rank_rest_options

    entries = _read_deck_entries(state)
    if entries is None:
        return None
    deck = [c for c in (build_card(e) for e in entries) if c is not None]
    if not deck or len(deck) / len(entries) < MIN_RESOLVED_FRACTION:
        return None

    options = state.get("options")
    if not isinstance(options, list) or not options:
        return None

    def _find(*wanted: str) -> int | None:
        for index, option in enumerate(options):
            text = _canonical(option.get("id") or option.get("action") or "")
            if any(w in text for w in wanted):
                return int(option.get("index", index))
        return None

    heal_index = _find("HEAL", "REST")
    smith_index = _find("SMITH", "UPGRADE")
    if heal_index is None or smith_index is None:
        return None  # only one real option; nothing to decide

    current_hp = _read_int(state, "hp", "current_hp", default=0)
    max_hp = _read_int(state, "max_hp", default=80) or 80
    upgradable = [c for c in deck if not getattr(c, "upgraded", False)]

    ranked = rank_rest_options(
        deck, upgradable, pilot,
        current_hp=current_hp, max_hp=max_hp,
        floor=_read_int(state, "total_floor", "floor", default=1),
        **kwargs,
    )
    if not ranked:
        return None
    return heal_index if ranked[0].kind == "rest" else smith_index


ELITE_VETO_WIN_RATE = 0.6
"""Below this measured elite win rate, do not walk into one.

Not tuned -- chosen as "clearly worse than a coin flip on the run". A starter deck
measures 33-42%, a starter plus five solid cards measures 100%, so the threshold
sits in a gap rather than on a slope.
"""


def veto_elite_rooms(state: Mapping[str, Any], pilot, **kwargs) -> bool:
    """True when this deck should not be taking elites yet.

    Deliberately a veto rather than a chooser. Ranking rooms by expected HP cost
    picks the rest site every time, because rest is the only room with negative
    cost and combat rewards are not modelled -- and "avoid all combat" wins
    nothing.
    """
    from sts2_env.evaluation.map_choice import elite_survivability

    entries = _read_deck_entries(state)
    if entries is None:
        return False
    deck = [c for c in (build_card(e) for e in entries) if c is not None]
    if not deck or len(deck) / len(entries) < MIN_RESOLVED_FRACTION:
        return False

    rate = elite_survivability(
        deck, pilot, floor=_read_int(state, "total_floor", "floor", default=1),
        max_hp=_read_int(state, "max_hp", default=80) or 80, **kwargs,
    )
    if rate < ELITE_VETO_WIN_RATE:
        logger.info("map: elite win rate %.0f%% -- avoiding elites", rate * 100)
        return True
    logger.info("map: elite win rate %.0f%% -- elites are fine", rate * 100)
    return False


def choose_upgrade_indexes(
    state: Mapping[str, Any], pilot, count: int = 1, **kwargs
) -> list[int] | None:
    """Which cards to upgrade on a smith screen, by measured HP value.

    ONLY safe when the caller knows this selection is an upgrade. The bridge's
    card_select payload carries `type`, `cards`, `min_select` and `max_select`
    and nothing about *why* it is selecting, so an upgrade screen and a removal
    screen are indistinguishable here. Ranking by "best to upgrade" on a removal
    screen would delete the best card in the deck. The caller tracks intent
    instead: it knows when it has just chosen smith.
    """
    from sts2_env.evaluation.rest_choice import rank_rest_options

    entries = _read_deck_entries(state)
    offered = state.get("cards")
    if entries is None or not isinstance(offered, list) or not offered:
        return None

    deck = [c for c in (build_card(e) for e in entries) if c is not None]
    if not deck or len(deck) / len(entries) < MIN_RESOLVED_FRACTION:
        return None

    # Match each offered card to a deck instance so the upgrade is evaluated
    # against the deck that actually exists.
    by_id: dict[Any, list] = {}
    for card in deck:
        by_id.setdefault(card.card_id, []).append(card)

    candidates, indexes = [], []
    for position, entry in enumerate(offered):
        card_id = _card_id(entry)
        pool = by_id.get(card_id) or []
        if not pool:
            continue
        candidates.append(pool.pop())
        indexes.append(int(entry.get("index", position)) if isinstance(entry, Mapping) else position)

    if not candidates:
        return None

    ranked = rank_rest_options(
        deck, candidates, pilot,
        current_hp=10**6,          # rest is irrelevant here; make it worthless
        max_hp=_read_int(state, "max_hp", default=80) or 80,
        floor=_read_int(state, "total_floor", "floor", default=1),
        **kwargs,
    )
    upgrades = [o for o in ranked if o.kind == "upgrade"]
    if not upgrades:
        return None

    chosen = [indexes[o.index] for o in upgrades[:count] if o.index is not None]
    logger.info(
        "smith: %s",
        "  ".join(f"{o.label}={o.hp_value:+.1f}hp" for o in upgrades[:4]),
    )
    return chosen or None
