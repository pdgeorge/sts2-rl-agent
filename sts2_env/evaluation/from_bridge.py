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

    winner = ranked[0].card
    if winner is None:
        return None
    for position, card in enumerate(context.candidates):
        if card is winner:
            return context.candidate_indexes[position]
    return None
