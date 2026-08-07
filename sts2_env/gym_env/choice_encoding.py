"""Encode the options a run decision is choosing between.

The agent could not see any of them. At a card reward offering EVIL_EYE, CINDER
and THUNDERCLAP its observation was 131 zeros plus twenty run-level scalars --
byte-identical to what it would be for any other three cards. So it could learn
combat, and a positional prior ("floor 5, low HP, take slot 1"), and nothing about
deckbuilding, because deckbuilding needs to know what is on offer.

THE RULE THAT MAKES THIS SAFE

The simulator and the live game describe the same card differently. RunManager's
pick_card carries `card_id`, `rarity` and `upgraded`; the mod's card entry carries
`id`, `type` and `cost`. Encoding from whichever fields a payload happens to have
would give the same card two different vectors, and a policy trained in the
simulator would misread the live game -- silently, showing up only as "she plays
worse on stream than in testing".

So a card is encoded from its **id**, looked up through create_card(), which since
the derivation work reads cost, type, damage and block from the decompiled game.
Both sides carry the id. Identical inputs produce identical features by
construction, not by two implementations agreeing.

Scales match gym_env/observation.py -- cost/5, damage/50, block/50 -- so a card
means the same thing here as it does in the combat block.

Layout (126 dims). Only the active decision's slots are populated; the rest are
zero, as the combat slice already is out of combat.

    card options    6 x 7  = 42   cost, is_attack, is_skill, is_power, dmg, blk, valid
    map nodes       6 x 10 = 60   MapPointType one-hot(9), valid
    rest / events   6 x 4  = 24   is_heal, is_smith, is_other_enabled, valid
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import numpy as np

from sts2_env.core.enums import CardId, CardType, MapPointType

CHOICE_SLOTS = 6

CARD_SLOTS = 20
"""Card option slots, separate from CHOICE_SLOTS and much larger.

A card *reward* offers three. A multi-select -- a deck transform, an enchant --
offers the whole deck, and one was observed live at 19 options on floor 16. Six
slots meant the agent could not see most of what it was choosing between.

Map nodes and rest/event options stay at CHOICE_SLOTS because they are genuinely
few. A screen with more options than slots leaves the tail unencoded, which
costs decision quality but not the ability to finish the screen -- the mask still
offers every option."""

CARD_FEATURES = 8
NODE_FEATURES = 10
OPTION_FEATURES = 4

CARD_BLOCK = CARD_SLOTS * CARD_FEATURES        # 160
NODE_BLOCK = CHOICE_SLOTS * NODE_FEATURES      # 60
OPTION_BLOCK = CHOICE_SLOTS * OPTION_FEATURES  # 24
CHOICE_OBS_SIZE = CARD_BLOCK + NODE_BLOCK + OPTION_BLOCK  # 244

COST_SCALE = 5.0
DAMAGE_SCALE = 50.0
BLOCK_SCALE = 50.0

MAP_POINT_TYPES = tuple(MapPointType.__members__)  # 9, order fixed by the enum

REST_HEAL = "HEAL"
REST_SMITH = "SMITH"

_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")


def normalize_enum_name(name: str) -> str:
    """PascalCase or UPPER_SNAKE in, UPPER_SNAKE out.

    The game says RestSite where the simulator says REST_SITE, VULNERABLE_POWER
    where it says VULNERABLE, VICIOUS where it says VICIOUS_CARD. Three separate
    naming gaps have been found this way, each initially looking like missing
    content. One normaliser, so the fourth does not.
    """
    if not name:
        return ""
    return _CAMEL_2.sub(r"\1_\2", _CAMEL_1.sub(r"\1_\2", name)).upper()


def resolve_card_id(name: str) -> CardId | None:
    """Bridge or simulator card name to CardId, or None if unknown.

    A card and a power sharing a name means the card gets a _CARD suffix here --
    Vicious the card is VICIOUS_CARD because PowerId already has VICIOUS -- while
    both payloads say the bare name.
    """
    if not name:
        return None
    upper = normalize_enum_name(name)
    for candidate in (upper, f"{upper}_CARD", f"{upper}_STATUS"):
        if candidate in CardId.__members__:
            return CardId[candidate]
    return None


def encode_card_options(
    card_names: Sequence[str],
    selected: Sequence[bool] | None = None,
) -> np.ndarray:
    """Card slots, from ids alone. Unknown cards occupy a slot marked invalid.

    Marked rather than skipped: a card the simulator has not implemented is still
    an option the game offered, and silently compacting the list would shift every
    slot after it out of step with the action indices.

    `selected` carries the multi-select state, and without it the agent cannot
    see the effect of its own action. A deck transform needs exactly three cards
    picked before it can be confirmed; with selection invisible the observation
    is byte-identical whether a card is chosen or not, so a deterministic policy
    toggles the same card on and off forever. Measured on `meta_ppo_v8`: 2.6% of
    episodes ran to the step cap and consumed ~73% of evaluation compute, and
    because transforms come from later rewards it struck the deepest runs.
    """
    from sts2_env.cards.factory import create_card

    out = np.zeros(CARD_BLOCK, dtype=np.float32)
    for slot, name in enumerate(card_names[:CARD_SLOTS]):
        base = slot * CARD_FEATURES
        card_id = resolve_card_id(str(name))
        if card_id is None:
            continue
        try:
            card = create_card(card_id)
        except Exception:  # noqa: BLE001 - an unimplemented card is data, not a crash
            continue
        out[base + 0] = max(0, card.cost) / COST_SCALE
        out[base + 1] = 1.0 if card.card_type == CardType.ATTACK else 0.0
        out[base + 2] = 1.0 if card.card_type == CardType.SKILL else 0.0
        out[base + 3] = 1.0 if card.card_type == CardType.POWER else 0.0
        out[base + 4] = (card.base_damage or 0) / DAMAGE_SCALE
        out[base + 5] = (card.base_block or 0) / BLOCK_SCALE
        out[base + 6] = 1.0
        if selected is not None and slot < len(selected) and selected[slot]:
            out[base + 7] = 1.0
    return out


def encode_map_nodes(node_types: Sequence[str]) -> np.ndarray:
    """Map slots as a MapPointType one-hot per node."""
    out = np.zeros(NODE_BLOCK, dtype=np.float32)
    for slot, raw in enumerate(node_types[:CHOICE_SLOTS]):
        base = slot * NODE_FEATURES
        name = normalize_enum_name(str(raw))
        if name in MAP_POINT_TYPES:
            out[base + MAP_POINT_TYPES.index(name)] = 1.0
        out[base + NODE_FEATURES - 1] = 1.0
    return out


def encode_simple_options(options: Sequence[tuple[str, bool]]) -> np.ndarray:
    """Rest-site and event slots as (option_id, enabled).

    HEAL and SMITH are distinguished because smithing is a card decision -- it
    upgrades one -- and a bare enabled flag could not tell the agent which of the
    two it was choosing. Everything else is 'some other enabled option'.
    """
    out = np.zeros(OPTION_BLOCK, dtype=np.float32)
    for slot, (option_id, enabled) in enumerate(list(options)[:CHOICE_SLOTS]):
        base = slot * OPTION_FEATURES
        name = normalize_enum_name(str(option_id))
        if name == REST_HEAL:
            out[base + 0] = 1.0
        elif name == REST_SMITH:
            out[base + 1] = 1.0
        elif enabled:
            out[base + 2] = 1.0
        out[base + 3] = 1.0
    return out


def encode_choices(
    card_names: Sequence[str] = (),
    node_types: Sequence[str] = (),
    options: Sequence[tuple[str, bool]] = (),
    selected: Sequence[bool] | None = None,
) -> np.ndarray:
    """The full CHOICE_OBS_SIZE block. Absent decisions stay zero."""
    return np.concatenate([
        encode_card_options(card_names, selected),
        encode_map_nodes(node_types),
        encode_simple_options(options),
    ])


# ---------------------------------------------------------------------------
# Extractors. One per side, deliberately adjacent: they are the thing that can
# drift, and tests/test_choice_encoding_parity.py holds them to the same output.
# ---------------------------------------------------------------------------

def choices_from_sim_actions(actions: Iterable[dict[str, Any]]) -> dict[str, list]:
    """Pull the choosable options out of RunManager.get_available_actions()."""
    cards: list[str] = []
    nodes: list[str] = []
    options: list[tuple[str, bool]] = []

    selected: list[bool] = []
    for action in actions:
        kind = action.get("action")
        if kind == "pick_card":
            cards.append(str(action.get("card_id", "")))
            selected.append(False)
        elif kind == "choose":
            # Multi-select screens -- deck transform, enchant, hand selection.
            # These were not read at all, so the whole screen encoded as zeros
            # and the agent could not see what it was choosing between, nor
            # what it had already picked.
            cards.append(str(action.get("card_id", "")))
            selected.append(bool(action.get("selected", False)))
        elif kind == "move":
            nodes.append(str(action.get("point_type", "")))
        elif kind in ("rest_option", "event_choice"):
            options.append((
                str(action.get("option_id", "")),
                bool(action.get("enabled", True)),
            ))
    return {"card_names": cards, "node_types": nodes, "options": options,
            "selected": selected}


def choices_from_bridge_state(state: dict[str, Any]) -> dict[str, list]:
    """Pull the same options out of a bridge state message.

    Field names differ from the simulator's throughout -- cards[].id against
    card_id, nodes[].type against point_type, options[].id against option_id --
    which is exactly why both sides feed one encoder rather than each building
    its own vector.
    """
    cards: list[str] = []
    nodes: list[str] = []
    options: list[tuple[str, bool]] = []

    selected: list[bool] = []
    for card in state.get("cards", []) or []:
        if isinstance(card, dict):
            cards.append(str(card.get("id", "")))
            # The mod sends `selected` on card_select screens; absent elsewhere,
            # which reads as not-selected and is correct for a single pick.
            selected.append(bool(card.get("selected", False)))
    for node in state.get("nodes", []) or []:
        if isinstance(node, dict):
            nodes.append(str(node.get("type", "")))
    for option in state.get("options", []) or []:
        if isinstance(option, dict):
            options.append((
                str(option.get("id", "")),
                bool(option.get("enabled", True)),
            ))
    return {"card_names": cards, "node_types": nodes, "options": options,
            "selected": selected}
