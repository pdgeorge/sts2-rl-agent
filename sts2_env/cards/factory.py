"""Helpers for instantiating cards from `CardId`.

This module builds a lazy `CardId -> make_*` registry by introspecting the
existing card modules. It is intentionally lightweight: the simulator already
has hundreds of per-card factories, but no central constructor registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from functools import lru_cache
from pathlib import Path
import re
from typing import Callable, Literal

import sts2_env.cards  # noqa: F401  # ensure card modules are imported
from sts2_env.cards import base as card_base
from sts2_env.cards.base import CardInstance
from sts2_env.core.card_pools import (
    ALL_CARD_POOL,
    CardPoolId,
    CHARACTER_CARD_POOLS_BY_ID,
    COLORLESS_CARD_POOL,
    CURSE_CARD_POOL,
    EVENT_CARD_POOL,
    QUEST_CARD_POOL,
    SHARED_CARD_POOLS_BY_ID,
    STATUS_CARD_POOL,
    TOKEN_CARD_POOL,
)
from sts2_env.characters.all import get_character
from sts2_env.core.enums import CardId, CardRarity, CardTag, CardType, OrbEvokeType, TargetType
from sts2_env.core.rng import Rng
from sts2_env.cards.reference_static_metadata import (
    MULTIPLAYER_CONSTRAINT_MULTIPLAYER_ONLY,
    MULTIPLAYER_CONSTRAINT_NONE,
    MULTIPLAYER_CONSTRAINT_SINGLEPLAYER_ONLY,
)

CardFactory = Callable[..., CardInstance]
GenerationContext = Literal["combat", "modifier"]
REFERENCE_TOKEN_RARITY_TEXT = "TOKEN"
REFERENCE_X_COST_TEXT = "X"

_CARD_MODULES = (
    "sts2_env.cards.ironclad_basic",
    "sts2_env.cards.ironclad",
    "sts2_env.cards.silent",
    "sts2_env.cards.defect",
    "sts2_env.cards.necrobinder",
    "sts2_env.cards.regent",
    "sts2_env.cards.colorless",
    "sts2_env.cards.status",
)

_COMBAT_GENERATION_EXCLUDED = frozenset({
    CardId.ALCHEMIZE,
    CardId.DISINTEGRATION,
    CardId.FEED,
    CardId.FRANTIC_ESCAPE,
    CardId.HAND_OF_GREED,
    CardId.MIND_ROT,
    CardId.NEOWS_FURY,
    CardId.ROYALTIES_CARD,
    CardId.SLOTH_STATUS,
    CardId.SOOT,
    CardId.THE_HUNT,
    CardId.WASTE_AWAY,
})

_MODIFIER_GENERATION_EXCLUDED = frozenset({
    CardId.ASCENDERS_BANE,
    CardId.BAD_LUCK,
    CardId.CURSE_OF_THE_BELL,
    CardId.ENTHRALLED,
    CardId.FOLLY,
    CardId.GREED,
    CardId.POOR_SLEEP,
    CardId.SPORE_MIND,
})

_REGISTERED_POOL_EXCLUDED_CARD_IDS = frozenset({
    CardId.DEPRECATED_CARD,
    CardId.PAIN,
    CardId.PARASITE,
})

_MODULE_CARD_POOL_ORDER: dict[str, tuple[CardId, ...]] = {
    "sts2_env.cards.ironclad": ALL_CARD_POOL,
    "sts2_env.cards.ironclad_basic": ALL_CARD_POOL,
    "sts2_env.cards.silent": ALL_CARD_POOL,
    "sts2_env.cards.defect": ALL_CARD_POOL,
    "sts2_env.cards.necrobinder": ALL_CARD_POOL,
    "sts2_env.cards.regent": ALL_CARD_POOL,
    "sts2_env.cards.colorless": COLORLESS_CARD_POOL,
    "sts2_env.cards.status": CURSE_CARD_POOL + EVENT_CARD_POOL + QUEST_CARD_POOL + STATUS_CARD_POOL + TOKEN_CARD_POOL,
}

_CARD_POOLS_BY_ID = {**CHARACTER_CARD_POOLS_BY_ID, **SHARED_CARD_POOLS_BY_ID}

_TYPE_CARD_POOL_ORDER: dict[CardType, tuple[CardId, ...]] = {
    CardType.CURSE: CURSE_CARD_POOL,
    CardType.QUEST: QUEST_CARD_POOL,
    CardType.STATUS: STATUS_CARD_POOL + TOKEN_CARD_POOL,
}

_RARITY_CARD_POOL_ORDER: dict[CardRarity, tuple[CardId, ...]] = {
    CardRarity.CURSE: CURSE_CARD_POOL,
    CardRarity.EVENT: EVENT_CARD_POOL,
    CardRarity.QUEST: QUEST_CARD_POOL,
    CardRarity.STATUS: STATUS_CARD_POOL + TOKEN_CARD_POOL,
}

_REFERENCE_VAR_ALIASES: dict[str, str] = {
    "vulnerable_power": "vulnerable",
    "weak_power": "weak",
    "strength_power": "strength",
    "dexterity_power": "dexterity",
    "poison": "poison_power",
    "plating_power": "plating",
    "doom_power": "doom",
    "calculation_base": "calc_base",
    "calculation_extra": "calc_extra",
    "calcify_power": "calcify",
    "countdown_power": "countdown",
    "danse_macabre_power": "danse_macabre",
    "debilitate_power": "debilitate",
    "lethality_power": "lethality",
    "sic_em_power": "sic_em",
    "sleight_of_flesh_power": "sleight_of_flesh",
    "devour_life_power": "devour_life",
    "vigor_power": "vigor",
    "neurosurge_power": "neurosurge",
    "sentry_mode_power": "sentry_mode",
    "prep_time_power": "prep_time",
    "knockdown_power": "knockdown",
    "rolling_boulder_power": "rolling_boulder",
    "arsenal_power": "arsenal",
    "black_hole_power": "black_hole",
    "parry_power": "parry",
    "furnace_power": "furnace",
    "stars_per_turn": "stars_per_turn",
    "block_for_stars": "block_for_stars",
}


@dataclass(frozen=True)
class CardMetadata:
    card_type: CardType
    rarity: CardRarity
    keywords: frozenset[str]
    can_be_generated_in_combat: bool
    can_be_generated_by_modifiers: bool
    has_turn_end_in_hand_effect: bool
    gains_block: bool
    orb_evoke_type: OrbEvokeType
    visual_card_pool: CardPoolId | None
    should_show_in_card_library: bool
    has_custom_playability: bool
    has_custom_should_play: bool
    has_custom_card_type: bool
    has_custom_target_type: bool
    multiplayer_constraint: str


@dataclass(frozen=True)
class ReferenceCardDefinition:
    card_id_text: str
    color: str
    cost: str
    card_type: str
    rarity: str
    target: str
    keywords: tuple[str, ...]
    tags: tuple[CardTag, ...]
    vars_text: str
    upgrade_text: str


def _apply_decompiled_static_metadata(card: CardInstance) -> CardInstance:
    static_metadata = _static_metadata_override(card.card_id)
    if static_metadata is not None:
        card.can_be_generated_in_combat = static_metadata.can_be_generated_in_combat
        card.can_be_generated_by_modifiers = static_metadata.can_be_generated_by_modifiers
        card.has_turn_end_in_hand_effect = static_metadata.has_turn_end_in_hand_effect
        return card
    card.can_be_generated_in_combat = card.card_id not in _COMBAT_GENERATION_EXCLUDED
    card.can_be_generated_by_modifiers = card.card_id not in _MODIFIER_GENERATION_EXCLUDED
    return card


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@lru_cache(maxsize=1)
def _reference_cards() -> dict[str, ReferenceCardDefinition]:
    text = Path("docs/CARDS_REFERENCE.md").read_text()
    entries = re.split(r"^### ", text, flags=re.MULTILINE)[1:]
    result: dict[str, ReferenceCardDefinition] = {}
    for entry in entries:
        fields: dict[str, str] = {}
        for line in entry.splitlines():
            match = re.match(r"- \*\*(.+?):\*\* (.+)", line)
            if match:
                fields[match.group(1)] = match.group(2)
        card_id_text = fields.get("ID")
        if not card_id_text:
            continue
        keywords = tuple(
            keyword.strip().lower()
            for keyword in fields.get("Keywords", "None").split(",")
            if keyword.strip() and keyword.strip() != "None"
        )
        tags = tuple(
            CardTag[_camel_to_snake(tag.strip()).upper()]
            for tag in fields.get("Tags", "None").split(",")
            if tag.strip() and tag.strip() != "None"
        )
        result[card_id_text] = ReferenceCardDefinition(
            card_id_text=card_id_text,
            color=fields.get("Color", "").lower(),
            cost=fields["Cost"],
            card_type=fields["Type"],
            rarity=fields["Rarity"],
            target=fields["Target"],
            keywords=keywords,
            tags=tags,
            vars_text=fields.get("Vars", "{}"),
            upgrade_text=fields.get("Upgrade", ""),
        )
    return result


def _reference_candidates(card_id: CardId) -> list[str]:
    candidates = [card_id.name]
    if card_id.name.endswith("_CARD"):
        candidates.append(card_id.name[:-5])
    if card_id.name.endswith("_STATUS"):
        candidates.append(card_id.name[:-7])
    if card_id == CardId.NULL_CARD:
        candidates.append("NULL")
    return candidates


def _reference_definition(card_id: CardId) -> ReferenceCardDefinition | None:
    refs = _reference_cards()
    for candidate in _reference_candidates(card_id):
        definition = refs.get(candidate)
        if definition is not None:
            return definition
    return None


def reference_card_entry(card_id: CardId) -> str:
    definition = _reference_definition(card_id)
    if definition is not None:
        return definition.card_id_text
    return card_id.name


@lru_cache(maxsize=1)
def _decompiled_card_static_metadata():
    from sts2_env.cards.reference_static_metadata import reference_metadata_by_card_id

    return reference_metadata_by_card_id()


def _static_metadata_override(card_id: CardId):
    return _decompiled_card_static_metadata().get(card_id)


def _coerce_reference_rarity(rarity_name: str) -> CardRarity:
    normalized = rarity_name.upper()
    if normalized == REFERENCE_TOKEN_RARITY_TEXT:
        return CardRarity.STATUS
    return CardRarity[normalized]


def _decompiled_effect_vars(card_id: CardId, upgraded: bool) -> dict[str, int] | None:
    """The card's dynamic vars as the game declares them, or None if it has none.

    None means "the decompile cannot answer for this card" -- a simulator-only
    construct, or one deleted upstream -- not "this card has no values", so callers
    fall back rather than treating it as an empty set.
    """
    from sts2_env.cards.reference_static_metadata import (
        reference_dynamic_vars_by_card_id,
        upgraded_reference_dynamic_vars_by_card_id,
    )

    table = (upgraded_reference_dynamic_vars_by_card_id() if upgraded
             else reference_dynamic_vars_by_card_id())
    return table.get(card_id)


def _build_reference_effect_vars(vars_text: str) -> dict[str, int]:
    effect_vars: dict[str, int] = {}
    for key, value_text in re.findall(r"([A-Za-z][A-Za-z0-9]*): ([^,}]+)", vars_text):
        value_text = value_text.strip()
        if re.fullmatch(r"-?\d+", value_text):
            normalized_key = _camel_to_snake(key)
            normalized_key = _REFERENCE_VAR_ALIASES.get(normalized_key, normalized_key)
            effect_vars[normalized_key] = int(value_text)
    return effect_vars


def _apply_upgrade_text(
    card: CardInstance,
    effect_vars: dict[str, int],
    upgrade_text: str,
) -> None:
    if not upgrade_text or upgrade_text == "No upgrade changes":
        return
    for part in [item.strip() for item in upgrade_text.split(";") if item.strip()]:
        if part.startswith("Add "):
            effect_vars_key = _camel_to_snake(part[4:])
            card.keywords = frozenset(set(card.keywords) | {effect_vars_key})
            continue
        if part.startswith("Remove "):
            effect_vars_key = _camel_to_snake(part[7:])
            card.keywords = frozenset(keyword for keyword in card.keywords if keyword != effect_vars_key)
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)([+-]\d+)", part)
        if not match:
            continue
        field_name = _camel_to_snake(match.group(1))
        field_name = _REFERENCE_VAR_ALIASES.get(field_name, field_name)
        delta = int(match.group(2))
        if field_name == "damage":
            card.base_damage = (card.base_damage or 0) + delta
            if field_name in effect_vars:
                effect_vars[field_name] += delta
        elif field_name == "block":
            card.base_block = (card.base_block or 0) + delta
            if field_name in effect_vars:
                effect_vars[field_name] += delta
        elif field_name == "cost":
            card.cost += delta
            card.original_cost += delta
        elif field_name == "star_cost":
            card.star_cost += delta
        else:
            effect_vars[field_name] = effect_vars.get(field_name, 0) + delta


def _build_reference_card(
    card_id: CardId,
    upgraded: bool = False,
    *,
    allow_generation: bool = False,
) -> CardInstance:
    definition = _reference_definition(card_id)
    if definition is None:
        raise KeyError(f"No reference card definition for {card_id!r}")

    static_metadata = _static_metadata_override(card_id)
    cost_text = definition.cost.split("|", 1)[0].strip()
    star_cost = 0
    star_cost_match = re.search(r"StarCost:\s*(-?\d+)", definition.cost)
    if star_cost_match is not None:
        star_cost = int(star_cost_match.group(1))

    has_energy_cost_x = cost_text == REFERENCE_X_COST_TEXT
    if static_metadata is not None:
        cost = static_metadata.cost
        has_energy_cost_x = static_metadata.has_energy_cost_x
    else:
        if cost_text == "Unplayable":
            cost = -1
        else:
            cost = 0 if has_energy_cost_x else int(cost_text)
    # Values come from the decompile. docs/CARDS_REFERENCE.md is prose and routing
    # only -- it used to define these as well, which made it a third copy of every
    # number, needing hand-edits in lockstep with the factories and the decompile.
    # It fell behind, and since the tests read it as an oracle they stayed green
    # against it while the simulator disagreed with the game that shipped.
    decompiled_vars = _decompiled_effect_vars(card_id, upgraded)
    effect_vars = (dict(decompiled_vars) if decompiled_vars is not None
                   else _build_reference_effect_vars(definition.vars_text))
    card_type = (static_metadata.card_type if static_metadata is not None
                 else CardType[definition.card_type.upper()])
    base_damage = effect_vars.get("damage", effect_vars.get("calc_base"))
    base_block = effect_vars.get("block")
    if base_damage is None and card_type == CardType.ATTACK:
        base_damage = 0
    if base_block is None and card_type in {CardType.SKILL, CardType.POWER}:
        base_block = 0

    # Upgraded values are already resolved when they come from the decompile; the
    # upgrade text only has to be replayed for cards it could not supply.
    can_upgrade = definition.upgrade_text != "Cannot be upgraded"
    card = CardInstance(
        card_id=card_id,
        cost=cost,
        card_type=card_type,
        target_type=(static_metadata.target_type if static_metadata is not None
                     else TargetType[_camel_to_snake(definition.target).upper()]),
        rarity=(static_metadata.rarity if static_metadata is not None
                else _coerce_reference_rarity(definition.rarity)),
        base_damage=base_damage,
        base_block=base_block,
        upgraded=upgraded and can_upgrade,
        keywords=frozenset(static_metadata.keywords if static_metadata is not None else definition.keywords),
        tags=frozenset(static_metadata.tags if static_metadata is not None else definition.tags),
        effect_vars=effect_vars,
        has_energy_cost_x=has_energy_cost_x,
        star_cost=star_cost,
        has_star_cost_x=static_metadata.has_star_cost_x if static_metadata is not None else False,
        has_turn_end_in_hand_effect=(
            static_metadata.has_turn_end_in_hand_effect if static_metadata is not None else False
        ),
    )
    if upgraded and can_upgrade and decompiled_vars is None:
        _apply_upgrade_text(card, effect_vars, definition.upgrade_text)
    card = _apply_decompiled_static_metadata(card)
    if not allow_generation:
        card.can_be_generated_in_combat = False
        card.can_be_generated_by_modifiers = False
    return card


def create_reference_card(
    card_id: CardId,
    upgraded: bool = False,
    *,
    allow_generation: bool = False,
) -> CardInstance:
    """Instantiate a card from reference metadata when no handwritten factory exists."""
    return _build_reference_card(card_id, upgraded=upgraded, allow_generation=allow_generation)


def _reference_source_module(card_id: CardId) -> str | None:
    definition = _reference_definition(card_id)
    if definition is None:
        return None
    color = definition.color
    if color in {"colorless"}:
        return "sts2_env.cards.colorless"
    if color in {"event", "token", "status", "curse", "quest"}:
        return "sts2_env.cards.status"
    if color == "silent":
        return "sts2_env.cards.silent"
    if color == "defect":
        return "sts2_env.cards.defect"
    if color == "necrobinder":
        return "sts2_env.cards.necrobinder"
    if color == "regent":
        return "sts2_env.cards.regent"
    if color == "ironclad":
        return "sts2_env.cards.ironclad"
    return None


def _probe_factory(factory: CardFactory) -> CardInstance:
    """Call a card factory without consuming the global instance counter."""
    saved_counter = card_base._next_instance_id
    try:
        return _apply_decompiled_static_metadata(factory())
    finally:
        card_base._next_instance_id = saved_counter


def _supports_upgraded_arg(factory: CardFactory) -> bool:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return "upgraded" in sig.parameters


@lru_cache(maxsize=1)
def _factory_registry() -> dict[CardId, tuple[CardFactory, bool, str]]:
    registry: dict[CardId, tuple[CardFactory, bool, str]] = {}

    for module_name in _CARD_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if not callable(obj) or not name.startswith("make_"):
                continue
            try:
                card = _probe_factory(obj)
            except TypeError:
                continue
            if not isinstance(card, CardInstance):
                continue
            registry[card.card_id] = (obj, _supports_upgraded_arg(obj), module_name)

    return registry


def create_card(card_id: CardId, upgraded: bool = False) -> CardInstance:
    """Instantiate a card by id using the existing `make_*` factories."""
    registry = _factory_registry()
    entry = registry.get(card_id)
    if entry is None:
        return create_reference_card(card_id, upgraded=upgraded, allow_generation=False)
    factory, supports_upgraded, _ = entry

    if supports_upgraded:
        return _apply_decompiled_static_metadata(factory(upgraded=upgraded))
    card = factory()
    if upgraded:
        definition = _reference_definition(card_id)
        if definition is not None and definition.upgrade_text not in {
            "",
            "No upgrade changes",
            "Cannot be upgraded",
        }:
            card.upgraded = True
            _apply_upgrade_text(card, card.effect_vars, definition.upgrade_text)
    return _apply_decompiled_static_metadata(card)


@lru_cache(maxsize=None)
def card_metadata(card_id: CardId) -> CardMetadata:
    registry = _factory_registry()
    if card_id in registry:
        factory, _, _ = registry[card_id]
        card = _probe_factory(factory)
    else:
        card = create_reference_card(card_id, allow_generation=False)
    return CardMetadata(
        card_type=card.card_type,
        rarity=card.rarity,
        keywords=card.keywords,
        can_be_generated_in_combat=card.can_be_generated_in_combat,
        can_be_generated_by_modifiers=card.can_be_generated_by_modifiers,
        has_turn_end_in_hand_effect=card.has_turn_end_in_hand_effect,
        gains_block=card.gains_block,
        orb_evoke_type=card.orb_evoke_type,
        visual_card_pool=card.visual_card_pool,
        should_show_in_card_library=card.should_show_in_card_library,
        has_custom_playability=card.has_custom_playability,
        has_custom_should_play=card.has_custom_should_play,
        has_custom_card_type=card.has_custom_card_type,
        has_custom_target_type=card.has_custom_target_type,
        multiplayer_constraint=(
            _static_metadata_override(card_id).multiplayer_constraint
            if _static_metadata_override(card_id) is not None
            else MULTIPLAYER_CONSTRAINT_NONE
        ),
    )


@lru_cache(maxsize=None)
def card_preview(card_id: CardId) -> CardInstance:
    """Return a probed card instance without advancing the global id counter."""
    registry = _factory_registry()
    if card_id in registry:
        factory, _, _ = registry[card_id]
        return _probe_factory(factory)
    return create_reference_card(card_id, allow_generation=False)


@lru_cache(maxsize=None)
def is_basic_strike_or_defend_card_id(card_id: CardId) -> bool:
    from sts2_env.cards.reference_static_metadata import reference_metadata_by_card_id

    metadata = reference_metadata_by_card_id().get(card_id)
    return (
        metadata is not None
        and metadata.rarity == CardRarity.BASIC
        and (CardTag.STRIKE in metadata.tags or CardTag.DEFEND in metadata.tags)
    )


def _coerce_rarity(rarity: str | CardRarity | None) -> CardRarity | None:
    if rarity is None or isinstance(rarity, CardRarity):
        return rarity
    return CardRarity[rarity.upper()]


def _matches_generation_context(
    metadata: CardMetadata,
    generation_context: GenerationContext | None,
) -> bool:
    if generation_context == "combat":
        return (
            metadata.can_be_generated_in_combat
            and metadata.rarity not in (CardRarity.BASIC, CardRarity.ANCIENT, CardRarity.EVENT)
        )
    if generation_context == "modifier":
        return metadata.can_be_generated_by_modifiers
    return True


def matches_player_count(card_id: CardId, *, is_multiplayer: bool | None) -> bool:
    if is_multiplayer is None:
        return True
    constraint = card_metadata(card_id).multiplayer_constraint
    if is_multiplayer:
        return constraint != MULTIPLAYER_CONSTRAINT_SINGLEPLAYER_ONLY
    return constraint != MULTIPLAYER_CONSTRAINT_MULTIPLAYER_ONLY


def is_multiplayer_only_card(card_id: CardId) -> bool:
    return card_metadata(card_id).multiplayer_constraint == MULTIPLAYER_CONSTRAINT_MULTIPLAYER_ONLY


def _registered_card_order(
    *,
    card_pool: CardPoolId | None,
    module_name: str | None,
    card_type: CardType | None,
    rarity_filter: CardRarity | None,
) -> tuple[CardId, ...]:
    if card_pool is not None:
        return _CARD_POOLS_BY_ID[card_pool]
    if module_name is not None:
        return _MODULE_CARD_POOL_ORDER.get(module_name, ALL_CARD_POOL)
    if card_type in _TYPE_CARD_POOL_ORDER:
        return _TYPE_CARD_POOL_ORDER[card_type]
    if rarity_filter in _RARITY_CARD_POOL_ORDER:
        return _RARITY_CARD_POOL_ORDER[rarity_filter]
    return ALL_CARD_POOL


def eligible_character_cards(
    character_id: str,
    *,
    card_type: CardType | None = None,
    rarity: str | CardRarity | None = None,
    require_keyword: str | None = None,
    generation_context: GenerationContext | None = "combat",
    is_multiplayer: bool | None = None,
) -> list[CardId]:
    """Return eligible class cards from the owning character's card pool."""
    rarity_filter = _coerce_rarity(rarity)
    config = get_character(character_id)
    eligible: list[CardId] = []

    for card_id in config.card_pool:
        if not matches_player_count(card_id, is_multiplayer=is_multiplayer):
            continue
        try:
            metadata = card_metadata(card_id)
        except KeyError:
            continue
        if not _matches_generation_context(metadata, generation_context):
            continue
        if card_type is not None and metadata.card_type is not card_type:
            continue
        if rarity_filter is not None and metadata.rarity is not rarity_filter:
            continue
        if require_keyword is not None and require_keyword not in metadata.keywords:
            continue
        eligible.append(card_id)

    return eligible


def eligible_registered_cards(
    *,
    card_pool: CardPoolId | None = None,
    module_name: str | None = None,
    card_type: CardType | None = None,
    rarity: str | CardRarity | None = None,
    exclude_ids: set[CardId] | None = None,
    generation_context: GenerationContext | None = "combat",
    is_multiplayer: bool | None = None,
) -> list[CardId]:
    """Return registered cards filtered by source module, reference pool, and metadata."""
    rarity_filter = _coerce_rarity(rarity)
    exclude_ids = exclude_ids or set()
    eligible: list[CardId] = []

    registry = _factory_registry()
    for card_id in _registered_card_order(
        card_pool=card_pool,
        module_name=module_name,
        card_type=card_type,
        rarity_filter=rarity_filter,
    ):
        if card_id in exclude_ids:
            continue
        if not matches_player_count(card_id, is_multiplayer=is_multiplayer):
            continue
        if card_id in _REGISTERED_POOL_EXCLUDED_CARD_IDS:
            continue
        if card_id == CardId.GENERIC:
            continue
        source_module = registry.get(card_id, (None, None, _reference_source_module(card_id)))[2]
        if source_module is None:
            continue
        if module_name is not None and source_module != module_name:
            continue
        try:
            metadata = card_metadata(card_id)
        except KeyError:
            continue
        if not _matches_generation_context(metadata, generation_context):
            continue
        if card_type is not None and metadata.card_type is not card_type:
            continue
        if rarity_filter is not None and metadata.rarity is not rarity_filter:
            continue
        eligible.append(card_id)

    return eligible


def create_cards_from_ids(
    card_ids: list[CardId],
    rng: Rng,
    count: int,
    *,
    distinct: bool = True,
) -> list[CardInstance]:
    """Create cards from a provided id pool."""
    if not card_ids or count <= 0:
        return []

    if distinct:
        shuffled_ids = list(card_ids)
        rng.shuffle(shuffled_ids)
        chosen_ids = shuffled_ids[:count]
    else:
        chosen_ids = [rng.choice(card_ids) for _ in range(count)]
    return [create_card(card_id) for card_id in chosen_ids]


def create_character_cards(
    character_id: str,
    rng: Rng,
    count: int,
    *,
    card_type: CardType | None = None,
    rarity: str | CardRarity | None = None,
    require_keyword: str | None = None,
    distinct: bool = True,
    generation_context: GenerationContext | None = "combat",
    is_multiplayer: bool | None = None,
) -> list[CardInstance]:
    """Create cards from the owning character pool with optional filtering."""
    eligible = eligible_character_cards(
        character_id,
        card_type=card_type,
        rarity=rarity,
        require_keyword=require_keyword,
        generation_context=generation_context,
        is_multiplayer=is_multiplayer,
    )
    return create_cards_from_ids(eligible, rng, count, distinct=distinct)


def create_distinct_character_cards(
    character_id: str,
    rng: Rng,
    count: int,
    *,
    card_type: CardType | None = None,
    rarity: str | CardRarity | None = None,
    require_keyword: str | None = None,
    generation_context: GenerationContext | None = "combat",
    is_multiplayer: bool | None = None,
) -> list[CardInstance]:
    """Create up to `count` distinct cards from a character pool."""
    return create_character_cards(
        character_id,
        rng,
        count,
        card_type=card_type,
        rarity=rarity,
        require_keyword=require_keyword,
        generation_context=generation_context,
        is_multiplayer=is_multiplayer,
        distinct=True,
    )


def eligible_transform_cards(
    original: CardInstance,
    *,
    character_id: str,
    generation_context: GenerationContext | None = None,
    is_multiplayer: bool | None = None,
) -> list[CardId]:
    """Return the decompiled-style transform pool for a specific original card."""
    registry = _factory_registry()
    source_module = registry.get(original.card_id, (None, None, _reference_source_module(original.card_id)))[2]
    use_colorless_pool = (
        original.card_type == CardType.QUEST
        or original.rarity in {CardRarity.EVENT, CardRarity.ANCIENT}
        or source_module == "sts2_env.cards.colorless"
    )
    if use_colorless_pool:
        candidates = eligible_registered_cards(
            card_pool=CardPoolId.COLORLESS,
            generation_context=generation_context,
            is_multiplayer=is_multiplayer,
        )
    elif original.card_id in set(get_character(character_id).card_pool):
        candidates = eligible_character_cards(
            character_id,
            generation_context=generation_context,
            is_multiplayer=is_multiplayer,
        )
    elif source_module is not None:
        candidates = eligible_registered_cards(
            module_name=source_module,
            generation_context=generation_context,
            is_multiplayer=is_multiplayer,
        )
    else:
        candidates = eligible_character_cards(
            character_id,
            generation_context=generation_context,
            is_multiplayer=is_multiplayer,
        )

    if original.rarity not in {CardRarity.EVENT, CardRarity.ANCIENT}:
        candidates = [
            card_id
            for card_id in candidates
            if card_metadata(card_id).rarity in {CardRarity.COMMON, CardRarity.UNCOMMON, CardRarity.RARE}
        ]

    candidates = [card_id for card_id in candidates if card_id != original.card_id]
    if is_basic_strike_or_defend_card_id(original.card_id):
        candidates = [
            card_id
            for card_id in candidates
            if not is_basic_strike_or_defend_card_id(card_id)
        ]
    if candidates:
        return candidates
    return [
        card_id
        for card_id in eligible_character_cards(
            character_id,
            generation_context=generation_context,
            is_multiplayer=is_multiplayer,
        )
        if card_id != original.card_id and card_metadata(card_id).rarity in {CardRarity.COMMON, CardRarity.UNCOMMON, CardRarity.RARE}
    ]


def create_transform_card(
    original: CardInstance,
    *,
    character_id: str,
    rng: Rng,
    generation_context: GenerationContext | None = None,
    is_multiplayer: bool | None = None,
) -> CardInstance:
    """Create a run-level transform result using the original-card-specific pool."""
    candidates = eligible_transform_cards(
        original,
        character_id=character_id,
        generation_context=generation_context,
        is_multiplayer=is_multiplayer,
    )
    if not candidates:
        raise ValueError(f"No valid transform candidates for {original.card_id.name}")
    return create_card(rng.choice(candidates))
