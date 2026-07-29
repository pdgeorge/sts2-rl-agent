"""Static card metadata parsed from decompiled card models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from sts2_env.core.card_pools import CardPoolId
from sts2_env.core.enums import CardId, CardRarity, CardTag, CardType, OrbEvokeType, TargetType


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CARD_DIR = Path("decompiled/MegaCrit.Sts2.Core.Models.Cards")

# The committed decompiled/ tree is a snapshot of whatever build it was taken
# from, which quietly turns this reference -- and every test comparing against it
# -- into a statement about the past. "4,609 tests pass" then means "the sim
# agrees with a decompile from some earlier patch", not "the sim matches the
# game". Point STS2_DECOMPILED_ROOT at a fresh decompile of the installed build
# and the suite becomes a live parity check instead.
#
# Not switchable on yet: a fresh tree contains cards CardId has no member for, and
# card_id_for_reference_class raises on the first one, which kills collection
# before a single test runs. Those members have to be added first.
DECOMPILED_ROOT_ENV = "STS2_DECOMPILED_ROOT"


def reference_card_dir() -> Path:
    override = os.environ.get(DECOMPILED_ROOT_ENV)
    if override:
        return Path(override) / REFERENCE_CARD_DIR.name
    return REPO_ROOT / REFERENCE_CARD_DIR


MULTIPLAYER_CONSTRAINT_NONE = "None"
MULTIPLAYER_CONSTRAINT_MULTIPLAYER_ONLY = "MultiplayerOnly"
MULTIPLAYER_CONSTRAINT_SINGLEPLAYER_ONLY = "SingleplayerOnly"
BASE_CONSTRUCTOR_RE = re.compile(
    r":\s*base\(\s*"
    r"(?P<cost>-?\d+)\s*,\s*"
    r"CardType\.(?P<card_type>[A-Za-z]+)\s*,\s*"
    r"CardRarity\.(?P<rarity>[A-Za-z]+)\s*,\s*"
    r"TargetType\.(?P<target_type>[A-Za-z]+)",
    re.DOTALL,
)
SHOULD_SHOW_IN_CARD_LIBRARY_FALSE_RE = re.compile(r"shouldShowInCardLibrary\s*:\s*false")
CUSTOM_PLAYABILITY_RE = re.compile(r"protected\s+override\s+bool\s+IsPlayable\s*=>")
CUSTOM_SHOULD_PLAY_RE = re.compile(r"public\s+override\s+bool\s+ShouldPlay\s*\(")
CUSTOM_CARD_TYPE_RE = re.compile(r"public\s+override\s+CardType\s+Type\s*=>")
CUSTOM_TARGET_TYPE_RE = re.compile(r"public\s+override\s+TargetType\s+TargetType\b")
ENERGY_X_RE = re.compile(r"override\s+bool\s+HasEnergyCostX\s*=>\s*true\s*;")
STAR_X_RE = re.compile(r"override\s+bool\s+HasStarCostX\s*=>\s*true\s*;")
STAR_COST_RE = re.compile(r"override\s+int\s+CanonicalStarCost\s*=>\s*(?P<star_cost>-?\d+)\s*;")
MAX_UPGRADE_LEVEL_RE = re.compile(
    r"override\s+int\s+MaxUpgradeLevel\s*=>\s*(?P<max_upgrade_level>-?\d+)\s*;"
)
CAN_BE_GENERATED_IN_COMBAT_RE = re.compile(
    r"override\s+bool\s+CanBeGeneratedInCombat\s*=>\s*(?P<value>true|false)\s*;"
)
CAN_BE_GENERATED_BY_MODIFIERS_RE = re.compile(
    r"override\s+bool\s+CanBeGeneratedByModifiers\s*=>\s*(?P<value>true|false)\s*;"
)
HAS_TURN_END_IN_HAND_EFFECT_RE = re.compile(
    r"override\s+bool\s+HasTurnEndInHandEffect\s*=>\s*(?P<value>true|false)\s*;"
)
GAINS_BLOCK_RE = re.compile(r"override\s+bool\s+GainsBlock\s*=>\s*(?P<value>true|false)\s*;")
ORB_EVOKE_TYPE_RE = re.compile(
    r"override\s+OrbEvokeType\s+OrbEvokeType\s*=>\s*OrbEvokeType\.(?P<value>[A-Za-z]+)\s*;"
)
VISUAL_CARD_POOL_RE = re.compile(
    r"override\s+CardPoolModel\s+VisualCardPool\s*=>\s*"
    r"ModelDb\.CardPool<(?P<pool>[A-Za-z]+)CardPool>\(\)\s*;"
)
MULTIPLAYER_CONSTRAINT_RE = re.compile(
    r"override\s+CardMultiplayerConstraint\s+MultiplayerConstraint\s*=>\s*"
    r"CardMultiplayerConstraint\.(?P<constraint>[A-Za-z]+)\s*;"
)
ENERGY_COST_UPGRADE_RE = re.compile(
    r"base\.EnergyCost\.UpgradeBy\((?P<delta>-?\d+(?:\.0+)?m?)\)"
)
ADD_KEYWORD_RE = re.compile(r"AddKeyword\(CardKeyword\.(?P<keyword>[A-Za-z]+)\)")
REMOVE_KEYWORD_RE = re.compile(r"RemoveKeyword\(CardKeyword\.(?P<keyword>[A-Za-z]+)\)")
CARD_KEYWORD_RE = re.compile(r"CardKeyword\.(?P<keyword>[A-Za-z]+)")
CARD_TAG_RE = re.compile(r"CardTag\.(?P<tag>[A-Za-z]+)")
CAMEL_WORD_BOUNDARY_RE = re.compile(r"(.)([A-Z][a-z]+)")
LOWER_TO_UPPER_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
VAR_CONSTRUCTOR_RE = re.compile(
    r"new\s+"
    r"(?P<type>[A-Za-z][A-Za-z0-9_]*)"
    r"(?:<(?P<generic>[A-Za-z][A-Za-z0-9_]*)>)?"
    r"\s*\("
)
INTEGER_LITERAL_RE = re.compile(r"-?\d+(?:\.0+)?m?")
DYNAMIC_VAR_UPGRADE_RE = re.compile(
    r"base\.DynamicVars"
    r"(?:\.(?P<property>[A-Za-z][A-Za-z0-9_]*)|\[\"(?P<name>[^\"]+)\"\])"
    r"\.UpgradeValueBy\((?P<delta>-?\d+(?:\.0+)?m?)\)"
)
REFERENCE_CLASS_ALIASES = {
    "Null": ("NULL_CARD",),
    "Sloth": ("SLOTH_STATUS",),
}
DYNAMIC_VAR_DEFAULT_NAMES = {
    "BlockVar": "Block",
    "CalculationBaseVar": "CalculationBase",
    "CalculationExtraVar": "CalculationExtra",
    "CardsVar": "Cards",
    "DamageVar": "Damage",
    "EnergyVar": "Energy",
    "ExtraDamageVar": "ExtraDamage",
    "ForgeVar": "Forge",
    "GoldVar": "Gold",
    "HealVar": "Heal",
    "HpLossVar": "HpLoss",
    "MaxHpVar": "MaxHp",
    "OstyDamageVar": "OstyDamage",
    "RepeatVar": "Repeat",
    "StarsVar": "Stars",
    "SummonVar": "Summon",
}
DYNAMIC_VAR_PROPERTY_NAMES = {
    "Block": "Block",
    "CalculationBase": "CalculationBase",
    "CalculationExtra": "CalculationExtra",
    "Cards": "Cards",
    "Damage": "Damage",
    "Dexterity": "DexterityPower",
    "Doom": "DoomPower",
    "Energy": "Energy",
    "ExtraDamage": "ExtraDamage",
    "Forge": "Forge",
    "Gold": "Gold",
    "Heal": "Heal",
    "HpLoss": "HpLoss",
    "MaxHp": "MaxHp",
    "OstyDamage": "OstyDamage",
    "Poison": "PoisonPower",
    "Repeat": "Repeat",
    "Stars": "Stars",
    "Strength": "StrengthPower",
    "Summon": "Summon",
    "Vulnerable": "VulnerablePower",
    "Weak": "WeakPower",
}
DYNAMIC_VAR_TYPES_WITH_DYNAMIC_VALUE = frozenset({
    "CalculatedBlockVar",
    "CalculatedDamageVar",
    "CalculatedVar",
})
REFERENCE_DYNAMIC_VAR_ALIASES = {
    "arsenal_power": "arsenal",
    "black_hole_power": "black_hole",
    "block_for_stars": "block_for_stars",
    "calcify_power": "calcify",
    "calculation_base": "calc_base",
    "calculation_extra": "calc_extra",
    "countdown_power": "countdown",
    "danse_macabre_power": "danse_macabre",
    "debilitate_power": "debilitate",
    "devour_life_power": "devour_life",
    "dexterity_power": "dexterity",
    "doom_power": "doom",
    "knockdown_power": "knockdown",
    "lethality_power": "lethality",
    "neurosurge_power": "neurosurge",
    "parry_power": "parry",
    "plating_power": "plating",
    "prep_time_power": "prep_time",
    "rolling_boulder_power": "rolling_boulder",
    "sentry_mode_power": "sentry_mode",
    "sic_em_power": "sic_em",
    "sleight_of_flesh_power": "sleight_of_flesh",
    "stars_per_turn": "stars_per_turn",
    "strength_power": "strength",
    "vigor_power": "vigor",
    "vulnerable_power": "vulnerable",
    "weak_power": "weak",
}


@dataclass(frozen=True)
class ReferenceCardStaticMetadata:
    card_id: CardId
    cost: int
    card_type: CardType
    target_type: TargetType
    rarity: CardRarity
    keywords: frozenset[str]
    tags: frozenset[CardTag]
    has_energy_cost_x: bool
    star_cost: int
    has_star_cost_x: bool
    max_upgrade_level: int
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


def snake_case(name: str) -> str:
    first = CAMEL_WORD_BOUNDARY_RE.sub(r"\1_\2", name)
    return LOWER_TO_UPPER_BOUNDARY_RE.sub(r"\1_\2", first).lower()


def card_id_for_reference_class(name: str) -> CardId:
    snake_name = snake_case(name).upper()
    aliases = {snake_name, f"{snake_name}_CARD", f"{snake_name}_STATUS"}
    aliases.update(REFERENCE_CLASS_ALIASES.get(name, ()))
    if name.endswith("Card"):
        stripped = name.removesuffix("Card")
        stripped_snake = snake_case(stripped).upper()
        aliases.update({stripped_snake, f"{stripped_snake}_CARD", f"{stripped_snake}_STATUS"})
    for alias in aliases:
        if alias in CardId.__members__:
            return CardId[alias]
    raise KeyError(f"No CardId alias for reference card class {name}")


def _property_expression(source: str, property_name: str) -> str:
    start = source.find(property_name)
    if start < 0:
        return ""
    end = source.find(";", start)
    if end < 0:
        return source[start:]
    return source[start : end + 1]


def _coerce_reference_rarity(name: str) -> CardRarity:
    if name == "Token":
        return CardRarity.STATUS
    return CardRarity[name.upper()]


def reference_metadata_from_source(path: Path) -> ReferenceCardStaticMetadata:
    source = path.read_text()
    constructor_match = BASE_CONSTRUCTOR_RE.search(source)
    if constructor_match is None:
        raise ValueError(f"{path} is missing a literal CardModel base constructor")

    keywords = frozenset(
        snake_case(keyword)
        for keyword in CARD_KEYWORD_RE.findall(_property_expression(source, "CanonicalKeywords"))
    )
    tags = frozenset(
        CardTag[snake_case(tag).upper()]
        for tag in CARD_TAG_RE.findall(_property_expression(source, "CanonicalTags"))
    )
    star_cost_match = STAR_COST_RE.search(source)
    max_upgrade_level_match = MAX_UPGRADE_LEVEL_RE.search(source)
    combat_generation_match = CAN_BE_GENERATED_IN_COMBAT_RE.search(source)
    modifier_generation_match = CAN_BE_GENERATED_BY_MODIFIERS_RE.search(source)
    turn_end_in_hand_match = HAS_TURN_END_IN_HAND_EFFECT_RE.search(source)
    gains_block_match = GAINS_BLOCK_RE.search(source)
    orb_evoke_type_match = ORB_EVOKE_TYPE_RE.search(source)
    visual_card_pool_match = VISUAL_CARD_POOL_RE.search(source)
    multiplayer_constraint_match = MULTIPLAYER_CONSTRAINT_RE.search(source)

    return ReferenceCardStaticMetadata(
        card_id=card_id_for_reference_class(path.stem),
        cost=int(constructor_match.group("cost")),
        card_type=CardType[constructor_match.group("card_type").upper()],
        target_type=TargetType[snake_case(constructor_match.group("target_type")).upper()],
        rarity=_coerce_reference_rarity(constructor_match.group("rarity")),
        keywords=keywords,
        tags=tags,
        has_energy_cost_x=ENERGY_X_RE.search(source) is not None,
        star_cost=int(star_cost_match.group("star_cost")) if star_cost_match is not None else 0,
        has_star_cost_x=STAR_X_RE.search(source) is not None,
        max_upgrade_level=(
            int(max_upgrade_level_match.group("max_upgrade_level"))
            if max_upgrade_level_match is not None
            else 1
        ),
        can_be_generated_in_combat=(
            combat_generation_match.group("value") != "false"
            if combat_generation_match is not None
            else True
        ),
        can_be_generated_by_modifiers=(
            modifier_generation_match.group("value") != "false"
            if modifier_generation_match is not None
            else True
        ),
        has_turn_end_in_hand_effect=(
            turn_end_in_hand_match.group("value") == "true"
            if turn_end_in_hand_match is not None
            else False
        ),
        gains_block=(
            gains_block_match.group("value") == "true"
            if gains_block_match is not None
            else False
        ),
        orb_evoke_type=(
            OrbEvokeType[snake_case(orb_evoke_type_match.group("value")).upper()]
            if orb_evoke_type_match is not None
            else OrbEvokeType.NONE
        ),
        visual_card_pool=(
            CardPoolId[snake_case(visual_card_pool_match.group("pool")).upper()]
            if visual_card_pool_match is not None
            else None
        ),
        should_show_in_card_library=SHOULD_SHOW_IN_CARD_LIBRARY_FALSE_RE.search(source) is None,
        has_custom_playability=CUSTOM_PLAYABILITY_RE.search(source) is not None,
        has_custom_should_play=CUSTOM_SHOULD_PLAY_RE.search(source) is not None,
        has_custom_card_type=CUSTOM_CARD_TYPE_RE.search(source) is not None,
        has_custom_target_type=CUSTOM_TARGET_TYPE_RE.search(source) is not None,
        multiplayer_constraint=(
            multiplayer_constraint_match.group("constraint")
            if multiplayer_constraint_match is not None
            else MULTIPLAYER_CONSTRAINT_NONE
        ),
    )


def upgraded_reference_metadata_from_source(path: Path) -> ReferenceCardStaticMetadata:
    metadata = reference_metadata_from_source(path)
    source = path.read_text()
    cost = metadata.cost + _energy_cost_upgrade_delta(source)
    keywords = set(metadata.keywords)
    body = _on_upgrade_body(source)
    keywords.update(snake_case(keyword) for keyword in ADD_KEYWORD_RE.findall(body))
    keywords.difference_update(snake_case(keyword) for keyword in REMOVE_KEYWORD_RE.findall(body))
    return replace(metadata, cost=cost, keywords=frozenset(keywords))


def _energy_cost_upgrade_delta(source: str) -> int:
    return sum(
        _integer_literal(match.group("delta"))
        for match in ENERGY_COST_UPGRADE_RE.finditer(_on_upgrade_body(source))
    )


def reference_dynamic_vars_from_source(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for var_type, generic_type, arguments in _canonical_var_constructors(path.read_text()):
        parsed = _dynamic_var_key_value(var_type, generic_type, arguments)
        if parsed is None:
            continue
        key, value = parsed
        if key in result:
            raise ValueError(f"{path} has duplicate dynamic var key {key!r}")
        result[key] = value
    return result


def upgraded_reference_dynamic_vars_from_source(path: Path) -> dict[str, int]:
    result = reference_dynamic_vars_from_source(path)
    for key, delta in _dynamic_var_upgrade_deltas(path.read_text()).items():
        result[key] = result.get(key, 0) + delta
    return result


def _dynamic_var_upgrade_deltas(source: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for match in DYNAMIC_VAR_UPGRADE_RE.finditer(_on_upgrade_body(source)):
        raw_name = match.group("name")
        if raw_name is None:
            raw_name = DYNAMIC_VAR_PROPERTY_NAMES.get(
                match.group("property"),
                match.group("property"),
            )
        key = _reference_dynamic_var_key(raw_name)
        result[key] = result.get(key, 0) + _integer_literal(match.group("delta"))
    return result


def _on_upgrade_body(source: str) -> str:
    match = re.search(r"protected\s+override\s+void\s+OnUpgrade\s*\(\)\s*\{", source)
    if match is None:
        return ""
    start = match.end()
    close_brace = _matching_delimiter(source, start - 1, "{", "}")
    if close_brace is None:
        return source[start:]
    return source[start:close_brace]


def _canonical_var_constructors(source: str) -> list[tuple[str, str | None, str]]:
    expression = _canonical_vars_expression(source)
    constructors: list[tuple[str, str | None, str]] = []
    search_from = 0
    while True:
        match = VAR_CONSTRUCTOR_RE.search(expression, search_from)
        if match is None:
            break
        open_paren = match.end() - 1
        close_paren = _matching_delimiter(expression, open_paren, "(", ")")
        if close_paren is None:
            break
        constructors.append(
            (
                match.group("type"),
                match.group("generic"),
                expression[open_paren + 1 : close_paren],
            )
        )
        search_from = close_paren + 1
    return constructors


def _canonical_vars_expression(source: str) -> str:
    property_start = source.find("CanonicalVars")
    if property_start < 0:
        return ""
    expression_start = source.find("=>", property_start)
    if expression_start < 0:
        return ""
    expression_start += 2
    expression_end = _expression_statement_end(source, expression_start)
    return source[expression_start:expression_end]


def _expression_statement_end(source: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ";" and depth == 0:
            return index
    return len(source)


def _matching_delimiter(source: str, start: int, opener: str, closer: str) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_arguments(arguments: str) -> list[str]:
    result: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    current: list[str] = []
    for char in arguments:
        if in_string:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        result.append("".join(current).strip())
    return result


def _dynamic_var_key_value(
    var_type: str,
    generic_type: str | None,
    arguments: str,
) -> tuple[str, int] | None:
    if var_type in DYNAMIC_VAR_TYPES_WITH_DYNAMIC_VALUE:
        return None
    args = _split_arguments(arguments)
    if not args:
        return None

    if args[0].startswith('"') and args[0].endswith('"'):
        name = args[0][1:-1]
        if len(args) < 2:
            return None
        value_text = args[1]
    elif var_type == "PowerVar" and generic_type is not None:
        name = generic_type
        value_text = args[0]
    else:
        name = DYNAMIC_VAR_DEFAULT_NAMES.get(var_type)
        value_text = args[0]

    if name is None:
        return None
    value = _integer_literal(value_text)
    if value is None:
        return None
    return _reference_dynamic_var_key(name), value


def _reference_dynamic_var_key(name: str) -> str:
    key = snake_case(name)
    return REFERENCE_DYNAMIC_VAR_ALIASES.get(key, key)


def _integer_literal(value_text: str) -> int | None:
    normalized = value_text.strip()
    if INTEGER_LITERAL_RE.fullmatch(normalized) is None:
        return None
    normalized = normalized.removesuffix("m")
    if "." in normalized:
        return int(float(normalized))
    return int(normalized)


@lru_cache(maxsize=1)
def reference_metadata_by_card_id() -> dict[CardId, ReferenceCardStaticMetadata]:
    return {
        metadata.card_id: metadata
        for metadata in (
            reference_metadata_from_source(path)
            for path in sorted(reference_card_dir().glob("*.cs"))
        )
    }


@lru_cache(maxsize=1)
def upgraded_reference_metadata_by_card_id() -> dict[CardId, ReferenceCardStaticMetadata]:
    return {
        metadata.card_id: metadata
        for metadata in (
            upgraded_reference_metadata_from_source(path)
            for path in sorted(reference_card_dir().glob("*.cs"))
        )
    }


@lru_cache(maxsize=1)
def reference_dynamic_vars_by_card_id() -> dict[CardId, dict[str, int]]:
    return {
        card_id_for_reference_class(path.stem): reference_dynamic_vars_from_source(path)
        for path in sorted(reference_card_dir().glob("*.cs"))
    }


@lru_cache(maxsize=1)
def upgraded_reference_dynamic_vars_by_card_id() -> dict[CardId, dict[str, int]]:
    return {
        card_id_for_reference_class(path.stem): upgraded_reference_dynamic_vars_from_source(path)
        for path in sorted(reference_card_dir().glob("*.cs"))
    }
