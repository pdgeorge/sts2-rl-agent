"""Render a card as text, deterministically, from code the simulator actually runs.

This text is the input to the embedding model, so it decides what the vectors
mean. Everything downstream inherits its blind spots.

WHY IT IS BUILT FROM CODE AND NOT FROM PROSE

Three text sources were available and two were rejected:

* ``docs/CARDS_REFERENCE.md`` has an ``Effect:`` field, but ``_reference_cards()``
  parses ID, Color, Cost, Type, Rarity, Target, Keywords, Tags, Vars and Upgrade
  -- and *not* Effect. Nothing consumes it, so nothing would notice it going
  wrong. It is also sometimes circular: Barricade's reads "Apply Barricade to
  self", which never mentions that block stops being cleared. ``factory.py``
  already records that this file fell behind once while the tests read it as an
  oracle and stayed green.
* The HuggingFace card set has real game text, but it is a third-party snapshot
  from May 2026 that cannot regenerate when the game patches.

What is left is the strongest source anyway: **the simulator's own
implementation**. A card's registered effect function is what the card *does*
here, so text derived from it cannot drift from behaviour -- it is behaviour.

WHERE EACH LINE COMES FROM

===================  =========================================================
 identity/cost/type   ``reference_static_metadata`` + ``card_preview``
 damage/block         ``card_preview`` (values come from the decompile)
 applies              ``PowerId`` references in the effect function's AST
 does                 method names called on ``CombatState`` in that AST
 power semantics      the power class: type, stacking, and which of
                      ``PowerInstance``'s ~78 hooks it overrides
 upgrade              diff of ``create_card(id, False)`` against
                      ``create_card(id, True)``
===================  =========================================================

The power-hook line is what rescues cards whose whole identity lives in code.
Barricade has no damage, no block and no effect vars; every structured field it
has is shared with any other 3-cost self-target Power. Introspecting its power
class yields ``should_clear_block``, which is exactly what makes it Barricade.

KNOWN LIMIT: STATUSES AND CURSES

37 of 577 cards (6.4%) render with no mechanical line -- every one of them a
status or a curse. For most that is correct: a Wound genuinely does nothing when
played, and ``keywords: unplayable`` says so.

For a handful it under-describes. Burn damages you at end of turn, but that
behaviour is special-cased in the combat engine rather than registered in any
``_CARD_*_HOOKS`` table, so introspection cannot reach it, and Burn currently
renders much like Wound.

The mitigation is the card name, which this template includes deliberately: the
embedding model knows what "burn" connotes and separates it from "wound" on the
text alone. That is a real reason to keep names in, beyond disambiguation.

If this turns out to matter -- statuses and curses are exactly the cards that
make a deck worse, so deck evaluation may be sensitive to it -- the fix is to
describe them from the engine's special-case paths rather than to guess.

FROZEN

``TEMPLATE_VERSION`` is part of the embedding artifact's identity. Changing what
this renders changes every vector, which invalidates every checkpoint trained
against them. Bump it, rebuild the table, retrain -- or do not change it.

Output must also be **deterministic**: same code in, same bytes out. Every set is
sorted before rendering. A template that reordered its own output between runs
would silently produce two different vector tables from one codebase.
"""

from __future__ import annotations

import ast
import inspect
import logging
from functools import lru_cache

from sts2_env.core.creature import get_power_class
from sts2_env.core.enums import CardId, PowerId
from sts2_env.powers.base import PowerInstance

logger = logging.getLogger(__name__)

TEMPLATE_VERSION = 1
"""Bump only alongside a rebuild of the vector table and a retrain."""

NONE = "none"

# A DENYLIST, deliberately, not an allowlist.
#
# An allowlist of "interesting" calls has to be guessed, and a first attempt at
# one here invented names the codebase does not use (deal_damage, draw_cards)
# while missing the ones it does (apply_damage, _draw_cards, _gain_block). Worse,
# it fails the way this project keeps getting hurt: a helper added by a future
# patch would be silently dropped, and the text would quietly get thinner with
# nothing to show for it. Same rot as the hardcoded keyword lists in
# deck_features.py.
#
# A denylist fails the other way. Something new shows up in the text and is at
# worst noise, which is visible and fixable. Named from an actual survey of all
# 577 effect functions.
_IGNORED_CALLS = frozenset({
    # builtins and plumbing
    "range", "list", "len", "max", "min", "sum", "getattr", "setattr", "hasattr",
    "isinstance", "int", "str", "bool", "float", "dict", "set", "tuple", "sorted",
    "reversed", "enumerate", "zip", "abs", "round", "any", "all", "print", "next",
    "iter", "super", "type",
    # container / string methods
    "get", "append", "extend", "items", "keys", "values", "copy", "format",
    "join", "add", "remove", "pop", "index", "count", "lower", "upper", "strip",
    "split", "insert", "clear", "update",
    # rng
    "choice", "shuffle", "next_int", "next_float", "next_bool", "randint",
    # registration decorators and owner/state lookup
    "register_effect", "register_late_effect", "register_chosen_hook",
    "owner", "combat_player_state_for", "acting_player_view",
})


def _sorted_unique(values) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@lru_cache(maxsize=None)
def _effect_ast(card_id: CardId) -> ast.AST | None:
    """Parse the registered effect function for a card, if it has one."""
    from sts2_env.cards.registry import _CARD_EFFECTS

    fn = _CARD_EFFECTS.get(card_id)
    if fn is None:
        return None
    try:
        return ast.parse(inspect.cleandoc(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        # A dynamically constructed effect has no retrievable source. Degrading
        # to "no applies/does lines" is correct; raising would make one odd card
        # break the whole table build.
        logger.debug("No source for the effect of %s", card_id.name)
        return None


def powers_applied(card_id: CardId) -> tuple[str, ...]:
    """``PowerId.X`` references inside the card's effect function."""
    tree = _effect_ast(card_id)
    if tree is None:
        return ()
    return _sorted_unique(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "PowerId"
    )


def actions_taken(card_id: CardId) -> tuple[str, ...]:
    """What the effect function calls -- both methods and module-level helpers.

    Both matter: ``apply_power_to`` is a method on ``CombatState``, while damage
    and block go through module-level helpers like ``calculate_damage`` and
    ``_gain_block``. Capturing only one kind loses half the vocabulary, which is
    how Body Slam first rendered with no ``does:`` line at all.

    Leading underscores are stripped so ``_gain_block`` and ``gain_block`` are
    one concept rather than two.
    """
    tree = _effect_ast(card_id)
    if tree is None:
        return ()

    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.append(node.func.id)

    cleaned = (name.lstrip("_") for name in names)
    return _sorted_unique(
        name for name in cleaned if name and name not in _IGNORED_CALLS
    )


# Game resources a card can read. Unlike _IGNORED_CALLS this IS an explicit set,
# and the asymmetry is deliberate: method names are implementation vocabulary
# that grows every patch, so guessing at them rots. These are the game's
# fundamental resources -- block, HP, energy, the piles -- which are stable
# because they are the rules, not the code.
#
# This line is what rescues cards whose damage is computed rather than declared.
# Body Slam has no base damage and calls the same two helpers as Whirlwind; the
# only thing distinguishing them is `base = owner.block`.
_STATE_READS = frozenset({
    "block", "current_hp", "max_hp", "energy", "max_energy",
    "hand", "draw_pile", "discard_pile", "exhaust_pile",
    "powers", "orbs", "stars", "gold", "potions", "relics", "deck",
})


def state_read(card_id: CardId) -> tuple[str, ...]:
    """Which game resources the effect reads. Synergy information, essentially:
    a card that reads ``block`` belongs with the cards that generate it."""
    tree = _effect_ast(card_id)
    if tree is None:
        return ()
    return _sorted_unique(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in _STATE_READS
    )


@lru_cache(maxsize=None)
def power_summary(power_name: str) -> str | None:
    """A power described by its type, stacking, and which hooks it overrides.

    The hooks are the point. They are read off the class rather than any
    document, so a power whose behaviour changes says so here without anyone
    remembering to update prose.
    """
    try:
        power_id = PowerId[power_name]
    except KeyError:
        return None
    cls = get_power_class(power_id)
    if cls is None:
        return None

    hooks = _sorted_unique(
        name
        for name in dir(PowerInstance)
        if not name.startswith("_")
        and callable(getattr(PowerInstance, name, None))
        and getattr(cls, name, None) is not getattr(PowerInstance, name, None)
    )
    power_type = getattr(getattr(cls, "power_type", None), "name", "unknown").lower()
    stack = getattr(getattr(cls, "stack_type", None), "name", "unknown").lower()

    text = f"{power_name} ({power_type}, {stack} stack)"
    if hooks:
        text += " hooks: " + ", ".join(hooks)
    return text


@lru_cache(maxsize=None)
def _card_colour(card_id: CardId) -> str:
    """Character pool, via the same lookup ``factory.py`` routes on."""
    from sts2_env.cards.factory import _reference_candidates, _reference_cards

    table = _reference_cards()
    for candidate in _reference_candidates(card_id):
        entry = table.get(candidate)
        if entry is not None and entry.color:
            return entry.color
    return "unknown"


def upgrade_delta(card_id: CardId) -> tuple[str, ...]:
    """What upgrading changes, by diffing the two constructed cards.

    Derived rather than described, so it cannot disagree with the card the
    simulator actually builds.
    """
    from sts2_env.cards.factory import create_card

    try:
        base = create_card(card_id, upgraded=False)
        up = create_card(card_id, upgraded=True)
    except Exception:  # noqa: BLE001 -- one uncreatable card must not stop the build
        logger.debug("Could not construct both variants of %s", card_id.name)
        return ()

    changes: list[str] = []
    for field in ("cost", "base_damage", "base_block"):
        before, after = getattr(base, field, None), getattr(up, field, None)
        if before != after and after is not None and before is not None:
            changes.append(f"{field} {before}->{after}")

    before_vars = dict(getattr(base, "effect_vars", {}) or {})
    after_vars = dict(getattr(up, "effect_vars", {}) or {})
    for key in sorted(set(before_vars) | set(after_vars)):
        if before_vars.get(key) != after_vars.get(key):
            changes.append(f"{key} {before_vars.get(key)}->{after_vars.get(key)}")

    return tuple(changes)


def render_card_text(card_id: CardId) -> str:
    """The frozen template. Deterministic: same code in, same bytes out."""
    from sts2_env.cards.factory import card_preview
    from sts2_env.cards.reference_static_metadata import reference_metadata_by_card_id

    preview = card_preview(card_id)
    meta = reference_metadata_by_card_id().get(card_id)

    card_type = getattr(getattr(preview, "card_type", None), "name", "unknown").title()
    rarity = getattr(getattr(meta, "rarity", None), "name", "unknown").title()
    target = getattr(getattr(preview, "target_type", None), "name", "unknown")
    # Colour comes from the reference table's Color field, which `factory.py`
    # parses and routes on -- exercised code, unlike the Effect field beside it,
    # which nothing reads. `visual_card_pool` on the static metadata is None for
    # every card, so it is not a substitute.
    colour = _card_colour(card_id)

    keywords = _sorted_unique(str(k).lower() for k in (getattr(meta, "keywords", ()) or ()))
    tags = _sorted_unique(
        getattr(t, "name", str(t)).lower() for t in (getattr(meta, "tags", ()) or ())
    )

    lines = [
        f"{card_id.name} | {card_type} | {rarity} | {colour} | "
        f"cost {preview.cost} | target {target}",
        f"keywords: {', '.join(keywords) if keywords else NONE} | "
        f"tags: {', '.join(tags) if tags else NONE}",
    ]

    damage = preview.base_damage
    block = preview.base_block
    lines.append(
        f"damage {damage if damage else NONE} | block {block if block else NONE}"
    )

    effect_vars = dict(getattr(preview, "effect_vars", {}) or {})
    if effect_vars:
        rendered = "; ".join(f"{k} {effect_vars[k]}" for k in sorted(effect_vars))
        lines.append(f"vars: {rendered}")

    applied = powers_applied(card_id)
    if applied:
        lines.append(f"applies: {', '.join(applied)}")
        for power_name in applied:
            summary = power_summary(power_name)
            if summary:
                lines.append(f"  {summary}")

    actions = actions_taken(card_id)
    if actions:
        lines.append(f"does: {', '.join(actions)}")

    reads = state_read(card_id)
    if reads:
        lines.append(f"reads: {', '.join(reads)}")

    changes = upgrade_delta(card_id)
    if changes:
        lines.append(f"upgrade: {'; '.join(changes)}")

    return "\n".join(lines)
