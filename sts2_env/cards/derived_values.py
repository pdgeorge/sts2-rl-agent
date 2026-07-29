"""Card scalars come from the decompiled game, not from hand-written literals.

The simulator used to carry the same numbers in three places: the card factories,
the committed decompile, and docs/CARDS_REFERENCE.md. All three had to be edited
together, so of course they drifted -- and because they drifted *together* the
test suite stayed green while every one of them was wrong about the live game.
4,609 passing tests meant "the repo agrees with itself".

There is only one thing that can be authoritative about what a card does, and it
is the assembly that shipped. So cost, type, target, rarity, damage and block are
read from the decompile at construction time and written over whatever the
factory passed. The literals still in the factories are inert; they are
documentation now, and they cannot make the simulator wrong.

The factories keep the part that genuinely cannot be derived: behaviour.

DELIBERATE DIFFERENCES

A few cards the simulator models differently on purpose. Those are declared in
MODELLING_OVERRIDES below, with the reason and -- more usefully -- the conditions
under which the difference stops being harmless. Before this existed, an
intentional difference and a stale number looked exactly alike: both were just a
literal sitting in a factory. Declaring them is the point of the table, not an
exemption from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from sts2_env.core.enums import CardId

# Fields that derive from the decompile. Anything not listed is the factory's.
DERIVED_FIELDS = frozenset({
    "cost", "card_type", "target_type", "rarity", "base_damage", "base_block",
    "keywords", "star_cost", "has_energy_cost_x", "has_star_cost_x", "effect_vars",
})


@dataclass(frozen=True)
class ModellingOverride:
    """A card the simulator represents differently from the game, on purpose."""

    reason: str
    fields: frozenset[str]
    # When the difference becomes visible in play. An override with no answer here
    # is probably drift wearing a disguise.
    observable_when: str


MODELLING_OVERRIDES: dict[CardId, ModellingOverride] = {
    CardId.CONFLAGRATION: ModellingOverride(
        reason=("the game deals Repeat(4) hits of Damage(2); the simulator deals "
                "one hit of 8"),
        fields=frozenset({"base_damage"}),
        observable_when=("thorns, per-hit block, and anything counting hits -- one "
                         "hit of 8 into a thorns enemy takes a quarter of the "
                         "retaliation it should"),
    ),
}


def _reference(card_id: CardId, upgraded: bool):
    from sts2_env.cards.reference_static_metadata import (
        reference_dynamic_vars_by_card_id,
        reference_metadata_by_card_id,
        upgraded_reference_dynamic_vars_by_card_id,
        upgraded_reference_metadata_by_card_id,
    )

    if upgraded:
        return (upgraded_reference_metadata_by_card_id().get(card_id),
                upgraded_reference_dynamic_vars_by_card_id().get(card_id, {}))
    return (reference_metadata_by_card_id().get(card_id),
            reference_dynamic_vars_by_card_id().get(card_id, {}))


def apply_derived_values(card) -> None:
    """Overwrite a card's scalars with the game's, in place.

    Cards the decompile has never heard of -- simulator-only constructs, cards
    deleted upstream -- keep whatever the factory gave them. Being silent about
    those here is deliberate: reporting them is scripts/diff_decompiles.py's job,
    and this runs on every card construction in every training step.
    """
    metadata, dynamic_vars = _reference(card.card_id, card.upgraded)
    if metadata is None:
        return

    override = MODELLING_OVERRIDES.get(card.card_id)
    skip = override.fields if override else frozenset()

    # A card whose C# overrides Type or TargetType computes them at runtime, so
    # the constructor argument the reference parsed is only the starting value.
    if "card_type" not in skip and not metadata.has_custom_card_type:
        card.card_type = metadata.card_type
    if "target_type" not in skip and not metadata.has_custom_target_type:
        card.target_type = metadata.target_type
    if "rarity" not in skip:
        card.rarity = metadata.rarity
    if "keywords" not in skip:
        # Upgrading can add or remove keywords, and the upgraded metadata has
        # already replayed AddKeyword/RemoveKeyword, so this is the final set.
        card.keywords = frozenset(metadata.keywords)
    if "star_cost" not in skip:
        card.star_cost = metadata.star_cost
    if "has_energy_cost_x" not in skip:
        card.has_energy_cost_x = metadata.has_energy_cost_x
    if "has_star_cost_x" not in skip:
        card.has_star_cost_x = metadata.has_star_cost_x
    # X-cost cards have no fixed cost to derive.
    if "cost" not in skip and not metadata.has_energy_cost_x:
        card.cost = metadata.cost

    # Only overwrite a value the game actually declares. A card that computes its
    # damage has no DamageVar, and clobbering the factory's number with None would
    # turn a working card into one that deals nothing.
    if "base_damage" not in skip and "damage" in dynamic_vars:
        card.base_damage = dynamic_vars["damage"]
    if "base_block" not in skip and "block" in dynamic_vars:
        card.base_block = dynamic_vars["block"]

    # effect_vars carries the rest of a card's numbers -- how much Vulnerable it
    # applies, how much HP it costs -- and those were hand-written too. The game's
    # values win; keys it does not declare are simulator-side bookkeeping and are
    # left alone.
    #
    # Only keys the factory already declares are touched. Adding every key the
    # game has looked tidier and broke nine more tests: effect functions branch on
    # whether a key is present, so introducing one changes behaviour rather than
    # just correcting a number. Correcting what is there is the safe half, and it
    # is the half that drifts.
    if "effect_vars" not in skip:
        for key, value in dynamic_vars.items():
            if key in card.effect_vars and key not in skip:
                card.effect_vars[key] = value
