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
    # When the difference becomes visible in play. An override with no answer here
    # is probably drift wearing a disguise.
    observable_when: str
    # Scalar fields (cost, base_damage, ...) NOT taken from the decompile.
    fields: frozenset[str] = frozenset()
    # Decompiled dynamic vars deliberately absent from effect_vars. Without this,
    # a var the game has and the factory lacks is indistinguishable from someone
    # simply not knowing about it -- which is how SPITE ran for months missing
    # `repeat` and hitting once instead of twice.
    omitted_vars: frozenset[str] = frozenset()
    # A difference in the EFFECT FUNCTION rather than in the data -- ordering,
    # timing, which hook fires. Needed because the data can match the decompile
    # perfectly while the behaviour does not, and that is the harder kind to see:
    # nothing in effect_vars is missing, so no data check can catch it.
    behaviour: str = ""


MODELLING_OVERRIDES: dict[CardId, ModellingOverride] = {
    CardId.CONFLAGRATION: ModellingOverride(
        reason=("the game deals Repeat(4) hits of Damage(2); the simulator deals "
                "one hit of 8"),
        fields=frozenset({"base_damage"}),
        omitted_vars=frozenset({"repeat"}),
        observable_when=("thorns, per-hit block, and anything counting hits -- one "
                         "hit of 8 into a thorns enemy takes a quarter of the "
                         "retaliation it should. `repeat` is omitted deliberately: "
                         "any consumer that multiplies damage by it -- the pilot's "
                         "effective_damage does -- would read 8 x 4 = 32."),
    ),
    CardId.DRUM_OF_BATTLE_CARD: ModellingOverride(
        reason=("the game gives its Energy from AfterCardExhausted when the card "
                "exhausts itself; the simulator gives it at the end of OnPlay"),
        behaviour=("energy is paid at the end of OnPlay instead of from the "
                   "self-exhaust hook"),
        observable_when=("the card is exhausted WITHOUT being played -- by another "
                         "card's exhaust effect, or a relic. The game pays the "
                         "energy then and the simulator does not. Identical "
                         "whenever the card is played normally, which is why this "
                         "is a modelling choice and not a bug. Fix properly by "
                         "extending fire_after_card_exhausted to reach cards; it "
                         "currently only reaches powers and relics."),
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
        omitted = override.omitted_vars if override else frozenset()
        for key, value in dynamic_vars.items():
            if key in skip or key in omitted:
                continue
            # `damage` and `block` already live in base_damage / base_block, so
            # adding them here would put a second copy of the same number beside
            # the first -- and for CONFLAGRATION a CONTRADICTORY one, since its
            # override deliberately carries base_damage 8 against the game's
            # Damage(2). Still updated if a factory declared them itself.
            if key in ("damage", "block") and key not in card.effect_vars:
                continue
            # ADD as well as update. This used to be `if key in card.effect_vars`,
            # so a var the game has and the factory omitted was silently dropped
            # -- no add, no warning, and the card just quietly did less than the
            # real one. SPITE lost `repeat` that way and hit once instead of
            # twice; DOMINATE lost `vulnerable` and applied none at all.
            card.effect_vars[key] = value


def declared_difference(card_id: CardId) -> ModellingOverride | None:
    """The declared reason this card differs from the game, if it does.

    The single place anything -- tests, docs, audits -- should ask. An
    undeclared difference is a bug by definition; that is the whole contract.
    """
    return MODELLING_OVERRIDES.get(card_id)


def expected_dynamic_vars(card_id: CardId, upgraded: bool) -> dict:
    """What the simulator SHOULD carry: the game's vars, minus declared omissions.

    Parity tests compare against this rather than against the raw decompile, so a
    deliberate difference passes and an accidental one fails. Without it the only
    ways to get a green suite are to fix the card or to weaken the test, and the
    second is always easier.
    """
    _, dynamic_vars = _reference(card_id, upgraded)
    override = MODELLING_OVERRIDES.get(card_id)
    if override is None:
        return dict(dynamic_vars or {})

    # A scalar field and its dynamic var are the same quantity under two names:
    # `base_damage` is carried as the `damage` var, `base_block` as `block`. An
    # override declaring `base_damage` therefore has to exclude `damage` too, or
    # CONFLAGRATION -- declared as one hit of 8 against the game's Damage(2) --
    # still fails the very test the declaration exists to satisfy.
    field_to_var = {"base_damage": "damage", "base_block": "block"}
    excluded = set(override.omitted_vars) | set(override.fields)
    excluded |= {field_to_var[f] for f in override.fields if f in field_to_var}
    return {
        key: value for key, value in (dynamic_vars or {}).items()
        if key not in excluded
    }


def undeclared_differences() -> dict[CardId, dict]:
    """Every card carrying fewer dynamic vars than the game, without saying why.

    Returns ``{card_id: {var: game_value}}``. Empty is the only acceptable state;
    `tests/test_modelling_differences.py` asserts exactly that.
    """
    from sts2_env.cards.factory import create_card

    from sts2_env.cards.reference_static_metadata import (
        reference_dynamic_vars_by_card_id,
    )

    found: dict[CardId, dict] = {}
    for card_id in reference_dynamic_vars_by_card_id():
        try:
            card = create_card(card_id)
        except Exception:  # noqa: BLE001 -- a card that will not build is a
            continue       # different failure, reported by its own tests
        expected = expected_dynamic_vars(card_id, upgraded=False)
        missing = {
            key: value for key, value in expected.items()
            # damage and block live in base_damage / base_block by design.
            if key not in ("damage", "block") and key not in (card.effect_vars or {})
        }
        if missing:
            found[card_id] = missing
    return found
