"""Who plays the test fights. The pilot defines what the battery measures.

A greedy-damage pilot scores a block deck as worthless -- not because it is, but
because a pilot that never blocks never converts block into survival. That bias
is the ceiling on any deck ranking built from it, so the pilot is an explicit
argument everywhere rather than a default buried in a function.

greedy is here because it is fast and, measured on this repo, not bad: 73.3% +/-
8.1% on act 1 encounters, statistically level with a trained PPO model. It is the
right choice for bulk work and the wrong choice for the final word.
"""

from __future__ import annotations

import numpy as np

from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import action_to_card_and_target, get_action_mask


def greedy_pilot(combat) -> int:
    """Highest base damage first, end turn when nothing damaging is playable."""
    valid = np.flatnonzero(get_action_mask(combat))
    if valid.size == 0:
        return ACTION_END_TURN

    best_action, best_damage = None, -1
    for action in valid:
        if action == ACTION_END_TURN:
            continue
        hand_index, _ = action_to_card_and_target(int(action))
        if hand_index is None or hand_index >= len(combat.hand):
            continue
        damage = combat.hand[hand_index].base_damage or 0
        if damage > best_damage:
            best_damage, best_action = damage, int(action)

    if best_action is not None:
        return best_action
    return ACTION_END_TURN if ACTION_END_TURN in valid else int(valid[0])


def search_pilot(rollouts: int = 6, max_rollout_turns: int = 8, seed: int = 0):
    """Flat Monte Carlo. Slower, and not archetype-biased the way greedy is.

    Not established as stronger than greedy -- measured at 63-67% against greedy's
    73.3%, a difference well inside noise at 30 seeds. Offered for spot-checking
    decks where the two disagree, which is where greedy's bias would show.
    """
    from sts2_env.search.flat_mc import FlatMonteCarloPolicy

    policy = FlatMonteCarloPolicy(
        rollouts=rollouts, max_rollout_turns=max_rollout_turns, seed=seed
    )
    return policy.act


# --- a pilot that can see a block card --------------------------------------

TURNS_AHEAD = 6.0
"""How many more turns of enemy output killing something is assumed to save.

This is the trade rate between damage and block, and it matters. Measured on a
13-card block deck, 30 seeds, act 1:

    TURNS_AHEAD    normal          elite
        1        84%  26.7hp     11%  65.7hp     blocks instead of killing
        3        90%  29.1hp     30%  55.9hp
        6        90%  31.6hp     33%  49.8hp     <- chosen
       20        89%  34.1hp     33%  55.8hp     effectively pure greedy

Too low and the pilot turtles while the enemy stays alive and keeps swinging;
too high and it is greedy again with extra steps. It flattens above ~6 because
by then damage dominates every block comparison anyway.
"""

VULNERABLE_MULTIPLIER = 0.5
"""Vulnerable makes the target take 50% more. Matches the power's own scaling."""


def _enemy_output(combat, enemy) -> float:
    """Damage per turn this enemy is currently threatening."""
    from sts2_env.monsters.intents import incoming_damage

    ai = getattr(combat, "enemy_ais", {}).get(getattr(enemy, "combat_id", None))
    if ai is None:
        return 0.0
    player = getattr(combat, "player", None)
    total = 0
    for intent in (getattr(ai.current_move, "intents", None) or ()):
        total += intent.effective_total(enemy, player, combat)
    return float(total)


def effective_damage(combat, card, target) -> float:
    """What this card would ACTUALLY hit this target for, right now.

    `base_damage` is not that number, in two separate ways, and both were live:

    Wrong by modifier. A Strike at 3 Strength into a Vulnerable target does not
    deal 6. Routing through the engine's own `calculate_damage` picks up
    Strength, Vulnerable, Weak and every power hook without this file needing to
    know any of them exist -- which is the only version of this that survives a
    patch. Five Ironclad cards were also carrying a frozen `base_damage` beside a
    real `calc_base` (PERFECTED_STRIKE reads 6 no matter how many Strikes are
    held); those are now computed rather than read.

    Wrong by absence. Three attacks compute their damage inside their effect
    function and carry `base_damage = 0` -- BODY_SLAM is `owner.block`, and no
    amount of `calculate_damage` recovers it, because there is no base to modify.
    The plan document claimed otherwise; it was wrong. For those the card is
    played on a *clone* and the enemy's HP drop is measured, which needs no list
    of which cards are special and so cannot rot when a patch adds a fourth.

    The dry run is only reached for an attack with no base damage, so it costs
    nothing on the ~97% of cards that have one.
    """
    from sts2_env.core.damage import calculate_damage
    from sts2_env.core.enums import CardType, ValueProp

    if target is None:
        return 0.0

    variables = getattr(card, "effect_vars", None) or {}
    # Multi-hit. SWORD_BOOMERANG is `repeat: 3` at 3 damage and deals 9;
    # DISMANTLE is `hits: 2` at 8 and deals 16. Reading base_damage alone
    # undervalues every multi-hit card by its own hit count -- and multi-hit is
    # precisely what Strength scales best, so the error compounds exactly where
    # the card is strongest.
    # Both keys mean TOTAL hits, not extra ones -- sword_boomerang does
    # `for _ in range(effect_vars["repeat"])`, so repeat:3 is three hits and not
    # four. Checked in the source rather than assumed; the off-by-one version of
    # this overvalued every multi-hit card by a full hit.
    hits = max(1, int(variables.get("repeat", 1) or 1), int(variables.get("hits", 1) or 1))

    base = card.base_damage or 0
    if base > 0:
        player = getattr(combat, "player", None)
        try:
            dealt = float(calculate_damage(base, player, target, ValueProp.MOVE, combat))
        except Exception:  # noqa: BLE001 -- a pilot must never take the fight down
            dealt = float(base)
        # Both paths must report HP REMOVED, not damage thrown. The dry run
        # measures an HP delta and so already nets off the enemy's block; this
        # path has to subtract it explicitly or the two disagree by exactly the
        # enemy's block, and a 14-damage hit into 13 block would be valued at 14
        # when it is worth 1.
        return max(0.0, dealt * hits - float(getattr(target, "block", 0) or 0))

    if card.card_type != CardType.ATTACK:
        return 0.0
    return _dry_run_damage(combat, card, target)


def _dry_run_damage(combat, card, target) -> float:
    """Play the card on a clone and see what it did. Last resort, measured.

    `clone_combat` rather than `deepcopy`: a plain deepcopy leaves the copy's
    monsters acting on the ORIGINAL's creatures, which would corrupt the live
    fight this is supposed to be reasoning about.
    """
    from sts2_env.cards.registry import _CARD_EFFECTS
    from sts2_env.search.cloning import clone_combat

    effect = _CARD_EFFECTS.get(card.card_id)
    if effect is None:
        return 0.0

    try:
        clone = clone_combat(combat)
        index = next(
            (i for i, c in enumerate(combat.hand) if c is card), None
        )
        clone_card = clone.hand[index] if index is not None and index < len(clone.hand) else None
        clone_target = next(
            (e for e in clone.enemies
             if getattr(e, "combat_id", None) == getattr(target, "combat_id", None)),
            None,
        )
        if clone_card is None or clone_target is None:
            return 0.0
        before = clone_target.current_hp
        effect(clone_card, clone, clone_target)
        return float(max(0, before - clone_target.current_hp))
    except Exception:  # noqa: BLE001
        return 0.0


_TURNS_CACHE: dict = {}


def turns_saved(combat) -> float:
    """Turns of enemy output a kill actually prevents. TURNS_AHEAD is the cap.

    `TURNS_AHEAD` was a flat 6 -- every kill credited with preventing six turns
    of damage, including on the last turn of a fight that is already won. That is
    what kept DEMON_FORM scoring 60 against a Strike's 40 with one turn left and
    the enemy on 5 HP: a scaling card cannot pay out over turns that will not
    happen.

    Memoised per (combat, turn, total enemy HP) because `remaining_turns` walks
    the hand through `effective_damage`, which dry-runs a clone for cards like
    BODY_SLAM. Recomputing that once per candidate card made a decision
    quadratic in hand size.
    """
    key = (id(combat), getattr(combat, "turn_count", 0),
           sum(getattr(e, "current_hp", 0) for e in (combat.enemies or [])))
    hit = _TURNS_CACHE.get(key)
    if hit is None:
        if len(_TURNS_CACHE) > 512:
            _TURNS_CACHE.clear()
        hit = _TURNS_CACHE[key] = min(TURNS_AHEAD, remaining_turns(combat))
    return hit


def _enemy_threat(combat, enemy) -> float:
    """The damage per turn this enemy is worth killing to stop.

    NOT the same question as `_enemy_output`, and conflating them was a real bug.
    Output is what is coming THIS turn, which is the right input for a blocking
    decision. Threat is what killing them prevents over the rest of the fight.

    They diverge exactly when it matters most. The act 1 elite BYGONE_EFFIGY
    opens asleep, so its current output is 0 -- and every damage value in the
    pilot collapsed to zero with it, on the fight the whole run turns on. Its
    move set contains a 15-damage attack, which is what it is actually worth
    killing to stop.

    `ai.states` holds the full move set, so this reads the peak rather than
    guessing from max HP.
    """
    ai = getattr(combat, "enemy_ais", {}).get(getattr(enemy, "combat_id", None))
    if ai is None:
        return _enemy_output(combat, enemy)

    player = getattr(combat, "player", None)
    peak = 0.0
    states = getattr(ai, "states", None)
    for state in (states.values() if isinstance(states, dict) else (states or ())):
        move = getattr(state, "move", None) or state
        total = 0.0
        for intent in (getattr(move, "intents", None) or ()):
            try:
                total += intent.effective_total(enemy, player, combat)
            except Exception:  # noqa: BLE001 -- one unreadable move is not fatal
                continue
        peak = max(peak, total)
    return max(peak, _enemy_output(combat, enemy))


def _damage_value(combat, card, target) -> float:
    """HP saved by dealing this card's damage to this enemy.

    Damage is not worth a flat amount per point. Ten damage into a 12 HP enemy
    about to hit for 20 removes almost all of that; the same ten into a 200 HP
    boss removes 5% of it. Scoring by raw `base_damage` -- which is what greedy
    does -- treats those as the same play, and it is the reason chip damage on a
    boss outranks a kill on the thing actually hurting you.
    """
    damage = effective_damage(combat, card, target)
    if damage <= 0 or target is None:
        return 0.0

    remaining = float(max(1, getattr(target, "current_hp", 1)))
    fraction_killed = min(1.0, damage / remaining)
    return fraction_killed * _enemy_threat(combat, target) * turns_saved(combat)


def _block_value(combat, card, already_pledged: float) -> float:
    """HP saved by this card's block -- capped by what is actually incoming.

    The cap is the entire point. Block above incoming damage is thrown away, so
    a pilot that values block linearly hoards Defends against a 3-damage attack
    and calls a block deck good. `already_pledged` carries the block this turn's
    earlier plays have committed, so the second Defend against one attack is
    correctly worth nothing.
    """
    from sts2_env.monsters.intents import incoming_damage

    block = float(card.base_block or 0)
    if block <= 0:
        return 0.0

    player = getattr(combat, "player", None)
    standing = float(getattr(player, "block", 0) or 0) + already_pledged
    unblocked = max(0.0, float(incoming_damage(combat)) - standing)
    return min(block, unblocked)


def _debuff_value(combat, card, target) -> float:
    """Vulnerable and Weak, priced by whose damage they change.

    Weak cuts what an enemy deals; Vulnerable raises what we deal to it, which
    shortens the fight by the same proportion. Both come out in HP, so a card
    like Taunt -- 6 block AND a Vulnerable for one energy -- is finally
    distinguishable from a plain Defend rather than tying with it at zero.
    """
    variables = getattr(card, "effect_vars", None) or {}
    value = 0.0

    weak = float(variables.get("weak", 0) or 0)
    if weak and target is not None:
        value += _enemy_threat(combat, target) * 0.25 * min(weak, 2.0)

    vulnerable = float(variables.get("vulnerable", 0) or 0)
    if vulnerable and target is not None:
        # Faster kill on this enemy => that much of its output never lands.
        value += (
            _enemy_threat(combat, target)
            * VULNERABLE_MULTIPLIER
            * min(vulnerable, 3.0) / 3.0
        )
    return value


DRAW_KEYS = ("cards", "draw")
"""Where a card records how much it draws. `cards` is the real one.

This term read `draw` alone from the day it was written, and NO CARD IN THE GAME
carries a `draw` key -- it is `cards` (BATTLE_TRANCE {'cards': 3}, OFFERING
{'cards': 3, 'energy': 2}). So `_draw_value` returned 0.0 for every card, always,
and nine Ironclad cards were invisible to it. `draw` is kept as a fallback in
case a patch introduces it, since checking two keys costs nothing.
"""


def _hp_cost(card) -> float:
    """HP the card spends on itself. A real cost the pilot was not paying.

    OFFERING is 6 HP, BLOOD_WALL 2, HEMOKINESIS 2, BREAKTHROUGH 1. Priced 1:1
    against the HP the card saves, since both sides are the same currency -- and
    against a weak enemy a 6 HP payment can easily exceed what the card prevents.
    """
    variables = getattr(card, "effect_vars", None) or {}
    return float(variables.get("hp_loss", 0) or 0)


MAX_REMAINING_TURNS = 12.0
"""Ceiling on the fight-length estimate. A 200 HP boss against a starter deck
divides out to 30+ turns, and nothing in this game pays out over 30 turns --
past a dozen the estimate is measuring how bad the deck is, not how long the
fight runs, and it would make every scaling card look infinitely good."""


def remaining_turns(combat) -> float:
    """How many more turns this fight probably lasts.

    The number every scaling card needs and the pilot did not have. `TURNS_AHEAD`
    was the same idea frozen as a constant; this is the state-dependent version,
    and it is what lets a Power be worth taking on turn 1 and worthless on the
    last turn of a short fight.

    Estimated as total enemy HP over what we remove per turn, with our output
    taken from the best few cards in hand -- roughly what one turn of energy
    buys. Crude, and it only has to separate "long fight" from "nearly over".
    """
    enemies = [e for e in getattr(combat, "enemies", []) or []
               if getattr(e, "is_alive", False)]
    if not enemies:
        return 1.0
    total_hp = float(sum(getattr(e, "current_hp", 0) for e in enemies))

    target = enemies[0]
    per_card = sorted(
        (effective_damage(combat, c, target) for c in (combat.hand or [])),
        reverse=True,
    )
    energy = max(1, int(getattr(combat, "energy", 3) or 3))
    output = float(sum(per_card[:energy])) or 1.0
    return max(1.0, min(MAX_REMAINING_TURNS, total_hp / output))


def _attacks_per_turn(combat) -> float:
    """Roughly how many attacks we land a turn. Strength scales per attack, so a
    deck of one big hit values it differently from a deck of many small ones."""
    hand = combat.hand or []
    attacks = sum(1 for c in hand if (c.base_damage or 0) > 0)
    energy = max(1, int(getattr(combat, "energy", 3) or 3))
    return float(max(1, min(attacks, energy)))


def _power_value(combat, card, target) -> float:
    """What a Power or scaling skill is worth, in the same HP currency.

    28 of 86 Ironclad cards score zero on damage, block, debuff and draw -- every
    Power among them -- so the pilot could never choose to play one. It is the
    ceiling on every measurement this repo produces, not only on drafting.

    Priced off the fight's remaining length, which is the whole point: Demon Form
    on turn 1 of a boss is enormous and on the last turn of a hallway fight is
    nothing, and no fixed number can say both.
    """
    variables = getattr(card, "effect_vars", None) or {}
    if not variables or target is None:
        return 0.0

    # A Power pays from the turn AFTER it is played -- deploying it costs this
    # turn's energy, and Demon Form at 3 cost leaves nothing behind it. So the
    # payout horizon is one turn shorter than the fight, which is what takes it
    # to exactly zero on the last turn instead of merely small.
    turns = max(0.0, remaining_turns(combat) - 1.0)
    if turns <= 0.0:
        return 0.0
    value = 0.0

    strength = float(variables.get("strength", 0) or 0)
    if strength:
        # Extra damage over the rest of the fight, converted the same way a
        # direct hit is, so the units match the rest of the pilot.
        extra = strength * _attacks_per_turn(combat) * turns
        remaining = float(max(1, getattr(target, "current_hp", 1)))
        value += min(1.0, extra / remaining) * _enemy_threat(combat, target) * turns_saved(combat)

    # Block every turn, capped by what is actually coming each turn.
    per_turn_block = float(variables.get("plating", 0) or 0)
    for key in ("juggernaut_power", "crimson_mantle_power", "inferno_power"):
        per_turn_block += float(variables.get(key, 0) or 0) * 0.5
    if per_turn_block:
        from sts2_env.monsters.intents import incoming_damage

        incoming = float(incoming_damage(combat))
        value += min(per_turn_block, max(incoming, per_turn_block)) * turns * 0.5

    # Energy is worth whatever a card costs to play, and cards are worth roughly
    # what the best one in hand is worth.
    energy = float(variables.get("energy", 0) or 0)
    if energy:
        best = max((effective_damage(combat, c, target) for c in (combat.hand or [])),
                   default=0.0)
        remaining = float(max(1, getattr(target, "current_hp", 1)))
        value += min(1.0, (energy * best * turns) / remaining) \
            * _enemy_threat(combat, target) * turns_saved(combat)

    return value


POWER_STACK_VALUE = 1.2
"""HP-equivalent per stack of a player power, per turn it will be active.

A blunt instrument, and deliberately so. It prices powers this file has no
specific model for -- BARRICADE, DARK_EMBRACE, CORRUPTION -- which would
otherwise score exactly zero and never be played at all. Wrong-but-positive beats
invisible: a card the pilot can never choose is one the battery scores as dead
weight, and 21 of 86 Ironclad cards were in that state.
"""


_STATIC_KEYS = ("strength", "energy", "plating", "vulnerable", "weak", "hp_loss",
                "juggernaut_power", "crimson_mantle_power", "inferno_power")
_NEEDS_DRY_RUN: dict = {}


def _needs_dry_run(card) -> bool:
    """Can this card ever be priced without playing it?

    Keyed on card_id and cached, because the answer is a property of the CARD and
    not of the situation. Getting that wrong cost 30x: the fallback originally
    fired whenever a card scored zero, which includes a Defend on a turn nothing
    is incoming -- so the common cards took a clone every decision and the
    battery went from 0.59s to 17.6s for 36 fights.
    """
    key = getattr(card, "card_id", card)
    cached = _NEEDS_DRY_RUN.get(key)
    if cached is None:
        variables = getattr(card, "effect_vars", None) or {}
        cached = _NEEDS_DRY_RUN[key] = not (
            (card.base_damage or 0) or (card.base_block or 0)
            or any(variables.get(k) for k in _STATIC_KEYS)
            or any(variables.get(k) for k in DRAW_KEYS)
        )
    return cached


def _dry_run_value(combat, card, target) -> float:
    """Play the card on a clone and price whatever changed.

    The generic fallback for cards whose entire effect lives in the function --
    empty `effect_vars`, or an ambiguous key like `power`, which means
    block-per-exhaust on FEEL_NO_PAIN and something else on CORRUPTION. There is
    no table that prices those; there is only playing them and looking.

    Deliberately last: it costs a `clone_combat` (~0.6ms) and only runs for cards
    the cheap terms already scored at zero.
    """
    from sts2_env.cards.registry import _CARD_EFFECTS
    from sts2_env.monsters.intents import incoming_damage
    from sts2_env.search.cloning import clone_combat

    effect = _CARD_EFFECTS.get(card.card_id)
    if effect is None:
        return 0.0
    index = next((i for i, c in enumerate(combat.hand) if c is card), None)
    if index is None:
        return 0.0

    try:
        clone = clone_combat(combat)
        if index >= len(clone.hand):
            return 0.0
        clone_card = clone.hand[index]
        clone_target = next(
            (e for e in clone.enemies
             if getattr(e, "combat_id", None) == getattr(target, "combat_id", None)),
            None,
        ) if target is not None else None

        def stacks(state):
            player = getattr(state, "player", None)
            powers = getattr(player, "powers", None) or {}
            return sum(getattr(v, "amount", 0) or 0 for v in powers.values())

        before = (float(getattr(clone.player, "block", 0) or 0),
                  float(getattr(clone, "energy", 0) or 0),
                  len(clone.hand), stacks(clone),
                  [getattr(e, "current_hp", 0) for e in clone.enemies])

        effect(clone_card, clone, clone_target)

        after = (float(getattr(clone.player, "block", 0) or 0),
                 float(getattr(clone, "energy", 0) or 0),
                 len(clone.hand), stacks(clone),
                 [getattr(e, "current_hp", 0) for e in clone.enemies])
    except Exception:  # noqa: BLE001 -- a pilot must never take the fight down
        return 0.0

    turns = max(0.0, remaining_turns(combat) - 1.0)
    incoming = float(incoming_damage(combat))

    block_gained = max(0.0, after[0] - before[0])
    energy_gained = max(0.0, after[1] - before[1])
    drawn = max(0, after[2] - before[2] - 1)      # -1: the card itself left hand
    new_stacks = max(0.0, after[3] - before[3])
    damage = float(sum(max(0, b - a) for b, a in zip(before[4], after[4])))

    value = min(block_gained, max(incoming, block_gained))
    value += energy_gained * 2.0
    value += drawn * 1.5
    value += new_stacks * POWER_STACK_VALUE * turns
    if damage > 0 and target is not None:
        remaining = float(max(1, getattr(target, "current_hp", 1)))
        value += min(1.0, damage / remaining) * _enemy_threat(combat, target) * turns_saved(combat)
    return value


def _draw_value(card) -> float:
    """Cards drawn are options, valued modestly and flatly.

    Deliberately small. Draw compounds in a deck built for it, and this pilot
    does not build decks -- overrating it here would recreate the failure it is
    meant to fix, in the other direction.
    """
    variables = getattr(card, "effect_vars", None) or {}
    drawn = next((variables[k] for k in DRAW_KEYS if variables.get(k)), 0)
    return float(drawn or 0) * 1.5


def value_pilot(combat) -> int:
    """Play the card worth the most HP this turn; end the turn when none are.

    Replaces greedy's `max(base_damage)`, which scores every skill at zero and
    therefore plays whichever one happens to sit earliest in hand. Under that
    rule Taunt, Armaments and a dead exhaust card are the same card, so the
    battery ranking them was reading noise -- and every decision built on the
    battery inherited it.
    """
    valid = np.flatnonzero(get_action_mask(combat))
    if valid.size == 0:
        return ACTION_END_TURN

    enemies = list(getattr(combat, "enemies", []) or [])
    pledged = 0.0
    best_action, best_value = None, 0.0

    for action in valid:
        if action == ACTION_END_TURN:
            continue
        hand_index, target_index = action_to_card_and_target(int(action))
        if hand_index is None or hand_index >= len(combat.hand):
            continue
        card = combat.hand[hand_index]

        target = None
        if target_index is not None and 0 <= target_index < len(enemies):
            target = enemies[target_index]
        if target is None or not getattr(target, "is_alive", False):
            alive = [e for e in enemies if getattr(e, "is_alive", False)]
            target = alive[0] if alive else None

        value = (
            _damage_value(combat, card, target)
            + _block_value(combat, card, pledged)
            + _debuff_value(combat, card, target)
            + _draw_value(card)
            + _power_value(combat, card, target)
            - _hp_cost(card)
        )
        # Only for cards that can never be priced statically -- see _needs_dry_run.
        if value <= 0.0 and _needs_dry_run(card):
            value = _dry_run_value(combat, card, target) - _hp_cost(card)
        if value > best_value:
            best_value, best_action = value, int(action)

    if best_action is not None:
        return best_action

    # Nothing is worth HP. Spend leftover energy anyway rather than banking it --
    # unplayed energy is worth exactly nothing at end of turn -- but only on a
    # card that does not exhaust, since burning a card for no gain is a real cost.
    for action in valid:
        if action == ACTION_END_TURN:
            continue
        hand_index, _ = action_to_card_and_target(int(action))
        if hand_index is None or hand_index >= len(combat.hand):
            continue
        keywords = combat.hand[hand_index].keywords or frozenset()
        if any(k.lower() == "exhaust" for k in keywords):
            continue
        return int(action)

    return ACTION_END_TURN if ACTION_END_TURN in valid else int(valid[0])
