"""Every line you could play this turn, tried, and the best one kept.

The idea the whole approach rests on. Slay the Spire tells you what the enemies
are about to do, and a turn has only a few hundred legal orderings of a few
cards, so there is no need to *predict* how a turn will go: play each ordering on
a copy, end the turn, let the enemies actually act, and look at what is left.

Measured on this repo: 49-469 orderings for a starter-deck turn, 20-225 ms
exhaustively. A live decision already waits 2.6-3.6 s on the game's animations.

WHAT THIS FINDS THAT A POLICY NET DOES NOT

Sequencing. Bash before the Strikes because Vulnerable multiplies what follows.
The exact block that survives an 11-damage intent rather than the block that
looks tidy. Killing the second enemy *this* turn because it removes 6 damage a
turn for the rest of the fight. These are combinatorial facts about the position,
recomputed from scratch every turn -- which is also why a rebalance patch does not
break it. It reads the new numbers out of the simulator.

WHAT IT DOES NOT SEE

Anything beyond the end of this turn plus the enemies' reply. It will not spend a
turn setting up a combo that pays off in three. That is a real limit and the
honest place to fix it is a deeper search, not a fudge factor here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import (
    apply_combat_action,
    get_action_mask,
    is_potion_action,
)
from sts2_env.search.cloning import clone_combat
from sts2_env.search.evaluate import DEFAULT_WEIGHTS, EvalWeights, evaluate
from sts2_env.search.potion_policy import forced_potion_action

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState

logger = logging.getLogger(__name__)

DEFAULT_MAX_NODES = 20_000
DEFAULT_TIME_BUDGET = 3.0
DEFAULT_MAX_DEPTH = 12
DEFAULT_LOOKAHEAD_TURNS = 2
MAX_PLAYOUT_ACTIONS_PER_TURN = 12

#: How many of the enumerated lines `SearchResult.considered` carries out.
#: A boss turn enumerates thousands, so this is a diagnostic sample, not the
#: shortlist -- eight is enough to see whether the played line won by a nose or
#: a mile, and small enough that a 10-turn fight costs ~80 rows rather than
#: tens of thousands. The journal is already the biggest artefact of a session.
CONSIDERED_LINES = 8

DEFAULT_TOP_K = 0
"""How many of the turn's best lines get played to the end of the fight.

ZERO, which is to say off, and the reason is recorded rather than assumed.

Rolling the top five lines out to the end -- three sampled futures each -- was
built to fix two things and fixed neither. Against the same 200 fights it moved
the win rate +0.5% +/- 1.1% (three fights won that were lost, two lost that were
won: a coin flip), left the power-play rate as flat in fight length as it was
before (boss 3.2 -> 4.2%, elite 2.3 -> 3.3%, hallway 4.6 -> 4.5%, where the
longest fights should show the highest rate and do not), and cost five times the
compute. It also lost a behaviour that was right: with rollouts on, the searcher
plays Strike before Bash and throws away the Vulnerable multiplier, because
three samples cannot resolve a three-damage difference and the noise decides.

It did buy -1.0 +/- 0.3 HP a fight, which is real. Not worth the rest.

The diagnosis is not depth, it is the playout. A rollout inherits every blind
spot of the policy playing it, and this one ranks Powers last and plays them only
when nothing else is legal -- so playing a fight to its end still never shows a
Power being used well. Making the playout competent is the prerequisite for this
being worth switching on, and for deck evaluation by simulation being honest
about the same cards.

Set it to 5 to turn it back on once that is true."""

DEFAULT_ROLLOUT_TURNS = 40
DEFAULT_ROLLOUT_SAMPLES = 3
TERMINAL_WEIGHT = 0.5
"""How much the end of the fight counts, against the state right after the
enemies reply. Same rule and the same reason as LOOKAHEAD_WEIGHT below: a
rollout that ends in death must not make dying now and dying in ten turns score
alike."""

LOOKAHEAD_WEIGHT = 0.5
"""How much the state after the playout counts, against the state right after the
enemies reply.

Not a tuning knob -- it is what stops the lookahead trading away this turn. Scored
on the playout alone, dying in two turns and dying right now are the same number,
so the searcher became indifferent to surviving: at 12 HP against 12 telegraphed
damage it played Strike and died at 0, where without lookahead it played Defend
and lived at 5. Both lines ended in death inside the crude playout, so both scored
about -10 and it took the marginally larger.

Keeping the immediate term at full weight makes dying now strictly worse than
dying later, which is true, and doubly true given the playout is a rough policy
whose predicted deaths are not to be trusted."""


@dataclass
class SearchResult:
    """The chosen line, and how sure the search was about it."""

    actions: tuple[int, ...]
    """The card plays, in order, NOT including the final end-turn."""

    score: float
    runner_up: float | None = None

    nodes: int = 0
    leaves: int = 0
    rollouts: int = 0
    elapsed: float = 0.0
    exhausted: bool = True
    """False when a budget cut the search short -- the answer is then the best of
    what was looked at, not the best there is. Reported rather than hidden."""

    considered: tuple[tuple[float, tuple[int, ...]], ...] = ()
    """The best few lines the enumeration looked at, `(score, actions)`, best
    first -- what she REJECTED, not only what she played.

    PHASE_TWO section 3.3 built this for card rewards and stopped there, so a
    lost boss fight shows the cards played and nothing about the alternatives.
    That is the difference between "she blundered" and "the position was
    already lost", and without it the mechanism behind a loss has to be
    guessed at -- which is how predictions 6, 7, 8 and 9 all came to be written
    against mechanisms that turned out to be flat.

    These are the LEAF scores from the enumeration, which is the ranking the
    shortlist is cut by. When `top_k` rescoring runs it reorders the top few by
    playing them out, and only the winner survives that step, so a chosen line
    is not always this list's first entry. `score` on the result is the final
    number; these are the field it was chosen from."""

    @property
    def gap(self) -> float | None:
        """How far ahead the best line finished.

        Feeds Cyra's `gut_phrase`: unlike a softmax over logits, this is a real
        margin between two lines the agent actually played out. The number still
        never leaves the process -- it becomes a phrase, not a percentage.
        """
        if self.runner_up is None:
            return None
        return self.score - self.runner_up

    def first_action(self) -> int:
        """What to do right now. End the turn if the line is empty."""
        return self.actions[0] if self.actions else ACTION_END_TURN


def _state_key(combat: "CombatState") -> tuple:
    """A cheap fingerprint for spotting positions already looked at.

    Two orderings of Strike, Strike, Defend reach the same place; without this the
    search re-explores every permutation of an interchangeable hand. Deliberately
    coarse and cheap -- it only has to be right about *sameness*, and anything it
    misses costs time rather than correctness.
    """
    player = combat.player
    return (
        player.current_hp,
        player.block,
        combat.energy,
        tuple(sorted((c.card_id.value, c.upgraded) for c in combat.hand)),
        len(combat.draw_pile),
        len(combat.discard_pile),
        tuple(sorted((p.value, inst.amount) for p, inst in player.powers.items())),
        tuple(
            (
                e.current_hp,
                e.block,
                tuple(sorted((p.value, inst.amount) for p, inst in e.powers.items())),
            )
            for e in combat.enemies
        ),
    )


def _reseed_futures(combat: "CombatState", salt: int) -> None:
    """Give this copy a different future: another shuffle, other enemy choices.

    Only the parts nobody can see yet. The enemies' *current* telegraphed intent
    is already fixed in the state and is not touched, so the searcher keeps
    reasoning about the hit it can actually see coming.
    """
    from sts2_env.run.run_state import RunRngSet

    state = getattr(combat, "current_player_state", None)
    player_state = getattr(state, "player_state", None)
    run_state = getattr(player_state, "run_state", None)
    if run_state is None:
        return

    # Derived from the position, never from `id(combat)`. An address changes
    # between processes, which would make the same fight play differently on
    # every run -- and reproducibility is the property the whole benchmark rests
    # on: two agents are only comparable if each faces the same fight twice.
    seed = (
        combat.turn_count * 1_000_003
        + max(0, combat.player.current_hp) * 10_007
        + sum(max(0, e.current_hp) for e in combat.enemies) * 101
        + salt
    ) & 0x7FFFFFFF
    run_state.rng = RunRngSet(seed)


def _is_power_card(card) -> bool:
    from sts2_env.core.enums import CardType

    return getattr(card, "card_type", None) == CardType.POWER


def _incoming_damage(combat: "CombatState") -> int:
    """What the enemies have telegraphed, AS IT WILL LAND.

    Summing the raw intent under-reads every buffed enemy. The hit itself has
    always applied Strength, Weak and the player's Vulnerable -- verified: a
    7-damage Chomp lands for 10 at +3 Strength, 5 under Weak, 10 into
    Vulnerable, 15 for both -- but this estimate did not, so the rollout policy
    decided whether it needed block from a number the turn would not produce.
    It reads worst against exactly the enemies that scale: Crusher's Adapt,
    Rocket's Charge Up, anything stacking Strength on the way to a boss turn.

    A bridge-built intent is ALREADY the final number -- the game telegraphs
    what will land -- so applying modifiers to it would double-count Strength on
    the live path. That is what `pre_modified` distinguishes.
    """
    from sts2_env.core.damage import calculate_damage
    from sts2_env.core.enums import ValueProp

    total = 0
    for enemy in combat.enemies:
        if not enemy.is_alive:
            continue
        ai = combat.enemy_ais.get(enemy.combat_id)
        if ai is None:
            continue
        for intent in ai.current_move.intents:
            base = intent.damage or 0
            if not base:
                continue
            hits = max(1, intent.hits or 1)
            if getattr(intent, "pre_modified", False):
                total += base * hits
                continue
            try:
                landed = calculate_damage(
                    base, enemy, combat.player, ValueProp.MOVE, combat)
            except Exception:
                landed = base
            total += landed * hits
    return total


def _incoming_damage_by_enemy(combat: "CombatState") -> dict[int, int]:
    """`_incoming_damage`, kept per enemy, so a kill can be priced.

    Killing the thing that is about to hit you removes its damage every turn
    that follows; blocking removes it once. The playout could not tell those
    apart while it only knew the total.
    """
    from sts2_env.core.damage import calculate_damage
    from sts2_env.core.enums import ValueProp

    per_enemy: dict[int, int] = {}
    for enemy in combat.enemies:
        if not enemy.is_alive:
            continue
        ai = combat.enemy_ais.get(enemy.combat_id)
        if ai is None:
            continue
        total = 0
        for intent in ai.current_move.intents:
            base = intent.damage or 0
            if not base:
                continue
            hits = max(1, intent.hits or 1)
            if getattr(intent, "pre_modified", False):
                total += base * hits
                continue
            try:
                landed = calculate_damage(
                    base, enemy, combat.player, ValueProp.MOVE, combat)
            except Exception:
                landed = base
            total += landed * hits
        per_enemy[enemy.combat_id] = total
    return per_enemy


def _effective_hp(creature) -> int:
    """What has to be removed to kill it, block included."""
    return max(0, (creature.current_hp or 0)) + max(0, (creature.block or 0))


def _landed_damage(combat: "CombatState", base: int, victim) -> int:
    """A card's damage AS IT WILL LAND, for deciding whether it is lethal.

    The same correction `_incoming_damage` documents, pointed the other way. A
    6-damage Strike is 8 at +2 Strength and 12 into Vulnerable, so reading
    `base_damage` alone under-reads exactly the swings that finish an enemy --
    and the whole point of pricing a kill is knowing when one is available.
    """
    from sts2_env.core.damage import calculate_damage
    from sts2_env.core.enums import ValueProp

    try:
        return calculate_damage(base, combat.player, victim, ValueProp.MOVE, combat)
    except Exception:
        return base


def _player_damage_per_turn(combat: "CombatState") -> float:
    """A rough read of how fast the player is closing the fight.

    Sum of the attack damage sitting in hand, which is what a turn can spend.
    It only has to be the right order of magnitude: it is the denominator of
    `_hp_per_damage`, where it converts damage dealt into HP saved.
    """
    total = 0
    for card in combat.hand:
        if (card.base_damage or 0) > 0:
            total += card.base_damage
    return float(max(1, total))


#: The least an point of damage may be worth. A floor, and it is load-bearing:
#: `_incoming_damage` reads only what is telegraphed for THIS turn, so an enemy
#: spending the turn buffing or defending reports zero -- and without a floor
#: every attack, including a lethal one, scored exactly nothing and the playout
#: stopped. It will attack again, and a fight is won by removing HP either way.
MIN_HP_PER_DAMAGE = 0.05


def _hp_per_damage(combat: "CombatState", incoming: int) -> float:
    """How much HP one point of damage dealt is worth. DERIVED, not tuned.

    Enemies holding H HP between them and dealing D a turn, against a player
    dealing P a turn, will land D * H/P before they die. Removing one point of
    that H shortens the fight by 1/P turns and so saves D/P HP.

    This is the whole of pd's thesis in one line, and it is why the old playout
    could not see it: the value of an attack depends on how hard the enemies hit
    and how long they will live, and the old policy scored it at face value. In
    a hallway fight D/P is small and blocking wins; in an elite it is large and
    killing wins. Nothing here is a knob -- both terms are read off the board.
    """
    return max(MIN_HP_PER_DAMAGE, incoming / _player_damage_per_turn(combat))


#: How many attacks a turn a Strength point gets to apply to, for pricing a
#: Power. Two is the median attacks played per turn across the 13,251 live card
#: plays of the 2026-08-15 session, not a guess.
ATTACKS_PER_TURN = 2.0

#: Per-turn HP value for a Power whose `effect_vars` say nothing this function
#: knows how to read. Small and positive: an unreadable Power is still usually
#: worth playing early in a long fight and not worth it in a short one, and
#: multiplying by the horizon is what expresses that.
UNKNOWN_POWER_HP_PER_TURN = 1.5


def _power_hp_per_turn(card, hp_per_damage: float) -> float:
    """What a Power is worth per turn, in HP, from its own declared amounts.

    Read off `effect_vars` rather than a card-name table, so it covers every
    character rather than the Ironclad list someone happened to write down.
    Strength converts through `hp_per_damage` because its payoff is damage;
    block-per-turn powers are already in HP.
    """
    effect_vars = getattr(card, "effect_vars", None) or {}
    value = 0.0
    readable = False
    for name, amount in effect_vars.items():
        if not isinstance(amount, (int, float)) or amount <= 0:
            continue
        readable = True
        key = str(name).lower()
        if "strength" in key or "dexterity" in key:
            value += amount * ATTACKS_PER_TURN * hp_per_damage
        else:
            # FeelNoPain's `power`, Juggernaut's `juggernaut_power` and friends
            # pay out in block or in damage-on-a-trigger. Counting the amount as
            # HP a turn is crude and is still enormously closer than the flat
            # 0.5 that made every Power rank last.
            value += float(amount)
    # The fallback is for a Power this function could not read AT ALL, not for
    # one it read and priced low. Applying it to the latter overrode the honest
    # answer with a bigger guess, and Inflame beat a Strike on the last turn of
    # the horizon -- exactly the "powers are always good" failure this is meant
    # to avoid.
    return value if readable else UNKNOWN_POWER_HP_PER_TURN


#: A rollout policy: given a combat and its legal action mask, return the action
#: to play, or None to fall back to the built-in heuristic for that step.
#: `SearchAgent(playout_policy=...)` threads one through to the playouts.
PlayoutPolicy = "Callable[[CombatState, np.ndarray], int | None]"


def _heuristic_playout_action(
    combat: "CombatState",
    actions: list[int],
    turns_remaining: int = 1,
) -> int | None:
    """Score every legal play in ONE unit -- HP -- and take the best.

    Split out of `_playout` so a pluggable policy can fall back to it per step
    rather than per playout -- a trained model that declines to act on one
    state should not cost the whole rollout its continuation.

    WHAT THIS REPLACED, AND WHY IT MATTERED MORE THAN DEPTH
    ------------------------------------------------------
    The old policy scored a card as `block if a hit is coming else damage` and
    gave every Power a flat 0.5, which ranked it below the worst attack in the
    deck. Three things followed, and all three are visible in the 13,251 live
    card plays of 2026-08-15:

    - Block and damage were compared as though they were the same unit, so a
      6-damage Strike beat a 5-block Defend on a turn that needed block. 6 > 5.
    - A Power was played only when nothing else was legal, so a playout whose
      entire purpose was to reveal a scaling card's payoff never once saw one
      used. The power-play rate came out FLAT in fight length -- 2.32% in
      1-2 turn fights against 2.25% in 8+ turn fights, where the long fights
      are exactly the ones a Power is for.
    - Nothing distinguished killing an enemy from chipping it, so the agent
      blocked its way through 5.2-turn elite fights at 7.2 damage a turn
      instead of closing them.

    `MODELS.md:97` and `DEFAULT_TOP_K` both record the attempt to fix this with
    depth instead. It moved the win rate +0.5% +/- 1.1% and left the power rate
    just as flat, because a deeper rollout of a policy that never plays a Power
    still never plays a Power.

    THE UNIT IS HP, FOR EVERY CARD
    ------------------------------
    - block: `min(block, unblocked)`. Block past the telegraphed hit saves
      nothing, which is the same thing `EvalWeights.block_unused` says.
    - a killing blow: the dead enemy's own damage, for every turn left in the
      horizon. This is pd's thesis priced -- an enemy on 6 HP intending 5 is
      worth 5 HP a turn forever, where blocking it is worth 5 once.
    - any other attack: damage * `_hp_per_damage`, which is derived from the
      board rather than tuned.
    - a Power: its declared per-turn value times the turns left to collect it.

    Every term is HP, so they can be compared, which is the thing the old
    version could not do.
    """
    from sts2_env.gym_env.action_space import action_to_card_and_target

    incoming = _incoming_damage(combat)
    unblocked = max(0, incoming - (combat.player.block or 0))
    per_enemy = _incoming_damage_by_enemy(combat)
    hp_per_damage = _hp_per_damage(combat, incoming)
    horizon = max(1, turns_remaining)

    enemies_by_id = {e.combat_id: e for e in combat.enemies if e.is_alive}

    best_action, best_value = None, 0.0
    for action in actions:
        hand_index, target = action_to_card_and_target(action)
        if hand_index is None or hand_index >= len(combat.hand):
            continue
        card = combat.hand[hand_index]
        damage = card.base_damage or 0
        block = card.base_block or 0

        value = 0.0
        if block:
            value += float(min(block, unblocked))
        if damage:
            victim = None
            if target is not None and 0 <= target < len(combat.enemies):
                candidate = combat.enemies[target]
                if candidate.is_alive:
                    victim = candidate
            if victim is None and len(enemies_by_id) == 1:
                victim = next(iter(enemies_by_id.values()))
            value += damage * hp_per_damage
            if victim is not None and _landed_damage(
                    combat, damage, victim) >= _effective_hp(victim):
                # ON TOP of the damage, not instead of it: the kill is worth
                # what the corpse stops doing for the rest of the horizon. This
                # term is zero for an enemy that is not attacking this turn,
                # which is why it adds to the damage value rather than
                # replacing it -- replacing it priced a lethal blow against a
                # buffing enemy at nothing at all.
                value += per_enemy.get(victim.combat_id, 0) * horizon
        if not damage and not block and _is_power_card(card):
            value += _power_hp_per_turn(card, hp_per_damage) * horizon

        if value > best_value:
            best_action, best_value = action, value

    if best_action is None or best_value <= 0:
        return None
    return best_action


def _playout(combat: "CombatState", turns: int, policy=None) -> None:
    """Play `turns` more turns quickly, in place, to see past this one.

    A Power costs energy now and pays for it over the turns that follow, and a
    boss is decided across a dozen of them. Scoring the state the moment the
    enemies have replied cannot see either, so Inflame reads as a wasted turn and
    is never played -- 15 power cards in 465 plays, against a benchmark where 73
    of 200 decks hold one.

    Deliberately cheap and deliberately not search: block when a hit is coming
    and there is block to be had, otherwise hit the thing in front. It exists to
    make a scaling card's payoff visible, not to play well -- the searcher is
    already doing that for the turn it can see.

    `policy` (Phase 2.3) replaces that heuristic with a callable -- a trained
    combat model, typically -- taking `(combat, mask)` and returning an action
    index, or None to defer to the heuristic for that step. Optional because
    `MODELS.md:97` found that lengthening the horizon with a *dumb* playout
    does not help; whether a *trained* one does is the open question this hook
    exists to let someone answer.
    """
    for turn_index in range(turns):
        if combat.is_over:
            return
        # How many turns are left to collect a Power's payoff, this one included.
        # A Power on the last turn of the horizon buys nothing, and the old
        # policy had no way to know that either.
        turns_remaining = turns - turn_index

        for _ in range(MAX_PLAYOUT_ACTIONS_PER_TURN):
            if combat.is_over:
                return
            mask = get_action_mask(combat)
            actions = [int(a) for a in np.where(mask == 1)[0] if a != ACTION_END_TURN]
            if not actions or combat.pending_choice is not None:
                break

            best_action = None
            if policy is not None:
                try:
                    chosen = policy(combat, mask)
                except Exception:
                    # A rollout policy is an optimisation, never a reason to
                    # take down a search that would otherwise have answered.
                    logger.debug("playout policy raised; using the heuristic",
                                 exc_info=True)
                    chosen = None
                # Trust it only if it is actually legal here -- a model handed
                # a state it was not trained on can return a masked action, and
                # applying that would corrupt the rollout silently.
                if chosen is not None and int(chosen) in actions:
                    best_action = int(chosen)

            if best_action is None:
                best_action = _heuristic_playout_action(
                    combat, actions, turns_remaining)

            if best_action is None:
                break
            if not apply_combat_action(combat, best_action):
                break

        if combat.is_over:
            return
        combat.end_player_turn()


def _branch_actions(combat: "CombatState") -> list[int]:
    """The moves worth branching on here, end-turn excluded.

    During a pending choice the mask means something different -- the options are
    choice indices, and index 0 is confirm rather than end-turn -- so the caller
    must not treat this as a list of card plays.
    """
    mask = get_action_mask(combat)
    actions = [int(a) for a in np.where(mask == 1)[0]]
    if combat.pending_choice is not None:
        return actions
    return [a for a in actions if a != ACTION_END_TURN]


def search_turn(
    combat: "CombatState",
    *,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    max_nodes: int = DEFAULT_MAX_NODES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    include_potions: bool = True,
    lookahead_turns: int = DEFAULT_LOOKAHEAD_TURNS,
    top_k: int = DEFAULT_TOP_K,
    rollout_turns: int = DEFAULT_ROLLOUT_TURNS,
    playout_policy=None,
    rollout_samples: int = DEFAULT_ROLLOUT_SAMPLES,
) -> SearchResult:
    """Find the best sequence of plays for the turn `combat` is in.

    Every reachable ordering is tried, and each one is scored by ending the turn
    on a copy and letting the enemies reply. The score is therefore what actually
    happened to a copy of the fight, not a guess about what would.
    """
    started = time.perf_counter()
    counters = {"nodes": 0, "leaves": 0}
    seen: set[tuple] = set()
    exhausted = True

    best_score = -float("inf")
    best_actions: tuple[int, ...] = ()
    second_score = -float("inf")
    # Every line and its cheap score, so the most promising few can be looked at
    # properly afterwards. Kept as paths rather than states: replaying three
    # actions from the root costs one clone, where holding hundreds of states
    # costs hundreds.
    shortlist: list[tuple[float, tuple[int, ...]]] = []

    def out_of_budget() -> bool:
        return (
            counters["nodes"] >= max_nodes
            or (time.perf_counter() - started) >= time_budget
        )

    def consider_ending(state: "CombatState", path: tuple[int, ...]) -> None:
        """Score 'stop here and end the turn', by doing exactly that."""
        nonlocal best_score, best_actions, second_score

        if state.pending_choice is not None:
            return  # cannot end a turn mid-choice

        counters["leaves"] += 1
        ended = clone_combat(state)
        ended.end_player_turn()
        score = evaluate(ended, weights)
        if lookahead_turns:
            future = clone_combat(ended)
            _playout(future, lookahead_turns, playout_policy)
            score += LOOKAHEAD_WEIGHT * evaluate(future, weights)

        shortlist.append((score, path))

        if score > best_score:
            second_score = best_score
            best_score, best_actions = score, path
        elif score > second_score:
            second_score = score

    def descend(state: "CombatState", path: tuple[int, ...]) -> None:
        nonlocal exhausted

        if out_of_budget():
            exhausted = False
            return

        key = _state_key(state)
        if key in seen:
            return
        seen.add(key)

        consider_ending(state, path)

        if len(path) >= max_depth or state.is_over:
            return

        for action in _branch_actions(state):
            if out_of_budget():
                exhausted = False
                return
            if not include_potions and is_potion_action(action):
                continue

            child = clone_combat(state)
            if not apply_combat_action(child, action):
                # Masked as legal but refused. A mask bug rather than a legal
                # outcome; skip the branch rather than recurse on a state that
                # did not change, which would not terminate.
                continue

            counters["nodes"] += 1
            descend(child, path + (action,))

    descend(combat, ())

    if best_score == -float("inf"):
        # Nothing was playable, or the budget went before the first leaf. Ending
        # the turn is always legal, and is what an empty line means.
        best_score, best_actions = 0.0, ()

    rollouts = 0
    if top_k and len(shortlist) > 1:
        best_actions, best_score, second_score, rollouts = _rescore_by_playing_to_the_end(
            combat, shortlist,
            weights=weights, top_k=top_k, rollout_turns=rollout_turns,
            playout_policy=playout_policy,
            rollout_samples=rollout_samples,
            fallback=(best_actions, best_score, second_score),
            deadline=started + time_budget,
        )

    return SearchResult(
        actions=best_actions,
        score=best_score,
        runner_up=second_score if second_score > -float("inf") else None,
        nodes=counters["nodes"],
        leaves=counters["leaves"],
        rollouts=rollouts,
        elapsed=time.perf_counter() - started,
        exhausted=exhausted,
        considered=tuple(sorted(shortlist, key=lambda e: -e[0])[:CONSIDERED_LINES]),
    )


def _rescore_by_playing_to_the_end(
    combat: "CombatState",
    shortlist: list[tuple[float, tuple[int, ...]]],
    *,
    weights: EvalWeights,
    top_k: int,
    rollout_turns: int,
    playout_policy=None,
    rollout_samples: int,
    fallback: tuple[tuple[int, ...], float, float],
    deadline: float,
) -> tuple[tuple[int, ...], float, float, int]:
    """Play the most promising lines to the end of the fight and rank on that.

    The cheap score stops two turns after this one, which is where a scaling
    card's whole value lives: measured across the benchmark, the searcher played
    Powers at 3.2% in bosses, 2.3% in elites and 4.6% in hallway fights -- a rate
    completely flat in fight length, when a correct agent would play them most in
    the longest fights. Playing a line to its actual conclusion prices Inflame
    without anyone deciding what Strength is worth: it either changed the result
    or it did not.

    Only the top few, because a rollout costs what a hundred leaf evaluations do
    and the cheap score is a good enough filter to pick which ones deserve it.

    The immediate term stays at full weight for the same reason it does in the
    two-turn lookahead: a rollout that ends in death cannot be allowed to make
    dying now and dying in ten turns look alike, or the searcher throws away
    survivable turns on the say-so of a rough playout policy.
    """
    ranked = sorted(shortlist, key=lambda entry: -entry[0])[:top_k]

    scored: list[tuple[float, tuple[int, ...]]] = []
    rollouts = 0
    for _, path in ranked:
        if time.perf_counter() >= deadline:
            break

        state = clone_combat(combat)
        replayed = True
        for action in path:
            if not apply_combat_action(state, action):
                replayed = False
                break
        if not replayed or state.pending_choice is not None:
            continue

        state.end_player_turn()
        immediate = evaluate(state, weights)

        # Several futures, not one. A rollout is a sample, and one sample carries
        # the whole variance: at weight 0.5 that was enough to drown a precise
        # 3-damage fact and make the searcher play Strike before Bash, throwing
        # away the Vulnerable multiplier it had just been shown how to use.
        #
        # Reseeding also removes a quiet cheat. A clone carries the real RNG, so a
        # single rollout knows the exact order of every future draw -- information
        # the live game will never hand over. Averaging over redrawn futures both
        # cuts the noise and measures the thing that transfers.
        outcomes = []
        for sample in range(max(1, rollout_samples)):
            future = clone_combat(state)
            if sample:
                _reseed_futures(future, sample)
            _playout(future, rollout_turns, playout_policy)
            outcomes.append(evaluate(future, weights))
            rollouts += 1

        terminal = sum(outcomes) / len(outcomes)
        scored.append((immediate + TERMINAL_WEIGHT * terminal, path))

    if not scored:
        actions, best, second = fallback
        return actions, best, second, rollouts

    scored.sort(key=lambda entry: -entry[0])
    best_score, best_actions = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else -float("inf")
    return best_actions, best_score, runner_up, rollouts


class SearchAgent:
    """A `CombatAgent` that plans a turn and then plays it out.

    Planning once per turn rather than once per card is not only cheaper, it is
    also correct here: the plan was produced by playing these exact actions from
    this exact state on a copy, and the copy carries the same RNG, so replaying
    them reproduces what search saw. The plan is still re-validated against the
    live mask before each action, and abandoned if the position is not the one
    that was planned for -- a plan that has stopped matching reality is the bug
    class this project has paid for most.
    """

    def __init__(
        self,
        weights: EvalWeights = DEFAULT_WEIGHTS,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        time_budget: float = DEFAULT_TIME_BUDGET,
        max_depth: int = DEFAULT_MAX_DEPTH,
        include_potions: bool = True,
        lookahead_turns: int = DEFAULT_LOOKAHEAD_TURNS,
        top_k: int = DEFAULT_TOP_K,
        rollout_turns: int = DEFAULT_ROLLOUT_TURNS,
        rollout_samples: int = DEFAULT_ROLLOUT_SAMPLES,
        playout_policy=None,
        name: str | None = None,
    ):
        self.weights = weights
        self.max_nodes = max_nodes
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.include_potions = include_potions
        self.lookahead_turns = lookahead_turns
        self.top_k = top_k
        self.rollout_turns = rollout_turns
        self.rollout_samples = rollout_samples
        # Phase 2.3: an optional trained rollout policy for the playouts. None
        # keeps the block-then-damage heuristic that every measured number in
        # MODELS.md was produced with.
        self.playout_policy = playout_policy
        self.name = name or (
            f"search(t<={time_budget}s, lookahead={lookahead_turns}, "
            f"top_k={top_k})")

        self._plan: list[int] = []
        self._plan_round: int | None = None
        """The `round_number` `_plan` was searched for. A plan is only valid for
        its own turn; see `act`."""
        self._last_gap: float | None = None
        self._last_result: SearchResult | None = None
        """The most recent search, kept so a caller can log what was rejected.
        Only ever read; nothing here branches on it."""
        self.searches = 0
        self.total_nodes = 0
        self.total_seconds = 0.0
        self.budget_exhausted_count = 0

    @property
    def last_gap(self) -> float | None:
        """The margin behind the most recent decision, for milestone phrasing."""
        return self._last_gap

    @property
    def last_result(self) -> "SearchResult | None":
        """The most recent `SearchResult`, for option-score logging."""
        return self._last_result

    def _replan(self, combat: "CombatState") -> None:
        result = search_turn(
            combat,
            weights=self.weights,
            max_nodes=self.max_nodes,
            time_budget=self.time_budget,
            max_depth=self.max_depth,
            include_potions=self.include_potions,
            lookahead_turns=self.lookahead_turns,
            top_k=self.top_k,
            rollout_turns=self.rollout_turns,
            playout_policy=self.playout_policy,
            rollout_samples=self.rollout_samples,
        )
        self._plan = list(result.actions)
        self._last_gap = result.gap
        self._last_result = result
        self.searches += 1
        self.total_nodes += result.nodes
        self.total_seconds += result.elapsed
        if not result.exhausted:
            self.budget_exhausted_count += 1

    def act(self, combat: "CombatState") -> int:
        mask = get_action_mask(combat)

        # A PLAN BELONGS TO THE TURN IT WAS MADE FOR. `search_turn` returns the
        # card plays for one turn, and the only thing stopping a leftover play
        # from being popped into the next one was "is that index still legal" --
        # which it usually is, on a completely different card. Offline the plan
        # always emptied before the turn ended so it never bit; live, where
        # `LiveSearch.decide` rebuilds the position from the bridge on every
        # call, a turn can advance underneath a plan that still has actions in
        # it. Cheap to guard, and the failure would be silent -- the same shape
        # as the CloneError that had the agent standing still for 8 turns.
        round_number = getattr(combat, "round_number", None)
        if round_number != self._plan_round:
            self._plan = []
            self._plan_round = round_number

        # Rules of thumb first, for the potions the evaluator is blind to. See
        # potion_policy for why, and for the standing instruction to replace any
        # of them with something measured. Checked before the plan because a
        # forced drink changes the position the plan was made for.
        if self.include_potions:
            forced = forced_potion_action(
                combat, {int(a) for a in np.where(mask == 1)[0]})
            if forced is not None:
                self._plan = []
                return forced

        # A pending choice was not part of the planned line; decide it on its own
        # terms and then plan afresh.
        if combat.pending_choice is not None:
            self._plan = []
            result = search_turn(
                combat,
                weights=self.weights,
                max_nodes=self.max_nodes,
                time_budget=self.time_budget,
                max_depth=self.max_depth,
                include_potions=self.include_potions,
                lookahead_turns=self.lookahead_turns,
                top_k=self.top_k,
                rollout_turns=self.rollout_turns,
            playout_policy=self.playout_policy,
                rollout_samples=self.rollout_samples,
            )
            self._last_gap = result.gap
            self._last_result = result
            self.searches += 1
            self.total_nodes += result.nodes
            self.total_seconds += result.elapsed
            action = result.first_action()
            return action if mask[action] else int(np.where(mask == 1)[0][0])

        if not self._plan:
            self._replan(combat)

        while self._plan:
            action = self._plan.pop(0)
            if action < len(mask) and mask[action]:
                return action
            # The position is not the one that was planned for.
            logger.debug("Plan diverged at action %d; replanning", action)
            self._plan = []
            self._replan(combat)
            if not self._plan:
                break

        return ACTION_END_TURN

    def stats(self) -> dict[str, float]:
        return {
            "searches": self.searches,
            "nodes_per_search": self.total_nodes / self.searches if self.searches else 0.0,
            "seconds_per_search": self.total_seconds / self.searches if self.searches else 0.0,
            "budget_exhausted": self.budget_exhausted_count,
        }


def model_playout_policy(model):
    """Wrap a trained MaskablePPO as a rollout policy for `SearchAgent`.

    Phase 2.3. `MODELS.md:97` showed that lengthening the horizon with the
    block-then-damage heuristic does not help, and the diagnosis recorded in
    `DEFAULT_TOP_K` above is that the playout, not the depth, is the limit: a
    rollout inherits every blind spot of the policy playing it, and the
    heuristic ranks Powers last, so playing a fight to its end still never
    shows a Power used well.

    This is the seam for testing that diagnosis rather than arguing it. Whether
    a trained playout actually helps is unmeasured -- use
    `score_combat_benchmark.py --search` with and without.

    Returns None on any failure, which the playout reads as "use the
    heuristic for this step".
    """
    from sts2_env.gym_env.observation import encode_observation

    def policy(combat, mask):
        obs = encode_observation(combat)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return policy
