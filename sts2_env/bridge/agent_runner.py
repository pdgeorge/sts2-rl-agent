"""Agent runner: connects a trained RL model to the real STS2 game.

Main loop:
  1. Connect to the game via TCP (bridge mod must be installed & running)
  2. Load a trained MaskablePPO model
  3. Receive game state -> encode observation -> model.predict -> send action
  4. Handle all game phases (combat, map, rewards, shop, rest, events)

Usage:
  python -m sts2_env.bridge.agent_runner --model-path models/combat_ppo.zip
  python -m sts2_env.bridge.agent_runner --model-path models/combat_ppo.zip --port 9002

The agent uses the trained model for combat decisions and simple heuristics
for non-combat decisions (map navigation, card rewards, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

from sts2_env.bridge.card_quality import SKIP_THRESHOLD, rank_cards
from sts2_env.bridge.client import STS2GameClient
from sts2_env.bridge.cyra_events import CyraPublisher
from sts2_env.bridge.journal import RunJournal
from sts2_env.bridge.raw_capture import RawCapture
from sts2_env.bridge.milestones import MilestoneWatcher
from sts2_env.bridge.protocol import (
    ActionType,
    BridgeStateType,
    MSG_TYPE_ERROR,
    MSG_TYPE_GAME_STATE,
    MSG_TYPE_PONG,
    Phase,
)
from sts2_env.bridge.run_adapter import RunStateAdapter
from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE
from sts2_env.parity.bridge_replay import BridgeReplayRecorder

# Bridge states a full-run model decides for itself. Combat is handled separately
# because its decode is richer than choose-by-index.
RUN_MODEL_STATES = frozenset({
    "map_select", "card_reward", "card_bundle", "reward_screen",
    "boss_relic", "rest_site", "event", "treasure",
})

logger = logging.getLogger(__name__)

DEFAULT_CHOICE_INDEX = 0
CARD_REWARD_LARGE_DECK_SIZE = 30
REST_HP_RATIO_THRESHOLD = 0.5
TERMINAL_PHASES = frozenset({
    BridgeStateType.GAME_OVER,
    BridgeStateType.RUN_COMPLETE,
})

ROOM_PRIORITY_HEALTHY = (
    "boss",
    "elite",
    "monster",
    "event",
    "unknown",
    "treasure",
    "shop",
    "restsite",
)
#: The health a room needs before it is worth entering, as a fraction of max HP.
#:
#: FITTED, not chosen. Over 116 live elite fights with a known max HP, gating on
#: the fraction of health the player had walking in:
#:
#:     rule                     taken   died   death rate
#:     hp > 0.50 * max  (old)      85     18      21%
#:     hp >= 0.75 * max            35      6      17%
#:     hp >= 0.80 * max            21      2      10%
#:     hp >= 0.85 * max            18      2      11%
#:
#: 0.80 is where the curve bends; past it the extra caution buys nothing. The old
#: 0.50 authorised precisely the fights that ended runs -- 32 of the 56 recorded
#: elite choices were made between 40 and 59 HP, and the measured death rate
#: there is 18-29% against 0% above 70.
#:
#: A FRACTION RATHER THAN AN ABSOLUTE HP FIGURE, which was the first attempt. The
#: safe band was observed on characters with 80-93 max HP, so "70 HP" and "87% of
#: max" describe the same measurement -- but only the fraction still means
#: anything once Max HP rewards push the ceiling to 103. Absolute thresholds
#: silently loosen as the character grows, which is backwards: the rooms get
#: harder, not easier.
ROOM_MIN_HP_FRACTION = {
    "boss": 0.80,
    "elite": 0.80,
    "monster": 0.40,
    "unknown": 0.40,
    "event": 0.40,
}
"""Monsters at 0.40 from the same table: their death rate is 35% at 20-29 HP,
17% at 30-39 and 6% at 40-49. Bosses share the elite figure aspirationally --
across 89 attempts the agent has NEVER entered a boss above 69 HP, so there is
no safe band in the data to fit, only a median entry of 47 and a 60-88% death
rate. 0.80 is the target to move that median toward, not an observation."""

#: Deliberately NO per-act scaling. The obvious move was to scale these by act,
#: and the data refuses it: act 2 elites measure a p90 of 24 against act 1's 54,
#: which reads as "act 2 is easier" and is really survivorship -- only strong
#: runs get there, n=4. Fitting a multiplier to that is fitting noise, and the
#: version of this that did made act 3 elites unaffordable at any HP and stopped
#: the agent ever upgrading a card again. Revisit when deep runs are common
#: enough to measure, which is what this policy exists to produce.

#: Rooms worth routing *to* when the next real room cannot be afforded.
RECOVERY_ROOMS = ("restsite", "shop", "treasure")

ROOM_PRIORITY_LOW_HP = (
    "restsite",
    "shop",
    "treasure",
    "monster",
    "event",
    "unknown",
    "elite",
    "boss",
)
#: DELETED: `CARD_REWARD_TYPE_PRIORITY = ("power", "attack", "skill")`. It was
#: defined here and referenced by nothing -- `_pick_card_reward_index` has
#: scored cards through `card_quality.rank_cards` since the rule it describes
#: was replaced (see that function's docstring: type-priority took
#: BLIGHT_STRIKE over SUNDER because it was listed first).
#:
#: Left as a comment because a stale constant is not harmless. Reading it as
#: live policy, the 43% rate at which live runs take a Power when one is on
#: offer looks like a bug -- the policy says Powers come first, so why 43%? It
#: is not a bug: `rank_cards` weighs a Power's SCALING_BONUS against what else
#: is on the table and often prefers the other card. Deleting the constant
#: removes the wrong answer rather than leaving it to be found again.

#: Removal FIRST, on measurement rather than taste. `removal_vs_relic.py` over
#: 30 real live boss decks: a removal is worth ~3.3 points of act 1 boss win and
#: a marginal relic ~2, and removal costs 75 gold against a shop relic's 150-300
#: -- roughly four times the win rate per gold. It was last here, behind
#: everything, and gated to curses.
#:
#: `buy_card` and `buy_potion` have NOT been priced against either, so their
#: order relative to each other is unchanged and unjustified. Only the one
#: comparison that was measured has been acted on.
SHOP_PURCHASE_ACTION_PRIORITY = (
    "remove_card",
    "buy_relic",
    "buy_card",
    "buy_potion",
    "buy_item",
)
SHOP_LEAVE_ACTION = "leave_shop"
REWARD_PROCEED_ACTION = "proceed"
REWARD_PICK_ACTION = "pick_reward"
CARD_BUNDLE_PICK_ACTION = "pick_card_bundle"
CRYSTAL_SPHERE_CELL_ACTION = "divine_cell"
REST_HEAL_OPTION_ID = "heal"
REST_SMITH_OPTION_ID = "smith"
STUCK_WARN_AFTER = 4
STUCK_ESCALATE_AFTER = 12
STUCK_ABANDON_AFTER = 24
"""Give up only after trying the one action that is always available.

The first threshold means "whatever we keep sending, the game is refusing".
That is not the same as "there is nothing to do": end-turn is legal in every
combat, and the usual cause of a refused play is a rule this side does not
model -- RINGING's one-card-a-turn limit, an unplayable status, a relic the
simulator has not learned. Ending the turn clears exactly that class of
stall, because next turn deals a new hand.

So `STUCK_ESCALATE_AFTER` switches to forcing end-turn, and only if *that*
is also refused for as many states again does the session stop. Abandoning
at the first threshold threw away a run that a single end-turn would have
rescued.

A screen the game cannot act on re-presents itself unchanged, and a
deterministic policy answers it the same way every time -- no exception, no
timeout, just a session that quietly stops making progress. These counters are
the only thing that notices."""

TREASURE_COLLECT_ACTION = "collect"
BOSS_RELIC_PICK_ACTION = "pick_relic"


def load_model(model_path: str) -> Any:
    """Load a trained MaskablePPO model.

    Args:
        model_path: Path to the saved model (.zip file).

    Returns:
        Loaded MaskablePPO model instance.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        logger.error(
            "sb3-contrib is required. Install with: pip install sb3-contrib"
        )
        raise

    logger.info("Loading model from %s", model_path)
    model = MaskablePPO.load(model_path)
    logger.info("Model loaded successfully.")
    return model


#: Every LiveSearch.decide failure, with the state that caused it. Read by
#: scripts/replay_search_failures.py, which rebuilds each one offline.
SEARCH_FAILURE_LOG = "output/live_search_failures.jsonl"


def _record_search_failure(state: dict, exc: BaseException) -> None:
    """Append the failing bridge state and traceback, one JSON object per line.

    The state is the whole point. `CombatSituation.from_bridge_state` /
    `to_combat` are pure functions of it, so one captured line reproduces the
    failure offline against the decompile, without the game running and without
    waiting for the same fight to come round again.
    """
    import traceback

    path = Path(SEARCH_FAILURE_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "t": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "state": state,
        }, default=str) + "\n")


def run_agent(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 9002,
    deterministic: bool = True,
    verbose: bool = False,
    record_replay_path: str | None = None,
    replay_factory: str | None = None,
    speed: str = "turbo",
    allow_random_fallback: bool = False,
    max_runs: int = 1,
    on_run_end: "Callable[[dict[str, Any]], None] | None" = None,
    tell_cyra: bool = False,
    combat_policy_path: str | None = None,
    journal_path: str | None = None,
    live_search: bool = False,
    search_rollout_model: bool = False,
    capture_raw_path: str | None = None,
    capture_raw_per_type: int = 25,
    capture_raw: "RawCapture | None" = None,
    force_seed: str | None = None,
    policy_path: str | None = None,
    on_journal_event: "Callable[[dict[str, Any]], None] | None" = None,
) -> None:
    """Main agent loop.

    Connects to the game, loads the model, and plays indefinitely
    until disconnected or interrupted.

    Args:
        model_path: Path to saved MaskablePPO model.
        host: Bridge server host.
        port: Bridge server port.
        deterministic: Whether to use deterministic action selection.
        verbose: Whether to log every action taken.
        max_runs: How many runs to play before returning. The mod already starts
            runs back to back, so anything above 1 simply stops this side from
            exiting at the first death -- which is the only reason a live session
            used to end after one run.
        on_run_end: Called with a summary dict as each run finishes.
        tell_cyra: Publish the four run milestones to cyra_brain. Off by default
            so training and evaluation never depend on a broker being up.
        journal_path: JSONL file recording every room, fight, card played and
            reward taken. The per-run summary says a run reached floor 11; this
            says what happened on the way, which is what a decision about what
            to fix next actually needs.
        live_search: Use the SearchAgent's turn planner for combat decisions
            instead of the trained model's single-step argmax. The trained
            model is one-turn myopic; the search enumerates legal orderings
            and lets the enemies reply on a clone, lifting boss win rate
            from 6.7% to ~20% on the harvested benchmark per
            docs/MODELS.md:120. Requires the mod patch from PR #6 (Phase 1.1)
            to send `encounter` / `encounter_seed` / `combat_seed`; without
            them, live_search raises ValueError on the first combat_action
            and the runner falls back to END_TURN. The model's path is
            unchanged when this is False.
    """
    model = load_model(model_path)
    adapter = StateAdapter()

    # THE POLICY IS A FILE, NOT THE GLOBALS. The module constants hold the
    # shipped defaults; the decision path runs whatever this file says, and the
    # journal stamps it. See sts2_env/policy_config.py for why nothing may
    # patch a global instead.
    from sts2_env.policy_config import PolicyConfig
    policy = PolicyConfig.load(policy_path)
    from sts2_env.policy_config import set_active_policy
    set_active_policy(policy)
    logger.info("policy %s loaded from %s (git %s)",
                policy.policy_version, policy.source_path, policy.git_sha)

    # The SearchAgent-backed combat decider, opt-in. Constructed once and
    # reset on each fight's combat_start by the main loop. Kept in scope
    # along with the previous combat action index so the local sim mirrors
    # the actions the runner sends to the live game.
    live_search_agent = None
    if live_search:
        from sts2_env.bridge.live_search import LiveSearch

        live_search_agent = LiveSearch(weights=policy.eval_weights)
        logger.info("LiveSearch enabled: combat decisions use the SearchAgent "
                    "planner instead of the trained model's argmax. Requires "
                    "the Phase 1.1 mod patch; if the mod does not send "
                    "`encounter` / `encounter_seed`, the first combat_action "
                    "will raise and the runner falls back to END_TURN per step.")

    # Which adapter to use is decided by the model, not a flag. A combat model
    # wants 131 dims and a full-run model 277; guessing wrong produces a shape
    # error at the first decision, and asking the model removes the guess.
    run_adapter = None
    expected_obs = int(model.observation_space.shape[0])
    if expected_obs == RUN_OBS_SIZE:
        run_adapter = RunStateAdapter()
        adapter = run_adapter          # combat also goes through it, same layout
        logger.info("Full-run model detected (%d dims): the model decides card "
                    "rewards, map, rest and events as well as combat.", expected_obs)
    elif expected_obs == COMBAT_OBS_SIZE:
        logger.info("Combat model detected (%d dims): non-combat decisions use "
                    "heuristics.", expected_obs)
    else:
        raise ValueError(
            f"Model expects {expected_obs} observation dims, which is neither the "
            f"combat size ({COMBAT_OBS_SIZE}) nor the full-run size ({RUN_OBS_SIZE}). "
            "It was probably trained against a different observation layout."
        )

    # Optional separate combat policy for hierarchical models.
    combat_model = None
    if combat_policy_path is not None:
        combat_model = load_model(combat_policy_path)
        c_expected = int(combat_model.observation_space.shape[0])
        if c_expected != COMBAT_OBS_SIZE:
            raise ValueError(
                f"Combat policy expects {c_expected} observation dims, but combat "
                f"observations are {COMBAT_OBS_SIZE}."
            )

    # Rollouts inside the search, once there is a model to roll with. Attached
    # here rather than where LiveSearch is built because the combat policy is
    # loaded below that point, and the combat policy is the one that should roll
    # a combat if a separate one was supplied.
    if live_search_agent is not None and search_rollout_model:
        from sts2_env.search.turn_search import model_playout_policy

        live_search_agent.set_playout_policy(
            model_playout_policy(combat_model or model))
        logger.info(
            "Search rollouts use the trained model rather than the "
            "block-then-attack heuristic. MODELS.md names this as the next "
            "thing to try: four turns of lookahead scored WORSE than two "
            "because the heuristic playout compounds its own errors, and it "
            "ranks Powers last, so a rollout never shows a Power used well. "
            "Affordable because the search spends ~0.08s of its 3.0s budget "
            "on a boss turn.")
        logger.info("Separate combat policy loaded from %s", combat_policy_path)

    logger.info("Connecting to STS2 at %s:%d...", host, port)

    with STS2GameClient(host=host, port=port) as raw_client:
        client: STS2GameClient | BridgeReplayRecorder
        if record_replay_path is not None:
            metadata = {
                "model_path": model_path,
                "host": host,
                "port": port,
            }
            if replay_factory is not None:
                metadata["scenario_factory"] = replay_factory
            client = BridgeReplayRecorder(raw_client, metadata=metadata)
            logger.info("Recording supported bridge states to %s", record_replay_path)
        else:
            client = raw_client

        # Session options, sent once before the game starts a run. The mod now
        # waits for a client rather than beginning on its own, so this arrives in
        # time to apply to the first run rather than the second.
        raw_client.send_action({
            "action": "configure",
            "speed": speed,
            # Off by default: with the fallback on, the mod plays randomly when the
            # agent does not answer, logs only to the game's own log, and the trace
            # records those actions as the model's. Silence in this console should
            # mean nothing is happening.
            "allow_random_fallback": allow_random_fallback,
            # Every run on one seed. The mod also reads STS2_RL_SEED, but that
            # is read in the GAME process, which Steam launches separately --
            # exporting it in front of this command sets it on the wrong
            # process and does nothing. This is the path that works.
            **({"seed": force_seed} if force_seed else {}),
        })
        logger.info("Requested speed=%s, random_fallback=%s", speed, allow_random_fallback)
        if force_seed:
            logger.info("FORCING every run onto seed %s", force_seed)

        logger.info("Connected. Starting agent loop.")

        step_count = 0
        combat_count = 0
        _was_in_combat = False
        last_enemy_id: str | None = None
        # The action we last returned from live_search; mirrors into the local
        # sim on the next decide call. Reset on each combat_start.
        _live_search_last_action: int | None = None
        # Two-strike policy for live_search failures: the first exception in a
        # combat is logged; the second switches the rest of that combat to the
        # trained model so the run keeps playing. Re-enabled at the next
        # combat_start by reset_for_new_fight clearing these.
        _live_search_failures = 0
        _live_search_disabled_for_combat = False
        run_index = 0
        run_started = time.monotonic()
        # Last run-level fields seen. game_over does not always carry them, so
        # the floor a run ended on is remembered as it goes rather than read off
        # the final message.
        progress: dict[str, Any] = {}
        _relic_warning_done = False
        cyra = CyraPublisher(enabled=tell_cyra)
        milestones = MilestoneWatcher()
        # How many times each event screen has been answered this run, so a
        # screen that re-presents itself and charges more each time (Hot Baths)
        # is not paid over and over.
        events_seen: dict[str, int] = {}
        last_fingerprint: tuple | None = None
        identical_states = 0
        _pending_removal = False
        stuck_log = "output/stuck_states.jsonl"
        journal = RunJournal(journal_path, model=model_path,
                             policy_version=policy.policy_version,
                             git_sha=policy.git_sha,
                             on_event=on_journal_event)
        journal.start_run(1)
        # The journal line doubles as the telemetry event downstream: which
        # policy loaded, stamped with the sha of the code playing it.
        journal.write("policy_version_loaded",
                      policy_version=policy.policy_version,
                      git_sha=policy.git_sha, source=policy.source_path)
        # The journal records decisions and drops the state behind them. This
        # keeps whole states, verbatim, so the bridge parsers can be replayed
        # against what the mod really sends rather than what we assumed.
        # A caller that spans several `run_agent` calls -- `live_eval`'s
        # restart-on-crash loop -- passes ITS capture in, because
        # `RawCapture` opens with "w" and constructing one per call truncates
        # the file on every relaunch. That cost the 2026-08-17 `postfix`
        # session all 68 of its boss fights: five segments, four restarts, and
        # the file left holding the last segment's single run. Sharing the
        # instance also shares the quota counters, so a crash-heavy session
        # cannot re-fill its floor-1 buckets once per segment either.
        owns_capture = capture_raw is None
        raw_capture = capture_raw or (
            RawCapture(capture_raw_path, per_type=capture_raw_per_type)
            if capture_raw_path else None
        )
        # Wrapped once, so every action the runner sends is recorded on the way
        # through rather than at each of the fourteen places that send one.
        client = journal.wrap(client)

        try:
            while True:
                try:
                    logger.info("Waiting for game state...")
                    state = client.receive_state()
                    logger.info("Received: type=%s", state.get("type", "?"))
                except TimeoutError:
                    logger.warning("Timeout waiting for state. Sending ping...")
                    if client.ping():
                        continue
                    else:
                        logger.error("Lost connection. Attempting reconnect...")
                        _reconnect_with_retry(client)
                        continue
                except ConnectionError:
                    logger.error("Connection lost. Attempting reconnect...")
                    _reconnect_with_retry(client)
                    continue

                # Captured here rather than beside journal.observe: this is the
                # first line after the state arrives, so what lands in the file
                # is the mod's own bytes, before any phase dispatch or early
                # `continue` can filter a screen out of the sample.
                if raw_capture is not None:
                    raw_capture.observe(state)

                msg_type = state.get("type", "")
                phase = _phase_for_state(state)
                step_count += 1

                if verbose and step_count % 10 == 1:
                    logger.info("Step %d: type=%s phase=%s", step_count, msg_type, phase)

                if verbose and msg_type:
                    logger.debug("Received: type=%s keys=%s", msg_type, list(state.keys()))

                if phase == MSG_TYPE_PONG:
                    continue
                # A mod older than the relic observation sends no "relics" field,
                # and the encoder then reads "owns nothing" -- wrong rather than
                # absent, and completely silent. The model would play every fight
                # relic-blind while the log looked healthy, so say it once, loudly.
                if not _relic_warning_done and "relic_count" in state:
                    _relic_warning_done = True
                    if "relics" not in state:
                        logger.error(
                            "This mod sends relic_count but not `relics`, so the "
                            "observation will read as owning NO relics. Rebuild the "
                            "mod; results from this run are not comparable.")
                    elif "potion_slots" not in state:
                        logger.warning(
                            "Mod sends no `potion_slots`; potions will read as empty.")
                    elif "deck" not in state:
                        logger.error(
                            "This mod sends deck_size but not `deck`, so the "
                            "observation will read as an EMPTY deck and every card "
                            "reward is decided blind. Rebuild the mod.")

                journal.observe(state)

                # A screen the game cannot act on presents itself again, the
                # policy is deterministic, and the same action goes back forever.
                # Seen live on 2026-08-05: end_turn sent six times at round 2 with
                # the hand, HP and round frozen, because a Necrobinder OstyAttack
                # sat unplayable in an Ironclad hand and end-turn was greyed out.
                # No exception, no timeout -- just a session that stops.
                fingerprint = _state_fingerprint(state)
                if fingerprint is not None and fingerprint == last_fingerprint:
                    identical_states += 1
                else:
                    identical_states = 0
                    last_fingerprint = fingerprint

                if identical_states == STUCK_WARN_AFTER:
                    logger.error(
                        "The game has not moved for %d states. The screen is "
                        "probably waiting on something the agent cannot send. "
                        "Dumping it to %s.",
                        identical_states, stuck_log,
                    )
                    _record_stuck_state(stuck_log, state, identical_states)

                if identical_states >= STUCK_ABANDON_AFTER:
                    # There is no abandon command: the mod abandons from the main
                    # menu, and this side cannot reach it. So the honest response
                    # is to stop cleanly rather than spin -- live_eval then prints
                    # its summary and every finished run is kept, instead of the
                    # session hanging until someone notices.
                    logger.error(
                        "Still stuck after %d identical states, including %d "
                        "with end-turn forced. There is no abandon command on "
                        "this side, so stopping here; the runs already finished "
                        "are kept and summarised. The screen is in %s.",
                        identical_states,
                        identical_states - STUCK_ESCALATE_AFTER,
                        stuck_log,
                    )
                    journal.write("stuck", states=identical_states,
                                  screen=state.get("type"),
                                  escalated=True,
                                  ended_session=True)
                    break

                if identical_states >= STUCK_ESCALATE_AFTER and phase in Phase.COMBAT_PHASES:
                    # Whatever we keep choosing, the game will not take it. End
                    # the turn instead: it is always legal, and it clears the
                    # whole class of "a rule this side does not model makes the
                    # card unplayable" -- next turn deals a new hand.
                    if identical_states == STUCK_ESCALATE_AFTER:
                        logger.error(
                            "Refused %d times; forcing end-turn rather than "
                            "abandoning. If the game takes it, the run "
                            "continues and the screen in %s is worth reading "
                            "for what it would not let us play.",
                            identical_states, stuck_log,
                        )
                        journal.write("stuck", states=identical_states,
                                      screen=state.get("type"),
                                      escalated=True,
                                      ended_session=False)
                    client.end_turn()
                    continue

                # Counted here rather than never: this was initialised, reset and
                # reported without ever being incremented, so every live record
                # written so far says "combats": 0.
                if phase in Phase.COMBAT_PHASES and not _was_in_combat:
                    combat_count += 1
                    # A new fight has started: reset the live-search mirror so
                    # the next decide call rebuilds the local sim fresh, and
                    # re-enable live_search in case it was disabled near the
                    # end of the previous fight by the two-strike fallback.
                    if live_search_agent is not None:
                        live_search_agent.reset_for_new_fight()
                        _live_search_last_action = None
                        _live_search_failures = 0
                        _live_search_disabled_for_combat = False
                _was_in_combat = phase in Phase.COMBAT_PHASES

                # Track the enemy the run is currently fighting, so the run-end
                # summary can say which one killed it. The bridge nests combat
                # fields inside `combat_state`; the same fallback as
                # state_adapter.py is used. Only the first alive enemy is
                # recorded -- enough to name the boss or the elite, and a
                # multi-enemy fight names the one the bridge lists first.
                if _was_in_combat:
                    combat_payload = state.get("combat_state") or state
                    for e in (combat_payload.get("enemies") or []):
                        if e.get("is_alive", True):
                            last_enemy_id = e.get("id")
                            break

                for event in milestones.observe(state):
                    logger.info("CYRA: %s", event["text"])
                    cyra.publish(event)

                # The deck's direction, checked on any state carrying a deck
                # rather than at the card-reward screen itself: at that point the
                # card just chosen is not in the deck yet, so committing there
                # would always lag a pick behind. MilestoneWatcher keeps this to
                # once a run.
                if state.get("deck"):
                    direction = _deck_direction(state)
                    committed = direction.committed if direction else None
                    if committed:
                        event = milestones.archetype_chosen(
                            committed, direction.confidence)
                        if event:
                            logger.info("CYRA: %s", event["text"])
                            cyra.publish(event)

                # room_type included so the log says WHERE a run ended, not just
                # on what floor. Two live runs both ended on floor 17 while the
                # simulator puts the act 1 boss on 16 -- so "floor >= 17 cleared
                # act 1" may be counting boss deaths as clears. Recording the room
                # settles that from data instead of from an off-by-one argument.
                for field in ("floor", "act", "act_floor", "run_hp", "run_max_hp",
                              "deck_size", "gold", "relic_count", "potion_count",
                              "room_type"):
                    if field in state:
                        progress[field] = state[field]

                if phase in TERMINAL_PHASES:
                    result = state.get("result", state.get("message", "unknown"))
                    logger.info("Run finished: %s", result)
                    run_index += 1
                    if on_run_end is not None:
                        summary = dict(progress)
                        summary.update({
                            k: v for k, v in state.items()
                            if k in ("floor", "act", "run_hp", "run_max_hp",
                                     "deck_size", "gold", "relic_count", "room_type")
                        })
                        # `act_cleared` is derived here so a JSONL record
                        # carries its own verdict and downstream reports do
                        # not have to re-derive it from `act` -- which is
                        # missing on some bridge versions, where the fallback
                        # `floor > ACT1_BOSS_FLOOR` in live_eval._cleared_act_1
                        # then has to do the work. Stamping it here records
                        # which path was taken.
                        run_hp = progress.get("run_hp")
                        died = isinstance(run_hp, int) and run_hp <= 0
                        summary.update({
                            "run": run_index,
                            "result": result,
                            "steps": step_count,
                            "combats": combat_count,
                            "seconds": round(time.monotonic() - run_started, 1),
                            "act_cleared": int(progress.get("act", 1) or 1) >= 2,
                            # `death_enemy_id` is the enemy the player was
                            # fighting when the run ended. None on a win or
                            # when the bridge never reported a combat state,
                            # which is why the field is present-and-None on
                            # old logs rather than absent -- a missing field
                            # could mean either, and that ambiguity is the
                            # kind of thing that costs a debugging hour.
                            "death_enemy_id": last_enemy_id if died else None,
                        })
                        # WAS THE SEARCH CUT SHORT? LiveSearch has counted
                        # this from the start and nothing ever read it, so
                        # whether live play is truncated has never been
                        # measured. It is a real live/offline difference:
                        # live runs on a 3s WALL CLOCK, offline on a hard
                        # node cap with 60s, and a wall clock is the thing
                        # that silently truncated two earlier sweeps.
                        #
                        # It matters for the 44-point boss gap (72% offline
                        # against 28% live). If the wide boss turns are the
                        # ones exhausting the budget, live is planning those
                        # fights on a partial tree while offline never does.
                        #
                        # `stats` is a PROPERTY. Calling it raised TypeError,
                        # which the bare `except: pass` here swallowed, so the
                        # fields were silently absent from a whole session and
                        # the measurement looked like "zero truncations" rather
                        # than "never recorded". Logged now, not swallowed.
                        #
                        # NOTE the counters reset per fight: reset_for_new_fight
                        # rebuilds the SearchAgent, so these are the LAST
                        # fight's numbers, not the run's.
                        if live_search_agent is not None:
                            try:
                                st = live_search_agent.stats
                                summary["searches"] = st.get("searches")
                                summary["searches_truncated"] = st.get("budget_exhausted")
                            except Exception:
                                logger.exception("could not read LiveSearch stats")
                        journal.record_run_end(summary)
                        on_run_end(summary)
                    if run_index >= max_runs:
                        break
                    # The mod returns to the menu and starts the next run on its
                    # own; this side only has to stay connected and keep answering.
                    logger.info("--- run %d finished; waiting for the next one ---",
                                run_index)
                    step_count = 0
                    combat_count = 0
                    _was_in_combat = False
                    last_enemy_id = None
                    _live_search_last_action = None
                    _live_search_failures = 0
                    _live_search_disabled_for_combat = False
                    if live_search_agent is not None:
                        live_search_agent.reset_for_new_fight()
                    progress = {}
                    milestones.reset()
                    journal.start_run(run_index + 1)
                    events_seen.clear()
                    # The pending "what did that choice cost" comparison only,
                    # not the learned costs -- an option that charges gold in
                    # one run charges it in the next.
                    _reset_event_gold_learning()
                    run_started = time.monotonic()
                    continue
                if phase == MSG_TYPE_ERROR:
                    logger.warning("Game error: %s", state.get("message", ""))
                    continue

                # ---- Full-run model: it decides everything, not just combat ----
                # A combat-shaped model leaves these to the heuristics below, which
                # is why runs stalled around floor 8: the deck was built by
                # ("power", "attack", "skill") and the map by room priority. A
                # run-shaped model has been trained to see the options and choose.
                if run_adapter is not None and msg_type in RUN_MODEL_STATES:
                    obs = run_adapter.encode_observation(state)
                    mask = run_adapter.compute_action_mask(state)
                    action, _states = model.predict(
                        obs, action_masks=mask, deterministic=deterministic,
                    )
                    decoded = run_adapter.decode_action(int(action), state)
                    if verbose:
                        logger.info("%s: model chose %s", msg_type.upper(), decoded)

                    if cyra.enabled and msg_type == "map_select":
                        event = milestones.map_choice(
                            state, int(decoded.get("index", -1)),
                            _decision_margin(model, obs, mask))
                        if event is not None:
                            logger.info("CYRA: %s", event["text"])
                            cyra.publish(event)
                    if decoded.get("action") == "skip":
                        client.skip()
                    else:
                        client.choose(int(decoded.get("index", 0)))
                    continue

                if phase in Phase.COMBAT_PHASES:
                    # ---- Combat: live search OR trained model ----
                    # Live search is SearchAgent-backed: enumerate legal
                    # orderings this turn, score by ending the turn on a
                    # copy and letting enemies reply, keep the best line.
                    # The action returned is in the same Discrete(115) layout
                    # the model uses, so the existing decode + send path is
                    # reused.
                    if live_search_agent is not None and not _live_search_disabled_for_combat:
                        # Build/mirror the local sim, ask the SearchAgent.
                        try:
                            action_int = live_search_agent.decide(
                                state, prev_action=_live_search_last_action,
                            )
                            # Attribute the fight, not just the run. A run with
                            # search enabled can still have its BOSS played by
                            # the trained model after two raises, and that was
                            # unrecoverable from the old journals.
                            try:
                                journal.note_search()
                            except Exception:  # noqa: BLE001 - never cost a run
                                pass
                        except Exception as search_exc:
                            try:
                                journal.note_search(failed=True)
                            except Exception:  # noqa: BLE001
                                pass
                            # PERSIST IT, WITH THE STATE THAT CAUSED IT. This
                            # only ever went to the console via logger.exception,
                            # so the one thing that would let it be fixed -- the
                            # bridge state the simulator could not rebuild --
                            # scrolled away with the session. `to_combat` is a
                            # pure function of that dict, so a captured state
                            # reproduces the failure offline with no game.
                            #
                            # This is not a cosmetic gap. A raise here does not
                            # degrade the search, it REPLACES it: the fight falls
                            # back to the trained combat model. Offline always
                            # searches, which is the leading explanation for
                            # boss win rate being 72% offline and 28% live.
                            try:
                                _record_search_failure(state, search_exc)
                            except Exception:  # noqa: BLE001 - never mask the original
                                pass
                            # Once-per-combat escalation. The first raise logs
                            # loudly; the second switches this combat to the
                            # trained model. Repeated END_TURN firings would
                            # have the player do nothing and die on the first
                            # encounter (the regression observed 2026-08-06 --
                            # an unnormalised potion id raised every step, the
                            # fallback to END_TURN was the actual cause of
                            # death, not the search being bad). Disable for
                            # this combat only; the combat_start transition
                            # will re-enable.
                            _live_search_failures += 1
                            if _live_search_failures == 1:
                                logger.exception(
                                    "LiveSearch.decide raised once; this "
                                    "combat will switch to the trained model "
                                    "if it raises again. Likely cause: the "
                                    "bridge sends a state the simulator "
                                    "cannot reconstruct (unknown potion, "
                                    "unmodelled power, mismatched Id.Entry). "
                                    "The model path is unaffected and plays "
                                    "the fight instead."
                                )
                            elif _live_search_failures >= 2:
                                logger.error(
                                    "LiveSearch.decide raised %d times in "
                                    "this combat; switching to the trained "
                                    "model for the rest of the fight.",
                                    _live_search_failures,
                                )
                                _live_search_disabled_for_combat = True
                                _live_search_last_action = None
                                # Fall through to the model branch below.
                            else:
                                # First failure: bounce to END_TURN once, the
                                # next step is the second-chance model switch.
                                client.end_turn()
                                _live_search_last_action = 0
                                continue
                        else:
                            decoded = adapter.decode_action(action_int, state)
                            if verbose:
                                logger.info("LIVE_SEARCH: chose %s", decoded)
                            _live_search_last_action = action_int
                            if decoded["type"] == ActionType.END_TURN:
                                client.end_turn()
                            elif decoded.get("out_of_hand"):
                                client.use_potion(
                                    decoded.get("slot", decoded.get("potion_slot", -1)),
                                    decoded.get("target_index", -1),
                                )
                            else:
                                client.play_card(
                                    decoded["card_index"],
                                    decoded.get("target_index", -1),
                                )
                            continue

                    # ---- Combat: use trained model ----
                    # Hierarchical models may delegate combat to a separate policy.
                    combat_adapter = adapter
                    combat_policy = model
                    if combat_model is not None:
                        combat_policy = combat_model
                        if run_adapter is not None:
                            combat_adapter = run_adapter._combat

                    obs = combat_adapter.encode_observation(state)
                    mask = combat_adapter.compute_action_mask(state)

                    # Ensure at least one action is valid
                    if mask.sum() == 0:
                        logger.warning("No valid actions! Defaulting to END_TURN.")
                        client.end_turn()
                        continue

                    action, _states = combat_policy.predict(
                        obs,
                        action_masks=mask,
                        deterministic=deterministic,
                    )
                    action_int = int(action)

                    decoded = combat_adapter.decode_action(action_int, state)

                    if verbose:
                        _log_combat_action(state, action_int, decoded)

                    if decoded["type"] == ActionType.END_TURN:
                        client.end_turn()
                    elif decoded.get("out_of_hand"):
                        client.use_potion(
                            decoded.get("slot", decoded.get("potion_slot", -1)),
                            decoded.get("target_index", -1),
                        )
                    else:
                        client.play_card(
                            decoded["card_index"],
                            decoded.get("target_index", -1),
                        )

                elif phase == Phase.MAP_SELECT:
                    choice = _pick_map_node(state)
                    if verbose:
                        logger.info("MAP: choosing node %d", choice)
                    client.choose(choice)

                elif phase == Phase.CARD_REWARD:
                    if msg_type == BridgeStateType.CARD_BUNDLE:
                        choice = _pick_card_bundle_index(state)
                        if verbose:
                            logger.info("CARD_BUNDLE: choosing bundle %s", choice)
                        client.choose(choice)
                    elif msg_type == BridgeStateType.CARD_SELECT:
                        indexes = _pick_card_select_indexes(
                            state, removing=_pending_removal)
                        # One screen per purchase: clear it however the screen
                        # resolves, so a later transform is never mistaken for
                        # a removal.
                        _pending_removal = False
                        if verbose:
                            logger.info("CARD_SELECT: choosing indexes %s", indexes)
                        if not indexes:
                            client.skip()
                        elif len(indexes) == 1:
                            client.choose(indexes[0])
                        else:
                            client.choose_many(indexes)
                    else:
                        choice = (
                            _pick_reward_screen_option(state)
                            if msg_type == BridgeStateType.REWARD_SCREEN
                            else _pick_card_reward_index(state)
                        )
                        if msg_type == BridgeStateType.CARD_REWARD:
                            # What she considered, not only what she chose.
                            _log_card_reward_options(journal, state, choice)
                        if verbose:
                            logger.info("CARD_REWARD: choosing option %s", choice)
                        _send_choice_or_skip(client, choice)

                elif phase == Phase.REST:
                    choice = _pick_rest_option(state)
                    if verbose:
                        logger.info("REST: choosing option %d", choice)
                    client.choose(choice)

                elif phase == Phase.SHOP:
                    choice = _pick_shop_option(state)
                    # The card_select that follows a bought removal is the only
                    # one we can identify, and identifying it is what lets that
                    # screen target a curse instead of guessing. Set here rather
                    # than inferred there, because this is where the knowledge
                    # actually exists.
                    _pending_removal = _is_removal_option(state, choice)
                    if verbose:
                        logger.info("SHOP: choosing option %d%s", choice,
                                    " (card removal)" if _pending_removal else "")
                    client.choose(choice)

                elif phase == Phase.EVENT:
                    choice = (
                        _pick_crystal_sphere_option(state)
                        if msg_type == BridgeStateType.CRYSTAL_SPHERE
                        else _pick_event_option(state, events_seen)
                    )
                    if verbose:
                        logger.info("EVENT: choosing option %d", choice)
                    client.choose(choice)

                elif phase == Phase.TREASURE:
                    choice = _pick_treasure_option(state)
                    if verbose:
                        logger.info("TREASURE: choosing option %d", choice)
                    client.choose(choice)

                elif phase == Phase.BOSS_RELIC:
                    choice = _pick_boss_relic_option(state)
                    if verbose:
                        logger.info("BOSS_RELIC: choosing option %d", choice)
                    client.choose(choice)

                elif phase == Phase.COMBAT_WAITING:
                    # Game is processing enemy turn / animations — just wait
                    pass

                else:
                    logger.debug("Unknown phase '%s', waiting...", phase)

                # Log progress periodically
                if step_count % 100 == 0:
                    logger.info("Step %d, combats seen: %d", step_count, combat_count)
        except KeyboardInterrupt:
            # Ctrl-C mid-run used to throw the run away. The summary is only
            # emitted when the bridge reports a terminal state, so interrupting
            # after the player died but before the game-over screen arrived
            # lost the whole run -- including, once, the first live boss kill
            # this project ever recorded. The journal and the raw capture
            # survived because they flush per event; the run record did not
            # exist yet.
            #
            # So build it here from the progress tracked so far and mark it
            # interrupted, rather than letting a keystroke decide whether a run
            # counts.
            logger.warning(
                "Interrupted mid-run. Recording run %d from the progress so "
                "far rather than discarding it.", run_index + 1)
            if on_run_end is not None:
                summary = dict(progress)
                summary.update({
                    "run": run_index + 1,
                    "result": "interrupted",
                    "steps": step_count,
                    "combats": combat_count,
                    "seconds": round(time.monotonic() - run_started, 1),
                    "act_cleared": int(progress.get("act", 1) or 1) >= 2,
                    "death_enemy_id": None,
                    "interrupted": True,
                })
                try:
                    journal.record_run_end(summary)
                    on_run_end(summary)
                except Exception:  # noqa: BLE001 - never mask the interrupt
                    logger.exception("could not record the interrupted run")
            raise
        finally:
            if raw_capture is not None:
                # In the finally so a Ctrl-C or a lost connection still lands
                # the trailer -- the counts of what was seen versus kept are
                # how you tell a rare screen from a truncated one.
                #
                # Only if we opened it. A borrowed capture belongs to a caller
                # that will keep using it after this run_agent returns, and
                # closing it here would land the trailer mid-session and drop
                # every state from the segments that follow.
                if owns_capture:
                    raw_capture.close()
                logger.info("Raw protocol capture: %s", raw_capture.summary)
            cyra.close()
            if isinstance(client, BridgeReplayRecorder):
                saved_path = client.save(record_replay_path)
                logger.info("Saved bridge replay trace to %s", saved_path)


# ----------------------------------------------------------------
# Heuristic decision functions for non-combat phases
# ----------------------------------------------------------------


def _decision_margin(model, obs, mask) -> float | None:
    """How far the top choice beat the runner-up, from the policy itself.

    This is the only honest introspection available: the policy really was nearly
    tied, or it really was not. The number stays inside cyra_game -- milestones.py
    turns it into a phrase -- because a softmax over action logits is not a
    calibrated confidence and must never be spoken as one.

    Returns None rather than raising: commentary is never worth risking a run.
    """
    try:
        import numpy as np
        import torch

        obs_tensor = torch.as_tensor(np.asarray([obs]), device=model.device)
        mask_tensor = torch.as_tensor(np.asarray([mask]), device=model.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            probs = dist.distribution.probs[0]
        if probs.numel() < 2:
            return None
        top2 = torch.topk(probs, 2).values
        return float(top2[0] - top2[1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read decision margin (%s)", type(exc).__name__)
        return None


def _phase_for_state(state: dict[str, Any]) -> str:
    msg_type = state.get("type", "")
    return {
        BridgeStateType.COMBAT_ACTION: Phase.COMBAT_PLAY,
        MSG_TYPE_GAME_STATE: state.get("phase", Phase.UNKNOWN),
        BridgeStateType.MAP_SELECT: Phase.MAP_SELECT,
        BridgeStateType.REWARD_SCREEN: Phase.CARD_REWARD,
        BridgeStateType.CARD_BUNDLE: Phase.CARD_REWARD,
        BridgeStateType.CARD_REWARD: Phase.CARD_REWARD,
        BridgeStateType.CARD_SELECT: Phase.CARD_REWARD,
        BridgeStateType.REST_SITE: Phase.REST,
        BridgeStateType.SHOP: Phase.SHOP,
        BridgeStateType.CRYSTAL_SPHERE: Phase.EVENT,
        BridgeStateType.EVENT: Phase.EVENT,
        BridgeStateType.TREASURE: Phase.TREASURE,
        BridgeStateType.BOSS_RELIC: Phase.BOSS_RELIC,
        BridgeStateType.GAME_OVER: BridgeStateType.GAME_OVER,
        BridgeStateType.RUN_COMPLETE: BridgeStateType.RUN_COMPLETE,
        MSG_TYPE_PONG: MSG_TYPE_PONG,
        MSG_TYPE_ERROR: MSG_TYPE_ERROR,
    }.get(msg_type, state.get("phase", Phase.UNKNOWN))


def required_hp_fraction(room_type: str) -> float:
    """How healthy the player should be before entering this kind of room.

    Rooms absent from the table -- shops, rest sites, treasure -- return 0.0 and
    are always eligible, which is what makes them the recovery options.

    `Unknown` takes the monster figure rather than 0: an Unknown node resolves to
    a fight often enough that treating it as free is how a run walks into one at
    12 HP, which the logs show happening four times.
    """
    return ROOM_MIN_HP_FRACTION.get(_canonical_text(room_type), 0.0)


def _plan_from_state(state: dict[str, Any], nodes: list[dict]) -> int | None:
    """Route to the boss using the full map, or None if it was not sent.

    The map arrives as `state["map"]` = {"nodes": [{"id", "type", "children"}]},
    and each choosable node carries `map_id` linking it into that graph. Callers
    that cannot supply it -- the live mod currently sends only the reachable
    next nodes -- fall through to the greedy chooser unchanged, so this is
    additive and cannot regress anything that does not opt in.
    """
    # OFF BY DEFAULT: this was MEASURED WORSE and must not ship silently.
    # 150 paired offline seeds, five parameter settings: reach 60.7% -> 56.0%,
    # paired net -2 (McNemar p=0.89), best variant +0 (p=1.00). Recorded as a
    # MISS in SCOREBOARD.md.
    #
    # The code stays because the mod now sends the map and the next attempt
    # should not have to rebuild the plumbing -- the likely defect is that it
    # routes on HP feasibility while the game's own guide keys on DECK
    # strength, which is not modelled here at all. Set STS2_MAP_PLANNING=1 to
    # try again. It stays off until a paired run beats greedy.
    import os
    if os.environ.get("STS2_MAP_PLANNING") != "1":
        return None

    graph = state.get("map") or {}
    map_nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not map_nodes:
        return None

    by_id = {str(n.get("id")): n for n in map_nodes if isinstance(n, dict)}
    starts = [(i, by_id.get(str(n.get("map_id")))) for i, n in enumerate(nodes)]
    starts = [(i, n) for i, n in starts if n is not None]
    if not starts:
        return None

    hp, max_hp = _read_hp_pair_from_state(state)
    if not hp or not max_hp:
        return None

    from sts2_env.bridge.map_planning import plan_route

    index_of = {id(n): i for i, n in starts}
    route = plan_route(
        children_of=lambda n: [by_id[c] for c in (n.get("children") or [])
                               if c in by_id],
        start_nodes=[n for _, n in starts],
        room_type_of=lambda n: n.get("type"),
        hp_fraction=hp / max_hp,
    )
    if route is None:
        return None
    chosen = index_of.get(id(route.first_step))
    if chosen is None:
        return None
    logger.info("MAP: planned route -- %s", route.describe())
    return _read_index(nodes[chosen], chosen)


def _can_afford(state: dict[str, Any], node: dict[str, Any]) -> bool:
    """Is the player healthy enough for this room to be worth entering?"""
    hp, max_hp = _read_hp_pair_from_state(state)
    if hp is None or not max_hp:
        return True
    return hp >= required_hp_fraction(node.get("type")) * max_hp


def _read_hp_pair_from_state(state: dict[str, Any]) -> tuple[int | None, int | None]:
    for container in _candidate_player_containers(state):
        hp, max_hp = _read_hp_pair(container)
        if hp is not None:
            return hp, max_hp
    return None, None


def _pick_map_node(state: dict[str, Any]) -> int:
    """Choose a reachable map node, refusing rooms the current HP cannot pay for.

    Rooms are still ranked by the same preference tables -- elites before
    monsters, because elites are where relics come from and a run that never
    takes one arrives at act 3 with nothing. What changed is that a room is only
    *eligible* if the player can survive its p90 damage.

    So the agent still wants the elite. It just takes it at 70 HP instead of 45,
    and goes to a rest site first when it cannot. Measured elite death rate at
    70-79 HP is 0%; at 40-59 it is 18-29%.

    Falling back to the full priority list when nothing is affordable is
    deliberate. Some map positions offer only fights, and refusing to move is not
    an option the game supports -- the least-bad fight beats a stalled run.
    """
    nodes = list(state.get("nodes", []))
    if not nodes:
        return DEFAULT_CHOICE_INDEX

    # PLAN THE WHOLE ROUTE when the map is available. Everything below this is
    # the greedy fallback for callers that can only see the adjacent nodes.
    planned = _plan_from_state(state, nodes)
    if planned is not None:
        return planned

    affordable = [node for node in nodes if _can_afford(state, node)]
    hp, _ = _read_hp_pair_from_state(state)

    # Recovery first whenever the map is offering something unaffordable: that is
    # the signal the run is being outpaced, and it is the moment the old policy
    # pressed on instead. The floor-45 run died here -- 76/97 read as "healthy",
    # it took an act 3 elite worth 58, and there was no rest before floor 45.
    if len(affordable) < len(nodes):
        for room_type in RECOVERY_ROOMS:
            for fallback_index, node in enumerate(nodes):
                if _canonical_text(node.get("type")) == room_type:
                    logger.info(
                        "MAP: routing to %s at %s HP -- %d of %d nodes cost more "
                        "than that.", room_type, hp,
                        len(nodes) - len(affordable), len(nodes))
                    return _read_index(node, fallback_index)

    for room_type in ROOM_PRIORITY_HEALTHY:
        for fallback_index, node in enumerate(affordable):
            if _canonical_text(node.get("type")) == room_type:
                return _read_index(node, nodes.index(node))

    # Nothing affordable and no recovery on offer: take the cheapest fight going.
    cheapest = min(nodes, key=lambda n: required_hp_fraction(n.get("type")))
    logger.info("MAP: nothing affordable at %s HP; taking the cheapest room (%s).",
                hp, cheapest.get("type"))
    return _read_index(cheapest, nodes.index(cheapest))


def _card_name(card: Any) -> str:
    """The card's id, however this screen happens to carry it."""
    if isinstance(card, dict):
        for key in ("id", "card_id", "name", "label"):
            value = card.get(key)
            if value:
                return str(value)
        return ""
    return str(card or "")


def _is_basic_card(card: Any) -> bool:
    """A starter Strike or Defend, for any character.

    The prefix is exactly the ten basics across all five characters and catches
    nothing else -- PERFECTED_STRIKE and BLIGHT_STRIKE do not start with STRIKE_.
    Written as a prefix rather than two Ironclad names so it keeps working the
    day this plays somebody else.
    """
    return _card_name(card).upper().startswith(("STRIKE_", "DEFEND_"))


def _is_upgraded(card: Any) -> bool:
    if isinstance(card, dict) and card.get("upgraded"):
        return True
    return _card_name(card).endswith("+")


def _is_curse(card: Any) -> bool:
    """A curse, by the card `type` the mod already sends.

    `RlCardSelector` writes `card.Type.ToString()` on every card, so this needs
    no lookup and stays right when a card is added. Falls back to the id for
    payloads that predate the field.
    """
    if isinstance(card, dict):
        if str(card.get("type", "")).upper() == "CURSE":
            return True
    return "CURSE" in _card_label(card).upper()


def _deck_has_curse(state: dict[str, Any]) -> bool:
    return any(_is_curse(c) for c in (state.get("deck") or []))


#: Never thin a basic below this many copies. The whole risk in removing
#: anything other than a curse is deleting the deck's last block, and a floor
#: answers it structurally rather than by judgement -- `_pick_card_select_indexes`
#: sees a card list and cannot know what the deck needs.
#:
#: 2 because that is the floor `scripts/removal_vs_relic.py` measured with. The
#: policy that ships has to be the policy that was priced, or the number does
#: not transfer.
MIN_BASIC_COPIES_TO_KEEP = 2

#: Strike before Defend, for the same reason: a deck that cannot block dies
#: faster than one that cannot kill.
REMOVAL_BASIC_ORDER = ("STRIKE_", "DEFEND_")


def _removal_order(cards: list[Any]) -> list[int]:
    """Indexes worth removing, worst first, or empty if nothing is safe to take.

    Curses first -- unambiguously worth losing. Then surplus basics, Strike
    before Defend, unupgraded copies before upgraded ones so a removal never
    throws away a rest-site smith.

    MEASURED, not assumed. `scripts/removal_vs_relic.py` ran exactly this order
    over 30 real live boss decks: 40% -> 44% -> 46% -> 50% boss win at 0/1/2/3
    removals (n=360 per cell, monotonic, 2.7 sigma end to end), against +2
    points for a marginal relic. That is what justifies removal outranking
    `buy_relic` in `SHOP_PURCHASE_ACTION_PRIORITY`, and it is why this is no
    longer curses-only.

    The old objection -- "pulling Strikes out of a strike-synergy deck weakens
    it" -- is real but is now answered by measurement rather than caution: it
    was priced across the decks this agent actually builds and came out ahead.
    The other half of that objection, the last Defend, is answered by the floor.
    """
    out = [i for i, c in enumerate(cards) if _is_curse(c)]
    for prefix in REMOVAL_BASIC_ORDER:
        same = [i for i, c in enumerate(cards)
                if _card_name(c).upper().startswith(prefix) and not _is_curse(c)]
        surplus = len(same) - MIN_BASIC_COPIES_TO_KEEP
        if surplus <= 0:
            continue
        same.sort(key=lambda i: (_is_upgraded(cards[i]), i))
        out.extend(same[:surplus])
    return out


def _has_removal_target(state: dict[str, Any]) -> bool:
    """Is there a card in the deck this policy would actually delete?

    Asked at the SHOP, before spending gold, because the screen that follows
    declines when it finds nothing worth taking -- and a removal bought against
    a deck with nothing to remove is 75 gold burned for no card.
    """
    return bool(_removal_order(list(state.get("deck") or [])))


def _worth_upgrading(card: Any) -> bool:
    return not _is_basic_card(card) and not _is_upgraded(card)


def _pick_card_select_indexes(state: dict[str, Any], *, removing: bool = False) -> list[int]:
    """Choose required card indexes for upgrade/transform/removal screens.

    The mod cannot tell us which of those this is. `RlCardSelector` implements a
    generic hook the *game* calls -- its own docstring lists "deck upgrade, deck
    transform, deck enchant, hand selection, and various other card selection
    prompts" -- so the payload carries only cards, min_select and max_select.
    `removing` is therefore supplied by the caller, which knows: the runner sets
    it after it has just bought a shop card-removal.

    ON REMOVAL: curses only. Not "basics", which was the tempting answer and is
    wrong twice over -- pulling Strikes out of a strike-synergy deck weakens it,
    and a Defend may be the only block the deck owns, which a card-select screen
    gives no way to know. A curse is unambiguously worth losing.

    ON EVERYTHING ELSE: basics last, and never a curse. Basics-last is right for
    upgrades -- a deck lists Strike, Strike, Strike, so taking the first card
    put every rest-site upgrade into a basic Strike. Avoiding curses is right
    for transforms, because transforming a curse yields another curse and
    sometimes a worse one. Both hold for a screen we cannot identify, which is
    why this is the default rather than the removal behaviour.
    """
    cards = list(state.get("cards", []))
    min_select = max(int(state.get("min_select", 1)), 0)
    max_select = max(int(state.get("max_select", min_select)), 0)
    if not cards or max_select == 0 or min_select == 0:
        return []
    count = min(min_select, max_select, len(cards))

    if removing:
        # Curses first, then surplus basics above the floor. Declining is still
        # the fallback: a screen with nothing safe on it means the runner should
        # not have bought the removal, and taking a card the deck needs is worse
        # than wasting the gold that is already spent.
        targets = _removal_order(cards)
        if not targets:
            logger.warning(
                "CARD_SELECT: asked to remove, but every card is either a "
                "non-basic or the last %d copies of a basic; declining rather "
                "than removing a card the deck may need.",
                MIN_BASIC_COPIES_TO_KEEP)
            return []
        return [_read_index(cards[i], i) for i in targets[:count]]

    # Stable ordering: real cards first, then basics, and curses last of all,
    # each keeping deck order so the choice stays reproducible.
    ranked = sorted(
        range(len(cards)),
        key=lambda i: (_is_curse(cards[i]), not _worth_upgrading(cards[i]), i),
    )
    chosen = ranked[:count]
    return [_read_index(cards[i], i) for i in chosen]


def _deck_direction(state: dict[str, Any]):
    """Which deck this run is building, read from the deck the bridge sent.

    Rebuilt per decision rather than carried across the run. The bridge sends
    the whole deck every time, so accumulating over it is the same answer with
    no state to drift -- the same reason live_search rebuilds its combat from
    the bridge on every call instead of keeping one.

    Returns None if the archetype data is unavailable, and every caller then
    behaves exactly as it did before there was any.
    """
    try:
        from sts2_env.search.archetypes import DeckDirection
    except Exception:  # noqa: BLE001 - deckbuilding help is never worth a crash
        return None
    try:
        direction = DeckDirection()
        direction.observe_deck(
            str(c.get("id") if isinstance(c, dict) else c)
            for c in (state.get("deck") or [])
        )
        return direction
    except Exception:  # noqa: BLE001
        logger.debug("could not read a deck direction", exc_info=True)
        return None


#: The bar a card must clear rises with the deck it would join.
#:
#: A FLAT BAR IS THE WRONG SHAPE, and a flat deck-size cap is worse. Deck size is
#: not intrinsically bad -- a cycling deck wants cards and a strength deck does
#: not -- so a cap tells the cycling deck to stop building the thing that makes
#: it work. Size should FALL OUT of quality: the same rule gives a cycling deck
#: thirty cards because thirty cards cleared the bar, and a strength deck twelve.
#:
#: A mediocre card is worth taking at ten cards and not at twenty-five, because
#: at twenty-five it dilutes the draws that matter. Expressed as: take it when
#:
#:     100 * score / QUALITY_BAR_SCALE  >  current deck size
#:
#: `rank_cards` scores between 1.00 and 5.90 across the 366 real card-reward
#: screens captured from live, so the scale converts a score into the largest
#: deck it is still worth joining. At scale 10 the median offer (2.50) is taken
#: up to 25 cards and refused after.
#:
#: SKIP_THRESHOLD remains an absolute floor for cards that are bad at any size.
#: It was 0.0 against observed scores of 1.00-5.90, so it never fired once and
#: the agent took every card it was offered -- which is why act 1 boss decks were
#: 21-22 cards with nine basic Strike/Defend still in them.
QUALITY_BAR_SCALE = 10.0


def card_is_worth_taking(score: float, deck_size: int) -> bool:
    """Is this card good enough for a deck of this size? See QUALITY_BAR_SCALE."""
    if score < SKIP_THRESHOLD:
        return False
    if deck_size <= 0:
        return True
    return (100.0 * score / QUALITY_BAR_SCALE) > deck_size


#: Signature of the last card-reward screen this policy asked to skip. A
#: deterministic policy that skips a screen the game re-presents will skip it
#: forever; this is what lets it notice and break out.
_last_card_reward_skipped: str | None = None


def _pick_card_reward_index(state: dict[str, Any]) -> int | None:
    """Choose a card reward, or return None when taking nothing is better.

    Scored by asking the simulator what each card is, rather than by the old
    rule -- prefer a Power, else an Attack, else a Skill, taking the first of
    that type on offer -- which took BLIGHT_STRIKE (8 damage for 1) over SUNDER
    (26 for 3) because Blight Strike was listed first.

    ON SKIPPING. The screen-driven path cannot skip: RlCardRewardScreenHandler
    reports `can_skip: false` and the game's own handler always takes a card.
    Returning None there would claim a decision the game cannot carry out, which
    is what leaves a screen open and loops a run forever (docs/KNOWN_ISSUES.md).
    So this only declines when the mod says a skip is real, and takes the least
    bad card otherwise -- and says so, because "took a card it rated as harmful"
    is worth seeing in the log rather than silently accepting.

    AND IT SKIPS THE SAME SCREEN ONLY ONCE. `can_skip: true` means the game
    renders a Skip button, NOT that clicking it consumes the reward. Live on
    2026-08-14 the reward screen kept re-offering the same three cards
    (UNMOVABLE / ANGER / CINDER) after every skip, and this function -- being
    deterministic and, on the merits, right -- skipped it again, and again,
    until the run was killed by hand. Same shape as the Crystal Sphere loop.

    Taking a card we would rather decline costs a little deck quality. Hanging
    the run costs the run. So the second time the identical screen appears, take
    the best card and say so loudly, leaving the underlying mod bug visible
    rather than absorbed.
    """
    global _last_card_reward_skipped
    cards = list(state.get("cards", []))
    can_skip = bool(state.get("can_skip", False))
    if not cards:
        return None if can_skip else DEFAULT_CHOICE_INDEX

    signature = json.dumps(cards, sort_keys=True, default=str)
    already_skipped_this = (signature == _last_card_reward_skipped)

    deck = state.get("deck") or []
    direction = _deck_direction(state)
    ranked = rank_cards(cards, deck, direction)
    best_score, best_index, best_card = ranked[0]
    if direction is not None and direction.committed and verbose_choice_logging():
        logger.info("CARD_REWARD: deck reads as %s; picking %s",
                    direction.committed, _card_label(best_card))

    deck_size = _read_deck_size(state)
    deck_is_bloated = deck_size > CARD_REWARD_LARGE_DECK_SIZE
    not_worth_it = not card_is_worth_taking(best_score, deck_size)

    if can_skip and (not_worth_it or deck_is_bloated) and not already_skipped_this:
        logger.info(
            "CARD_REWARD: skipping (best was %s at %.2f, deck %d)",
            _card_label(best_card), best_score, _read_deck_size(state),
        )
        _last_card_reward_skipped = signature
        return None

    if already_skipped_this:
        logger.error(
            "CARD_REWARD: this screen was already skipped and the game is "
            "still offering it -- the skip click is not consuming the reward. "
            "Taking %s to break the loop; a hung run is worse than a bad card.",
            _card_label(best_card),
        )
        _last_card_reward_skipped = None

    if not_worth_it:
        logger.warning(
            "CARD_REWARD: every option looks bad (best %s at %.2f) and this "
            "screen cannot skip; taking the least bad.",
            _card_label(best_card), best_score,
        )

    return _read_index(best_card, best_index)


def _log_card_reward_options(journal: Any, state: dict[str, Any], choice: int | None) -> None:
    """Log every option she considered, with scores, not only the choice.

    PHASE_TWO section 3.3: the 68 unplayable cards would have been obvious the
    moment an option list was logged, because a card that cannot resolve is
    conspicuously ABSENT from every offer it should appear in -- and absence is
    invisible in a chosen-only log. Same scoring path as the decision itself
    (`rank_cards` over the same state), so the numbers here are the numbers she
    actually ranked by. Never raises: a log line cannot cost the run.
    """
    try:
        cards = list(state.get("cards", []))
        if not cards:
            return
        ranked = rank_cards(cards, state.get("deck") or [], _deck_direction(state))
        options = [{"index": index, "card": _card_label(card),
                    "score": round(score, 3)}
                   for score, index, card in ranked]
        journal.write("card_reward_options", options=options,
                      deck_size=_read_deck_size(state),
                      can_skip=bool(state.get("can_skip", False)),
                      chosen=choice, skipped=choice is None)
    except Exception:
        logger.debug("card_reward_options logging failed", exc_info=True)


def _state_fingerprint(state: dict[str, Any]) -> tuple | None:
    """What would have to change for the game to have moved on.

    Deliberately coarse: the screen, the floor, the round, the HP and the hand.
    Anything finer (a request id, a timestamp) differs on every message and would
    make a frozen screen look like progress.
    """
    if not isinstance(state, dict):
        return None
    player = state.get("player") or {}
    return (
        str(state.get("type", "")),
        state.get("floor"),
        state.get("round"),
        state.get("run_hp"),
        player.get("energy"),
        tuple(str(c.get("id")) for c in (state.get("hand") or []) if isinstance(c, dict)),
        tuple(str(e.get("hp")) for e in (state.get("enemies") or []) if isinstance(e, dict)),
    )


def _record_stuck_state(path: str, state: dict[str, Any], repeats: int) -> None:
    """Write the whole screen out, once, so the next one is diagnosable.

    The journal records decisions, not the states behind them, which was enough
    to see the loop and not enough to say which card caused it.
    """
    import json as _json

    try:
        from pathlib import Path as _Path

        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"repeats": repeats, "state": state},
                                 default=str) + "\n")
    except Exception:
        logger.debug("Could not write the stuck state", exc_info=True)


def verbose_choice_logging() -> bool:
    """Whether to narrate a pick. Cheap indirection so the log line above stays
    one condition rather than threading `verbose` through four call sites."""
    return logger.isEnabledFor(logging.INFO)


def _card_label(card: Any) -> str:
    if isinstance(card, dict):
        return str(card.get("id") or card.get("label") or card)
    return str(card)


def _pick_reward_screen_option(state: dict[str, Any]) -> int:
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX
    option = _first_matching_option(options, actions=(REWARD_PICK_ACTION,))
    if option is not None:
        return _read_index(option, DEFAULT_CHOICE_INDEX)
    option = _first_matching_option(options, actions=(REWARD_PROCEED_ACTION,)) or options[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


def _pick_card_bundle_index(state: dict[str, Any]) -> int:
    bundles = [
        bundle
        for bundle in state.get("bundles", [])
        if bool(bundle.get("enabled", True))
    ]
    if not bundles:
        return DEFAULT_CHOICE_INDEX
    option = _first_matching_option(bundles, actions=(CARD_BUNDLE_PICK_ACTION,))
    if option is None:
        option = bundles[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


#: Last floor of each act, so a rest site can tell whether a boss is next.
#: Confirmed from live data rather than assumed: two runs both ended on floor 17
#: in a room the bridge reported as `Boss` while the simulator puts the act 1
#: boss on 16, and `room_type` in the run records settles it at 17/33/50.
ACT_BOSS_FLOORS = frozenset({17, 33, 50})


def _boss_is_next(state: dict[str, Any]) -> bool:
    """Is the floor after this one an act boss?

    An explicit `boss_is_next` wins when the caller supplies one, because the
    floor rule below is tied to the LIVE game's numbering and does not travel.
    The simulator does not count the opening Ancient/Neow room as a floor -- live
    reports it as floor 1 -- so its act 1 boss is total_floor 16 against the
    game's 17, and `floor + 1 in {17, 33, 50}` never fires offline. The
    "heal rather than smith before the boss" rule was therefore silently absent
    from every simulated run while being active in every live one, which is
    exactly the kind of divergence that stops an offline result transferring.

    `scripts/live_policy.py` computes the flag from the simulator's own map
    (`run_state.map.boss_point`), so each side answers in its own terms rather
    than sharing a constant neither can honour.
    """
    explicit = state.get("boss_is_next")
    if explicit is not None:
        return bool(explicit)
    try:
        floor = int(state.get("floor") or 0)
    except (TypeError, ValueError):
        return False
    return (floor + 1) in ACT_BOSS_FLOORS


def _pick_rest_option(state: dict[str, Any]) -> int:
    """Choose a rest-site option by option identity, not display order.

    HEAL whenever the next boss would out-damage the current HP. This used to ask
    the flat `hp < 0.5 * max_hp`, and on the floor before an act boss that test
    said SMITH 17 times at a median 49 HP -- upgrading a card immediately before
    a fight the measured death rate puts at 88% from that position.

    Boss HP entering, over 89 live attempts: median 47, and never once above 69.
    Every boss has been fought from a losing position, so the rest site before it
    is not a choice between two goods.
    """
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX
    hp, _ = _read_hp_pair_from_state(state)

    # Smithing is only worth a rest if there is something worth smithing. A deck
    # of nothing but basics has no upgrade that changes a fight, so the HP is
    # worth more -- which is the whole reason to be at a rest site.
    deck = state.get("deck") or []
    has_real_upgrade = any(_worth_upgrading(card) for card in deck) if deck else True

    # What is coming, not what a ratio says about now. A rest site's whole value
    # is that it is the last chance to pay for the next room, so it is held to
    # the same bar the map uses to decide whether that room is enterable.
    _, max_hp = _read_hp_pair_from_state(state)
    needed = required_hp_fraction("boss" if _boss_is_next(state) else "elite")
    outmatched = hp is not None and max_hp and hp < needed * max_hp

    preferred = (
        REST_HEAL_OPTION_ID
        if outmatched or not has_real_upgrade
        else REST_SMITH_OPTION_ID
    )
    if outmatched and has_real_upgrade:
        logger.info("REST: healing at %s/%s -- %s next wants %.0f%%.",
                    hp, max_hp,
                    "the act boss" if _boss_is_next(state) else "an elite",
                    100 * needed)
    option = _first_matching_option(options, option_ids=(preferred,))
    if option is None and preferred == REST_SMITH_OPTION_ID:
        option = _first_matching_option(options, option_ids=(REST_HEAL_OPTION_ID,))
    if option is None:
        option = options[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


def _is_removal_option(state: dict[str, Any], chosen_index: int) -> bool:
    """Did the shop choice we just made buy a card removal?"""
    for option in _enabled_options(state):
        if _read_index(option, -1) == chosen_index:
            return str(option.get("action", "")) == "remove_card"
    return False


def _pick_shop_option(state: dict[str, Any]) -> int:
    """Buy an enabled shop item when one exists; leave when only exit remains."""
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX
    for action in SHOP_PURCHASE_ACTION_PRIORITY:
        if action == "remove_card" and not _has_removal_target(state):
            # Nothing this policy would delete: no curse, and every basic is
            # down to its last MIN_BASIC_COPIES_TO_KEEP. Buying the removal
            # anyway spends 75 gold on a screen that will decline.
            continue
        option = _first_matching_option(options, actions=(action,))
        if option is not None:
            return _read_index(option, DEFAULT_CHOICE_INDEX)
    option = _first_matching_option(options, actions=(SHOP_LEAVE_ACTION,)) or options[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


_MARKUP = re.compile(r"\[/?[a-zA-Z]+\]")
_LOSE_HP = re.compile(r"lose\s+(\d+)\s+(max\s+)?hp", re.IGNORECASE)
_HEAL_HP = re.compile(r"heal\s+(\d+)\s+hp", re.IGNORECASE)
_SET_MAX_HP = re.compile(r"max\s+hp\s+(?:becomes|is set to|to)\s+(\d+)", re.IGNORECASE)

#: "Pay 40 Gold". Events charge gold as readily as HP and the safety check only
#: ever parsed HP, so a paid option read as free.
#:
#: Endless Conveyor is the case that found this. Every option is "Pay 40 Gold.
#: <reward>. Continue feasting!" against one free alternative, "Observe the
#: Chef". Nothing in the text mentions HP, so every grab scored as harmless and
#: the agent fed the belt 120 gold in one visit -- most of a shop relic -- and
#: on an earlier session answered the same event 83 times.
_PAY_GOLD = re.compile(r"pay\s+(?:\[[^\]]*\])?\s*(\d+)\s*(?:\[[^\]]*\])?\s*gold",
                       re.IGNORECASE)

# Never walk out of an event below this much of your maximum. An event is worth
# HP, but no event reward is worth arriving at the next elite unable to survive
# its opening turn -- which is how these runs actually end.
# ===========================================================================
# ===  TEMPORARY WORKAROUND FOR A GAME CRASH -- DELETE WHEN MEGACRIT FIXES  ==
# ===========================================================================
#
# Event options that segfault the game. Not our bug, and not a balance call:
# picking one of these ENDS THE SESSION, losing every run that would have
# followed it.
#
# NUTRITIOUS_SOUP, on the Tezcatara ancient event. Four selections across four
# independent sessions, four SIGSEGVs, and every game log terminates one line
# later on the identical asset load:
#
#     [AutoSlay] Selecting event option: TEZCATARA...options.NUTRITIOUS_SOUP
#     [WARN] Asset not cached: res://scenes/vfx/vfx_card_enchant.tscn
#     <process dies>
#
# `NutritiousSoup.AfterObtained()` loops the whole deck and calls
# `NCardEnchantVfx.Create(item)` + `AddChildSafely` for EACH basic Strike --
# N vfx nodes spawned in one frame, for cards with no on-screen node during an
# event. This agent never removes Strikes, so it always carries the full five
# and hits the worst case every time. Reproduces at both 5x and 1x animation
# speed, so it is not our AnimationSpeedPatch.
#
# TAKING IT WOULD BE CORRECT PLAY. TezcatarasEmber on every basic Strike is
# exactly what the `strike-synergy` archetype is built around, so this costs us
# a genuinely strong pick. It is here only because a crash costs more.
#
# TO REMOVE: delete this block and the one call to
# `_event_option_crashes_the_game` in `_pick_event_option`. Nothing else refers
# to either. Verify with a live session that reaches Tezcatara and takes Soup.
#
# Matched on the display label because that is all the bridge sends -- the mod's
# event payload carries `label` and `event_id` but not the option's TextKey. So
# this is language-dependent, and would silently stop working on a non-English
# client. Acceptable for a workaround; if it ever needs to be robust, add
# `text_key` to the payload in `RlEventRoomHandler.ChooseEventOption`.
CRASHING_EVENT_OPTION_LABELS = frozenset({"nutritious soup"})


def _event_option_crashes_the_game(option: dict[str, Any]) -> bool:
    """Does picking this option crash the game? See the block above."""
    label = str(option.get("label") or "").strip().lower()
    return label in CRASHING_EVENT_OPTION_LABELS


# ===========================================================================
# ===  END OF TEMPORARY WORKAROUND  =========================================
# ===========================================================================

EVENT_HP_FLOOR_RATIO = 0.5
EVENT_HP_FLOOR_ABSOLUTE = 25

# Hot Baths re-presents the same screen and charges more each time. Taking the
# paid option twice is a decision; taking it seven times is a loop, and it is
# what killed run 1 (68 -> 41 HP and still going).
EVENT_MAX_REPEATS = 1

_EXIT_WORDS = ("leave", "skip", "ignore", "walk", "refuse", "decline", "depart",
               "exit", "nothing", "abstain", "proceed", "continue")


def _plain(text: Any) -> str:
    """Option text with the game's colour markup stripped."""
    return _MARKUP.sub("", str(text or ""))


def _event_option_text(option: dict[str, Any]) -> str:
    """Everything readable about an option.

    `id` is always the literal string `event_choice`, and `description` is
    sometimes empty, so a heuristic reading either alone sees nothing. The human
    text is in `label`, and the numbers are in `description`.
    """
    return _plain(
        f"{option.get('label', '')} "
        f"{option.get('description', '')} "
        f"{option.get('text', '')}"
    )


def _event_gold_cost(option: dict[str, Any]) -> int:
    """Gold this option charges, as far as the text admits."""
    text = _event_option_text(option)
    return sum(int(a) for a in _PAY_GOLD.findall(text))


def _event_hp_cost(option: dict[str, Any]) -> tuple[int, int, int | None]:
    """(hp lost, max hp lost, max hp set to) as far as the text admits."""
    text = _event_option_text(option)
    hp_lost = max_hp_lost = 0
    for amount, is_max in _LOSE_HP.findall(text):
        if is_max:
            max_hp_lost += int(amount)
        else:
            hp_lost += int(amount)
    for amount in _HEAL_HP.findall(text):
        hp_lost -= int(amount)
    set_to = _SET_MAX_HP.search(text)
    return hp_lost, max_hp_lost, int(set_to.group(1)) if set_to else None


def _event_option_is_exit(option: dict[str, Any]) -> bool:
    text = _event_option_text(option).lower()
    return any(word in text for word in _EXIT_WORDS)


# ===========================================================================
# ENDLESS CONVEYOR
# ===========================================================================
# Read off `decompiled/MegaCrit.Sts2.Core.Models.Events/EndlessConveyor.cs`
# rather than guessed at, because three rounds of guarding by text guessed
# wrong three times.
#
#   GenerateInitialOptions() -> [ GrabSomethingOffTheBelt, ObserveChef ]
#
#   GrabSomethingOffTheBelt(): LoseGold(40) unless the dish is GOLDEN_FYSH,
#       runs the dish, rolls a new one, then SetEventState(...) with
#       [ GrabSomethingOffTheBelt, Leave ]. It re-opens. Forever.
#
#   ObserveChef(): upgrades a random upgradable card and SetEventFinished().
#       Free, and it ENDS the event.
#
#   GenerateGrabSomethingOffTheBeltOption(): once Gold < 40 this returns an
#       option whose action is NULL, labelled `...options.LOCKED`.
#
# So index 0 is always the grab and index 1 is always the way out, and the
# right play is never to grab:
#
#   - Observe the Chef is a FREE card upgrade, measured in this repo at ~5
#     points of act 1 boss win, and it leaves immediately.
#   - A grab is 40 gold for one weighted roll over +4 max HP, an upgrade, a
#     transform, a colourless card, a potion or a 10 heal. A free upgrade is
#     already at the top of that distribution.
#   - 40 gold is over half a shop card removal (75), which `removal_vs_relic.py`
#     puts at ~3.3 points of boss win.
#   - GOLDEN_FYSH is the one free dish and RollDish only adds it once
#     NumOfGrabs > 1, so it cannot be reached without paying first.
#
# WHY THE EXISTING GUARDS ALL MISSED IT. Every one of them is reactive --
# `repeats >= 1`, `EVENT_MAX_REPEATS`, `EVENT_HARD_CAP` -- and this costs the
# run on the FIRST answer. The gold guard never fires because the mod sends
# `label` = "Grab Suspicious Condiment off the Belt" with the 40 gold nowhere
# in the text, so `_event_gold_cost` reads 0, both options score as safe, and
# `safe[0][0]` returns index 0. Live 2026-08-16, floor 11, 250 gold: grabbed,
# rolled SUSPICIOUS_CONDIMENT, which calls RewardsCmd.OfferCustom -- a reward
# screen nested inside a still-open event -- and the run ended 30s later at
# full HP.

#: `option.Event.Id.Entry` for this event, normalised by `_normalised_event_id`.
#: Both spellings are accepted because the mod sends the game's id and the
#: simulator uses its own (`sts2_env/events/act2.py:222` -> "EndlessConveyor"),
#: and no capture of the live value exists -- the journal has never recorded
#: `event_id`. Matching on the id is preferred over the label because the label
#: is English; the label is the fallback, not the primary.
ENDLESS_CONVEYOR_EVENT_IDS = frozenset({"ENDLESSCONVEYOR"})

#: Every grab option is "Grab <dish> off the Belt". The dish changes each roll,
#: the tail does not. Only used when the event id does not resolve.
_CONVEYOR_GRAB_MARKER = "off the belt"


def _normalised_event_id(value: Any) -> str:
    """An event id comparable across the mod's spelling and the simulator's."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _is_endless_conveyor(event_id: str, options: list[dict[str, Any]]) -> bool:
    if _normalised_event_id(event_id) in ENDLESS_CONVEYOR_EVENT_IDS:
        return True
    for option in options:
        if _normalised_event_id(option.get("event_id")) in ENDLESS_CONVEYOR_EVENT_IDS:
            return True
    # Fallback for a build whose id we do not recognise: the belt names itself.
    return any(
        _CONVEYOR_GRAB_MARKER in _event_option_text(option).lower()
        for option in options
    )


def _endless_conveyor_index(options: list[dict[str, Any]]) -> int | None:
    """The option that leaves the belt: Observe the Chef, or Leave.

    Anything that is not a grab ends the event, so this takes the first
    non-grab rather than trying to tell Observe and Leave apart.
    """
    for i, option in enumerate(options):
        if _CONVEYOR_GRAB_MARKER not in _event_option_text(option).lower():
            return _read_index(option, i)
    return None


# ===========================================================================
# LEARNING WHAT THE MOD DOES NOT SEND
# ===========================================================================
# `_event_gold_cost` can only read a cost the option text admits to, and the
# conveyor proves the text does not have to admit to anything. The gold total
# does not lie, so watch it: remember what was chosen and how much gold was
# held, and when the next event state arrives with less gold, record that
# option as a purchase.
#
# Generic on purpose. It is not a second conveyor patch -- it catches any event
# that charges gold silently and re-presents itself, which is the shape of the
# whole failure class, and it needs no vocabulary.

#: (normalised event id, option label) seen to reduce gold, this process.
_event_options_that_charged_gold: set[tuple[str, str]] = set()

#: What the last event choice was, so the next state can price it.
#: (normalised event id, option label, gold held before choosing).
_last_event_choice: tuple[str, str, int] | None = None


def _reset_event_gold_learning() -> None:
    """Between runs. The learned set is deliberately NOT cleared -- an option
    that charges gold in one run charges it in the next -- but the pending
    comparison is, because the next state belongs to a different run."""
    global _last_event_choice
    _last_event_choice = None


def _price_the_previous_event_choice(event_id: str, gold: Any) -> None:
    """Charge the last choice with whatever gold went missing since."""
    global _last_event_choice
    pending = _last_event_choice
    if pending is None or not isinstance(gold, int):
        return
    previous_event, label, gold_before = pending
    _last_event_choice = None
    if previous_event != _normalised_event_id(event_id):
        return  # a different event; the gold difference is not attributable
    if gold < gold_before:
        if (previous_event, label) not in _event_options_that_charged_gold:
            logger.info(
                "EVENT %s: %r cost %d gold, which its text never said. "
                "Not taking it again in this session.",
                event_id or "?", label, gold_before - gold)
        _event_options_that_charged_gold.add((previous_event, label))


def _remember_event_choice(event_id: str, option: dict[str, Any], gold: Any) -> None:
    global _last_event_choice
    if isinstance(gold, int):
        _last_event_choice = (
            _normalised_event_id(event_id),
            str(option.get("label") or ""),
            gold,
        )


def _event_option_charges_gold(event_id: str, option: dict[str, Any]) -> bool:
    """Either the text says so, or we have watched it happen."""
    if _event_gold_cost(option) > 0:
        return True
    key = (_normalised_event_id(event_id), str(option.get("label") or ""))
    return key in _event_options_that_charged_gold


def _pick_event_option(state: dict[str, Any], seen: dict[str, int] | None = None) -> int:
    """Choose an event option, refusing the ones that end the run.

    The previous version could not work on this mod's payload. It matched
    keywords against `id` (always the literal `event_choice`) and `description`
    (often empty) while the readable text sits in `label`, so nothing ever
    matched. It also only looked at all below 50% HP, and its avoidance path
    `continue`d past a harmful option and then returned `options[0]` -- the
    option it had just rejected.

    Two guards now, deliberately independent, because each covers what the other
    cannot. The parsed cost catches "Lose 6 HP" wherever the text says so; the
    repeat guard catches Hot Baths, whose labels are `Linger` and `Exit Baths`
    and contain no warning at all.
    """
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX

    hp = state.get("run_hp")
    max_hp = state.get("run_max_hp")
    event_id = str(state.get("event_id") or "")
    if not event_id:
        for option in options:
            if option.get("event_id"):
                event_id = str(option["event_id"])
                break

    floor = None
    if isinstance(hp, int) and isinstance(max_hp, int) and max_hp > 0:
        floor = max(EVENT_HP_FLOOR_ABSOLUTE, int(max_hp * EVENT_HP_FLOOR_RATIO))

    repeats = (seen or {}).get(event_id, 0)

    # Price whatever we chose last time against the gold now on the screen,
    # before deciding anything. See _price_the_previous_event_choice.
    _price_the_previous_event_choice(event_id, state.get("gold"))

    # ENDLESS CONVEYOR, ANSWERED FROM THE DECOMPILED SOURCE. Ahead of every
    # other guard because they are all reactive and this one costs the run on
    # the FIRST answer. See the block above ENDLESS_CONVEYOR_EVENT_IDS.
    if _is_endless_conveyor(event_id, options):
        exit_index = _endless_conveyor_index(options)
        if exit_index is not None:
            logger.info(
                "EVENT: Endless Conveyor -- leaving the belt. Every grab is 40 "
                "gold for one roll and re-opens the event; the other option is "
                "a free card upgrade that ends it.")
            if seen is not None and event_id:
                seen[event_id] = repeats + 1
            return exit_index
        # Only grabs on offer means the belt has already been ridden and the
        # Leave option is missing or disabled. Nothing safe to pick; fall
        # through to the guards below rather than grab by default.
        logger.error(
            "EVENT: Endless Conveyor is offering only grab options (%r). "
            "Falling through to the generic guards.",
            [o.get("label") for o in options][:4])

    # HARD CAP NEXT. Checked before any scoring, because the whole point is
    # that it does not depend on understanding the options -- Endless Conveyor
    # was answered 83 times by a scorer that understood every one of them.
    escape = _event_escape_index(options, event_id, repeats)
    if escape is not None:
        if seen is not None and event_id:
            seen[event_id] = repeats + 1
        return escape

    safe: list[tuple[int, dict]] = []
    unsafe: list[tuple[int, int, int, dict]] = []  # (hp cost, gold cost, index, option)

    for i, option in enumerate(options):
        hp_lost, max_hp_lost, max_hp_set = _event_hp_cost(option)
        index = _read_index(option, i)

        # >>> TEMPORARY CRASH WORKAROUND -- see CRASHING_EVENT_OPTION_LABELS <<<
        # Treated as lethal because it is: it kills the process, not the run.
        # Delete this branch when the game is fixed.
        if _event_option_crashes_the_game(option):
            logger.warning(
                "EVENT: refusing %r -- it segfaults the game (4 of 4 sessions). "
                "This is a TEMPORARY workaround; see CRASHING_EVENT_OPTION_LABELS.",
                option.get("label"))
            unsafe.append((10 ** 6, 10 ** 6, index, option))
            continue
        # >>> END TEMPORARY CRASH WORKAROUND <<<

        lethal = False
        # An option that dictates a new maximum, or strips most of it, is a death
        # sentence with extra steps: the run continues and cannot survive a fight.
        if max_hp_set is not None and isinstance(max_hp, int) and max_hp_set < max_hp * 0.5:
            lethal = True
        if isinstance(max_hp, int) and max_hp_lost >= max_hp * 0.25:
            lethal = True
        if floor is not None and isinstance(hp, int) and hp_lost > 0 and hp - hp_lost < floor:
            lethal = True
        # Nothing in the text said it costs anything, but this screen has already
        # been paid for once. Hot Baths looks exactly like this.
        if hp_lost <= 0 and repeats > EVENT_MAX_REPEATS and not _event_option_is_exit(option):
            lethal = True

        # ONE PAID OPTION PER EVENT. An option that charges gold is a purchase,
        # and a screen that re-presents itself after each purchase is a shop
        # with no exit sign. Endless Conveyor charges 40 a time and says
        # "Continue feasting!"; taking it twice is a decision, taking it every
        # time it is offered is how 120 gold left in one room.
        #
        # Deliberately not scaled by how much gold is held: the failure is
        # buying REPEATEDLY, not buying once, and a threshold on gold would let
        # a rich run empty itself just as thoroughly.
        # `_event_option_charges_gold` rather than `_event_gold_cost`, so an
        # option we have WATCHED take gold counts even when its text is silent
        # about it. That silence is exactly what let the conveyor through.
        if repeats >= 1 and _event_option_charges_gold(event_id, option):
            lethal = True

        if lethal:
            # Gold is the SECOND key. When every option is unsafe the
            # fallthrough takes the cheapest, and with hp cost tied at zero it
            # used to break the tie on index -- which on Endless Conveyor is
            # the 40-gold grab, sitting at index 0 ahead of the free option.
            unsafe.append((hp_lost, _event_gold_cost(option), index, option))
        else:
            safe.append((index, option))

    if seen is not None and event_id:
        seen[event_id] = repeats + 1

    if safe:
        # Prefer leaving once the screen has been taken before; otherwise the
        # first option that is not going to kill us.
        if repeats > EVENT_MAX_REPEATS:
            for index, option in safe:
                if _event_option_is_exit(option):
                    _remember_event_choice(event_id, option, state.get("gold"))
                    return index
        # A free option beats a paying one at equal safety, and on the FIRST
        # visit the guard above has not fired yet. This is what makes the
        # learned gold cost worth anything: without it the knowledge would only
        # ever be applied on a repeat, and the conveyor is decided on visit one.
        for index, option in safe:
            if not _event_option_charges_gold(event_id, option):
                _remember_event_choice(event_id, option, state.get("gold"))
                return index
        _remember_event_choice(event_id, safe[0][1], state.get("gold"))
        return safe[0][0]

    # Everything on offer is dangerous. Take the cheapest rather than the first,
    # and never fall back to an option that was explicitly rejected.
    logger.warning(
        "Every option in event %s looks harmful at %s HP; taking the cheapest.",
        event_id or "?", hp,
    )
    cheapest = min(unsafe)
    _remember_event_choice(event_id, cheapest[3], state.get("gold"))
    return cheapest[2]


#: A single event may not be answered more times than this in one run.
#:
#: The repeat guard above prefers an EXIT option once a screen has been seen
#: twice, and that only works if an exit can be recognised. Endless Conveyor
#: defeated it live on 2026-08-14: every option reads "Grab <food> off the
#: Belt", none of them contains an exit word, so all of them were marked unsafe
#: and the fallthrough kept picking one. The agent answered the same event 83
#: times and the run ended at floor 4 on FULL HEALTH -- 84/84, no death, just a
#: run that could not leave a room.
#:
#: The word list will always be incomplete; a hard cap does not depend on
#: guessing the vocabulary. 8 is far above any legitimate event -- the longest
#: real one is a handful of choices -- so tripping it means something is wrong,
#: and it is logged as such.
EVENT_HARD_CAP = 8


def _event_escape_index(options: list[dict], event_id: str, count: int) -> int | None:
    """The way out of an event we have answered too many times, or None.

    Prefers a recognised exit, then the LAST option, which is where events
    conventionally put "leave". Breaking out on the wrong option costs one bad
    outcome; not breaking out costs the run.
    """
    if count < EVENT_HARD_CAP or not options:
        return None
    for i, option in enumerate(options):
        if _event_option_is_exit(option):
            logger.error(
                "EVENT %s answered %d times -- taking the exit option %r. "
                "The repeat guard should have caught this sooner.",
                event_id or "?", count, option.get("label"))
            return _read_index(option, i)
    last_index = len(options) - 1
    logger.error(
        "EVENT %s answered %d times and NO option looks like an exit "
        "(labels: %r). Taking the last option to break the loop -- this is a "
        "hard cap, not a decision, and the event needs looking at.",
        event_id or "?", count,
        [o.get("label") for o in options][:6])
    return _read_index(options[last_index], last_index)


#: The last Crystal Sphere screen we answered. If the next one is identical, our
#: answer did nothing and repeating it will do nothing again.
_last_crystal_sphere: str | None = None


def _pick_crystal_sphere_option(state: dict[str, Any]) -> int:
    """A cell to divine, or proceed once divining stops doing anything.

    PROCEED WHEN THE SCREEN DOES NOT MOVE. The mod lists every hidden cell as an
    option whether or not the player has divinations left to spend, so a spent
    board still offers 85 cells and one proceed. This preferred a cell whenever
    one existed, clicked a cell that could not be clicked, got the same screen
    back, and chose the same cell again -- 24 identical states, then the runner
    gave up and ended the session on its first run.

    Comparing against the previous screen is what distinguishes "there are cells
    to divine" from "there are cells, and clicking them is a no-op". The state
    carries no counter that would say so directly.
    """
    global _last_crystal_sphere

    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX

    signature = json.dumps(state.get("options"), sort_keys=True, default=str)
    repeated = signature == _last_crystal_sphere
    _last_crystal_sphere = signature

    proceed = _first_matching_option(options, actions=(REWARD_PROCEED_ACTION,))
    if repeated and proceed is not None:
        logger.info("Crystal Sphere did not change after the last pick; proceeding.")
        return _read_index(proceed, DEFAULT_CHOICE_INDEX)

    option = _first_matching_option(options, actions=(CRYSTAL_SPHERE_CELL_ACTION,))
    if option is not None:
        return _read_index(option, DEFAULT_CHOICE_INDEX)
    option = proceed or options[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


def _pick_treasure_option(state: dict[str, Any]) -> int:
    option = _first_matching_option(
        _enabled_options(state),
        actions=(TREASURE_COLLECT_ACTION,),
    )
    return _read_index(option, DEFAULT_CHOICE_INDEX) if option is not None else DEFAULT_CHOICE_INDEX


def _pick_boss_relic_option(state: dict[str, Any]) -> int:
    option = _first_matching_option(
        _enabled_options(state),
        actions=(BOSS_RELIC_PICK_ACTION,),
    )
    return _read_index(option, DEFAULT_CHOICE_INDEX) if option is not None else DEFAULT_CHOICE_INDEX


def _send_choice_or_skip(client: Any, choice_index: int | None) -> None:
    if choice_index is None:
        client.skip()
    else:
        client.choose(choice_index)


def _enabled_options(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        option
        for option in state.get("options", [])
        if bool(option.get("enabled", True))
    ]


def _first_matching_option(
    options: list[dict[str, Any]],
    *,
    option_ids: tuple[str, ...] = (),
    actions: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    option_id_set = {_canonical_text(value) for value in option_ids}
    action_set = {_canonical_text(value) for value in actions}
    for option in options:
        if option_id_set and _canonical_text(option.get("id")) in option_id_set:
            return option
        if action_set and _canonical_text(option.get("action")) in action_set:
            return option
    return None


def _read_deck_size(state: dict[str, Any]) -> int:
    run_state = state.get("run_state", {})
    if isinstance(run_state, dict):
        deck = run_state.get("deck")
        if isinstance(deck, list):
            return len(deck)
    return int(state.get("deck_size", 0) or 0)


def _read_hp_ratio(state: dict[str, Any]) -> float | None:
    for container in _candidate_player_containers(state):
        hp, max_hp = _read_hp_pair(container)
        if hp is not None and max_hp and max_hp > 0:
            return hp / max_hp
    return None


def _candidate_player_containers(state: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    for key in ("player", "run_state", "combat_state"):
        value = state.get(key)
        if isinstance(value, dict):
            if isinstance(value.get("player"), dict):
                containers.append(value["player"])
            containers.append(value)
    containers.append(state)
    return containers


def _read_hp_pair(container: dict[str, Any]) -> tuple[int | None, int | None]:
    hp_value = container.get("hp")
    if isinstance(hp_value, str) and "/" in hp_value:
        hp_text, max_hp_text = hp_value.split("/", 1)
        return _optional_int(hp_text), _optional_int(max_hp_text)

    # `run_hp` as well as `hp`, because outside combat there is no player block
    # and only the run-level pair is sent. Without this, _read_hp_ratio returned
    # None at every rest site, shop, event and map screen -- so the low-HP branch
    # of the rest and routing heuristics could not fire at all, and the agent
    # smithed at 20 HP and walked into elites as though it were healthy.
    hp = _optional_int(hp_value)
    max_hp = _optional_int(container.get("max_hp"))
    if hp is None:
        hp = _optional_int(container.get("run_hp"))
    if max_hp is None:
        max_hp = _optional_int(container.get("run_max_hp"))
    return hp, max_hp


def _read_index(option: dict[str, Any], fallback: int) -> int:
    value = _optional_int(option.get("index"))
    return fallback if value is None else value


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_text(value: Any) -> str:
    return str(value or "").replace("_", "").replace(" ", "").casefold()


# ----------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------


def _reconnect_with_retry(
    client: STS2GameClient, max_retries: int = 3, delay: float = 3.0
) -> None:
    """Attempt to reconnect to the game server with retries.

    THREE, NOT TEN. Each attempt blocks on a ~58s socket connect timeout before
    it raises, so `delay` is nearly irrelevant and ten attempts is ten minutes.
    Measured on 2026-08-14: attempts landed at 16:16:46, 16:17:47, 16:18:48 --
    61 seconds apart.

    That ten minutes is spent waiting for a game that has already died and is
    not coming back on its own. `--restart-on-crash` relaunches it via Steam and
    the bridge is back up **10 seconds later** ("Bridge is back up; resuming",
    16:25:57 -> 16:26:07). So the retry loop was costing seven minutes per crash
    to avoid a fix that takes ten seconds, and an overnight session with several
    crashes lost most of an hour to it.

    Three attempts is still ~3 minutes, which is ample for a transient blip. A
    real crash falls through to the restart that much sooner.

    Args:
        client: The game client to reconnect.
        max_retries: Maximum reconnection attempts.
        delay: Seconds between attempts.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Reconnect attempt %d/%d...", attempt, max_retries)
            client.reconnect()
            logger.info("Reconnected successfully.")
            return
        except ConnectionError:
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.error("Failed to reconnect after %d attempts.", max_retries)
                raise ConnectionError(
                    f"Could not reconnect to STS2 bridge after {max_retries} attempts. "
                    "The game may have crashed or the bridge mod is no longer running."
                )


def _log_combat_action(
    state: dict[str, Any], action_int: int, decoded: dict[str, Any]
) -> None:
    """Log a combat action with context for debugging."""
    combat = state.get("combat_state") or state
    player = combat.get("player", {})
    hand = combat.get("hand", [])
    enemies = combat.get("enemies", [])

    if decoded["type"] == ActionType.END_TURN:
        logger.info(
            "COMBAT [HP:%d/%d E:%d] -> END_TURN (round %d)",
            player.get("hp", 0),
            player.get("max_hp", 0),
            player.get("energy", 0),
            combat.get("round", 0),
        )
    elif decoded["type"] == ActionType.POTION or decoded.get("out_of_hand"):
        slot = decoded.get("slot", decoded.get("potion_slot", -1))
        ti = decoded.get("target_index", -1)
        potions = combat.get("potions", [])
        potion_name = "?"
        for potion in potions:
            if int(potion.get("slot", -1)) == slot:
                potion_name = potion.get("id", "?")
                break
        target_name = enemies[ti].get("id", "?") if 0 <= ti < len(enemies) else "N/A"
        logger.info(
            "COMBAT [HP:%d/%d E:%d] -> POTION %s (slot=%d) -> %s (idx=%d)",
            player.get("hp", 0),
            player.get("max_hp", 0),
            player.get("energy", 0),
            potion_name,
            slot,
            target_name,
            ti,
        )
    else:
        ci = decoded.get("card_index", -1)
        ti = decoded.get("target_index", -1)
        card_name = hand[ci].get("id", "?") if ci < len(hand) else "?"
        target_name = enemies[ti].get("id", "?") if 0 <= ti < len(enemies) else "N/A"
        logger.info(
            "COMBAT [HP:%d/%d E:%d] -> PLAY %s (idx=%d) -> %s (idx=%d)",
            player.get("hp", 0),
            player.get("max_hp", 0),
            player.get("energy", 0),
            card_name, ci,
            target_name, ti,
        )


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------


def main() -> None:
    """CLI entry point for the agent runner."""
    parser = argparse.ArgumentParser(
        description="Run a trained RL agent on the real STS2 game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained MaskablePPO model (.zip file).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bridge server hostname.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9002,
        help="Bridge server port.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic action selection (no exploration).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic action selection (for diversity).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Log every action taken.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--record-replay",
        default=None,
        help="Optional path to save a bridge replay trace JSON while the agent runs.",
    )
    parser.add_argument(
        "--replay-factory",
        default=None,
        help="Optional module:function factory to store in replay metadata for later comparison.",
    )
    parser.add_argument(
        "--speed",
        choices=["turbo", "fast", "normal", "slow"],
        default="turbo",
        help=(
            "How fast the game plays itself. turbo is for gathering traces and is "
            "unwatchable; normal is the game's own pace with fast mode on; slow "
            "pauses between actions so a viewer can follow along."
        ),
    )
    parser.add_argument(
        "--allow-random-fallback",
        action="store_true",
        help=(
            "Let the game keep playing randomly when the agent does not answer. Off "
            "by default: the fallback logs only to the game's log, so this console "
            "stays silent while the mod plays on, and the trace records those "
            "actions as the model's."
        ),
    )
    parser.add_argument(
        "--journal", default=None,
        help="JSONL recording every room, fight, card played and reward taken.",
    )
    parser.add_argument(
        "--capture-raw", default=None,
        help=(
            "JSONL of the raw states the mod sends, verbatim and unparsed. The "
            "journal records decisions and drops the state behind them; this is "
            "the other half, and it is what lets the bridge parsers be replayed "
            "offline against real payloads instead of assumed ones."
        ),
    )
    parser.add_argument(
        "--capture-raw-per-type", type=int, default=25,
        help=(
            "How many states of each message type to keep (default 25). Quotas "
            "rather than everything, so one short session yields every kind of "
            "screen instead of ten thousand combat_actions from one long fight."
        ),
    )
    parser.add_argument(
        "--combat-policy",
        default=None,
        help=(
            "Optional separate combat policy (.zip). When the main model is a full-run "
            "model, this overrides combat decisions so the main model only handles "
            "map, rewards, shop, rest, and events."
        ),
    )
    parser.add_argument(
        "--live-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the SearchAgent turn planner for combat decisions instead of "
            "the trained model's single-step argmax. The search enumerates "
            "legal orderings this turn and lets the enemies reply on a clone, "
            "lifting boss win rate from 6.7%% to ~20%% on the harvested "
            "benchmark (docs/MODELS.md:120).\n"
            "\n"
            "ON BY DEFAULT since 2026-08-14, and that is the fix for a "
            "measurement error rather than a tuning choice. It was opt-in, so "
            "only 93 of 508 recorded live runs -- 18%% -- actually used the "
            "search. Every pooled number this project has quoted was therefore "
            "dominated by the TRAINED MODEL, not by the agent under "
            "development. Split apart, the search reaches the boss 58.4%% of "
            "the time against the model's 45.4%%, and offline reaches 60.7%% -- "
            "so offline was predicting live reach correctly all along and we "
            "were comparing it against a different agent.\n"
            "\n"
            "Pass --no-live-search to run the trained model instead."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    deterministic = not args.stochastic

    run_agent(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        deterministic=deterministic,
        verbose=args.verbose,
        record_replay_path=args.record_replay,
        replay_factory=args.replay_factory,
        speed=args.speed,
        allow_random_fallback=args.allow_random_fallback,
        combat_policy_path=args.combat_policy,
        journal_path=args.journal or None,
        live_search=args.live_search,
        capture_raw_path=args.capture_raw or None,
        capture_raw_per_type=args.capture_raw_per_type,
    )


if __name__ == "__main__":
    main()
