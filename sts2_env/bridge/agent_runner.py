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
import logging
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from sts2_env.bridge.client import STS2GameClient
from sts2_env.bridge.cyra_events import CyraPublisher
from sts2_env.bridge.journal import RunJournal
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
CARD_REWARD_TYPE_PRIORITY = ("power", "attack", "skill")
SHOP_PURCHASE_ACTION_PRIORITY = (
    "buy_relic",
    "buy_card",
    "buy_potion",
    "remove_card",
    "buy_item",
)
SHOP_LEAVE_ACTION = "leave_shop"
REWARD_PROCEED_ACTION = "proceed"
REWARD_PICK_ACTION = "pick_reward"
CARD_BUNDLE_PICK_ACTION = "pick_card_bundle"
CRYSTAL_SPHERE_CELL_ACTION = "divine_cell"
REST_HEAL_OPTION_ID = "heal"
REST_SMITH_OPTION_ID = "smith"
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
            says what happened on the way, which is what a decision about what to
            fix next actually needs.
    """
    model = load_model(model_path)
    adapter = StateAdapter()

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
        })
        logger.info("Requested speed=%s, random_fallback=%s", speed, allow_random_fallback)

        logger.info("Connected. Starting agent loop.")

        step_count = 0
        combat_count = 0
        _was_in_combat = False
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
        journal = RunJournal(journal_path, model=model_path)
        journal.start_run(1)
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

                # Counted here rather than never: this was initialised, reset and
                # reported without ever being incremented, so every live record
                # written so far says "combats": 0.
                if _phase_for_state(state) in Phase.COMBAT_PHASES and not _was_in_combat:
                    combat_count += 1
                _was_in_combat = _phase_for_state(state) in Phase.COMBAT_PHASES

                for event in milestones.observe(state):
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
                        summary.update({
                            "run": run_index,
                            "result": result,
                            "steps": step_count,
                            "combats": combat_count,
                            "seconds": round(time.monotonic() - run_started, 1),
                        })
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
                    progress = {}
                    milestones.reset()
                    journal.start_run(run_index + 1)
                    events_seen.clear()
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
                        indexes = _pick_card_select_indexes(state)
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
                    if verbose:
                        logger.info("SHOP: choosing option %d", choice)
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
        finally:
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


def _pick_map_node(state: dict[str, Any]) -> int:
    """Choose a reachable map node from the bridge state's node list."""
    nodes = list(state.get("nodes", []))
    if not nodes:
        return DEFAULT_CHOICE_INDEX
    hp_ratio = _read_hp_ratio(state)
    priority = (
        ROOM_PRIORITY_LOW_HP
        if hp_ratio is not None and hp_ratio < REST_HP_RATIO_THRESHOLD
        else ROOM_PRIORITY_HEALTHY
    )
    for room_type in priority:
        for fallback_index, node in enumerate(nodes):
            if _canonical_text(node.get("type")) == room_type:
                return _read_index(node, fallback_index)
    return _read_index(nodes[0], DEFAULT_CHOICE_INDEX)


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


def _worth_upgrading(card: Any) -> bool:
    return not _is_basic_card(card) and not _is_upgraded(card)


def _pick_card_select_indexes(state: dict[str, Any]) -> list[int]:
    """Choose required card indexes for upgrade/transform/select screens.

    Basics go last. This used to take the first cards in the list, and a deck is
    listed Strike, Strike, Strike... so every upgrade at a rest site went into a
    basic Strike -- the least valuable upgrade available and one that does
    nothing for a deck's actual problems.
    """
    cards = list(state.get("cards", []))
    min_select = max(int(state.get("min_select", 1)), 0)
    max_select = max(int(state.get("max_select", min_select)), 0)
    if not cards or max_select == 0 or min_select == 0:
        return []
    count = min(min_select, max_select, len(cards))

    # Stable ordering: real cards first, then basics, each keeping deck order so
    # the choice stays reproducible.
    ranked = sorted(
        range(len(cards)),
        key=lambda i: (not _worth_upgrading(cards[i]), i),
    )
    chosen = ranked[:count]
    return [_read_index(cards[i], i) for i in chosen]


def _pick_card_reward_index(state: dict[str, Any]) -> int | None:
    """Choose a card reward, or return None when skipping is the best action."""
    cards = list(state.get("cards", []))
    can_skip = bool(state.get("can_skip", False))
    if not cards:
        return None if can_skip else DEFAULT_CHOICE_INDEX
    if can_skip and _read_deck_size(state) > CARD_REWARD_LARGE_DECK_SIZE:
        return None
    for card_type in CARD_REWARD_TYPE_PRIORITY:
        for fallback_index, card in enumerate(cards):
            if _canonical_text(card.get("type")) == card_type:
                return _read_index(card, fallback_index)
    return _read_index(cards[0], DEFAULT_CHOICE_INDEX)


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


def _pick_rest_option(state: dict[str, Any]) -> int:
    """Choose a rest-site option by option identity, not display order."""
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX
    hp_ratio = _read_hp_ratio(state)

    # Smithing is only worth a rest if there is something worth smithing. A deck
    # of nothing but basics has no upgrade that changes a fight, so the HP is
    # worth more -- which is the whole reason to be at a rest site.
    deck = state.get("deck") or []
    has_real_upgrade = any(_worth_upgrading(card) for card in deck) if deck else True

    preferred = (
        REST_HEAL_OPTION_ID
        if (hp_ratio is not None and hp_ratio < REST_HP_RATIO_THRESHOLD)
        or not has_real_upgrade
        else REST_SMITH_OPTION_ID
    )
    option = _first_matching_option(options, option_ids=(preferred,))
    if option is None and preferred == REST_SMITH_OPTION_ID:
        option = _first_matching_option(options, option_ids=(REST_HEAL_OPTION_ID,))
    if option is None:
        option = options[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


def _pick_shop_option(state: dict[str, Any]) -> int:
    """Buy an enabled shop item when one exists; leave when only exit remains."""
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX
    for action in SHOP_PURCHASE_ACTION_PRIORITY:
        option = _first_matching_option(options, actions=(action,))
        if option is not None:
            return _read_index(option, DEFAULT_CHOICE_INDEX)
    option = _first_matching_option(options, actions=(SHOP_LEAVE_ACTION,)) or options[0]
    return _read_index(option, DEFAULT_CHOICE_INDEX)


_MARKUP = re.compile(r"\[/?[a-zA-Z]+\]")
_LOSE_HP = re.compile(r"lose\s+(\d+)\s+(max\s+)?hp", re.IGNORECASE)
_HEAL_HP = re.compile(r"heal\s+(\d+)\s+hp", re.IGNORECASE)
_SET_MAX_HP = re.compile(r"max\s+hp\s+(?:becomes|is set to|to)\s+(\d+)", re.IGNORECASE)

# Never walk out of an event below this much of your maximum. An event is worth
# HP, but no event reward is worth arriving at the next elite unable to survive
# its opening turn -- which is how these runs actually end.
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

    safe: list[tuple[int, dict]] = []
    unsafe: list[tuple[int, int, dict]] = []       # (cost, index, option)

    for i, option in enumerate(options):
        hp_lost, max_hp_lost, max_hp_set = _event_hp_cost(option)
        index = _read_index(option, i)

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

        if lethal:
            unsafe.append((hp_lost, index, option))
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
                    return index
        return safe[0][0]

    # Everything on offer is dangerous. Take the cheapest rather than the first,
    # and never fall back to an option that was explicitly rejected.
    logger.warning(
        "Every option in event %s looks harmful at %s HP; taking the cheapest.",
        event_id or "?", hp,
    )
    return min(unsafe)[1]


def _pick_crystal_sphere_option(state: dict[str, Any]) -> int:
    options = _enabled_options(state)
    if not options:
        return DEFAULT_CHOICE_INDEX
    option = _first_matching_option(options, actions=(CRYSTAL_SPHERE_CELL_ACTION,))
    if option is not None:
        return _read_index(option, DEFAULT_CHOICE_INDEX)
    option = _first_matching_option(options, actions=(REWARD_PROCEED_ACTION,)) or options[0]
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
    client: STS2GameClient, max_retries: int = 10, delay: float = 3.0
) -> None:
    """Attempt to reconnect to the game server with retries.

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
        "--combat-policy",
        default=None,
        help=(
            "Optional separate combat policy (.zip). When the main model is a full-run "
            "model, this overrides combat decisions so the main model only handles "
            "map, rewards, shop, rest, and events."
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
    )


if __name__ == "__main__":
    main()
