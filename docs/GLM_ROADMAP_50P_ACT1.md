# GLM Roadmap — 50% Act 1

Goal: live agent beats Act 1 with ≥50% confidence measured across 20 consecutive live runs, with structured per-action logs sufficient to diagnose every failure.

Building on the four pieces the project already contains — `CombatSituation` + `SearchAgent` + `RunJournal` + `score_combat_benchmark`. We construct the smallest missing seams. Each phase has a measurable live-game checkpoint.

Reference background on the LLM-router idea: <https://huggingface.co/blog/t22000t/slay-the-spire-ai-collection>. Likely only partly useful because that model was trained several months ago and the game has had substantial updates since — the simulator-parity work in this repo is current, the HF model is a snapshot.

## What's already in place (don't rebuild)

Glancing over the codebase before writing new code turned up that most of the infrastructure I would have proposed building already exists. Recording it here so it doesn't get rewritten by accident.

- `sts2_env/search/situation.py:108` — `CombatSituation` dataclass + `from_run_manager` + `to_combat` + JSON serialize. This is the "combat from real situations" loader. Done.
- `scripts/harvest_combat_benchmark.py` — walks runs and snapshots real fights with floor-band quotas. Done.
- `scripts/score_combat_benchmark.py --model X.zip --against Y.zip` — paired comparison on real situations. Already used; `MODELS.md:122` records the search-v2 +9% / -7.8 HP result. Done.
- `sts2_env/search/turn_search.py:494` — `SearchAgent.act(combat)` returns an action int, replans per turn, replays the plan against the live mask, abandons and replans when the plan diverges. Designed to be called from the live path. Done — just not plumbed into `agent_runner.py`.
- `SearchResult.gap` at `turn_search.py:124` — margin between best and runner-up lines, explicitly flagged at `:128` as "feeds Cyra's `gut_phrase`". Done.
- `sts2_env/bridge/cyra_events.py:112` `CyraPublisher.publish` — fire-and-forget. Commit `3df5cb8 "bridge: tell Cyra the four moments that matter, and nothing else"` already wired the milestone hooks. Done.
- `sts2_env/bridge/journal.py:325` `RunJournal.wrap()` — proxies the bridge client so every action gets journaled with offered-vs-chosen. Already applied to play_card / end_turn / use_potion / choose / choose_many / skip. Done.

## What "never blocks, never plays Powers" actually is

Not a learning bug. A distribution problem.

- The reward is correct: `gym_env/reward_config.py:79` has `COMBAT_HP_WEIGHT = 1.0`, `:80` `COMBAT_ENEMY_WEIGHT = 0.5` — player HP counts double, and the comment at `:82` says explicitly "Player HP is weighted double because enemy HP resets every combat and yours does not." Block IS rewarded.
- The action mask is correct: `gym_env/action_space.py:121-136` walks the hand honestly and exposes skill and power cards unconditionally. Nothing structural is hiding them.
- The training distribution isn't. `gym_env/combat_env.py:78` reads `deck = create_ironclad_starter_deck()`. The Ironclad starter deck contains 0 Powers and 4 block cards. All training is against random Act 1 encounters at full HP. PPO had no training situation in which playing a Power was ever the right call. The observation in `MODELS.md:147` — "15 power cards in 465 plays, against a benchmark where 73 of 200 decks hold one" — is exactly what you get from that. The model literally couldn't learn it.
- The hierarchical-env reward leak is at `gym_env/hierarchical_env.py:239` and `:256`: both `CombatSolver.solve` implementations initialize `total_reward = 0.0` and never accumulate. The `MODELS.md:240` diagnosis was correct.

## Phase 0 — measurement is step zero (half a day)

Before any new training, fix the loop so we can tell what each change did.

### 0.1 Capture the actual bridge payload

One short session prints the first `combat_action` and `combat_start` JSON the mod emits, stamps it to `output/bridge_protocol_sample.json`. This fixes the spec we're coding against before we touch any C#.

### 0.2 Stop the silently-broken counters

- `combats` field in `live_eval.py` is initialized but never incremented — explicit note at `agent_runner.py:307-312`. Three lines to fix.
- Add `death_floor`, `death_room_type`, `death_enemy_id` to the run-end summary in `live_eval.py`. Currently `death_floor` is `None` across all 172 logged runs.
- Add `act_cleared` (`act >= 2`) to the JSONL record, not just the report aggregator — `tests/test_live_eval_act_clear.py` fixes the report; the raw event is still missing.

### 0.3 First baseline live session

Run `output/combat_v3_overnight` against the live game for ~20 runs. Examine the journal. Confirm the mod sends offered-vs-chosen on card rewards via `_OPTION_KEYS` at `journal.py:39`. If it doesn't, extend the mod to emit them; if it does, we're done.

**Exit criterion:** we can read a journal line and know what the agent did, what it was offered, how much HP it spent in every fight, and where the run died. Without this, everything below is unverifiable.

## Phase 1 — extend bridge protocol to enable CombatSituation reconstruction (1-2 days)

### 1.1 Extend `bridge_mod/` to send

In the `combat_start` payload add: `deck` as a list of `{"id", "upgraded"}` pairs, plus `encounter` (the `setup_*` function name) and `encounter_seed`. `live_journal_run1.jsonl:7` shows `relics` already comes through; this adds the last two pieces.

### 1.2 Add `CombatSituation.from_bridge_state(state)` in `sts2_env/search/situation.py`

~80 LoC. Parses the same deck/relic/potion/enemies/encounter fields as `from_run_manager`, but from JSON. The goal is: `state_json["combat_start"] → CombatSituation → to_combat() → CombatState` produces a real fight the SearchAgent can clone.

### 1.3 Parity test

`tests/test_combat_situation_from_bridge.py` — feed a captured `combat_start` payload, call `to_combat()`, assert the enemies and opening hand and player HP match what the live game reported.

**Exit criterion:** from a live `combat_start` message, the simulator can build the same fight.

## Phase 2 — wire live search into `agent_runner.py` (2-3 days)

### 2.1 New module `sts2_env/bridge/live_search.py` (~150 LoC)

Wraps `SearchAgent`:

- On `combat_start`: call `CombatSituation.from_bridge_state(state).to_combat()` to seed a local `CombatState`.
- On each `combat_action`: check the local sim's player state against the bridge's (HP, block, energy, powers, hand size). If they diverge beyond tolerance, rebuild from the new bridge state — drift recovery.
- On each action from `agent_runner`: feed it into the local sim AND into the live SearchAgent; return the SearchAgent's action.
- The local sim mirrors *what the live game did*, so the SearchAgent's clone always starts from a position the live game is actually in.

### 2.2 Edit `agent_runner.py:396-419`

The combat phase dispatch becomes: if `_live_search` is configured, route to `live_search.action(state)`; else fall back to the existing PPO-combat-model path. Same return-shape, same downstream journal.

### 2.3 Train the rollout policy

The `_playout` in `turn_search.py:217` is the block-then-damage heuristic. Leave it for now — `MODELS.md:97` proved extending the horizon with a dumb playout doesn't help. But add an optional `--playout-policy` to `SearchAgent` so a trained PPO can be passed as the rollout function. ~30 LoC edit to `turn_search.py` + a wrapper in `live_search.py`.

### 2.4 Live session #2 (SearchAgent online, combat_v3_overnight as rollout policy)

Run 20 runs. Compare to baseline from Phase 0.3.

- Expected: same hallway win rate, much lower HP-per-fight, occasional boss win.
- **Decision point:** if boss win rate moves from 0% to ≥10% across 20 runs, continue. If still 0%, the deckbuilding phase is the gating failure — go to Phase 3 with that knowledge.

**Exit criterion:** live agent uses real lookahead in combat, not just policy argmax.

## Phase 3 — retrain combat on real situations (3 days + one overnight training)

### 3.1 Add `--situation-set` to `scripts/train_combat.py`

~20 LoC. When set, `STS2CombatEnv.reset` calls `CombatSituation.from_dict` from the file instead of `create_ironclad_starter_deck()` + `ALL_ACT1_ENCOUNTERS`. May need to expand `CombatSituation.to_combat()` to optionally skip the `start_combat()` call when the situation is mid-fight (not needed today — `to_combat` calls it, which is fine).

### 3.2 Re-harvest a larger fixture

Run `scripts/harvest_combat_benchmark.py --combat-model output/combat_v3_overnight/final_model.zip --count 2000 --max-floor 16`. The existing one is 200; 2000 is still a small file but enough to land deck variety. The harvest script handles floor quotas already.

### 3.3 Train

Resume from `combat_v3_overnight` (existing fine-tune path in `scripts/refinetune.sh` and `train_combat.py --resume-from`). Target ~2M steps, single overnight.

- Eval on `--situation-set` of the held-out 200 (which `score_combat_benchmark.py` already reads).
- Real metric: `combat_v3_overnight` is 74% overall / 6.7% boss / 53 HP lost per boss on the 200-fight benchmark. **Pass: ≥80% overall / ≥30% boss / ≤35 HP per boss.**

### 3.4 Live session #3 (new combat model + live search)

20 runs. The 50% Act 1 milestone target is here.

**Exit criterion:** trained combat improves the held-out benchmark; live runs prove it transfers.

## Phase 4 — fix the reward leak so deckbuilding can learn (2-3 days, runs in parallel with Phase 3's training)

This is the meta-policy path. It matters because if it works, Phase 5's LLM is reduced from "decider" to "tie-breaker".

### 4.1 Fix `HierarchicalRunEnv.step`

At `hierarchical_env.py:239` and `:256`, have the combat solver call `self._inner.step(action)` (the public API) instead of `self._inner._step_combat(action)` (private, bypasses reward). Then accumulate `obs, reward, terminated, truncated, info` across the fast-forward and surface the sum to the meta-policy.

### 4.2 Replace `HeuristicCombatSolver` default with `FrozenRLCombatSolver` from `combat_v3_overnight`

The 7 dead meta-policy versions were trained with a solver that never blocks (`hierarchical_env.py:45-122`); we now have a better model and a search path.

### 4.3 Optional: `FrozenSearchCombatSolver`

~80 LoC. Replaces frozen PPO with `SearchAgent` as the fast-forward solver. Slower training (search is ~3 s/turn inside a fast-forward loop runs 1000s of turns per rollout) but the meta-policy would learn against real combat play rather than a frozen snapshot. **Skip unless 4.1+4.2 alone reach the 50% milestone.**

**Exit criterion:** a meta-policy training run's eval reward is *not flat across N evals* for the first time. Then we have signal.

## Phase 5 — LLM router over sparse-reward meta decisions (1-2 weeks, conditional)

Only after Phase 4 gives non-flat meta-policy reward signal. Without that, the LLM is comparing against a flat baseline where any choice looks equivalent.

### 5.1 New module `sts2_env/bridge/decision_router.py` (~400-600 LoC)

For phases `card_reward`, `boss_relic`, `rest_site`, `shop`, `event`, calls a constrained-decoding LLM (Qwen-7B-Instruct-Q4 via `llama-cpp-python`, T=0) with a structured prompt built from the same `choice_encoding` features the simulator uses. Falls back to the run-adapter PPO on parse error.

The Kimi-2.7 / Opus failures on combat are not evidence against LLMs here — combat is too long-horizon for token-level reasoning; deck selection is not.

### 5.2 Constrained decoding

Out-of-band options masked via BBE token masking or via post-hoc rejection sampling. Skip the long-prose LLM trap entirely. The HF blog post above is a useful reference for prompt structure, but the model weights there are months out of date — build on the simulator and search for the actual decisions, not on a dated HF snapshot.

## Phase 6 — Cyra milestones (already 90% done, half a day)

Already four milestone events publish via `cyra_events.py` per commit `3df5cb8`. Audit that they're firing; add `decision_router` explanations on close-gap decisions (uses `SearchResult.gap` and LLM-router margin).

## Update-robustness (stays continuous)

- `scripts/on_update.sh` (+ `diff_decompiles.py`) on each game patch — already exists.
- `check_card_parity.py` runs in CI — already exists.
- Observation layout fingerprint raises `ValueError` on mismatch — already exists at `agent_runner.py:181`.
- `scripts/refinetune.sh` resumes from `best_model.zip` on patch — already exists.

## Files the plan touches

| Touch | File | LoC |
|---|---|---|
| EDIT | `sts2_env/bridge/agent_runner.py:307-322` (counters, death fields) | ~30 |
| EDIT | `sts2_env/bridge/live_eval.py:48-65` (act_cleared raw event) | ~10 |
| EDIT | `sts2_env/bridge/agent_runner.py:396-419` (combat dispatch to search) | ~40 |
| NEW | `sts2_env/bridge/live_search.py` (SearchAgent wrapper) | ~150 |
| EDIT | `sts2_env/search/situation.py:244` (add `from_bridge_state`) | ~80 |
| EDIT | `scripts/train_combat.py:69` (add `--situation-set`, swap env.reset) | ~25 |
| EDIT | `sts2_env/gym_env/combat_env.py:69` (situation-loaded reset) | ~20 |
| EDIT | `sts2_env/gym_env/hierarchical_env.py:239,256` (reward leak) | ~20 |
| NEW (optional) | `sts2_env/gym_env/frozen_search_solver.py` | ~80 |
| NEW (Phase 5) | `sts2_env/bridge/decision_router.py` | ~400-600 |
| NEW | `bridge_mod/RlCombatHandler.cs` (extend payload) | ~30-50 C# |
| NEW | `tests/test_combat_situation_from_bridge.py` | ~80 |
| NEW | `tests/test_live_search_round_trip.py` | ~120 |

Total: ~1100 LoC new code, ~200 LoC edits, on top of ~50,000 LoC preserved. One small mod patch (which is the only true external dependency).

## Time estimate, honest

- Phase 0: half a day → tight feedback loop online
- Phase 1: 1-2 days
- Phase 2: 2-3 days → first measurable live win
- Phase 3: 3 days + 1 overnight training → second measurable live win, this is where 50% Act 1 has its best shot
- Phase 4: 2-3 days (can run in parallel with Phase 3's training) → fixes meta-policy learning
- Phase 5: 1-2 weeks conditional on Phase 4

**Realistic: ~3 weeks of focused part-time work to evaluate whether 50% Act 1 is achievable** — that's phases 0-3 + a Phase 4 stub. The live session after Phase 3 is the diagnostic: either we've cleared it or we know exactly why not.

## What this plan doesn't promise

- Beating the Act 3 boss in this timeline. P0-P3 should beat Act 1. Phase 4 makes Act 2 possible.
- LLM Qwen is sub-second for a constrained choice. But the mod setup — bridge mod compilation, Godot 4.5.1 Mono, .NET 9 — is a yak shave. Verify the mod compiles on this machine on day 1 (per `PARITY_GAPS.md:249`, dotnet wasn't on PATH here as of 2026-05-22).

## Note on stale content

The Doormaker boss (`sts2_env/monsters/act3.py:1921`, `encounters/act3.py:170-173`) was removed from STS2 some time ago but is still present in this repo's simulator. Doesn't affect the Act 1 milestone but matters for any Act 3 work that follows. Worth pruning when it becomes the crosshair; not before.

## First concrete PRs (small, testable)

1. **PR #1: Phase 0 instrumentation** — fix counters + add death fields + add `act_cleared` raw event + test. Pure-python, no mod dependency. One evening's work, plus one live session to validate.
2. **PR #2: Phase 1.2 `CombatSituation.from_bridge_state`** + parity test. Pure-python, tests against a captured JSON sample. No mod dependency if we mock the sample.
3. **PR #3: Phase 1.1 mod patch** (when we're ready to compile) — the mod patch + a marshaller.

## PR #1 changelog — Phase 0 instrumentation

Landed on `glm52`. Pure-python; no mod change required.

What actually needed doing turned out smaller than the plan above: the `combats` counter was already incrementing (the `MODELS.md` note that said it wasn't is historical, fixed in a later commit). What was still missing was the death context and the act-clear raw event.

**Changes:**

- `sts2_env/bridge/journal.py`: `RunJournal.observe` now tracks `act` and writes an `act_clear` event when `act` increments. The event carries `act_from`, `act_to`, the floor where the boss was (the previous floor, not the new one — tracked before the floor block on purpose), `room_type`, `hp`, `max_hp`. First-act-seen is recorded silently, not as a clear.
- `sts2_env/bridge/agent_runner.py`: tracks `last_enemy_id` while in combat (first alive enemy from `state["combat_state"]["enemies"]`, with the same `state.get("combat_state") or state` fallback as `state_adapter.py`). On run end, the summary now includes:
  - `act_cleared: bool` — `progress["act"] >= 2`. Stamped on the JSONL record so downstream reports don't have to re-derive it.
  - `death_enemy_id: str | None` — `last_enemy_id` if `run_hp == 0` else `None`. `None` on a win or when the bridge never reported a combat state. Present-and-`None` rather than absent, so a missing field can't be confused with a "didn't track" later.
  - `last_enemy_id` resets at the start of each run, so it doesn't leak from run N into run N+1.
- The `_was_in_combat = _phase_for_state(state) in Phase.COMBAT_PHASES` recomputation was replaced with the already-computed `phase` variable from the top of the loop. Same result, three fewer dict lookups per state.

**Tests added:**

- `tests/test_live_journal.py`: 4 new tests for `act_clear` — the increment case, the first-act-silent case, the no-change case, and the hp-on-crossing case.
- `tests/test_live_eval_loop.py`: 6 new tests — `act_cleared` True/False, `death_enemy_id` captured on death, `None` on win, no leak between runs, and `combats` counter increments per fight (with the real-protocol detail that two combat states back-to-back count as one fight, so a non-combat state has to separate them).

**What's not in this PR:**

- The C# mod patch (Phase 1.1) — separate PR.
- `CombatSituation.from_bridge_state` (Phase 1.2) — separate PR.
- The live session validation (Phase 0.3) — needs the mod compiled; deferred until PR #3.

**Pre-existing test failures noted but not addressed:** 123 parity tests in `test_regent_*`, `test_silent_*`, `test_status_curse_*` fail on `glm52` independent of these changes (verified by stashing the edits and re-running). They're content-parity regressions from earlier work, not regressions from this PR.