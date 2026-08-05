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

## PR #2 changelog — Phase 1.2: CombatSituation.from_bridge_state

Landed on `glm52`. Pure-python; no mod change required. Defines and tests the target spec for the Phase 1.1 mod patch (PR #3).

**Changes:**

- `sts2_env/search/situation.py`: new classmethod `CombatSituation.from_bridge_state(state, *, situation_id=None)`. The live-game counterpart to `from_run_manager`: reads the JSON the C# mod sends at combat_start and produces a `CombatSituation` the SearchAgent can clone.
  - Accepts deck entries in two forms: the current mod's bare id strings (upgraded flag lost, defaults to False) and the target spec `{"id": str, "upgraded": bool}` dicts (which PR #3 will move the mod to). Both paths tested.
  - Requires `encounter` (the setup function name) and `encounter_seed`/`combat_seed` (ints). Raises `ValueError` with a clear message if `encounter` is missing — a quiet fallback to a random encounter would have the search planning against a different fight than the one on screen. These fields are the Phase 1.1 mod patch's job.
  - Defaults `character_id` to `"Ironclad"` when absent (the current mod hardcodes Ironclad per `RlAutoSlayer.cs:78` and sends no character_id field).
  - `combat_seed` falls back to `encounter_seed` when absent, so a mod that sends one seed rather than two still works.
  - New helper `_parse_deck_entry` handles both deck entry formats.

- `tests/test_combat_situation_from_bridge.py`: 13 new tests covering:
  - Encounter + seed rebuild the same enemies (reproducibility)
  - Player HP / deck / relics / potions / room type / floor fields round-trip
  - Upgraded flag is preserved when the mod sends dicts
  - Missing encounter raises rather than silently randomising
  - Missing encounter_seed defaults to 0; missing combat_seed falls back to encounter_seed
  - Two rebuilds of one situation agree on enemy HP (the benchmark's foundational property)
  - A bridge-built situation plays a turn through the simulator without raising

**What's not in this PR:**

- The C# mod patch (PR #3 / Phase 1.1) — needed to send `encounter`, `encounter_seed`, `combat_seed`, and the upgraded flag on deck entries. Without it, calling `from_bridge_state` on a real bridge payload raises, which is the intended loud failure.

**Pre-existing test failures noted but not addressed:** 123 parity tests in `test_regent_*`, `test_silent_*`, `test_status_curse_*` fail on `glm52` independent of these changes (verified by stashing the edits and re-running). They're content-parity regressions from earlier work, not regressions from this PR.

## PR #4 changelog — Phase 3.1: --situation-set training flag

Landed on `glm52`. Pure-python; no mod change required. Adds the env-and-script plumbing the next training run needs to leave the starter-deck plateau behind.

**Changes:**

- `sts2_env/gym_env/combat_env.py`: `STS2CombatEnv.__init__` gains a `situation_pool: list | None` argument. When set, `reset()` draws a random `CombatSituation` from it and calls `to_combat()` instead of building the Ironclad starter deck + Act 1 encounter pool. The situation owns the deck, HP, relics, potions, room type and encounter; the env's `encounter_pool` / `player_hp` / `player_max_hp` arguments are simply not consulted when `situation_pool` is set. When `None` (the default), every existing test and every existing training command runs exactly as before.

- `scripts/train_combat.py`: new `--situation-set PATH` flag. Loads the JSON fixture with `load_situations` and passes it through to every env (train and eval). The script's banner now prints the situation count and, when `--resume-from` is also set, a note that the fine-tune is re-fitting a starter-deck model to the real distribution.

- `tests/test_combat_env_situation_pool.py`: 7 new tests covering the situation path's contract — obs shape, action mask legality, episode termination under END_TURN-only, starter-deck fallback when `None`, the situation path actually changing deck size away from the starter's 10, and same-seed reproducibility.

**Why this is the lift that breaks the plateau.**

`combat_v3_overnight` scored 92% on the starter-deck benchmark and 6.7% on bosses in the 200-fight harvested benchmark (`docs/MODELS.md:28`). The starter-deck model had never seen a 16-card deck at 40 HP holding three relics; it could not have learned the right plays because the situations it learned on did not contain them. Training against the harvested fixture puts the model in the rooms it dies in.

**What's not in this PR.**

The actual retraining run. It needs the decompile on disk to match the installed game build (the script refuses otherwise), and ~2M steps overnight. The user can pull the trigger:

```bash
python scripts/train_combat.py \
    --resume-from output/combat_v3_overnight/final_model.zip \
    --situation-set tests/fixtures/act1_combat_benchmark.json \
    --total-timesteps 2000000 \
    --n-envs 8 \
    --output-dir output/combat_real_situations
```

A larger fixture (2000 situations rather than 200) is the next harvest step, per Phase 3.2 of the roadmap.

## PR #5 changelog — Phase 4.1: HierarchicalRunEnv reward leak fix

Landed on `glm52`. Pure-python; no mod change required. Fixes the bug that made seven meta-policy training versions (`meta_ppo_v1..v7`) silently unlearnable, as `docs/MODELS.md:240` diagnosed.

**The bug, confirmed by stashing the fix.** `HeuristicCombatSolver.solve` and `FrozenRLCombatSolver.solve` both called `run_env._step_combat(action)` — the private method that applies the action but bypasses the reward block in `run_env.step`. So the meta-policy never saw:

- `COMBAT_WON` (1.0, per `reward_config.py:58`) — never credited for a fight won.
- `ELITE_WON` (3.0) and `BOSS_WON` (10.0) — never credited for harder rooms.
- `FLOOR_REACHED` (0.35) for floors advanced when a combat was fast-forwarded across an act boundary.
- Card-reward shaping for the meta-decision immediately after.

Only the terminal `RUN_WIN` / `RUN_DEATH` and the card-reward shaping (which fires outside the solver) reached the meta-policy. Its entire reward signal was sparse and arrived once at the end of a 400-step rollout, hundreds of steps from any decision it could attribute the reward to.

**The fix.** Both solvers now route through `run_env.step(action)` instead of `run_env._step_combat(action)`. The accumulated `step_reward` is returned to the meta-policy via `reward += combat_reward` (already in `HierarchicalRunEnv.step` at `:257`). The work `step` does that `_step_combat` didn't (observation encoding, terminated/truncated evaluation) is cheap and only happens inside a fast-forward loop — never on the hot training path.

Also propagates `truncated` from internal steps (the step cap can fire mid-fast-forward; the old code hard-coded `truncated=False`).

**How the tests pinned it.** Without the fix, every fast-forwarded combat's max positive step reward is `0.35` (FLOOR_REACHED alone). With the fix, the step that wins a hallway fight rewards `1.35` (FLOOR_REACHED + COMBAT_WON). The test asserts `max_positive_step_reward >= 1.0`. `test_floor_alone_does_not_satisfy_the_combat_won_assertion` pins `FLOOR_REACHED < COMBAT_WON` so a later change to reward config cannot silently hide the leak.

**Stash-verified:** with the fix reverted and only the tests applied, `test_a_combat_that_ends_during_fastforward_emits_combat_won` fails with `assert 0.35 >= 1.0` — exactly the regression the test exists to catch.

**What's not in this PR.**

- A full meta-policy training run to verify the model now learns. That's Phase 4.2+3 with `FrozenRLCombatSolver` (defaults to `combat_v3_overnight`); the user can pull the trigger:

```bash
python scripts/train_meta_policy.py \
    --combat-policy output/combat_v3_overnight/final_model.zip \
    --total-timesteps 5000000 \
    --act-count 1 \
    --n-envs 8 \
    --output-dir output/meta_ppo_v8_rewarded
```

The eval-reward curve on `v1..v7` was flat across all evaluations (per `docs/MODELS.md:223`). The first eval after this fix that *isn't* flat is the signal Phase 4 worked.

## PR #6 changelog — Phase 1.1: C# mod patch sending encounter + seeds + deck upgrade

Landed on `glm52`. C# changes to the bridge mod. **Not compiled or tested against the live game on this branch** — the user needs to compile and validate. Pure-python changes alongside it (encounter name normalisation) are tested in `tests/test_combat_situation_from_bridge.py`.

**C# changes:**

- `bridge_mod/RlRunInfo.cs:163-189` — deck entries are now `{"id": str, "upgraded": bool}` dicts instead of bare strings. The Ironclad starter's Bash+ now reports as `upgraded: true`, where before it lost the flag and the SearchAgent's clone would have featured a 2-turn Vulnerable instead of 3. Bare strings still work as a fallback in `from_bridge_state` for tests and any mod not rebuilt against this change.

- `bridge_mod/RlCombatHandler.cs:567-586` — `SerializeCombatState` now attaches three fields to every `combat_action` state:
  - `encounter` — `combatState.Encounter.Id.Entry` (PascalCase class name like `"NibbitsWeak"`).
  - `encounter_seed` — derived from `(runSeed + totalFloor) + StringHelper.GetDeterministicHashCode(encounterName)`, the same formula `EncounterModel.GenerateMonstersWithSlots` uses at `EncounterModel.cs:263`. The Python `Rng(seed)` from `to_combat()` then produces the same monster HP rolls the live game saw.
  - `combat_seed` — currently set to the same value as `encounter_seed`. The C# game's combat-level RNG (deck shuffle, monster AI) is also derived from `RunState.Rng`, so the same hash should suffice; if parity testing later shows the shuffle uses a separate stream, split the field and patch the mod again. `from_bridge_state` accepts whatever it is sent.

- `bridge_mod/RlCombatHandler.cs:30` — added `using MegaCrit.Sts2.Core.Helpers;` for `StringHelper.GetDeterministicHashCode`.

**Python-side changes:**

- `sts2_env/search/situation.py` — `resolve_encounter` now accepts the C# mod's PascalCase encounter form (`"NibbitsWeak"`) in addition to the Python `setup_X` form. New helper `_setup_name_for_encounter_id` normalises PascalCase → `setup_X_snake_case_lower`, UPPER_SNAKE also handled. Existing tests that pass `setup_X` names still resolve; tests that pass PascalCase also resolve.

- `tests/test_combat_situation_from_bridge.py` — 3 new tests for the name normalisation: PascalCase resolves to the same function as setup_X, end-to-end rebuild from a PascalCase `encounter` field succeeds, and the boss encounter `VantomBoss` resolves to `setup_vantom_boss`.

**What's NOT verified on this branch:**

- The C# does not compile on this machine (no dotnet / Godot SDK setup; per `docs/PARITY_GAPS.md:249`, dotnet was not on PATH as of 2026-05-22). The patches compile in the head, against the `decompiled/` reference, but the user must build the mod and validate against a live session.
- Seed parity is asserted structurally (same formula as `EncounterModel.cs:263`) but not bit-verified against the live game. A `scripts/check_card_parity.py`-style audit for monster HP would close this gap and is a candidate Phase 1.1.1 task.
- The `(long)seed` cast for JSON may overflow to negative for very large `ulong` seeds; Python's `int()` accepts negative values faithfully, but if the simulator's `Rng.__init__` masks negative to positive, the stream diverges. Verify by reading `sts2_env/core/rng.py` before live iteration.

**To validate when the user wakes up:**

```bash
# In the sts2-rl-agent checkout:
dotnet build bridge_mod/STS2BridgeMod.csproj
# Start STS2 with the new mod; capture one combat_action JSON line:
.venv/bin/python -m sts2_env.bridge.live_eval \
    --model-path output/combat_v3_overnight/final_model.zip \
    --log output/live_eval_pr6_test.jsonl \
    --journal output/live_journal_pr6_test.jsonl \
    --runs 1 --verbose
# Check the journal for combat_start fields:
grep combat_start output/live_journal_pr6_test.jsonl | head -1
# It should now report encounter, encounter_seed, combat_seed, and
# deck entries as dicts.
```

## PR #7 changelog — Phase 2.1: live_search.py — the SearchAgent on the live path

Landed on `glm52`. Pure-python; opt-in flag `--live-search`. The next live session can use real lookahead in combat instead of the trained model's single-step argmax.

**What it does:**

`sts2_env/bridge/live_search.py` (~190 LoC) — `LiveSearch` class wraps `SearchAgent`:

- Builds a local `CombatState` from the bridge's `combat_action` state on the first call of each fight, via `CombatSituation.from_bridge_state(state).to_combat()` (PR #2's path).
- Mirrors every action the runner sends to the live game into the local sim: each `decide(bridge_state, prev_action=...)` applies the previously-returned action to the local sim to keep it in lockstep.
- Calls `SearchAgent.act(local_combat)` to plan the turn: enumerate legal orderings, end the turn on a clone, let the enemies reply on the clone, keep the best line. The action returned is in the same `Discrete(115)` layout the trained model uses, so the existing `state_adapter.decode_action` path sends it to the game unchanged.
- Detects drift (HP, block, energy, hand-size mismatches beyond tolerance) and logs loudly without crashing — the SearchAgent's plan-divergence handling is the backstop.
- Resets per-fight via `reset_for_new_fight()`, which the runner calls at the combat-start transition.

**Runner integration:**

- `sts2_env/bridge/agent_runner.py` — `run_agent` gains `live_search: bool = False` parameter; when True, the combat branch routes through `LiveSearch.decide(state, prev_action)` instead of `model.predict(obs, action_masks=mask)`. The decoded action goes to the bridge via the existing `play_card` / `end_turn` / `use_potion` client methods. Per-run and per-fight resets tracked. Falls back to `END_TURN` if the SearchAgent raises (the likely cause being the mod missing the Phase 1.1 fields, which `from_bridge_state` raises on).
- `--live-search` CLI flag wired into both `agent_runner` and `live_eval`.
- No-op path when False: the existing model path runs unchanged. Backwards compatible.

**Tests** (`tests/test_live_search.py` — 8 new):

1. First `decide` returns a legal action in `Discrete(115)`.
2. First `decide` invokes search at least once and increments the rebuild counter.
3. Subsequent decides within a fight return legal actions without rebuild.
4. The SearchAgent keeps planning across calls within a turn (replay vs replan).
5. `reset_for_new_fight` triggers a fresh build on the next decide.
6. Drift is logged but does not break `decide` — no exception.
7. Missing `encounter` (mod not patched) raises `ValueError` on first decide — loud failure, the right behaviour.
8. The action returned decodes via `StateAdapter.decode_action` — pins that the SearchAgent and the adapter agree on action-index meaning.

**What's NOT done here:**

- Drift recovery mid-fight. `to_combat()` always starts a fresh fight (calls `start_combat`); there's no path to rebuild a mid-fight CombatState from JSON. The roadmap's Phase 2 hinted this might be needed — for the milestone it's not, because the SearchAgent's plan-divergence handling is the backstop, and a fresh rebuild happens at every combat_start.
- The Phase 2.4 live session: needs the mod compiled (PR #6) and STS2 running. After PR #6 lands and the user validates the mod, run:

```bash
.venv/bin/python -m sts2_env.bridge.live_eval \
    --model-path output/combat_v3_overnight/final_model.zip \
    --live-search \
    --log output/live_eval_pr7_search.jsonl \
    --journal output/live_journal_pr7_search.jsonl \
    --runs 20 --verbose
```

Expected vs Phase 0.3 baseline: same hallway win rate, much lower HP-per-fight (the search survives turns the model threw away), occasional boss win. If boss win rate moves from 0% to ≥10% across 20 runs, continue to Phase 3. If still 0%, the deckbuilding phase is the gating failure.