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

- ~~A full meta-policy training run to verify the model now learns.~~ **Done 2026-08-06** — the eval curve is non-flat for the first time (see PR #14). Note the command below is wrong: `--act-count` is not a flag `train_meta_policy.py` has. That's Phase 4.2+3 with `FrozenRLCombatSolver` (defaults to `combat_v3_overnight`):

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

- ~~The C# does not compile on this machine (no dotnet / Godot SDK setup; per `docs/PARITY_GAPS.md:249`, dotnet was not on PATH as of 2026-05-22).~~ **Wrong, and wrong when written** — .NET 9 was installed at `~/.dotnet` the whole time, with the `PATH` export in a file the login shell never reads. It compiles with 0 errors. See PR #12.
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

## Overnight summary — 6 PRs landed

All six phases through Phase 2.1 were committed on `glm52` in one autonomous session. Pure-python work fully tested; the C# mod patch (PR #6) is committed but **uncompiled and unvalidated against the live game** — it's the only piece needing the user's machine.

```
bb5ba8c  PR #1  bridge: name the enemy a run died to, + act_clear event (Phase 0)
721ce30  PR #2  search: CombatSituation.from_bridge_state (Phase 1.2)
e4c33f3  PR #4  combat_env: --situation-set training flag (Phase 3.1)
bbcf294  PR #5  hierarchical_env: combat solver routes through step() (Phase 4.1)
2631d43  PR #6  bridge: encounter/seeds + upgraded deck (Phase 1.1, C# uncompiled)
35e6d8b  PR #7  bridge: live_search -- opt-in SearchAgent on the live path (Phase 2.1)
```

**302 bridge/search/gym-env tests pass**; 3387 across the wider suite pass. The 97 pre-existing `test_regent_*` parity failures (documented in PR #1's changelog) are unchanged by these commits — verified by stashing the edits and re-running.

**What's deferred to the user:**

> **Status as of 2026-08-06:** 1, 2, 3 and 5 are done. Only **4 (Phase 3.4)**
> remains. The `(long)seed` worry in item 1 turned out to be a non-issue for the
> reason stated — but the seeds diverged anyway, for a different and larger
> reason: the generator itself was not the game's. See PR #14.

1. **Compile PR #6** — `dotnet build bridge_mod/STS2BridgeMod.csproj`. Validate the combat_action JSON includes `encounter` / `encounter_seed` / `combat_seed` and that deck entries are now `{id, upgraded}` dicts. The `(long)seed` cast may overflow for very large `ulong` seeds — if the simulator's `Rng` masks negatives, the streams diverge; check `sts2_env/core/rng.py` before iterating.
2. **Phase 2.4 live session** — run `live_eval --live-search` for 20 runs and compare to Phase 0.3 baseline. Same command shape as Phase 0.3 but with `--live-search`.
3. **Phase 3.2 + Phase 3.3 retraining** — `scripts/harvest_combat_benchmark.py --count 2000 --combat-model output/combat_v3_overnight/final_model.zip`, then `train_combat.py --resume-from output/combat_v3_overnight/final_model.zip --situation-set <harvest_output>.json --total-timesteps 2000000 --n-envs 8`. Needs the decompile-to-game-build match the trainer already enforces.
4. **Phase 3.4 live session** — `live_eval --combat-policy <new_model> --live-search` for 20 runs. This is the diagnostic: either 50% Act 1 is on the table or it isn't, and the journal will say why.
5. **Phase 4.1+4.2 meta-policy training** — `train_meta_policy.py --combat-policy output/combat_v3_overnight/final_model.zip ...` to verify the eval-reward curve is now non-flat (was flat for `meta_ppo_v1..v7`).

**What's safe to skip on first live iteration:** Phase 5 (LLM decision router). It's a 1-2 week conditional piece that only matters if Phase 4 alone doesn't close the milestone. Try the simpler stack first.

The next decision the user makes — whether to ship Phase 2.4 first or jump to Phase 3.3 first — depends on which question they want to answer first: "does live search help on its own?" (Phase 2.4) or "does training on real situations help?" (Phase 3.3). Both feed into the same Phase 3.4 session where the answer to "is 50% achievable?" lands.

## PR #8 changelog — Phase 2.4 hotfix: potion naming + non-catastrophic fallback

Landed on `glm52` after the first live Phase 2.4 session regressed badly. The "live search dying on the first encounter repeatedly" symptom from the user's first `--live-search` session was **not** the search being bad — it was the search never running.

**Root cause.** The C# mod sends every model `Id.Entry` as `StringHelper.Slugify(type.Name)` — CamelCase → UPPER_SNAKE (`ModelDb.cs:537`). Cards already match the simulator's enum names; relics have `coerce_relic_id` that handles both forms; encounters got `_setup_name_for_encounter_id` in PR #6. **Potions had no normalizer.** `_POTION_MODELS` keys on PascalCase (`StrengthPotion`); the bridge sent `STRENGTH_POTION`; `create_potion` raised `KeyError('STRENGTH_POTION')` inside `to_combat()`; `LiveSearch.decide` raised; the runner's fallback fired `END_TURN` every combat step; the player did nothing but end turn until dying on the first encounter.

**Two fixes:**

1. `sts2_env/potions/base.py` — new `coerce_potion_id(potion_id)` mirror of `coerce_relic_id`. Accepts the simulator's PascalCase id (`"StrengthPotion"`) and the bridge's UPPER_SNAKE slug (`"STRENGTH_POTION"`); falls back to the input unchanged when neither resolves so the caller's KeyError-at-lookup behaviour for genuinely unknown ids is preserved.

2. `sts2_env/search/situation.py:_instantiate_potion` — an id that does not resolve after coercion now drops the slot to `None` with a `WARNING` log line instead of raising. A searcher running against a fight missing one potion is still useful; a searcher crashing tank every combat step of a live run via the END_TURN fallback, which is the regression the user saw. A future patch wants to keep the loud failure for a brand-new potion the simulator should support, and the warning is loud — it just doesn't kill the run.

**Plus a runner resilience fix** independent of potions:

`sts2_env/bridge/agent_runner.py` — the `--live-search` fallback when `LiveSearch.decide` raises now uses a two-strike policy: the first raise in a combat logs loudly and replies END_TURN once; the second switches the rest of that combat to the trained model so the run keeps playing rather than stalling on END_TURN until the player dies. `_live_search_disabled_for_combat` resets at the next `combat_start`, so search is re-enabled for every new fight.

The previous one-strike policy was catastrophic: a single bad state — an unknown potion, a parity gap on a relic, a power the simulator silently no-ops — turned every combat step into END_TURN and guaranteed death on the first encounter. With two-strike the worst case is "live_search got one turn wrong, the model finishes the fight, the run continues."

**Tests:**

- `tests/test_combat_situation_from_bridge.py` — 3 new tests: `STRENGTH_POTION` resolves to `StrengthPotion`; a bridge state with UPPER_SNAKE potion ids rebuilds without raising; a genuinely unknown potion drops with a warning rather than crashing.
- `tests/test_search_situation.py:test_an_unknown_potion_is_dropped_with_a_warning_not_a_crash` replaces the old `test_an_unknown_potion_is_a_clear_error` which pinned the prior crash behaviour.

**Phase 2.4 post-fix expectation:** re-run the same `live_eval --live-search` command. Boss win rate should now move from 0% toward 10%+ across 20 runs (per `MODELS.md:120`). If it still dies on the first encounter, the new failure mode will be a per-step log line rather than a class of state the simulator cannot reconstruct — easier to diagnose and to fix piecewise.

## PR #9 changelog — Phase 2.1 (actual fix): rebuild from bridge on every decide

Landed on `glm52` after PR #8's potion coercion got `--live-search` past the first crash, but the next session exposed the real design bug: the local sim kept across `decide` calls **drifted to a frozen fiction** and the search planned against it instead of the live game. The user's log shows the symptom clearly — 11 turns of the search replaying `END_TURN` against a state it thought had energy 0 and a 3-card hand, while the live game reported energy 3 and 5 cards in hand on every step. The player died on a hallway fight at full HP.

**The actual fix**, replacing the kept-sim model entirely:

- `sts2_env/search/situation.py` — new `CombatSituation.to_combat_mid_fight(bridge_state)`. Builds a fresh `CombatState` via `to_combat()` (which sets up deck/relics/potions/encounter properly), then **overwrites mutable state** from the bridge's report:
  - Player HP, block, energy, base_max_energy — direct assignments.
  - Player powers — full reset, then applied verbatim from the bridge's powers list via a new `_coerce_power_id` (mirrors `coerce_relic_id`/`coerce_potion_id`, handles `STRENGTH_POWER` → `STRENGTH` and the reverse).
  - Hand — cleared and rebuilt as fresh `CardInstance` objects from the bridge's hand list. The sim's own draw is discarded.
  - Enemy HP / max_hp / block / powers — same direct assignments and full-reset-and-replace for powers.
  - Enemy intent — bridge reports the live game's next move. `_override_enemy_intent` re-points the AI's current state to a `MoveState` that already exists in the simulator's state machine (installing the bridge's intent over the simulator's), so `roll_move` still works. If the bridge's `intent_move_id` is not in the simulator's state machine (a parity gap), the override is *skipped* rather than installing a follow-up-less synthetic that would crash `roll_move`.
  - Combat round_number and turn_count — set from the bridge's `round` field.

- `sts2_env/bridge/live_search.py` — rewritten **without any kept local sim**. `LiveSearch.decide` rebuilds from the bridge every single call: `CombatSituation.from_bridge_state(state).to_combat_mid_fight(state)` then `SearchAgent.act(combat)`. The previous design's `_local`, `_last_action`, `_last_round`, `_drift_count`, `_rebuild_count`, `_check_drift`, `_DriftException`, `DRIFT_TOLERANCE`, `DRIFT_DISABLE_THRESHOLD` all gone. The bridge is ground truth; the local sim matches it every step; no drift to "tolerate" because drift is the sim being wrong and the sim is thrown away per call.

- `tests/test_live_search.py` — rewritten. The new contract is one method (`decide`), one stat block (`searches`, `budget_exhausted`). Six tests pin the contract: legal action in `Discrete(115)`, search invocation, no stale-state carry between calls, energy seen on every call, reset clears the search plan, action decodes via state_adapter, mid-fight overlay gets bridge-reported HP/block/energy into the sim.

- `tests/test_combat_situation_from_bridge.py` — 7 new tests for `to_combat_mid_fight`: player HP/block/energy overlay, player powers overlay, hand overlay, enemy HP/block/powers overlay, dead-enemy handling, round/turn_count, end-to-end search invocation without raising.

**Why the old code looked right and was wrong.** `_check_drift` logged the drift (HP 91 → 81, energy 0 → 3, hand 3 → 5) and the comment said "SearchAgent will replan if the mask diverges." But the SearchAgent was *never told the mask had diverged* — it was given the same `self._local` combat, whose mask naturally matched the (wrong) local state, so plans played out uneventfully. The bridge's ground truth was logged and ignored. The lesson, bluntly: **when the game tells you the state, you set the state to what the game told you.** There is no drift to tolerate; there is only being wrong, and the fix for being wrong is to stop being wrong.

**Phase 2.4 (real) post-fix expectation:** re-run the same `live_eval --live-search` command. This time `LIVE_SEARCH` should run every combat step, plan against the bridge's reported hand+energy+enemy intents, and the death spiral should be gone. The first reachable bug after this is *not* "the search picks END_TURN forever" — that was the drift symptom, fixed — but "the search picks a card the live game rejects," which surfaces as a logged plan-divergence and a re-plan, not as a stuck state.

**Cleanup pass before commit** (no behaviour change, tests unchanged and still passing):

- `situation.py` — the mid-file `PowerId` import and the three function-local imports (`PowerInstance` ×2, `Intent`) hoisted to the module header; verified no circular-import risk (`powers.base`, `monsters.intents`, `monsters.state_machine` import standalone and none of them reach back into `search.situation`). `logger` moved below the imports where it belongs.
- `situation.py` — two stale docstrings corrected. Both said `_override_enemy_intent` "synthesises a MoveState", which is exactly what the implementation deliberately refuses to do (a synthetic has no `follow_up_id` and would crash `roll_move`). They now describe the install-onto-the-real-MoveState path and the skip-on-unknown-move-id fallback that the code actually implements. `_override_enemy_intent` also now bails on `ai is None` / missing `move_id` before building the `Intent` rather than after.
- `live_search.py` — dropped the unused `CombatState` / `get_action_mask` imports and the `try/except` around `from_bridge_state` whose only body was `raise`. The bare `return 0` for the END_TURN fallback now uses the imported `ACTION_END_TURN` constant (which is `0`, verified) so a future action-space renumber cannot silently turn the fallback into "play card 0".

**Test state at commit:** 369 pass across the bridge/search/gym-env selection, 4832 pass suite-wide. The pre-existing parity failures are **149**, not the 97/123 quoted in the PR #1 and overnight-summary changelogs above — that count has grown with later content work and those earlier numbers should be read as historical. Verified pre-existing by stashing `sts2_env/` + `tests/` and re-running: identical 149 failures, 4825 passing. So this PR is +7 passing tests and zero regressions.

## PR #10 changelog — closing three open questions, and the harvest that wasn't blocked

Landed on `glm52`. Picks up where the overnight session stopped. The theme: three of the items the overnight summary handed to the user turned out **not to need the user's machine at all**, and one open risk turned out to be a non-issue.

### 10.1 — PR #9's seed question, closed (`tests/test_rng_parity.py`)

PR #6 flagged this and PR #9 inherited it: the C# mod serialises `encounter_seed` as `(long)seed`, so a seed above int32's range arrives in the JSON negative, and if `Rng.__init__` masked that to a different 32-bit pattern than the game used, the monster HP rolls would diverge silently. The searcher would then plan against a fight that is not the one on screen — the worst class of bug, because every individual piece looks correct.

**It does not diverge.** `Rng.__init__` does `_to_uint32(seed)` then `_to_int32(...)`, which round-trips any int32 bit pattern whatever sign it arrived with, and masks a wider value the way C# int arithmetic wraps. Verified across `-1`, `-5`, `-123456789` and `int.MinValue`, plus a genuine long. Two tests pin it. No production change — this closes a question, it does not fix a bug.

### 10.2 — Phase 0.1, finally (`sts2_env/bridge/raw_capture.py`)

Phase 0.1 asked for a recorded protocol sample *before* any C# was touched. It was never done, and the cost is on the record: every parser on the bridge path was written against a guessed payload shape, and two needed a live session plus a bug-fix round to correct — PR #8's UPPER_SNAKE potion ids, PR #9's drift. Neither was reachable from the unit tests, **because the unit tests were built from the same guess as the code**. A recorded payload is what breaks that circle: it is the mod's answer rather than ours.

`--capture-raw PATH` on both `agent_runner` and `live_eval` writes whole states verbatim to JSONL. Quota'd per message type (default 25) so one short session yields every *kind* of screen rather than ten thousand `combat_action`s from the longest fight; the trailer carries seen-vs-kept counts so a truncated type does not read as a rare one. Captured on the first line after `receive_state`, before any phase dispatch can filter a screen out; flushed per state and closed in a `finally`, so a Ctrl-C or a lost connection still leaves a usable file. Never raises — a diagnostic that can end a 20-run session is worse than no diagnostic.

**This is the highest-value thing to run first when the mod compiles**, ahead of the 20-run session:

```bash
.venv/bin/python -m sts2_env.bridge.live_eval \
    --model-path output/combat_v3_overnight/final_model.zip \
    --capture-raw output/bridge_protocol_sample.jsonl \
    --runs 1 --verbose
```

One run is enough. The resulting file makes `from_bridge_state` and `to_combat_mid_fight` checkable offline against real payloads, with no game running — which is how the PR #8 and PR #9 classes of bug get found before a session is spent on them, rather than after.

### 10.3 — Phase 3.2 harvest: never actually blocked

The overnight summary listed the 2000-situation harvest as needing the user's machine. It does not — `harvest_combat_benchmark.py` walks the simulator and touches neither the game nor the decompile gate. Run here: 1240 runs, all four floor bands filled to their 500 quota, **2000 situations** — 1575 MONSTER / 260 ELITE / 165 BOSS across 22 distinct encounters, deck size mean 13.6 (max 21), HP fraction mean 0.69 (min 0.16). 1797 of 2000 hold a deck that is not the 10-card starter, which is the entire point.

Written to `tests/fixtures/act1_combat_train_2000.json`, **not** the script's default path — the default is `act1_combat_benchmark.json`, the held-out 200 that `score_combat_benchmark` reads and that every number in `MODELS.md` is measured against. Harvesting over it would have destroyed the ability to compare this model to the last one. Harvested at seed 1234 against the eval set's seed; verified zero leakage, on full identity (encounter + both seeds + deck + HP) and on the weaker encounter+seed pair.

### 10.4 — the eval set was the training set (`scripts/train_combat.py`)

Found while setting up Phase 3.3. PR #4 passed `--situation-set` to *every* env, eval included. `best_model.zip` is selected on the eval number, so a run would have picked whichever checkpoint memorised the training fights hardest and reported it as skill — and the Phase 3.3 pass bar (≥80% overall / ≥30% boss) would have been measured against the same situations the gradient steps came from.

New `--eval-situation-set` flag. Phase 3.3 is now train-on-2000 / eval-on-the-held-out-200, which are verified-disjoint, so the eval number is comparable to the `combat_v3_overnight` figures in `MODELS.md` measured on that same 200. Omitting the flag keeps the old behaviour rather than erroring, but now says out loud that eval is drawing from the training pool.

### 10.5 — Phase 4.2 needs no change

The roadmap lists "replace `HeuristicCombatSolver` default with `FrozenRLCombatSolver`" as Phase 4.2 work. It is already wired: `train_meta_policy.py:86-93` selects `FrozenRLCombatSolver(args.combat_policy)` whenever `--combat-policy` is passed, which PR #5's own suggested command already does. The library-level default stays `HeuristicCombatSolver` deliberately — it is the sensible fallback for a caller with no model, and hardcoding a checkpoint path into the env would be worse. **Nothing to do; the flag is the mechanism.**

### 10.6 — Phase 3.3 is not an overnight run

The decompile gate **passes on this machine** (`check_decompile_matches_installed()` → `True`, build `9f6869d1f4e27dc4`), so the trainer does not refuse. Bare training throughput fine-tuning from `combat_v3_overnight` on the 2000-situation pool is ~1600 fps; with the held-out eval callback firing every 10k steps the observed end-to-end rate puts 2M steps at roughly **1–1.5 hours** on 8 envs, not the overnight the plan assumed. The estimate in Phase 3 should be read down accordingly — it means the Phase 3.3 → 3.4 loop is a same-sitting iteration rather than a next-day one, and that eval frequency, not gradient steps, is the thing to trade if it needs to be faster.

### Cleanup applied to PR #9's code before commit

Hoisted the mid-file `PowerId` import and three function-local imports in `situation.py` to the module header (verified no circular-import risk); moved `logger` below the imports. Corrected two docstrings that described `_override_enemy_intent` as "synthesising a MoveState" — exactly what the implementation deliberately refuses to do, since a synthetic has no `follow_up_id` and would crash `roll_move`. Dropped `live_search.py`'s unused imports and a `try/except` whose only body was `raise`; its END_TURN fallback now uses the `ACTION_END_TURN` constant rather than a bare `0`.

### What still genuinely needs the user's machine

> **Superseded — all three of these were done on 2026-08-06.** The mod compiles
> (the block was a `PATH` export in `~/.bashrc` while the login shell is fish),
> the protocol was captured, and the Phase 2.4 session ran and passed. See
> "PHASE 2.4 — PASSED" below. Kept as written because the reasoning about what
> was blocked, and how wrong it turned out to be, is the useful part.

1. ~~**Compile PR #6**~~ — done; 0 errors.
2. ~~**Capture the protocol**~~ — done; it found a live bug on its first run.
3. ~~**Phase 2.4 live session**~~ — done and passed. **3.4 is still outstanding.**

## PR #11 — Phase 3.3 ran, and the answer is no

The retrain from 10.6 completed 1.19M of its 2M steps before the run was killed, and was scored against `combat_v3_overnight` on the held-out 200. Full numbers in `docs/MODELS.md`; the summary is that **Phase 3.3's hypothesis did not survive its own measurement.**

```
                  v3_overnight   real_situations
win rate             74.0%          72.5%     -1.5% +/- 1.5%   inside the noise
BOSS                  6.7%           6.7%     (15 fights -- 1 fight either way)
ELITE                42.3%          46.2%     (26 fights, +/-10%)
```

The boss row — the row the retrain existed for — did not move. And it is not a question of the missing 810k steps: the held-out eval reward is flat across the entire 1.19M, first quarter +0.397 against last quarter +0.428, oscillating around +0.45 from the first evaluation onward. The curve never climbed. A fine-tune from an already-converged model began at its ceiling and stayed.

### What this changes in the plan

The "What 'never blocks, never plays Powers' actually is" section above argues the failure is a distribution problem, and concludes the model "literally couldn't learn it". The first half of that is well-evidenced — the starter deck really does contain 0 Powers, and the model really was never shown a situation where one was right. **The second half is now disconfirmed.** Shown 2000 real situations for 1.19M steps, it still did not learn it. Being unable to learn from the old distribution did not imply being able to learn from the better one.

Set that against the turn-search rows in `MODELS.md`: 83% overall and 20–26.7% boss on the *same* 200 fights, against this model's 72.5% / 6.7%. Lookahead moves bosses; combat training on a better distribution does not. That is the Phase 2.4 decision point — "if boss win rate is still 0%, the deckbuilding phase is the gating failure" — arrived at from the training side without needing the live session to tell us.

**So Phase 3 is answered and should not be re-run bigger.** The remaining Act 1 gap is in search (Phase 2) and deckbuilding (Phase 4), not in the combat policy's training data. Phase 3.2's fixture keeps its value regardless: it is the situation set the *search* should be evaluated on, and `--situation-set` remains the right plumbing.

### The pass bar was not measurable

Phase 3.3 set "≥80% overall / ≥30% boss" as its gate. The benchmark holds 15 boss fights. A proportion near 30% on n=15 carries a standard error of roughly 12 points, so the bar and the observed 6.7% sit about one standard error apart — **no run could have cleared that gate on this fixture**, including a run that genuinely deserved to.

This is worth fixing before any further boss claims are made, and it is cheap: the harvester already produces boss fights (the 2000-set contains 165), it just needs a boss-weighted held-out draw at a fresh seed. Until that exists, treat every boss percentage in this document — including the 20–26.7% attributed to turn search — as one or two fights of resolution.

### Operational note

The 2M run was killed at 1.19M steps because a backgrounded shell does not outlive the agent process that started it. Long runs want `setsid nohup … &` so they survive; the benchmark that produced the numbers above was launched that way and did.

## PR #12 — the mod compiles, and always could have

Every "needs the user's machine" note above rests on one claim, first written on 2026-05-22 in `docs/PARITY_GAPS.md:249` and repeated at `:352`, `:422`, `:429` and `:559` of this document: that the C# cannot be built here. **It is wrong, and it was wrong when it was written.**

.NET 9.0.316 has been installed at `~/.dotnet` the whole time. The `PATH` export was added to `~/.bashrc`; the login shell is zsh, which never reads that file. So `dotnet` was "not found" in exactly the way an installed toolchain is not found when it is configured for a shell you do not use. Fixed by moving the export to `~/.zshrc`, verified with `zsh -lic 'dotnet --version'`.

**PR #6's C# patch compiles: 0 errors.** The 125 warnings are all nullable-reference analysis and all pre-existing. The build deploys `STS2BridgeMod.dll` straight into the game's `mods/` directory, and the deployed binary carries the Phase 1.1 fields — `encounter`, `encounter_seed`, `combat_seed` and `upgraded` are present as UTF-16 literals in the DLL, which is the check worth doing because plain `strings` reads ASCII and finds none of them.

One warning is worth understanding rather than ignoring: `GodotPath is not configured; skipping .pck export. The existing .pck will be reused.` That is benign here. The `.pck` carries Godot resources; the handler code lives in the DLL, which did rebuild. It would matter for a change touching scenes or assets, and PR #6 touches neither.

### What this means for the plan

The item at the top of every "deferred to the user" list for the last two sessions is done. The only genuinely remaining external dependency is **the game being launched** — `live_eval` connects to the mod over port 9002, and nothing on this side can start Steam.

So the sequence, once STS2 is running:

```bash
# 1. Capture the protocol first -- one run, before anything else.
.venv/bin/python -m sts2_env.bridge.live_eval \
    --model-path output/combat_v3_overnight/final_model.zip \
    --capture-raw output/bridge_protocol_sample.jsonl \
    --runs 1 --verbose
```

This is the step that has been skipped twice and cost two bug-fix rounds. The capture makes `from_bridge_state` and `to_combat_mid_fight` checkable offline against real payloads — and given PR #6 has now changed the wire format, the odds that the first live session hits a shape mismatch are higher than usual, not lower.

Only then the 20-run session with `--live-search`.

### A note on the estimate

Three items this session were listed as blocked on hardware and were not: the Phase 3.2 harvest, the Phase 3.3 training, and now the mod build. In each case the block was a configuration detail or an unchecked assumption rather than a real constraint. The lesson is cheap to state and was expensive to learn twice: **verify the blocker before planning around it.**

## PR #13 — working the phases in order, 1 through 4

The instruction was to stop jumping ahead and go 1, 2, 3, 4, 5. Doing that immediately found that **Phase 1 was not finished** — its exit criterion is "from a live `combat_start` message, the simulator can build the same fight", and the first capture proved it could not. Everything below follows from taking that criterion literally.

### Phase 1 — closed, after two real bugs

**1a. The generator was not the game's.** `sts2_env/core/rng.py` ran a `System.Random` clone seeded from the low 32 bits, with the game's *deprecated* string hash. The game's `Rng(ulong)` wraps `MegaRandom` — xoshiro256\*\* seeded by four Splitmix64 draws from the full 64-bit seed — and hashes stream names with XxHash64. Three things wrong at once, so all three changed at once. XxHash64 is checked against published reference vectors, so that half is anchored outside this repo rather than against itself.

**1b. Enemy HP was never reconstructable from a seed.** This is the part worth remembering. `CombatState.cs:499` rolls monster HP from `RunState.Rng.Niche` — a *run-level* stream whose position depends on every draw earlier in the run — not from the encounter's RNG. No amount of generator fidelity recovers it. It also excludes HP values already taken by siblings on the same side, which is why the two Corpse Slugs in the capture have different HP.

So `CombatSituation` now carries `enemy_max_hp`, read from the bridge and applied in `to_combat`. Which is the "if the game says 67, you have 67" rule, applied one level below where it had already been applied. `to_combat_mid_fight` had been quietly covering for this on the live path — the search worked while `to_combat` was wrong for every fight, which is exactly how a bug like this survives.

**Result: 18/18 comparable captured states match, from 0/25.** The seven skipped are mid-fight states where an enemy had died, so the game has dropped it while `to_combat` rebuilds the opening roster; comparing those says nothing and the checker now says so instead of scoring them as failures.

### Why this was invisible for months

Every seeded test in the suite compared the simulator against itself. `test_rng_parity.py` asserted thirteen stream seeds and a five-value sequence, all of them generated by running the implementation under test. They passed. They were wrong. The file now asserts published hash vectors, the derivation rules transcribed from the decompile, and structural properties — nothing whose only source is this code.

Ten behavioural tests broke on the new stream. None was a real regression; each is fixed by asserting the invariant its own name claims rather than the draw that used to satisfy it. Two were fragile for unrelated reasons and are better now: the elite-relic test assumed base potion odds of 0.0 suppress potions when `PotionRewardOdds` adds `ELITE_BONUS` to the *return* check, and the shop test failed as a bare `StopIteration` when rolled prices exceeded starting gold. Where an exact sequence was never game-verified and cannot be re-derived (Fabricator spawn order), the mechanism is asserted and the sequence dropped rather than re-baked into a fresh unverifiable constant.

**149 pre-existing failures unchanged, zero new, throughout.**

### Phase 2 — 2.3 done, 2.4 is yours

`SearchAgent(playout_policy=...)` takes a callable `(combat, mask) -> action | None`, falling back to the block-then-damage heuristic per step when it declines, raises, or names a masked action. `model_playout_policy(model)` wraps a MaskablePPO. Default `None`, so every recorded number still describes the default build.

This is the seam for testing the `DEFAULT_TOP_K` diagnosis — that the playout, not the depth, is what limits the rollouts — not a claim that it works. Measuring it needs a paired benchmark run.

**2.4 cannot be run from here.** It needs the game listening on 9002, and nothing on this side can launch Steam.

### Phase 3 — re-baselined, and the pass bar made measurable

The RNG change means every fixture seed rolls different enemy HP, so all pre-2026-08-06 numbers describe different fights. `combat_v3_overnight` re-measured: **71.0% overall, 23.1% elite, 6.7% boss**. Full table and the caveat in `docs/MODELS.md`. The elite row halving (42.3% → 23.1%, ~2 se on 26 fights) is probably the missing `SetUniqueMonsterHpValue` rule, which matters most in multi-monster fights — a real remaining parity gap, recorded rather than fixed.

`harvest_combat_benchmark.py --room-types BOSS,ELITE` added, and a boss/elite-weighted held-out fixture harvested. The default fixture holds 15 boss fights, which puts roughly a 12-point standard error on any boss rate near 30% — the Phase 3.3 gate was never measurable on it, by anyone, including a run that deserved to pass.

### Phase 4 — running

`train_meta_policy.py --combat-policy combat_v3_overnight` launched, 5M steps, `FrozenRLCombatSolver`, on GPU. Note the roadmap's own suggested command was wrong: `--act-count` is not a flag the script has.

`v1..v7` had a flat eval curve because of the reward leak PR #5 fixed. **The first non-flat eval is the Phase 5 gate**, and nothing before it justifies starting Phase 5.

### Phase 5 — untouched, as instructed

## PR #14 — the loaded die, and Phase 4's answer

### The RNG is now in line with the game, and here is exactly what that means

Four things were wrong. Three are fixed; the fourth cannot be fixed and did not need to be.

1. **The generator.** A `System.Random` clone where the game runs xoshiro256\*\* seeded by Splitmix64. Fixed.
2. **The seed width.** Masked to 32 bits, discarding half of every 64-bit live seed. Fixed.
3. **The stream-name hash.** The game's *deprecated* 32-bit hash instead of XxHash64. Fixed, and anchored to published reference vectors.
4. **Monster HP uniqueness.** `SetUniqueMonsterHpValue` picks from the HP range *minus the totals siblings already hold*, so the game cannot deal two enemies on a side the same HP. This simulator rolled each monster independently and collided in **11.4% of multi-enemy fights** — measured over 600 harvested Act 1 situations. Now 0%.

### The distinction that matters for "training on a loaded die"

Exact stream parity for monster HP is **unreachable, permanently**. The game rolls it from `RunState.Rng.Niche`, a run-level stream whose position depends on every draw earlier in the run. No generator work recovers a specific fight's HP from a seed.

That is fine, because **training needs the same distribution, not the same sequence.** An agent learns nothing from which particular seed produced a 43-HP Nibbit; it learns from facing Nibbits whose HP is distributed as the game distributes it. The loaded die was real and was items 1–4 above — a wrong generator and a missing uniqueness rule genuinely skew what the agent trains against. Those are closed.

Where a *specific* fight must be reproduced — a live bridge state, a harvested fixture — the answer is not derivation but recording. Both paths now store the enemies' actual HP and replay it. `to_combat()` reads it instead of re-rolling. Which is the same rule as everywhere else: if the game says 43, it is 43.

That also makes fixtures stable across future generator work. They were not: correcting the generator silently changed every enemy in both fixtures on 2026-08-06. The two on disk predate the field and still fall back to rolling, so **re-harvesting is worth doing before the next measured comparison** — they are internally consistent but no longer faithful to the runs they came from.

### Two things the fallout taught

**Decimillipede segments have their own uniqueness rule** — step HP by 2 until no teammate matches — and a random re-draw lands between its rungs. Exempted.

**Uniqueness is enforced across the whole enemy list, not per species.** `CombatState.cs:496` passes `_enemies`. A Zapbot holding 24 really does stop a Stabbot being 24. Two ascension tests asserted an exact fixed roll a sibling had already taken; they assert the range now, which is what they were actually about.

### Phase 4 — the gate is open

`meta_ppo_v8_rewarded`, 27 evaluations through 540k steps:

```
min +0.770   max +7.890   -> NON-FLAT
```

`v1..v7` were flat across every evaluation, which is what the reward leak PR #5 fixed produced. **This is the first non-flat meta-policy eval curve in the project**, and it is the condition Phase 5 was gated on. The run is still going; the curve is noisy and not yet obviously climbing, so "the signal exists" is the claim, not "the meta-policy is good".

Phase 5 is now unblocked. It is still not started, and should not start on 27 noisy evals — let the run finish and look at the shape.

## PHASE 2.4 — PASSED. Live search works. 2026-08-06

**The reference point for this project.** Raw data in `docs/milestones/2026-08-06-live-search-works/`.

Twenty live runs, all twenty finished — the first session that neither stalled nor crashed.

```
                            baseline      --live-search
mean floor                     9.1            14.7
median floor                     8              17
reached the act 1 boss     2/20  10%      10/20  50%     +40 pts +/-13.0  (~3 se)
cleared act 1              0/20   0%       2/20  10%     +/-6.7 -- two runs
boss win rate               0/2            2/10  20%     +/-12.6
deepest run                    --          floor 33 (act 3)
```

**Exit criterion met:** the live agent uses real lookahead in combat, and it is worth 40 points of boss-reach rate at roughly 3 standard errors.

**Decision point resolved — continue.** The plan said: *"if boss win rate moves from 0% to ≥10% across 20 runs, continue. If still 0%, the deckbuilding phase is the gating failure."* It moved 0% → 20%, and act 1 clears 0% → 10%. Deckbuilding is **not** the sole gate; search alone bought most of the run.

### The result that changes how to work

`MODELS.md` puts turn search v2 at **20% boss win rate** on the offline benchmark. Live: **2/10 = 20%**.

The offline benchmark predicts live performance. Every future change should be screened against the 200-fight fixture *before* spending an hour of live runs — this project has repeatedly discovered in a live session what a benchmark run would have caught in minutes. One corroboration at n=10, so treat it as a working assumption rather than proven, but act on it.

### Where the remaining Act 1 gap actually is

8 of 20 died **to the act 1 boss**; 6 died in hallways. Arriving is close to solved at 50%. Beating it is not, at 20% of arrivals.

That reframes the rest of the plan. The gap is one fight, not the whole run, and it is measurable offline — which is exactly what the new boss/elite-weighted fixture (`act1_boss_elite_benchmark.json`, 27 boss / 123 elite) was built for.

### What it cost to get here

Three stalls across three sessions, one mistake each time, and it is the same mistake: **the game was sending the truth and this side was computing its own instead.**

1. A local sim kept across calls, drifting into a frozen fiction (PR #9).
2. Enemy HP re-derived from a seed that cannot produce it — plus phantom dead enemies and mismatched enemy indices.
3. Playability re-derived while the game was marking every card `playable: false` under RINGING.

The durable fix is the fourth one: the stuck detector now escalates to end-turn before abandoning, so the *next* unmodelled rule costs a turn rather than a session.

### Against the goal

Target is 50% Act 1 over 20 consecutive runs. This is **10% ± 6.7%** — two runs, which establishes "not zero" and nothing more.

## Phase 5 — REDESIGNED: retrieval, not a generative router (design note, 2026-08-06)

The plan above specifies a **Qwen-7B-Instruct-Q4 generative router** with constrained decoding, prompted with the options and asked to name one. That is not what we are building. This section supersedes 5.1 and 5.2; they are kept above as the reasoning that got us here.

### What changed

A 7B is roughly ten times larger than needed, and there is an STS2-specific asset already built: [`t22000t/slay-the-spire-2-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings) — **576 cards** (essentially the whole pool; the simulator knows 577), **1024-dim**, unit-normalised so a dot product *is* cosine similarity, encoded from prettified-JSON mechanics fields with card self-references replaced by `~`, via **Qwen3-Embedding-0.6B**.

Three consequences, all favourable:

- **The vectors are precomputed.** No model runs at decision time — load a matrix, take a dot product. Microseconds, so the live-path latency objection to a 7B disappears.
- **Constrained decoding becomes unnecessary.** Section 5.2 exists only to stop a generative model naming an option that isn't on offer. Retrieval scores *only the offered cards*, so an invalid answer is structurally impossible. That subsystem is deleted, not deferred.
- **It targets what the heuristic cannot do.** `card_quality.py` scores cards in isolation — "26 damage beats 8" — with no concept of the deck having a plan. Cosine against an archetype vector is exactly "does this belong in what I am building".

### What the reference implementation actually does, and how ours must differ

[`t22000t/slaythespire-build-me-a-deck`](https://huggingface.co/spaces/t22000t/slaythespire-build-me-a-deck) embeds the **user's prompt** ("build me a strike deck"), scores every card by cosine to it, and picks greedily under mana-curve and type-mix constraints. Its own description: *"greedy similarity-based selection with constraint-feasibility checks; not an optimization solver"*. The 72B LLM it loads is invoked **only after** the deck is built, to write prose about boss matchups — it plays no part in selection.

That demonstrably produces coherent, synergistic decks. Perfected Strike lands near a "strike deck" query because both texts talk about Strikes. So *embeddings do capture synergy*, via textual co-occurrence.

Three things it gets for free that we do not:

1. **Intent comes from a human.** The query vector is the user's sentence. The agent has nobody to ask.
2. **It picks from the whole pool.** We pick 1 of 3 the game offers — far more constrained, far less forgiving.
3. **It has no notion of power level.** Pure theme-matching takes a weak on-theme card over a strong off-theme one.

The narrow limitation that survives: similarity says Perfected Strike is *about* Strikes; it does not say it hits for 6 more because you hold eight. That is a number, and **the simulator can compute it exactly** — a signal neither an embedding nor an LLM has.

### The design

**Where intent comes from — two stages.**

- **Picks 1–3: no archetype yet.** The Ironclad starter is 5 Strike, 4 Defend, Bash. A centroid over the whole deck is 10 parts starter to 1 part signal and would read "attack deck" for every run. So the centroid is computed over **non-starter cards only**, and not trusted until there are ~3 of them.
- **Picks 4+: archetype fit.** Centroid of the non-starter cards becomes the query; offered cards are scored by cosine to it.

**Early picks favour deck-*defining* cards, not merely strong ones.** Feast is a great card that commits to nothing; Barricade defines a deck. Computable from the same vectors: cosine each card against every archetype label and look at the *shape* — Barricade spikes on one and is near-zero elsewhere, Feast is moderately similar to everything. `max(sims) − mean(sims)` is that in one number. High = defining, low = generically fine.

So: **picks 1–3 maximise `quality × peakedness`; picks 4+ maximise `quality × fit-to-centroid`.** Same ingredients, reweighted by stage.

**Archetype labels are kept for legibility.** A dozen plain-text archetype descriptions, embedded once offline. The deck's nearest label is logged — "floor 7, deck reads as: exhaust" — so a strange pick can be diagnosed against what the agent thought it was building. Costs one dot product; the scoring itself can use the raw centroid.

**`card_quality.py` is not replaced.** It stops being a placeholder and becomes the permanent *quality* term, with embeddings supplying the *direction* it never had. That also means there is no fallback problem: if the embedding half fails, behaviour degrades to today's.

### HARD CONSTRAINT: no deck thinning

**Cyra cannot remove cards, so no archetype may depend on removing Strikes or Defends.**

This is not merely unimplemented — it is currently *harmful*. `remove_card` sits 4th in `SHOP_PURCHASE_ACTION_PRIORITY`, so it fires whenever relics, cards and potions are unaffordable. The card it removes comes from `_pick_card_select_indexes`, which sorts **basics last** — correct for the *upgrade* screens it was written for, exactly inverted for removal. The agent would keep its Strikes and remove its Bludgeon. **Separate defect, worth fixing independently of Phase 5.**

The constraint is a useful filter rather than only a limitation. It selects archetypes where the starter cards are *assets*:

- **Strike-synergy** (Perfected Strike) — actively wants the 5 Strikes. Best possible fit.
- **Block scaling** (Barricade) — Defends are on-theme; deck size matters less when not racing.
- **Power/scaling** (Inflame, Demon Form) — survive, then deploy; deck size is a mild tax.

And it rules out exhaust loops, precise draw combos, and anything needing to see a specific card every turn. Those read as viable to a naive similarity score and then quietly lose runs. **The archetype vocabulary must be short and screened for fat-deck viability**, not "every archetype in the game".

### Validation before building (next step)

Cheap, offline, no live time:

1. Pull the embeddings; check coverage against the simulator's card ids.
2. Define the short fat-deck-viable archetype list.
3. Classify a sample of the 2000 harvested real decks by nearest label and read the results. If floor-14 decks come back as plausible archetypes, the signal is real; if everything reads "attack deck", it is not.
4. Check peakedness actually separates Barricade from Feast.

If 3 or 4 fail, the approach is wrong and we have spent an afternoon rather than two weeks.
