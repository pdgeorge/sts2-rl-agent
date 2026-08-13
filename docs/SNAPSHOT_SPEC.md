# Spec: in-memory combat snapshot/restore for sts2-cli

A work order for a fork of [`wuhao21/sts2-cli`](https://github.com/wuhao21/sts2-cli). Self-contained: everything needed to do the job and to prove it was done is here.

## 1. What is being asked for

Add two operations to the headless simulator: **take an in-memory snapshot of the current combat**, and **restore the combat to a previously taken snapshot**. Nothing else about the project changes.

## 2. Why, in one paragraph

The consumer is a turn-search agent. Slay the Spire telegraphs enemy intents, so the agent does not predict a turn — it plays every legal ordering of its playable cards on a copy, ends the turn, lets the enemies act, evaluates what is left, and keeps the best line. That requires branching: from position P play card A, evaluate, **discard that future**, return to P, play card B, and so on. Today the agent does this against a hand-written Python reimplementation of the game, which drifts from the real game and is the suspected cause of a large unexplained discrepancy in results. If the real engine can branch, the reimplementation can be deleted and the drift stops existing. Snapshot/restore is the one primitive that makes that possible.

## 3. Prerequisite fixes (already found — apply these first or nothing builds)

The upstream repo does not build or run against the current game build. Four fixes, all verified working:

1. `src/Sts2Headless/RunSimulator.cs` — `CreatureCmd.Damage` gained a trailing `CardPlay?` parameter. In `NeutralizeSafe`, pass `play` as the final argument.
2. `src/Sts2Headless/RunSimulator.cs` — `RunManager.SetUpSavedSinglePlayer` is now `SetUpSavedSingleplayer` (lowercase `p`).
3. `setup.sh` — add `Sentry.Godot.dll` to the `DLLS` list. Without it, `start_run` dies in a `<Module>` type initializer.
4. `src/Sts2Headless/RunSimulator.cs`, in `EnsureModelDbInitialized` — `ModManager.State` stays `None` headless, so `ReflectionHelper.ModTypes` throws as soon as `RunState.CreateShared` reaches `ModelDb.BadgeModels`. Set it by reflection to `ModManagerState.Skipped`, which the game documents as "We're in test mode".

Also recommended: `Program.cs` catch-all reports only `ex.GetType().Name: ex.Message`, which for a `TypeInitializationException` names nothing useful. Walk `InnerException` and print `ex.ToString()` to stderr. Debugging this task without that is painful.

Known unfixed bug, out of scope but do not be confused by it: roughly 45% of random-agent runs get stuck on the Neow event, re-presenting the identical choice forever. Suspected cause is options requiring a follow-up screen (card removal, curse grants). Use seeds that get past Neow, or drive the agent to pick a plain relic option.

## 4. The API

Three new commands on the existing stdin/stdout JSON protocol, alongside `start_run`, `action`, `get_map` and the rest.

```jsonc
// Capture the current combat. Valid only during combat.
{"cmd": "snapshot"}
→ {"type": "snapshot", "id": "snap_1"}

// Return the combat to a captured state. The snapshot remains valid for reuse.
{"cmd": "restore", "id": "snap_1"}
→ {"type": "ok"}

// Free a snapshot. Snapshots are also freed when combat ends.
{"cmd": "drop_snapshot", "id": "snap_1"}
→ {"type": "ok"}
```

Errors use the existing shape: `{"type": "error", "message": "..."}`. Restoring an unknown or freed id is an error, not a silent no-op. Calling `snapshot` outside combat is an error.

A snapshot must be **restorable more than once**. The search restores the same position tens to hundreds of times in a row; a one-shot snapshot is useless.

## 5. What must be captured

Everything that can affect how the rest of the fight plays out:

- player and enemy HP, block, and all powers/buffs/debuffs with their stack counts
- hand, draw pile, discard pile, exhaust pile — contents **and order**
- energy, turn number, and whose turn it is
- enemy intents, including any already-rolled next intent
- per-combat counters and any accumulated per-turn state (cards played this turn, attacks this turn, and similar)
- **the combat RNG state**

The RNG is the one most likely to be missed and the one whose absence is hardest to notice. If it is not captured, restoring and replaying the same action draws a different card, and the search silently evaluates positions that cannot occur. Test 4 exists to catch exactly this.

Relics, potions and deck are run-level and outside the combat, but if the engine mutates them mid-combat (a relic that counts triggers, a potion consumed) that mutation must be captured too.

## 6. Performance budget

Measured on real act-1 boss positions with the shipping agent, 858 searches:

| | nodes per search |
|---|---|
| median | 44 |
| mean | 98 |
| p90 | 252 |
| max | 1167 |

One node is one `restore` plus a few `action` calls. A live decision has about 3 seconds. Measured cost of an `action` during combat today is **1.2 ms median, 5.2 ms p90**, so the budget for snapshot/restore is what is left.

**Target: 500 restore-plus-action cycles in under 1 second, single process, warm.** That comfortably covers p90 and leaves headroom for the worst case. If restore lands above about 5 ms this approach does not work and that is a valid, useful answer — say so rather than shipping something that technically functions.

Report the achieved number. "It works" without a timing is an incomplete result.

## 7. Acceptance tests

These are the deliverable as much as the code is. Each must be an automated test in `tests/`, and each must be able to fail — a test that passes against an unmodified build is not testing anything.

**Test 1 — restore actually rewinds.** In combat, snapshot. Record full state. Play a card. Assert state changed. Restore. Assert state is **field-for-field identical** to the recording, including pile contents and order.

**Test 2 — the original is not mutated by exploring.** Snapshot. Play a long line of several cards, ending the turn if possible. Restore. Play a *different* line. Assert the second line's result is exactly what it would have been had the first line never been played — compare against a control run that plays only the second line from a fresh identical position.

**Test 3 — repeated restore.** Snapshot once, then restore and play the same single card 100 times. Assert all 100 outcomes are identical to each other. This is the one that catches state leaking across restores, which is the most likely subtle failure.

**Test 4 — RNG is captured.** Snapshot at a point where a draw is pending. Restore and draw, twice. Assert the same card arrives both times. Then, from the same snapshot, play a line involving a shuffle or a random effect twice and assert both produce identical results.

**Test 5 — independence of two snapshots.** Take snapshot A, advance, take snapshot B, restore A, advance differently, restore B. Assert B restores to what was captured, unaffected by anything done after it.

**Test 6 — performance.** The 500-cycle benchmark from §6, asserting the wall-clock target and printing the achieved per-restore time.

For "field-for-field identical", serialize the combat to a canonical string or hash and compare that. Comparing a handful of fields by eye will pass while the powers list quietly diverges.

## 8. Implementation notes and known landmines

- `CombatState` is **not** `Serializable` and there is no existing `Clone`/`DeepCopy`/`MemberwiseClone` on it. There is no route to lean on; this is a build.
- The game serializes *runs*, not mid-combat positions, so `load_save` / `write_continue_save` cannot be reused for this.
- `CombatManager.Instance` is a **singleton**. A clone must not mutate what the live combat reads, and restore must put the singleton's view back consistently.
- Card effects execute **asynchronously through an action queue**. The repo already IL-patches `CombatManager.WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction` to `Task.CompletedTask` for this reason. Snapshot must be taken with the queue quiesced, or a half-executed effect will be captured.
- `CombatStateTracker` and the combat history entries are part of the state. They may be safe to drop for search purposes, but if they are dropped, **say so explicitly** — silently omitting state is how Test 3 passes today and something breaks in three weeks.
- Approaches worth considering, in rough order of preference: a hand-written deep copy of the combat object graph; a generic reflection-based deep clone with a cycle-tracking identity map; or a copy-on-write / command-journal undo. The first is fastest and most work, the second is quickest to write and may well hit the performance target, the third avoids copying entirely but is easy to get subtly wrong. Any of them is acceptable if the tests pass and the budget is met.

## 9. Non-goals

- Do not fix the Neow bug (separate work).
- Do not add snapshot/restore for run-level state — combat only.
- Do not change the existing command surface or output format.
- Do not optimise anything else.
- Do not add a rendering or UI path.

## 10. What to report back

1. Whether the acceptance tests pass, individually.
2. The achieved per-restore timing and the 500-cycle number.
3. Which approach from §8 was taken and why.
4. **Anything deliberately not captured**, and the reasoning.
5. An honest statement if the performance target was missed. A negative result here is genuinely valuable — it decides an architecture — and is much more useful than a passing test suite hiding a 40 ms restore.
