# Where this project is — 2026-08-07

A standing snapshot, meant to be rewritten rather than appended to. For how we
got here read `docs/GLM_ROADMAP_50P_ACT1.md`; for what is broken read
`docs/KNOWN_ISSUES.md`; for what has been trained and measured read
`docs/MODELS.md`.

## The one number

**10% ± 6.7% Act 1 clear rate, live, over 20 runs.** Target is 50%.

That is two runs out of twenty. It establishes "not zero" and nothing more —
do not quote it as a rate.

Reaching the boss is close to solved: **50%**, up from 10% before live search.
Beating it is not: **20%** of arrivals. The remaining Act 1 gap is one fight.

Tagged `live-search-works`; the session's raw data is preserved under
`docs/milestones/2026-08-06-live-search-works/`.

## What actually plays the game

The best live result used **no neural network for any decision**:

| Decision | Made by |
|---|---|
| Combat | `SearchAgent` — exhaustive turn search, no learning |
| Card rewards | `card_quality.py` — hand-written floor |
| Map, shop, rest, event, treasure | heuristics in `agent_runner.py` |
| Trained combat PPO | loaded, consulted **zero** times in 239 combats |

The learned components sat on the bench. That is the honest summary of where
the strength currently comes from.

## Phase status

| Phase | State |
|---|---|
| 0 — instrumentation | Done. Raw protocol capture exists and earns its keep. |
| 1 — bridge protocol | Done, properly. Closing it required fixing the RNG and reading enemy HP from the game rather than re-deriving it. |
| 2 — live search | **Done, passed.** +40 points of boss-reach at ~3 se. |
| 3 — retrain combat | 3.1–3.3 done. **3.3 is a null result** — training on real situations did not help. 3.4 as written is moot. |
| 4 — meta-policy | Plumbing fixed and proven. `v8` trained 5M steps, moved one floor, 0% win rate, **discarded**. |
| 5 — deckbuilding | Designed, validated in principle, not built. |

## The simulator now matches the game's RNG

It did not, for months. It ran a `System.Random` clone from a 32-bit-truncated
seed with the game's *deprecated* string hash; the game runs xoshiro256\*\*
seeded by Splitmix64 from a full 64-bit seed, hashing stream names with
XxHash64. All three are fixed.

Two things follow that are easy to forget:

- **Monster HP cannot be reproduced from a seed.** The game rolls it from the
  run-level `Niche` stream, whose position depends on the whole run. It is now
  *recorded* by both the bridge and harvest paths, never re-derived.
- **Every number measured before 2026-08-06 describes different fights.**
  `MODELS.md` carries the re-baseline.

The reason this survived so long: every seeded test compared the simulator
against itself. The RNG tests now assert published hash vectors and rules
transcribed from the decompile.

## What is known-broken

Six items in `docs/KNOWN_ISSUES.md`. The two that gate further training:

1. **Multi-select selection state is absent from the run observation.** The
   policy cannot see what it has selected, so it toggles one card forever. 2.6%
   of episodes, **~73% of eval compute**, and it strikes at floor 16 — the good
   runs. Reproduce with seed 9004.
2. **Stalling is nearly free.** `COMBAT_TURN_COST × 200 turns` exactly cancels
   `COMBAT_WON`. Needs to be genuinely negative — a 20-turn stall should score
   worse than taking damage. This is a watchability requirement as much as a
   scoring one.

And one that is simply wrong: **`PHASE_BOSS_RELIC` is a Slay the Spire 1
mechanic that STS2 does not have.** No `BossRelic` type exists in the decompile;
what follows a boss is an Ancient, which is an `EventModel`. The simulator
models the Ancients correctly *and also* awards StS1 boss relics from a
fabricated pool.

## What needs a human

Only the live game. The mod compiles (the blocker was a `PATH` export in
`~/.bashrc` while the login shell is fish), the protocol is captured, the bridge
connects. Nothing on this side can launch Steam.

## Test state

**4882 passing, 149 failing.** The 149 are pre-existing content-parity failures
in card and monster reference suites — `test_all_cards_unit_coverage` (26),
`test_ironclad_combat_edge_card_model_parity` (14), Defect/Regent/Silent card
effects. None are in the bridge, search, RNG or gym-env work. They predate all
of it and are their own project.
