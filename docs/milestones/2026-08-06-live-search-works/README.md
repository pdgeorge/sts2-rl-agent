# 2026-08-06 — live search works, in the real game

The first 20-run live session that finished all 20 runs, and the first evidence that turn search transfers from the benchmark to the live game.

Reference point. Everything here is the raw data, not a summary of it — if a later change claims an improvement, this is what it has to beat.

## The numbers

Both sessions: `combat_v3_overnight/final_model.zip`, 20 live runs, Ironclad, Act 1.

| | Phase 0.3 baseline (model only) | Phase 2.4 (`--live-search`) |
|---|---|---|
| mean floor | 9.1 | **14.7** |
| median floor | 8 | **17** |
| reached the Act 1 boss | 2/20 — 10% | **10/20 — 50%** |
| cleared Act 1 | 0/20 — 0% | **2/20 — 10%** |
| boss win rate (of runs that arrived) | 0/2 | **2/10 — 20% ±12.6** |
| deepest run | — | floor 33 (Act 3) |

**Reached-boss: +40 points ± 13.0, about 3 standard errors.** That one is real. The median run went from dying on floor 8 to standing in front of the boss.

**Act 1 clear: 10% ± 6.7%.** That is two runs. It establishes "not zero"; it does not establish a rate. Do not quote it as one.

## The finding worth more than the headline

`docs/MODELS.md` records turn search v2 at **20% boss win rate** on the offline 200-fight benchmark. The live boss win rate came in at **2/10 = 20%**.

The offline benchmark predicted live performance. That means a change can be pre-screened against 200 fixture fights in minutes instead of an hour of live runs, and the screening can be trusted. Given how many live sessions this project has spent discovering things a benchmark run would have caught, this is the more useful result.

Treat it as one corroboration, not a law — n=10 boss fights live, ±12.6.

## Where the wall moved to

Deaths by floor band, live-search session:

```
3x  Act 1 boss — WATERFALL_GIANT
2x  Act 1 boss — VANTOM
1x  Act 1 boss — KIN_FOLLOWER
1x  Act 1 boss — LAGAVULIN_MATRIARCH
1x  Act 1 boss — (unnamed)
2x  floor 13   — FOGMOG
2x  floor 7/14 — SEAPUNK
1x  floor 9    — WRIGGLER
1x  floor 11   — (unnamed)
1x  past Act 1
```

**8 of 20 died to the Act 1 boss.** Six died in hallways. The agent now reliably arrives and reliably loses — a far better problem than the baseline's "dies on floor 8", and it says the remaining Act 1 gap is concentrated in one fight rather than spread across the run.

## What had to be fixed to get a session to finish at all

Three stalls, three sessions, one mistake each time: **the game was sending the truth and this side was computing its own instead.**

1. **The drifting local sim** (PR #9) — a sim kept across calls diverged from the live game within a few turns, and the search planned against a frozen fiction. Fixed by rebuilding from the bridge every step.
2. **Re-derived enemy HP** — monster HP comes from the run-level `Niche` stream and cannot be reproduced from an encounter seed. Fixed by recording what the game reported. Also surfaced a roster bug: dead enemies stayed alive as phantom targets, and the sim's enemy indices did not match the game's compacted list.
3. **Re-derived playability** — the player held `RINGING` (one card a turn), had spent it, and the game marked every card `playable: false`. The sim offered plays anyway. Fixed by treating the bridge's flag as final.

Plus the stuck detector now escalates to end-turn before abandoning a session, rather than throwing away 20 runs over one turn the game would not allow. That one is the general fix — it catches the *next* unmodelled rule, whatever it turns out to be.

## Files here

| file | what it is |
|---|---|
| `live_eval.jsonl` | the 20 run records of the live-search session |
| `baseline_live_eval.jsonl` | the Phase 0.3 baseline, model only, for comparison |
| `live_journal.jsonl.gz` | every room, fight, card played and reward taken across the 20 runs |
| `bridge_protocol.jsonl.gz` | raw states the mod sent, verbatim — the protocol reference |

`output/` is gitignored, so these are copies. They are the point of the directory: the session itself is not reproducible, and without them this page would be a claim rather than a record.

## Reproducing the session

```bash
.venv/bin/python -m sts2_env.bridge.live_eval \
    --model-path output/combat_v3_overnight/final_model.zip \
    --live-search \
    --log output/live_eval_next.jsonl \
    --journal output/live_journal_next.jsonl \
    --capture-raw output/bridge_protocol_next.jsonl \
    --capture-raw-per-type 60 \
    --runs 20 --verbose
```

Needs the game running with the bridge mod listening on 9002. Roughly 0.7 h for 20 runs at 2.2 min/run.

## Against the goal

The target is 50% Act 1 across 20 consecutive runs. This is **10% ± 6.7%**.

Reaching the boss is close to solved: 50%, up from 10%. Beating it is not: 20% of arrivals. Both remaining levers — a better boss policy, or a deck that arrives stronger — are now measurable offline first, which is what the benchmark result above buys.
