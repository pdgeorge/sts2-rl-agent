# SCOREBOARD

One number decides whether work counted: **pooled live clear rate, with n and a 95% confidence interval.**

Not offline. Not "explained a gap". Not "shipped a change". Those can all be true while this number sits still, which is exactly what happened in the week of 2026-08-10.

## The number

| date | n | reach boss | win \| reach | **clear** | what changed |
|---|---|---|---|---|---|
| 2026-08-13 | 496 | 48.0% +/- 4.4 | 26.5% +/- 5.6 | **12.7% +/- 2.9** | baseline, every live journal to date |
| 2026-08-14 | 89 | 58.4% +/-10.2 | 25.0% +/-11.8 | **14.6% +/- 7.3** | the subset that actually ran the SEARCH |
| 2026-08-14 | 410 | 45.4% +/- 4.8 | 26.9% +/- 6.4 | **12.2% +/- 3.2** | the subset that ran the v3 TRAINED MODEL |
| 2026-08-15 | 100 | 66.0% +/- 9.3 | 30.3% +/-11.1 | **20.0% +/- 7.8** | `--live-search` on by default, first clean n=100 (1001 search fights, 0 model) |

**Prediction 4 is a HIT on reach** (45.4% -> 66.0%, predicted 58.4%) and clear roughly doubled. 20.0% +/- 7.8 overlaps the 25.2% +/- 7.9 search-era estimate, so they are the same rate; the 48% session (n=25) is resolved as noise and is not to be built on.

**Why this session came in at 20 and not 25: mean HP on boss arrival fell 82% -> 79%.** Boss win below 80% is 3% (1/31); at or above it is 54% (19/35). Nothing else moved.

## Target

**25% clear by Sunday 2026-08-16.**

`clear = reach x win`, so 25% needs roughly **62% reach x 40% boss win**. Neither half alone reaches it: routing at 62% reach with today's 26.5% win is 16%; today's 48% reach with a 40% win is 19%. Both have to move.

## The rule

Every change gets a prediction written **before it is built**, in the table below. After measurement the row is marked HIT or MISS, and a MISS stays in the table.

This exists because the failure mode is mine and it is documented: quoting a favourable subset as the rate, and calling a null a win after the fact. A prediction written afterwards is not a prediction.

| # | change | predicted effect | measured | verdict |
|---|---|---|---|---|
| 1 | shop card removal, curses -> worst-card | +2 clear points | pending | — |
| 2 | reconstruction audit fixes | unknown until the audit runs; predict before fixing | audit clean at the boss (0/60 wrong); no fix needed | **CLOSED, no change** |
| 4 | `--live-search` on by default | reach 45.4% -> 58.4% (the rate search already achieves on the 89 runs that used it) | pending | — |
| 3 | map routing with lookahead (`map_planning.py`) | reach 48% -> 62% live; offline reach 54% -> 65% | offline reach 60.7% -> 56.0%; paired net -2, p=0.89; best of 5 variants +0, p=1.00 | **MISS** |
| 5 | demote `elite` below `monster` in `ROOM_PRIORITY_HEALTHY` | elites/run 0.87 -> ~0.58 (only 29 of 87 elite picks had a non-elite alternative); ~6 fewer run-ending deaths per 100; reach 66% -> 72%; clear 20% -> 25% | pending | — |
| 6 | route planner re-targeted at ONE objective: fewest elites on the path to the boss | the other 58 of 87 elites were forced by the greedy adjacent-node view. elites/run -> ~0.3, mean HP at boss 79% -> 88%, win\|reach 30% -> 48%, clear -> 35% | pending | — |

### The evidence behind 5 and 6

Act 1, pre-boss, n=100 live:

| room | entries | ended the run | mean chip |
|---|---|---|---|
| elite | 82 | **19 (23%)** | 31.8 HP |
| monster | 705 | 14 (2.0%) | 6.3 HP |

An elite room is 11x more likely to end the run than a monster room and costs five times the HP. 19 of the 34 pre-boss deaths were elites; among survivors elites are 22 of the ~55 HP missing at the boss, which is most of the distance to the 80% cliff.

Winners and losers fight the SAME rooms -- 7.75 vs 7.61 monsters, 0.75 vs 0.89 elites. The only difference is damage taken: 49 vs 69 total chip. Not deck, not relics, not route length.

Note 6 is NOT a repeat of 3. Prediction 3 sorted routes by a generic survivability score and was measured offline, where the boss gap is 43 points. 6 has a single objective that the live data names, and only live measurement counts.

## Measurement rules

- **Live only.** Offline is a bug detector and a cheap filter; it has never predicted live on boss questions, and the gap is 43 points.
- **Pooled**, never a favourable session. A 20-run session resolves nothing finer than ~25 points.
- **n >= 100** for any claim, and n ~290 to resolve 25% at +/- 5.
- Report the interval every time. A number without one is not a result.

## What does NOT count as progress

- an offline improvement that has not been confirmed live
- a hypothesis eliminated (useful, but it is not the number)
- a change shipped whose effect has not been measured
- the best session out of several
