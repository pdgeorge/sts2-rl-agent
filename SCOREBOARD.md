# SCOREBOARD

One number decides whether work counted: **pooled live clear rate, with n and a 95% confidence interval.**

Not offline. Not "explained a gap". Not "shipped a change". Those can all be true while this number sits still, which is exactly what happened in the week of 2026-08-10.

## The number

| date | n | reach boss | win \| reach | **clear** | what changed |
|---|---|---|---|---|---|
| 2026-08-13 | 496 | 48.0% +/- 4.4 | 26.5% +/- 5.6 | **12.7% +/- 2.9** | baseline, every live journal to date |
| 2026-08-14 | 89 | 58.4% +/-10.2 | 25.0% +/-11.8 | **14.6% +/- 7.3** | the subset that actually ran the SEARCH |
| 2026-08-14 | 410 | 45.4% +/- 4.8 | 26.9% +/- 6.4 | **12.2% +/- 3.2** | the subset that ran the v3 TRAINED MODEL |

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
