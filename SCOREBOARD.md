# SCOREBOARD

One number decides whether work counted: **pooled live clear rate, with n and a 95% confidence interval.**

Not offline. Not "explained a gap". Not "shipped a change". Those can all be true while this number sits still, which is exactly what happened in the week of 2026-08-10.

## The number

| date | n | reach boss | win \| reach | **clear** | what changed |
|---|---|---|---|---|---|
| 2026-08-13 | 496 | 48.0% +/- 4.4 | 26.5% +/- 5.6 | **12.7% +/- 2.9** | baseline, every live journal to date |

## Target

**25% clear by Sunday 2026-08-16.**

`clear = reach x win`, so 25% needs roughly **62% reach x 40% boss win**. Neither half alone reaches it: routing at 62% reach with today's 26.5% win is 16%; today's 48% reach with a 40% win is 19%. Both have to move.

## The rule

Every change gets a prediction written **before it is built**, in the table below. After measurement the row is marked HIT or MISS, and a MISS stays in the table.

This exists because the failure mode is mine and it is documented: quoting a favourable subset as the rate, and calling a null a win after the fact. A prediction written afterwards is not a prediction.

| # | change | predicted effect | measured | verdict |
|---|---|---|---|---|
| 1 | shop card removal, curses -> worst-card | +2 clear points | pending | — |
| 2 | reconstruction audit fixes | unknown until the audit runs; predict before fixing | pending | — |
| 3 | map routing with lookahead | reach 48% -> 62% | pending | — |

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
