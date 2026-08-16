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
| 5 | demote `elite` below `monster` in `ROOM_PRIORITY_HEALTHY` | reach 66% -> 72%, clear 20% -> 25% | never measured -- **withdrawn and reverted before it ran** | **WITHDRAWN, wrong direction** |
| 6 | playout policy priced in HP (`_heuristic_playout_action`) | power-play rate stops being flat in fight length and rises with it; damage per elite fight falls; live clear 20% -> 28% | offline paired A/B, 150 situations: wins 134 -> 134, damage/fight 13.03 -> 12.69, power rate 0.82/1.21/1.10/1.71% -> 1.30/1.62/1.01/1.23% by fight length. Power-holding decks only (n=73): wins 54 -> 54, damage/fight 19.79 -> 19.23, power rate 7.25/4.28/4.11/5.41% -> 10.96/4.66/4.39/4.62% | **MISS on the behavioural gate** |

### Why 6 missed, and what the diagnosis got wrong

The claim was that the live power-play rate is flat in fight length (2.32% at 1-2 turns against 2.25% at 8+) BECAUSE the playout scoring the lookahead priced every Power at a flat 0.5 and so never played one. Rewriting the playout to price everything in HP did not move the rate, in either arm, and did not change a single win.

The measurement that should have come first: **live decks hold about 1.4 Powers** (138 taken across the 100 runs; a Power is on offer at 30.2% of card rewards and taken 43.0% of the time it is). In a ~20-card deck over a 5-turn fight the agent draws roughly 1.3 Powers and plays about 0.5. The flat 2-3% rate is mostly **deck composition, not play** -- there are barely any Powers in the deck to play. A rate that low cannot rise with fight length no matter how the playout scores.

The offline benchmark is worse than useless for this question in its default form: **110 of the first 150 decks hold no Power at all**. `--powers-only` exists now so this cannot be diluted to nothing again.

Damage per fight fell ~3% in BOTH arms, which is the sort of thing that gets written up as a small win. Paired sign test over the 73 fights: **mean -0.99 HP, better 16, worse 12, tied 45, z=0.76, p=0.450 -- not resolvable.** 45 of 73 fights come out identical. It is noise, and it would have been the sixth false positive of the same shape as the five in `WEEKEND_DECISIONS.md` section 5.

What survives: the playout had a genuine unit error (block and damage compared as one quantity, so a 6-damage Strike beat a 5-block Defend) and no term separating a kill from a chip. Fixing those is defensible on its own and measured null. **Null is not a reason to take it live.**

Note for anyone rerunning `ab_playout_policy.py`: the searcher is time-budgeted, so it is not bit-deterministic and the per-bucket percentages move a little between runs. That is another reason to read the paired test rather than the summary rows.

### 5 was wrong, and the reason is worth keeping

The reasoning was: elite rooms end 23% of the runs that enter them against 2.0% for monsters, so take fewer. pd rejected it and the [map guide](https://sts2.untapped.gg/en/guides/how-to-make-the-best-map-choices-in-slay-the-spire-2) says why -- **target 2-3 elites in act 1**, and "avoiding elites purely to preserve HP is a trap" because relics are what carries a run past act 1. We already take **0.87 elites a run**. We are under-fighting them, not over-fighting them.

The 23% is real. It is a statement about how badly the agent PLAYS elites, not about whether to enter them, and reading it the other way would have optimised act 1 by giving up acts 2 and 3. Prediction 6 is the same evidence pointed at the fight instead of the map.

Correction to the numbers used in 5: `damage_taken` in the journal is computed as `hp_before - hp_after`, and `hp_after` is captured AFTER Burning Blood's +6 refund, so every chip figure in this repo dated before 2026-08-16 is net of that heal. Gross, act 1 pre-boss: elite **37.9** damage over 5.2 turns (7.2/turn), monster **11.6** over 3.1 (3.8/turn). The elite is 1.9x worse per turn, not the 3x first reported.

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
