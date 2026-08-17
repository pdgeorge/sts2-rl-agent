# PHASE TWO — a plan to take act 1 from 14% to 50%

This is a gameplan, not a proposal and not a rebuttal. It exists because the last
several weeks have been spent changing things and then being unable to tell
whether they helped, and that is a fixable problem rather than a hard one.

A separate document will cover the road to act 3 at 10%. Everything here is
chosen so that it still pays off there — nothing in this plan is act 1 tuning
that has to be thrown away later.

---

## 1. Where we actually are

Pooled over **every live journal, 514 runs**:

| | rate |
|---|---|
| reach the act 1 boss | 47.1% +/- 2.2 |
| win it, given reached | 28.9% +/- 2.9 |
| **clear act 1** | **13.6% +/- 1.5** |

Recent era only (last 6 journals, 123 runs): reach 54%, boss win 27%,
clear 14.6% +/- 3.2. Not distinguishable from the pooled figure.

Offline, same agent, 400 paired seeds: reach 64%, boss win 74%, clear 48%.

**Read those two together.** Reach agrees within about 10 points. Boss win is
**45 points apart**. Offline is a usable instrument for the first half of the
funnel and a broken one for the second, and no plan should spend offline runs
answering a boss question until that is explained.

### What 50% actually requires

`clear = reach x win`, so:

| reach | boss win needed for 50% |
|---|---|
| 55% | 91% |
| 65% | 77% |
| 75% | 67% |
| 80% | 62% |

At today's 54% reach, no achievable boss win rate gets there. **Both halves have
to move.** Any plan that targets only one is arithmetically incapable of the
goal, and that is worth checking every proposal against.

---

## 2. What we have learned, stated as evidence

### 2.1 The pattern that separates wins from nulls

Everything that moved the number removed something the agent was
**structurally unable to do**:

| fix | what was impossible |
|---|---|
| card id resolution | 68 of 600 cards could not be played at all |
| card reward skip | `can_skip:false` always; every card taken, forever |
| `?` room RoomType | KeyError -> whole fight handed to the weak model |
| act 1 variant | 57% of real boss fights were against unmodelled bosses |

Everything that came back null adjusted a number that was already roughly right:

| change | paired result |
|---|---|
| eval weights | null; raising `enemy_hp` actively hurt (-2.6 se) |
| archetype card picking | +0.0% +/- 3.3%, 104/120 seeds identical |
| waterfall giant siphon + potion rules | +1.3% +/- 1.2% (399 seeds) |
| incoming-damage modifiers | +0.8% +/- 1.0% (400 seeds) |
| search truncation | measured at 1.6% of searches; not a factor |

**Five consecutive nulls from tuning. Four wins from removing impossibilities.**

This is the single most useful thing we know, and it should drive where effort
goes. Hunt for options the agent never had. Those are invisible in the win rate
AND invisible in the tests, because nothing asserts "she could have done this
and did not".

### 2.2 What is already known to work

- **Search over a trained policy.** `MODELS.md`: boss win 6.7% -> ~20%. Three
  independent full-run RL attempts plateaued at mean floor 9.5-9.7, and all are
  now unloadable because the observation layout moved. Search is the agent.
- **Parity against the decompile.** Live clear went ~3% -> ~25% as disparities
  went 157 -> 2. The only intervention class with a demonstrated effect.

### 2.3 What we cannot currently measure

Live throughput is **2.2 min/run**. To detect an improvement from 15%:

| improvement | live runs needed | game time |
|---|---|---|
| +5 points | 2188 | 80 h |
| +10 points | 590 | 22 h |
| +20 points | 166 | 6 h |

So a 40-run live session resolves roughly a **25-point** change and nothing
finer. Every conclusion drawn from a 30-run session this month was unsupported,
including several of mine. Live is a **bug detector**, not a measuring device.

Offline is 400 paired runs in ~3 h and resolves +/-1.2%. That is the measuring
device — for reach. Not yet for the boss.

---

## 3. Infrastructure to build first

Four are borrowed from the external proposal, which is better than us on
engineering discipline. Four are things I would insist on from scratch. None of
them make the agent smarter; all of them make the next twenty experiments
cheaper and harder to get wrong.

### 3.1 Weights in versioned config, not code constants

Today `QUALITY_BAR_SCALE`, `SKIP_THRESHOLD`, `EvalWeights`,
`ROOM_MIN_HP_FRACTION` are module constants, and every sweep monkey-patches
globals inside worker processes. That is how a sweep shipped with its baseline
arm doing the exact opposite of its name for 400 runs.

A policy is a config file with a version. Sweeps select configs. Nothing patches
a global.

### 3.2 Policy version stamped on every run

We cannot currently tell which code produced which journal; I have inferred it
from timestamps against git log and got it wrong. Every run record carries
`policy_version` and `git_sha`.

### 3.3 Log every option's score, not just the choice

We log what she picked. We need what she considered, with scores. This is not
bookkeeping: **the 68 unplayable cards would have been obvious immediately**,
because they would be conspicuously absent from every option list they should
have appeared in.

### 3.4 Holdout seeds

We tune and evaluate on the same seeds. Split them, report both, and treat a
result that only appears on the tuning half as noise.

### 3.5 Identifier audit in CI

`scripts/audit_bridge_ids.py` exists and found the 68-card bug. It must run
automatically, not when someone suspects something.

**Note this is NOT the same as version diffing.** Flame Barrier was never a
version change: the game always sent `FLAME_BARRIER`, we always spelled it
`FLAME_BARRIER_CARD`. A diff between game builds reports "no change", correctly,
while 68 cards stay unplayable. The audit asks a different question — does every
name the game sends resolve in our model, right now.

### 3.6 Capture the state that caused a failure

Done for `LiveSearch.decide`; extend to every decision path. `decide raised`
told us nothing for weeks. The state that caused it made the cause obvious in
one run.

### 3.7 Derive constants at construction, never copy them

`derived_values.py` already does this for cards. Monster and relic constants are
still hand-copied, and that is where the wrong Siphon, nine wrong HP values and
thirteen wrong damages came from. Extend it.

### 3.8 One function per decision, never a copy

Now true for all seven non-combat decisions. Keep it true — the test in
`tests/test_bridge_offline_alignment.py` fails if a second implementation
appears.

**Cost:** ~2 days. **Payoff:** every experiment after it is faster and harder to
misread.

---

## 4. The plan to 50%

### Track A — close the offline/live boss gap (blocking)

45 points apart on the fight that decides the run. Until this is explained,
offline cannot judge any boss-facing change, which is most of what is left.

The instrument exists: `scripts/replay_live_boss_fights.py` rebuilds the boss
positions the live agent actually faced and plays them offline with the live
search budget. Same deck, same HP, same relics, same intents.

- search wins them -> the fault is in the live path (reconstruction, bridge),
  and the boss model is fine
- search loses them -> offline's optimism is upstream, in ARRIVING at the boss
  in better shape than live does

At n=2 it won both fights live lost. That is a lead, not a result. The capture
now keys per fight, so one session yields ~30 fights and the answer.

**Gate:** the gap is explained, or offline is formally restricted to reach-rate
questions.

**Status: partially answered, and the answer is not the one this section
expected.** `scripts/deck_or_play.py` and the relic capture together give:

| | relics at boss | HP at boss | boss win |
|---|---|---|---|
| offline | 2.8 | 91% | 74% |
| live | 4.7 | 81% | 29% |

Live and offline decks are equivalent relic-for-relic (27% vs 26% at a
controlled 80% HP, both carrying exactly 9.0 basics). Offline wins carrying
FEWER relics, so relic acquisition is not the gap. At the measured exchange rate
of ~1 win point per 1% of max HP and ~4.5 per relic, HP and relics together
account for ~19 of the 45 points. **~26 points remain unexplained**, and no
current hypothesis covers them. The gate stays shut.

**2026-08-16, second reading: the gate's instrument exists, and the gap now has
a name.** The sunday session captured 13 boss fights whole (the quota is now
keyed per fight AND boss buckets keep 100 states, because 25 cut both Waterfall
Giant captures at round five). Replaying the live arrival positions: offline
5/13 = 38% +/- 13% against live 2/13. Arrival therefore explains ~36 of the
45-point funnel gap (74% fresh-run offline vs 38% at live arrivals), and ~23
points stayed. Those 23 were not a mystery: the mid-fight reconstruction
installed every bridge power as a bare ``PowerInstance`` with no hooks, so the
search clone let Waterfall Giant die quietly -- the STEAM_ERUPTION detonation
lives in ``after_death`` -- and scored every kill a clean win. Run 7 arrived at
80/80 and died killing the giant at 37 HP against a ~37 bank. That is the
execution side of the gap, fixed in ``situation._bridge_power_instance`` and
pinned by four tests.

**What the replay still cannot see: the deal.** The bridge sends
``draw_pile_count``, never the pile order, and the game's shuffle stream is
run-level, so every replay reshuffles. One replay scores one arbitrary deal and
cannot separate play from luck; ``scripts/replay_boss_seed_sweep.py`` answers
the honest question instead -- win rate across many reshuffles of the same
arrival. Full fidelity needs the draw pile order on the wire; that is a mod
change and the next instrument work.

**2026-08-16, third reading: the sweep closed the gate.** Each of the 13
captured arrivals replayed across 20 reshuffles, same search budget as live:
offline pooled **50% +/- 3%** against live **15% +/- 10%** on the identical
positions. The spread names the shape: Waterfall at 80/80 with deck 22 wins
**100%**, including the reshuffle k=0, which IS the live game's own
``combat_seed`` -- same deck, same draw order, offline wins the fight live
lost, so that one is pure execution, and the dropped power hooks are the named
mechanism. Waterfall at 80/80 with deck 18 loses **0%** in every world: some
arrivals are simply dead, and no policy change rescues them. Kin at 53/80:
90% offline, lost live. Beast at 34/80: 85% offline, lost live. So: the
arrival term is real (deck 18 vs deck 22), but the majority of the gap between
the funnel's 74% and live's 15% is execution the reconstruction used to drop.
G1 is open for the next live session's purpose: confirm prediction 9, and get
the draw pile order onto the wire so replays stop gambling on deals.

### Track B — reach rate, 54% -> 75%+

Offline agrees with live here, so this can be tuned offline with confidence.

1. **HP gate re-fit.** `ROOM_MIN_HP_FRACTION` was fitted on 116 live elite
   fights, then nine monster HP constants and thirteen damages were corrected
   underneath it. It has never been validly re-swept — both attempts were
   instrumentation failures.
2. **Where runs die.** Of 21 pre-boss deaths in the last session, 12 were in
   monster rooms and 8 in elites (Bygone Effigy 4, Phantasmal Gardener 2).
   Elites are a minority of rooms and a plurality of deaths.
3. **Scored on act 2 reach as well**, so a routing change that buys act 1 by
   starving act 2 cannot pass.
4. **Routing has no lookahead at all, and that is an impossibility rather than a
   tuning gap.** `_pick_map_node` ranks the immediately reachable nodes by room
   type and takes the best one. It cannot plan a route, cannot set up an elite
   followed by a rest site, and diverts to a recovery room whenever *any*
   visible node is unaffordable — so it zigzags rather than committing to a
   path. The game's own beginner guide opens with the opposite advice: plan
   backward from the boss, and favour routes with intersections. By §2.1's
   pattern this is the highest-yield category we have, and it was missing from
   this plan until the gate work surfaced it.
5. **The elite gate keys on HP; the guide keys on deck strength** — "fight
   elites when your deck is strong enough, they drop game-changing relics".
   There is no deck-strength signal in the routing decision at all. Worth
   knowing before re-fitting the gate, because it may be that the gate is the
   wrong variable rather than the wrong number.

**Status: B.1 is done and it is a NULL — the sixth consecutive tuning null.**
`scripts/ab_elite_gate.py`, 3 arms x 150 paired seeds:

| gate | elites | 0-elite | reach | win\|reach | clear |
|---|---|---|---|---|---|
| 0.80 (ships) | 1.12 | 32% | 54% | 72% | 39% +/- 4.6 |
| 0.60 | 1.25 | 29% | 51% | 78% | 40% +/- 4.6 |
| 0.45 | 1.34 | 27% | 50% | 79% | 40% +/- 4.6 |

Paired: 0.60 gains 8 and loses 5 (McNemar p=0.58); 0.45 gains 9 and loses 6
(p=0.61). Nothing.

**The elites column is the finding.** Halving the gate bought 0.22 elites per
run and moved zero-elite runs 32% -> 27%. **The HP gate was never what
suppressed elite count** — the greedy path simply does not pass elites. So the
live correlation behind this experiment (0/1/2+ elites -> 7%/9%/23% clear) was
essentially all reverse causation, exactly as flagged before running it.

This promotes B.4. Routing is now the only remaining explanation for 39% of live
runs fighting no elites, and it is the impossibility-class item.

Two methodological notes. Offline takes 1.12-1.34 elites against live's 1.28,
which is good 1:1 evidence for the routing code. And **the holdout half is
useless at this size** — 37 runs gave 62% clear against tuning's 39%. `seed % 4
== 3` does NOT correlate with the act 1 variant (49.6% vs 50.8% underdocks over
4000 seeds), so it is small-n variance, but future sweeps need more seeds or a
larger holdout share for §3.4 to do its job.

### Track C — deck quality (running now)

The measured cause of boss losses: act 1 boss decks are 21-22 cards, **43%
basic Strike/Defend**, against 173-222 HP bosses. She enters at 81% HP, survives
6-14 turns, and cannot kill the boss.

Two of the three causes are fixed (the mod can now click Skip; the policy can
now decline). The third — what quality bar — is the sweep in flight, using pd's
rule: `100 * score / scale > deck_size`, so the bar rises with the deck and size
falls out of quality rather than being capped.

**Open risk, stated in advance:** a stricter bar does not remove the nine basics
already in the deck, so a smaller deck is a MORE basic deck — 9 of 15 is 60%
against 9 of 21 at 43%. If stricter loses, the answer is removal and upgrades,
not declining, and Track C becomes that instead.

**Status: closed, negative — and the stated open risk is what happened.** The
quality-bar sweep came back at -5.7%, and `deck_or_play.py` then showed live and
offline decks winning identically relic-for-relic. **Deck composition is not the
act 1 problem**, so "are the decks shit" is answered: no, or at least no more
than offline's, which clear 48%.

The upgrade half of the fallback is also now measured, and also negative for act
1. `scripts/upgrades_vs_hp.py` (2160 fights, n=180/cell, no errors) gives ~5 win
points per upgrade against ~1 per 1% of max HP; a rest site heals 30% of max HP,
so heal (~30 points) beats smith (~5) everywhere except within ~5% of full.
**Loosening the smith gate would lose act 1 runs.** Upgrades still compound
across acts 2–3 in a way this act-1 grid cannot see, so this closes Track C for
M1 only, not for the act 3 plan.

**The removal half is POSITIVE, and is the one thing here that shipped.**
`scripts/removal_vs_relic.py`, 2880 fights over 30 real live boss decks:

| basics removed | deck | all relics | one fewer relic |
|---|---|---|---|
| 0 (live) | 20.2 | 40% +/- 3 | 39% +/- 3 |
| -1 | 19.2 | 44% +/- 3 | 42% +/- 3 |
| -2 | 18.2 | 46% +/- 3 | 44% +/- 3 |
| -3 | 17.2 | 50% +/- 3 | 48% +/- 3 |

A removal is worth ~3.3 points and a marginal relic ~2. Monotonic across all
four levels, 2.7 sigma end to end. It is **linear, not compounding** — which was
named in advance as the thing that would have made removal a priority rather
than a nicety, and it did not happen.

Removal also costs 75 gold against a shop relic's 150-300, so it is roughly 4x
the win rate per gold. `remove_card` moved to the front of
`SHOP_PURCHASE_ACTION_PRIORITY` and is no longer restricted to curses; it now
takes curses first, then surplus basics, Strike before Defend, unupgraded first,
never below 2 copies. `tests/test_shop_card_removal.py` pins that exact order,
because the measurement only transfers if the shipped policy is the policy that
was priced.

**Honest magnitude: ~+4 boss-win points, ~+2 clear points.** A tool that makes
good work better, not a route to 50%. It is a lower bound, though — the grid
prices only the boss fight, and a thinner deck also helps every fight on the way
there, which lands in reach and is uncounted.

### Track D — keep hunting impossibilities

The highest-yield activity we have, and it needs a standing method rather than
luck. Sources that have each produced a real bug: the disparity log, the
stuck-state dumps, the search-failure captures, the identifier audit, and pd
watching a run and saying "why did she not play that".

---

## 5. Gates

Each gate is set where our instruments can actually resolve it.

| gate | criterion | instrument |
|---|---|---|
| G0 | infrastructure in place | config-driven policies, versions in logs, audit in CI |
| G1 | boss gap explained | replay ~30 captured live boss fights |
| G2 | reach >= 70% | 400 paired offline seeds, act 2 reach not regressed |
| G3 | deck quality decided | quality-bar sweep, act 2 reach not regressed |
| G4 | offline clear >= 55% | 400 paired seeds, holdout half agrees |
| G5 | live confirmation | 100+ live runs with restart-on-crash |

**G5 is deliberately 100+, not 20.** At 40 runs a 50% claim has +/-8 error;
at 100 it is +/-5. A 20-run live check can only catch catastrophe, which is a
real job but not this one.

---

## 6. What this plan refuses to do

- **Train a full-run policy.** Three attempts, floor 9.5-9.7, all now unloadable.
- **Tune weights as the primary engine.** Five consecutive nulls.
- **Draw conclusions from 30-run live sessions.** They cannot resolve anything
  smaller than 25 points.
- **Report a favourable subset as the rate.** Pooled, with n and error bars, or
  it does not go in a report.

---

## 7. Act 3 is a separate document

Everything above is chosen to still be worth having at act 3, but the act 3 plan
is a different problem and needs its own analysis:

- act 2 (Hive) matches the game; **act 3 (Glory) does not** — the simulator
  rolls `doormaker_boss`, which exists nowhere in the decompile, and cannot roll
  `AeonglassBoss`, which the game has. Same shape as the act 1 variant bug that
  left 57% of act 1 boss fights unmodelled. That must be fixed before any act 3
  number is trusted.
- act 2/3 content has had almost none of the parity attention act 1 received,
  and the week's pattern says unaudited content hides exactly this class of bug.
- rare-event measurement: a 10% target needs far more runs per decision than a
  50% one, and the offline funnel is the only affordable way to get them.
