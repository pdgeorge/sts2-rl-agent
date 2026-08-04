# Plan: from floor 11.6 to a win

Written 2026-08-04, from 142 recorded live runs. Supersedes nothing — `PLAN.md` holds the infrastructure phases and `PLAN_DECKBUILDING.md` the drafting ones. This is the document that says **which of them to do, in what order, and why**, and it is written to be falsifiable rather than encouraging.

---

## Where we actually are

142 live runs across every recorded session:

```
mean floor 11.6    median 11.5    max 31    wins 0

  0-4   ###                                              3
  5-9   #########################################################   57
 10-14  #########################################        41
 15-19  ######################################           38
 20-24  ##                                               2
 30-34  #                                                1
```

Three numbers matter more than the mean:

```
reached the act 1 boss (floor 17+)    34 / 142    24%
got PAST it (floor 18+)                3 / 142     2%
reached floor 34+                      0 / 142     0%
```

**The act 1 boss kills 91% of the runs that reach it.** Nothing else in the data is close to that as a wall. The distribution is not a smooth decay — it is a pile-up at 15-19 that does not continue.

### What a win costs, arithmetically

A win is 3 acts × ~15 rooms + 3 bosses ≈ 48 floors. Treating survival as uniform per floor (it is not — see below — but it bounds the problem):

```
                       per-floor survival    implied mean floor
today                        0.921                  11.6
P(win) =  5%                 0.940                  16
P(win) = 20%                 0.967                  29
P(win) = 50%                 0.986                  69
```

The encouraging reading: 5% win rate needs per-floor survival to go from 0.921 to 0.940. That is not a different project, it is a 2-point improvement compounded 48 times.

The honest correction: survival is **not** uniform. Bosses are ~10x deadlier than hallways in our data, so the real model is a handful of hard gates with easy stretches between. That means the average is the wrong target and **the gates are the whole game**. Improving hallway survival moves the mean and does not move P(win); getting past the act 1 boss moves both.

---

## The bottleneck chain, in the order it must be fixed

Each link invalidates measurement of everything below it. That is the entire argument for the ordering — it is not preference.

### 0. The simulator is wrong about the cards that matter — 22 failing Ironclad parity tests

```
BREAK  CINDER  COLOSSUS  CRIMSON_MANTLE  DEMON_FORM  DOMINATE  DRUM_OF_BATTLE
FIGHT_ME  HEMOKINESIS  HOWL_FROM_BEYOND  MANGLE  SETUP_STRIKE  TREMBLE  TAUNT
UNRELENTING
```

(149 tests fail in total; 81 are for characters an Ironclad MVP does not play and are out of scope, 46 are shared/generic and need triage.)

**These are almost exactly the cards the pilot cannot see** (all 20 Powers plus the scaling skills). The overlap is not a coincidence — a card whose effect is complex enough to be mis-simulated is a card whose value is complex enough to be invisible to a heuristic pilot. Fixing the pilot first would make it play these cards *using wrong rules*, and every measurement would look better while being less true.

**Gate.** Ironclad parity failures at 0 — resolved in the direction the decompile says, which is not necessarily the code's. Not the other 81.

**Cost: smaller than it looks, and the direction of the fix is the surprise.** Spot-checked before committing to this phase. Every failure is a small value disagreement:

```
assert 4 == 5    assert 3 == 2    assert 4 == 3    assert 80 == 82
assert 7 == 8    assert 70 == 75  assert 85 == 86  assert 74 == 80
```

`DEMON_FORM` was checked against the decompile directly. `DemonForm.cs` has `OnUpgrade → UpgradeValueBy(1m)` on a canonical base of 3, so upgraded Demon Form gives **4 Strength**. The simulator produces 4. The test asserts 3 — it encodes *Slay the Spire 1's* Demon Form.

**So the first sample is a stale test, not a simulator bug.** `apply_derived_values` pulls the real numbers from game data, and these tests hardcode expectations from before that existed. If that generalises, phase 0 is largely deleting wrong assertions rather than fixing the engine — hours, not days.

It must still be done card by card against the decompile, because "the test is probably stale" is exactly the reasoning that let `docs/CARDS_REFERENCE.md` drift while the tests read it as an oracle and stayed green. **Every one of the 22 gets checked against `decompiled/`, and the count of test-was-wrong versus code-was-wrong gets reported**, because that ratio tells us whether the simulator can be trusted at all — which is the real question this phase answers.

### 1. The pilot is blind to 29 of 86 cards — every Power

This is `PLAN_DECKBUILDING.md` phase D0, and it is the ceiling on **everything this repo measures**, not only drafting. The battery, the gauntlet, `eval_drafting.py`, the card rankings, `DEFENCE_CAP_BY_ACT` — all of them are reports on what a pilot that cannot play a third of the pool can do.

Today's null result on untapped priors is the clearest evidence of the cost: real-run card values from 27,000 games made the agent *worse* (8.7 → 7.4 floors), because a card worth +4% to a human who can build around it is worth nothing to a pilot that will never play it.

**Gate.** Handed a deck with `DEMON_FORM` or `INFLAME`, the pilot plays it on turn 1-2 of a long fight and declines it on the last turn of a short one. Blind-card count under 10.

### 2. Combat is the binding constraint, not drafting

```
act 1 elite win rate    greedy heuristic  36%    trained v3 model  33%
```

A competent human wins act 1 elites at ~95%. **At 36%, meeting two elites and a boss in act 1 is a ~4% proposition before drafting is considered at all** — which is the floor-17 wall, arithmetically.

That the trained model is *level with a greedy heuristic after 40M steps* is the finding that should drive everything here. Three training runs (6M, 20M, 40M) produced null results. The fourth will too, unless something upstream changes.

**Gate.** Act 1 elite win rate above 70% with a fixed reference deck. If more training cannot do that — and the evidence says it cannot — the answer is search at decision time (`flat_mc` measured 63-67%, currently *worse* than greedy, and worth revisiting only once the pilot inside it can see the whole card pool).

### 3. The act 1 boss is an upgrade problem, and it is already solved on paper

Measured in `rest_choice`: upgrading eight cards moved the act 1 boss from **13.9% to 69.4%**. That is a 55-point swing, the largest single measured effect anywhere in this project, and the agent does not systematically pursue it.

Baalorlord's target is 33-50% upgrade density for winning runs. **We do not record upgrade density in run records, so we do not know ours.** That is an instrumentation hole on the single highest-leverage quantity identified so far, and it is one line to fix.

**Gate.** Act 1 boss win rate above 50% live. Upgrade density at floor 16 above 30%.

### 4. Drafting — where most of today went, and it is fourth for a reason

Deck bloat is real (live decks reach 17-23 cards; the guide says skip liberally) and the block-density and cycle-time terms now address it. But drafting sits below combat in the chain: a perfect deck flown at a 36% elite win rate still dies at floor 17.

**Do not spend another day here until gates 0-3 pass.**

---

## What has been tried and did not work

Recorded so it is not retried. Every one of these cost real time.

| Attempt | Result |
|---|---|
| More training: 6M, 20M, 40M steps | Null. No learning trend in any of them |
| Better observation (v2 frozen embeddings) | Null on floors; card identity now survives patches |
| Intent damage fix (v3) | Real but small: ~3 HP/fight, +10 points elite win rate |
| Flat Monte Carlo combat search | 63-67% vs greedy's 73.3% — worse, inside noise |
| untapped card priors leading the draft | **Negative**: 8.7 → 7.4 floors over 30 paired runs |
| `best_model` selection over noisy evals | Artifact every time; `final` is never worse |

The pattern: **every attempt to improve the agent by giving it more of something (steps, data, search) has failed, and the two that worked were bug fixes.** That is a strong prior about where the next win comes from.

---

## Instrumentation to add first — hours, not days

None of this is a phase; it is the cost of not repeating today.

1. **Record upgrade density and block density in every run record.** The highest-leverage quantity in the project is currently unmeasured live.
2. **Record which fight killed the run, and the deck at death.** `room_type` exists; the enemy does not.
3. **Run `scripts/eval_drafting.py` before every drafting change.** It exists now, it takes minutes, and it would have caught today's regression before it reached the game.
4. **Make the harness reach real deck sizes.** Its runs stop at ~13 cards because it routes at random, so the bloat and density terms — both aimed at 17-23 card decks — are currently **untestable in it**. This is a known hole in today's work: those two terms are shipped and unmeasured.

---

## Order of work

```
0  Ironclad parity        22 failing tests, triage first          gate: 0 failures
1  Pilot sees the pool    D0 from PLAN_DECKBUILDING               gate: <10 blind cards
2  Combat quality         elite win 36% -> 70%                    gate: elite win >70%
3  Upgrade density        boss 9% -> 50%                          gate: past act 1 boss
4  Drafting               already largely built                   gate: eval_drafting A/B
```

Instrumentation runs alongside 0 and is not optional.

---

## What would make me say this is not winnable

Honesty about the abandon conditions, because "keep going" is not a plan.

- **If gate 2 cannot be reached.** If the pilot can see every card, the simulator is right, and act 1 elite win rate still sits under 50%, then the gap is the combat policy itself and no amount of drafting or routing closes it. The project would then be a combat-AI problem wearing a roguelike costume, and should be scoped to that or stopped.
- **If parity cannot be held.** If patches break Ironclad card parity faster than it can be fixed, every offline measurement decays to noise and the whole measured-decision approach fails. The scrape-and-rederive design exists to prevent this; it has not been tested against a real patch yet.
- **If the act 1 boss stays a wall after upgrade density is fixed.** That would mean the 13.9% → 69.4% result does not transfer live, and the most reliable measurement in the project is wrong about the real game.

A win is not close. 0 wins in 142 runs, and floor 34 has never been reached. But the chain above is four links, three of them are bug-shaped rather than research-shaped, and the arithmetic says a 5% win rate needs a 2-point per-floor improvement rather than a miracle.
