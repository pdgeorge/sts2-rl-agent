# Is it the decks, or is it the play?

**Neither, mostly.** The decks are fine — offline's and live's are equivalent
once measured fairly. What differs is the condition she arrives in, and the live
path itself.

I got this wrong at 1am and corrected it at 10am. The wrong version is in the
git history; this is what the measurements actually support.

---

## 1. The decks are NOT the problem

`scripts/deck_or_play.py` fights the six act 1 bosses with real decks, under the
same search that plays live, at a controlled 80% HP. Same search, same HP, same
bosses — only the deck differs.

| deck | boss win |
|---|---|
| live decks, with their real relics | **46% +/- 2** (n=841) |
| live decks, relics stripped | **27% +/- 3** (n=243) |
| offline decks, relics absent | **26% +/- 3** (n=201) |

Relic-for-relic, **live's decks and offline's decks win at the same rate**
(27% against 26%). Composition agrees too:

| | size | basic Strike/Defend | upgraded | HP at boss |
|---|---|---|---|---|
| offline | 19.1 | 9.0 (47%) | 2.2 (12%) | **93%** |
| live | 20.4 | 9.0 (44%) | 1.5 (7%) | **81%** |

Identical basics. So "her decks are shit" is not what the data says — or rather,
they are, but offline's are equally shit and offline wins 74%.

### The mistake worth recording

My first pass reported "offline decks are 20 points WORSE than live's". That was
my own harness: `_offline_boss_decks` set `relics: []`, because the funnel
captured `boss_deck` and not relics. Live decks fought with their real 4.7
relics; offline decks fought bare. Stripping live's relics reproduces the entire
difference (46% -> 27%), which is how the artifact was caught.

---

## 2. What relics are worth: 22 points

    live decks WITH relics    46%
    live decks WITHOUT        27%

Live carries **4.7 relics** at the act 1 boss (n=488). Offline's count was never
recorded — the funnel is capturing it now. **This is the single most important
open number**, because relics are worth more than any deckbuilding change
measured so far, and if offline accumulates more of them that is most of the gap
on its own.

---

## 3. The decomposition, corrected

| | boss win | what differs from the row below |
|---|---|---|
| offline, its own everything | 74% | HP 93%, its own relic count (unknown) |
| live decks + relics, at 80% HP, played offline | 46% | **28 points: arrival condition** |
| live, actually playing | 29% | **17 points: the live path** |

- **28 points are arrival condition** — HP (93% against 81%) and relics. Not
  deck composition, which is equivalent.
- **17 points are the live path** — identical deck, identical HP, identical
  relics, and the offline search still wins more. That is reconstruction,
  bridge, or execution.
- **0 points are deck composition.**

---

## 4. The rest-site finding still stands, but its size is now unclear

`_pick_rest_option` smiths only at `hp >= 0.8 * max_hp`, and she reaches rest
sites at a median 42% HP, so **it can fire on 11% of visits** (15 of 137). That
is real, and it is the same shape as every other genuine bug here — a rule whose
condition is nearly unreachable:

| rule | condition | could fire |
|---|---|---|
| card reward skip | `score < 0.0` | 0 of 366 |
| deck bloat skip | `deck > 30` | never (decks are 21) |
| **rest upgrade** | **`hp >= 80%`** | **11% of visits** |

But the upgrade difference it explains is only 2.2 against 1.5 cards, and the
decks perform the same. So fixing it may buy little. `scripts/upgrades_vs_hp.py`
is measuring exactly that — win rate against upgrades (0/+2/+4/+6) crossed with
HP (80/65/50%) — and an early n=144 sample suggested upgrades outrun the HP they
cost by a wide margin. **Not yet trustworthy; the full grid is still running.**

`_boss_is_next` already exists and already branches, but between
`required_hp_fraction("boss")` and `("elite")`, which are **both 0.80** — so it
selects between two identical numbers and does nothing.

---

## 5. Both open questions have now been answered, and both came back against me

### 5.1 The upgrade/HP grid: the rest-site fix is dead

`scripts/upgrades_vs_hp.py`, n ≈ 128 per cell, no errors:

| | 80% HP | 65% HP | 50% HP |
|---|---|---|---|
| **+0 (live)** | 45% | 30% | 13% |
| **+2** | 57% | 41% | 22% |
| **+4** | 59% | 48% | 30% |
| **+6** | 72% | 59% | 48% |

The exchange rate falls straight out: **~1 point of boss win per 1% of max HP**,
and **~5 points per upgrade**.

A rest site buys one or the other. Heal is **30% of max HP** (`rest_site.py:35`),
so heal ≈ **30 points** and smith ≈ **5 points**. Healing is worth **six times a
smith**, and smithing only wins within about 5% of full HP, where the heal would
be wasted anyway.

So `hp >= 0.80` is not too strict. If anything it is too loose. **The rest-site
fix I proposed in §4 would have made act 1 worse**, and the only reason that is
not now shipped is that this grid ran before the sweep did — which is the same
service the quality-bar sweep performed by coming back negative.

The one thing this does NOT settle: it measures the act 1 boss. Upgrades persist
into acts 2–3 while HP is regained at every rest site, so the trade is more
favourable to smithing over a full run than it is here. That is an act 3
question and this grid must not be quoted at it.

### 5.2 Offline's relic count: the relic hypothesis is dead too

|  | relics at boss | HP at boss | boss win |
|---|---|---|---|
| offline | **2.8** | 91% | 74% |
| live | **4.7** | 81% | 29% |

Offline wins 74% carrying **two fewer relics**. I predicted the opposite. So
relic acquisition does not explain the offline/live gap, and §4's framing of it
as "arrival condition" is only half right — the HP half survives, the relic half
is inverted.

Relics being worth 22 points still stands; that was a controlled measurement and
this does not touch it. It just is not the gap.

Applying §5.1's exchange rate to what is left: 10 points of HP ≈ 10 win points,
2 relics ≈ 9. That is ~19 of the 45. **The remaining ~26 points are still
unexplained**, and no current hypothesis covers them.

---

## 6. What I would do next, in order

1. **Re-fit the elite HP gate** (`scripts/ab_elite_gate.py`, running). 39% of
   live runs fight ZERO elites and the gate is 0.80 against a median rest-site
   arrival of 42%. This is PHASE_TWO Track B.1, which records that the gate was
   fitted before nine monster HP values and thirteen damages were corrected
   underneath it and has never been validly re-swept.
   **The supporting correlation is confounded and is not evidence**: 0/1/2+
   elites gives 7%/9%/23% clear, but a healthy run is both the one that clears
   and the one a 0.80 gate admits. Only the paired intervention separates them.
2. **The ~26 unexplained points** of the offline/live boss gap.
3. **Map routing has no lookahead at all.** `_pick_map_node` ranks the
   immediately reachable nodes by room type and takes the best one; it cannot
   plan a route, and it diverts to recovery whenever *any* visible node is
   unaffordable. The game's own beginner guide says to plan backward from the
   boss and prefer intersections. This is the structural-impossibility class
   that produced all four of this project's wins, and it is currently unlisted
   in PHASE_TWO.
4. **The elite gate keys on HP; the guide keys on deck strength.** There is no
   deck-strength signal in the routing decision at all.

## 6. Loose end, now a real bug

`CloneError: a turn-setup callback is pending` killed two separate harness runs
(841/1020 and 201/360). The search cannot clone certain legal positions. It has
not been seen live yet, but a raise inside `LiveSearch.decide` is precisely how a
fight gets handed to the trained model — and that mechanism is part of the 17
points.
