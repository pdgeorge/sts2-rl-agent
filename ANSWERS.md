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

## 5. What I would do next, in order

1. **Count offline's relics.** Running now. If offline carries meaningfully more
   than 4.7, that plus the 12-point HP difference is the 28 points, and the work
   is relic acquisition and HP economy — not deckbuilding at all.
2. **Read the upgrade/HP grid.** It says whether the rest-site fix is worth
   building before we spend a sweep on it.
3. **Then the 17 points of live-path loss**, which is a separate investigation.

## 6. Loose end, now a real bug

`CloneError: a turn-setup callback is pending` killed two separate harness runs
(841/1020 and 201/360). The search cannot clone certain legal positions. It has
not been seen live yet, but a raise inside `LiveSearch.decide` is precisely how a
fight gets handed to the trained model — and that mechanism is part of the 17
points.
