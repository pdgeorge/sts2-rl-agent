# Is it the decks, or is it the play?

Both. The deck is the bigger half, and the deck problem has one dominant cause
that is a rule which almost never fires — the same shape as every other real bug
this project has found.

---

## 1. The decomposition

`scripts/deck_or_play.py` takes the decks LIVE actually carried to the act 1
boss — 85 distinct decks read out of the captured protocol — and fights the six
act 1 bosses with them offline, under the same search that plays live, at a
controlled 80% HP (live's own measured mean at boss entry is 81%).

841 fights:

| | boss win rate |
|---|---|
| offline, its OWN decks | **74% +/- 3** |
| offline, LIVE's decks | **46% +/- 2** |
| live, live's decks | **29% +/- 3** |

    74 -> 46   28 points lost to THE DECK   (same play, worse cards)
    46 -> 29   17 points lost to THE PLAY   (same cards, live path)

So the decks cost roughly 1.6x what the play costs. Neither is small.

Caveat, stated rather than buried: the 74% figure comes from the funnel at
`max_nodes=2000`, while the 46% run used live's own 20,000. The middle row
therefore had the BETTER searcher and still won less, which strengthens the deck
conclusion rather than weakening it — but the two are not a perfectly controlled
pair.

---

## 2. Why the decks are bad

81 distinct live decks at floors 15-17:

| | |
|---|---|
| size | mean 20.4, median 21 |
| basic Strike/Defend | **9.0 cards = 44% of the deck** |
| **upgraded** | **1.5 cards = 7% of the deck** |

Nine basics and one and a half upgrades, going into a 173-222 HP boss.

The most-played cards in those fights were Defend (85) and Strike (80) out of
385 total — she is playing 6-damage Strikes at a 200 HP boss because 44% of her
deck is 6-damage Strikes.

---

## 3. The dominant cause: she can almost never upgrade

`_pick_rest_option` smiths only when `hp >= 0.8 * max_hp`. Measured over 137
captured rest-site visits:

    HP on arrival at a rest site:  p10 20%   median 42%   p90 82%

| threshold | rest visits that would SMITH |
|---|---|
| **0.80 (current)** | **11%** |
| 0.70 | 21% |
| 0.60 | 27% |
| 0.50 | 39% |
| 0.40 | 55% |

**Nine rest visits in ten are spent healing.** That is exactly the 7% upgraded
deck, arrived at honestly: she cannot upgrade because she is never healthy
enough at the moment the choice is offered.

This is the same failure as the two before it — a rule set where its condition
is nearly unreachable, so a policy that exists never influences a run:

| rule | condition | how often it could fire |
|---|---|---|
| card reward skip | `score < 0.0` | 0 of 366 screens |
| deck bloat skip | `deck > 30` | never; act 1 decks are 21-22 |
| **rest upgrade** | **`hp >= 80%`** | **11% of visits** |

---

## 4. Why this is not the same as "just lower the threshold"

The 0.80 came from somewhere real. Its docstring records that a flat
`hp < 0.5 * max_hp` test "said SMITH 17 times at a median 49 HP — upgrading a
card immediately before an act boss" and got runs killed. Healing before a boss
is correct. The bug is that the SAME threshold is applied to every rest site,
including the ones nowhere near a boss.

`_boss_is_next` already exists and is already wired through
(`agent_runner.py`, plumbed via `live_policy._boss_is_next`). It is currently
used to pick between `required_hp_fraction("boss")` and `("elite")` — which are
**both 0.80**, so the branch selects between two identical values and does
nothing at all.

So the fix is not a new threshold, it is making the existing branch mean
something: heal before a boss, upgrade otherwise.

---

## 5. What I would change, in order

1. **Split the rest thresholds.** Boss-next stays at 0.80. Ordinary rest sites
   drop to something that can actually fire. The sweep decides the value, and it
   is a cheap one — it needs the same 3 arms x 70 seeds the last one used.
2. **Then re-measure the deck.** Upgrades should move the 7% figure directly;
   if they do not, this analysis is wrong and I want to know quickly.
3. **Then the 17 points of play.** That is the offline/live gap and it is a
   separate investigation, not a deckbuilding one.

## 6. What I am NOT proposing

- **Removal.** You were right and the sweep agreed from the other direction:
  declining cards made act 1 *worse* (-5.7% +/- 2.8%), because a smaller deck is
  a more-basic deck when the basics are what is diluting it. Upgrades attack the
  same problem from the side that works — they improve the cards already there
  instead of refusing new ones.
- **A new heuristic.** Everything above uses machinery that already exists and
  is already plumbed. `_boss_is_next` is live and currently branches between two
  equal numbers.

---

## 7. Loose end found on the way

`deck_or_play` died at 841/1020 with `CloneError: a turn-setup callback is
pending`. That is the search failing to clone a legal position. It has not been
seen live yet, but a raise inside `LiveSearch.decide` is exactly how a fight
gets handed to the trained model, and that is part of the 17 points. Worth
chasing after the rest-site work.
