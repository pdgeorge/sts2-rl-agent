# Bridge and offline: where they agree, and where they do not

One of these has to be the source of truth for training. They are not the same
agent today. This is the audit, and the divergences are listed so they can be
closed rather than rediscovered.

## The decision points

| decision | live (bridge) | offline (`scripts/live_policy.py`) | aligned |
|---|---|---|---|
| map node | `_pick_map_node` | `_pick_map_node` (same function) | YES |
| rest site | `_pick_rest_option` | `_pick_rest_option` (same function) | YES |
| card reward | `_pick_card_reward_index` | same function | YES |
| shop | `_pick_shop_option` | same function | YES |
| event | `_pick_event_option` | same function | YES |
| treasure | `_pick_treasure_option` | same function | YES |
| boss relic | `_pick_boss_relic_option` | same function | YES |
| combat | `SearchAgent`, 20000 nodes / 3s | `SearchAgent`, 2000 nodes / 60s | **NO -- budget** |

Seven of eight agree. Every non-combat decision now calls the same function on
both sides; the delegation was mechanical because both already speak the same
action vocabulary (`buy_relic`, `buy_card`, `buy_potion`, `remove_card`,
`leave_shop`, `collect`, `pick_relic`, `event_choice` are identical strings on
the wire and in RunManager), so `_bridge_options` relabels rather than
translates.

Exercised over 60 offline runs: MAP_CHOICE 324, CARD_REWARD 279, EVENT 43,
SHOP 33, REST_SITE 24, TREASURE 7 decisions, with 17 of ~710 falling through to
the old fallback -- all of them sub-screens that share a phase (the relic and
potion pickers inside CARD_REWARD), not the decisions this is about.

WAS: two of eight agreed. The fallback is `harvest_combat_benchmark._noncombat_action`,
whose own docstring calls it "a plausible amateur's non-combat choice -- not an
attempt at good play".

## The card reward divergence, in detail

Both sides score with `card_quality.rank_cards`, so the *ranking* matches. The
**skip rule does not**:

    live     skip if best_score < SKIP_THRESHOLD  OR  deck_size > LARGE_DECK_SIZE
    offline  skip if best_score <= 0

Offline has no deck-size rule at all. Deck size is the thing that decides an act
1 boss fight, so this is the divergence that matters most.

## Both skip rules are unreachable in act 1

Measured over 366 real card-reward screens from the captured protocol:

    best-on-offer score:  min 1.00   median 2.50   max 5.90
    screens where the best scores below SKIP_THRESHOLD (0.0):  0 of 366

and `CARD_REWARD_LARGE_DECK_SIZE = 30` against act 1 decks of 21-22 cards.

So the agent takes every card it is offered, and would continue to even now that
the mod can actually click Skip. That is why boss decks look like this:

    VANTOM      21 cards, 9 basic Strike/Defend (43%)
    LAGAVULIN   22 cards, 9 basic Strike/Defend (41%)

The thresholds are not wrong by a little. They are set where they can never
fire, so the skip policy has never influenced a single run.

## What NOT to do about it

Do not pick new thresholds by eye. The right values are an empirical question
and now a cheap one: both sides can skip, the offline funnel is paired on 400
seeds and resolves +/-1.2%, and it reports act 2 reach alongside act 1 clear so
a threshold that buys act 1 by gutting act 2 is visible immediately.

## Closing the divergences

The fix for the copies is not to keep them in step -- it is to delete them.
`live_policy` should build the bridge-shaped state and call the LIVE chooser, as
it already does for map and rest. Anything else drifts the moment one side is
edited, which is exactly what happened to the card reward path.

The combat budget divergence is different: 2000 nodes offline is a deliberate
cost bound for sweeps, not an accident. It should be stated in every report that
compares the two, because the weaker searcher currently wins more, and any
explanation of that gap has to account for it.
