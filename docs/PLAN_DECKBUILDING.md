# Plan: drafting a deck that builds on itself

Written 2026-08-04. Companion to `docs/PLAN.md`, which this does not replace — it fills in what that plan calls Phase 5 ("Deck knowledge: MAP-Elites archive over the battery. Measured synergy table.") and adds the two things that have to happen before a synergy table can mean anything.

Prompted by <https://sts2.untapped.gg/en/guides/getting-started> and the question of how Cyra could draft a *meta deck* — a deck whose cards make each other better — rather than a pile of individually-good cards.

Revised 2026-08-04 after reading <https://sts2.untapped.gg/en/articles/core-deckbuilding-concepts-in-slay-the-spire> (Baalorlord), which changes the shape of this plan substantially. See **Phase D-1**, which did not exist in the first draft and is now the cheapest and highest-value phase in the document. The synergy table, which was the centrepiece, has been repriced by about 100x and demoted to optional.

---

## The cheapest thing in this document, and it was missing from the first draft

Baalorlord's article proposes four deck-level metrics. Every one of them is computable **directly from a decklist in microseconds** — no pilot, no simulation, no seeds, no noise, no synergy table:

1. **Damage output**, split into *frontload* (available turn 1–2 without setup) and *scaling* (grows over the fight). With an explicit tradeoff: "to get scaling, you must sacrifice frontload."
2. **Cycle time** — `(deck size − card draw effects) / cards drawn per turn`. How many turns to see your whole deck. Adding any card raises it. A card that draws exactly one card does not improve it *versus skipping*. Powers and Exhaust cards lower it, because they leave the cycle.
3. **Block density** — the *fraction* of the deck that is block. Recommends ~33%.
4. **Upgrade density** — fraction upgraded. 33–50% for winning runs at high difficulty.

This matters far more than it looks, because the first draft of this plan proposed spending 100+ hours of simulation to reach conclusions that three of these four give away for free.

### The block density numbers check out, computed independently

Baalorlord's figures reproduce under exact hypergeometric draw (not the binomial approximation), opening hand of 5, N=20:

```
density   P(no block)   P(>=4 of 5 are block)      his figures
  20%         28%              0.1%                 "30% chance of none"
  33%          8%              3.1%                 "89% chance of >=1",  "3%"
  50%          2%             15.2%                 "17% all-block hands"
```

The trough is real and it is shallow-bottomed: below ~25% you brick on defence, above ~45% you draw hands that cannot kill anything. `scripts/` has no equivalent of this and it costs nothing to add.

### This reframes the defence question — and resolves the conflict found today

`DEFENCE_FLOOR_BY_ACT = {1: 1, 2: 2, 3: 2}` counts *cards*. Density is a *fraction*. Those diverge exactly as a run progresses:

```
10-card starter, 4 Defends           40% block density
25-card act 2 deck, same 4 Defends   16% block density  -- half the healthy floor
```

So "two defensive cards by act 2" is not a stable target at all; it silently becomes a weaker and weaker requirement as the deck grows. The count formulation was measuring the wrong thing, and the failure pdgeorge reported live — *"taking big hits but having no block cards in the deck"* — is precisely what a diluted density looks like from the inside.

It also dissolves the conflict found while wiring up `card_priors`. Untapped says act 2 block cards are negative (`BLOOD_WALL` -4% over 10,000 offers, `FLAME_BARRIER` -4%). That is a *card-level marginal value averaged over every deck that took it* — decks already at 45% block density and decks at 10%, added together. Density is a *deck-level* quantity, and the two are not in conflict: a block card is bad at 45% and good at 15%, and averaging over both produces exactly the small negative number untapped reports.

**A conditional quantity was being estimated by an unconditional statistic.** That is the whole disagreement, and no synergy table is needed to fix it.

---

## What "builds on itself" means, stated so it can be measured

Let `V(D)` be the deck score the battery already produces: mean HP surviving a gauntlet (`mean_gauntlet_hp`).

Today `card_choice.rank_candidates` picks `argmax_c V(D + c)`. That is the **marginal value** of one card against the deck as it stands.

A deck that builds on itself is one where marginal value is *increasing* rather than decreasing — where a card is worth more because of what is already there, and worth more still because of what it makes worth taking later. The two-card version is superadditivity:

```
S(a, b) = V(D + a + b) - V(D + a) - V(D + b) + V(D)
```

`S > 0` is synergy, `S < 0` is anti-synergy (two win conditions competing for the same energy), `S = 0` is a pile of good cards. Nothing in the current pipeline computes or approximates `S`, and `card_choice`'s own docstring says so:

> It is greedy over one card. It cannot see that two cards are only good together, which is exactly what a synergy measurement would add later.

The whole-run version — the actual objective — is the value of the best deck *reachable* from here:

```
V*(D, R) = E_offers[ max_c V*(D + c, R - 1) ]        R = card rewards remaining
```

Everything below is a way of approximating `V*` cheaply enough to run live.

### The tension with the guide, and how it resolves

The untapped guide says the opposite of what "build a meta deck" sounds like:

> Draft cards that solve your immediate problems, rather than hoping to build an ideal combo later in the game.

> Taking every card bloats your deck with filler, making it much harder to reliably draw your strongest, most upgraded cards when you actually need them.

These are not in conflict with `V*` — they *are* `V*`, correctly discounted. The expectation is over offers you will actually get. A payoff card whose enabler is rare, with four card rewards left, has almost no probability of ever being enabled, so `V*` prices it as the dead card it is. The same formula that rewards Barricade next to Body Slam refuses to take Barricade on floor 44 with no Body Slam in sight. **The discount is the guide's advice, written as arithmetic.** Any design that adds a flat "synergy bonus" without the reachability discount will produce exactly the greedy-combo drafting the guide warns about.

The bloat point is already handled and should stay that way: `include_skip=True` makes declining a real candidate, and gauntlet-HP scoring prices dilution directly, because a diluted deck draws its good cards less often and loses more HP.

---

## Why this cannot work today — four reasons, in order of cost

### 1. The pilot cannot play a self-building deck. This blocks everything else.

`value_pilot` scores each playable card as `_damage_value + _block_value + _debuff_value + _draw_value` and picks the maximum, with `best_value` initialised to `0.0`. A card scoring exactly 0 is therefore *never chosen in the main loop* — it can only be played by the leftover-energy fallback at the bottom of the function, after every card worth HP has already been played, and only if it does not exhaust.

Measured over the 86 registered Ironclad cards:

| Finding | Count |
|---|---|
| Cards scoring exactly 0 to `value_pilot` | **38 of 86 (44%)** |
| Powers scoring 0 | **20 of 20** — every Power in the character, including `INFLAME`, `DEMON_FORM_CARD`, `BARRICADE_CARD`, `JUGGERNAUT_CARD`, `FEEL_NO_PAIN_CARD`, `CORRUPTION_CARD`, `RUPTURE_CARD` |
| Non-Power cards also scoring 0 | 18 — so the blindness is not only a Powers problem |
| Cards carrying `calc_base` | 8 |
| Of those, genuinely **invisible** (`base_damage` 0 or None) | 3 — `BODY_SLAM`, `DEMONIC_SHIELD`, `EXPECT_A_FIGHT` |
| Of those, **visible with a frozen wrong number** | 5 — `PERFECTED_STRIKE` reads 6 regardless of how many Strikes are held; `TEAR_ASUNDER` reads 5 against a `calc_base` of 0 |

The first draft of this table said all eight `calc_base` cards were invisible. Five of them are worse than invisible: they are *mis-priced*, which is the same bug class as `Intent.damage` carrying a frozen literal — the one that invalidated three experiments and forced layout v3. An unseen card is skipped and the loss is legible. A mis-priced card gets played at the wrong moment and corrupts the measurement without ever looking wrong.

Two of the four value terms never fire at all:

- `_draw_value` reads `effect_vars["draw"]`. **No card in the game has a `draw` key** — the draw variable is called `cards` (`BATTLE_TRANCE {'cards': 3}`, `OFFERING {'cards': 3, 'energy': 2}`). The term returns 0.0 for every card, always. Nine Ironclad cards carry `cards`.
- `_debuff_value` reads `weak` and `vulnerable`. No Ironclad card carries `weak`; only 5 carry `vulnerable`. Half of that term is dead for this character.

So the instrument that decides what a card is worth is blind to precisely the cards that make a deck build on itself. Demon Form is a 3-cost card the pilot plays last or not at all. Every "the battery says scaling is worthless" result to date is uninterpretable until this is fixed — and it will keep being uninterpretable, because a synergy table built on this pilot measures the pilot's blind spot with extra decimal places.

`battery.py`'s docstring already predicted this exact failure mode, one archetype over:

> A greedy-damage pilot cannot pilot a block deck, so it will score block cards as worthless -- not because they are, but because it never converts block into survival. This is the single largest bias in the whole design and it is not fixable by adding encounters or seeds.

`value_pilot` was written to fix that for block. The same fix has not been done for scaling, draw, or energy.

### 2. The measurement is too short to contain a payoff.

`score_candidate` plays 5 consecutive fights (`GAUNTLET_FIGHTS`), from full HP, of the current act's *normal* tier, with the deck exactly as it is and no upgrades. `tiers_for_floor` explicitly excludes bosses:

> Bosses are excluded: with a mid-act deck most candidates score zero against them, and a column of zeros ranks nothing.

That is the right call for the measurement as built, and it is also the reason scaling cannot be seen. A scaling card's payoff is in long fights (elites, bosses) and in later acts, after upgrades and after four more cards have been added around it. Five act-1 hallway fights at full HP is the single cell where a scaling card looks worst: it costs a turn and pays nothing back before the fight ends.

`rest_choice` has already shown how large this effect is, from the other direction — upgrading eight cards moved the act 1 boss from **13.9% to 69.4%**. Deck maturity is worth 55 points of boss win rate and the draft evaluator does not model it at all.

### 3. The score has no lookahead. (It is NOT, as first drafted, unconditional.)

The first draft said "there is no term anywhere that depends on a pair" and concluded the score was additive by construction. That is wrong in a way that matters for costing this whole plan.

`V(D + c) − V(D)` is **already conditional on every card in `D`**. If the pilot could play Barricade, the battery would already price Body Slam higher in a deck that holds one — no pair table involved, because the gauntlet plays the actual deck. What `rank_candidates` genuinely lacks is **lookahead**: it cannot value a card for a partner it does not yet hold.

So the missing quantity is not conditioning. It is the *prospective* half of it:

```
realised     synergy with partners ALREADY in the deck   -- already inside V(D+c), free once D0 lands
prospective  synergy with partners not yet drawn         -- genuinely needs a pair table
```

This split is the reason Phase D2 is now optional rather than central. See its revised cost.

### 4. The prior is unconditional, and it is exactly the wrong quantity for this.

`card_priors.prior_score` returns untapped's run-winrate delta, averaged over every deck that ever took the card. That is *average marginal value across all decks*, which is the quantity a synergy-aware drafter is trying to beat. It is a good prior on card quality — 27,000 runs versus 8 seeds, and it should keep leading, exactly as `BATTERY_POINTS_PER_UNIT`'s docstring argues. But it can never say "Barricade is +12 in this deck and -3 in that one", and untapped publishes no pair statistics: the card pages carry Card Reward / Shop / Smith stats plus a curated "RELATED CARDS" list with no numbers attached. **Conditional value has to come from our own simulation. There is no shortcut.**

---

## Phases and gates

Same rules as `docs/PLAN.md`: each phase ends with a gate, and if the gate fails, stop rather than build on it. Rule 6 applies throughout — selection requires clearing 2 sem on a held-out seed set.

### Phase D-1 — Deck-level densities. Do this first; it is nearly free.

Not in the first draft. It should be, because it is the only phase here that produces a usable drafting signal **without depending on the pilot**, and the pilot is what blocks everything else.

- **Compute the four metrics from a decklist.** New module, pure functions, no simulation: `block_density`, `upgrade_density`, `cycle_time`, and `frontload_damage` / `scaling_damage`. Cycle time is `(deck_size − draw_per_cycle) / cards_drawn_per_turn` straight out of the article. Block density keys off `base_block >= HIGH_QUALITY_BLOCK`? **No** — off *all* block, including Defends, because density is about what you draw, not about card quality. This is the opposite of the choice `card_choice.HIGH_QUALITY_BLOCK` makes, and deliberately so: the two answer different questions and both are right.
- **Replace `DEFENCE_FLOOR_BY_ACT` with a density band.** Target ~33%, penalty rising below ~25% and above ~45%, from the hypergeometric table at the top of this document. A band, not a threshold, because the article is explicit that "not every run wants the same amount of block" — a high-draw deck sees more of itself per turn and tolerates less.
- **Give `rest_choice` an upgrade-density target.** 33–50%. It currently prices each upgrade individually and has no notion of the deck being under-upgraded as a whole, which is the state the agent is in every time it dies to a boss. Compare against `rest_choice`'s own measured result: eight upgrades moved the act 1 boss from 13.9% to 69.4%.
- **Price bloat through cycle time in the skip decision.** `include_skip` already makes skipping a real candidate and gauntlet-HP scoring already penalises dilution indirectly. Cycle time makes it *direct* and *free*, and it expresses the article's sharpest point: a card that draws exactly one card does not beat skipping on cycle time, so it has to justify itself on its other text alone.

**Gate.** Two, both cheap:

1. On a recorded live deck that died to a boss, the metrics say something a human would recognise — e.g. block density well under 25%, upgrade density under 20%. If the numbers do not flag a deck that visibly failed, the metrics are not measuring what the article claims.
2. Adding the density term changes at least one pick on a replayed draft, in the direction of the band. If it never changes a pick it is decoration; delete it rather than keep it.

**Why this goes first.** It is hours, not days; it needs no pilot fix; it cannot be contaminated by pilot blindness because it never plays a card; and it directly targets the failure pdgeorge has watched twice. If everything else in this document is abandoned, this phase still stands.

### Phase D0 — Make the pilot able to fly a self-building deck

Blocking. Nothing below is interpretable until this passes.

- **Give the pilot a turn horizon.** One number: `remaining_turns ≈ total_enemy_hp / max(1, our_damage_per_turn)`, clamped to something like `[1, 12]`. This is the quantity every scaling card needs and the pilot does not have. `TURNS_AHEAD = 6.0` is the same idea already hardcoded as a constant; make it a state-dependent estimate and reuse it in both places.
- **Price powers in the pilot's existing HP currency.** Strength: `strength × expected_attacks_per_turn × remaining_turns`, cashed out through `_damage_value`'s "fraction of enemy output removed" logic so the units match. Per-turn block (`plating`, `FEEL_NO_PAIN_CARD`'s `power`, `JUGGERNAUT_CARD`): `block_per_turn × remaining_turns`, capped by incoming as `_block_value` already does. Energy (`PYRE`, `FORGOTTEN_RITUAL`): value the energy at the deck's mean HP-per-energy.
- **Fix `_draw_value` to read `cards`, not `draw`.** One-line change that turns on 9 cards. Then price draw as expected value of the cards drawn rather than a flat 1.5 — the flat constant was chosen deliberately conservative to avoid overrating draw, but 1.5 × 0 is 0 either way.
- **Ask the engine for damage instead of reading `base_damage`.** `sts2_env/core/damage.py:calculate_damage` exists and takes `(base_damage, dealer, target, props, combat)`. Routing the pilot through it fixes `BODY_SLAM` and the other 7 `calc_base` cards, *and* makes Strength and Vulnerable show up in the pilot's damage numbers automatically instead of needing separate cases. This is the change that makes the pilot see its own scaling.
- **Keep `greedy_pilot` untouched** as the fast bulk pilot and the control. Every claim below should be checkable against "does the fixed pilot disagree with greedy, and in the direction we expect".

**Gate.** Two checks, both cheap and both falsifiable:

1. Handed a deck containing `DEMON_FORM_CARD` or `INFLAME`, the pilot plays it on turn 1–2 of a long fight and *declines* to play it on the last turn of a short one. Trace one fight with `scripts/show_fight.py` and read it.
2. On a matched pair — scaling deck versus a cost-matched vanilla-attack deck — the battery ranks vanilla above scaling on `act1_weak` and scaling above vanilla on `act1_boss`. If the ordering does not invert somewhere, the pilot still cannot convert scaling and the gate has failed.

### Phase D1 — Make the measurement long enough for a payoff to land

- **Make the boss cell discriminate instead of returning zeros.** Replace win-rate-only scoring for boss/elite cells with a graded score: `min(1, damage_dealt / boss_max_hp)` blended with turns survived. A deck that takes the act-1 boss to 20% and one that never scratches it currently score identically at 0.0, which is why the column had to be excluded. Graded, it becomes the most informative cell for scaling and can come back into `tiers_for_floor`.
- **Score the deck at maturity, not only as it stands.** Evaluate `D + c` *and* `mature(D + c, floor)` = the deck plus `k` median-quality future cards and the upgrades it will plausibly have by the boss, then blend by remaining floors. `k` = expected remaining card rewards, which the run structure knows. Cheapest defensible version: a fixed +3 cards / +2 upgrades "act tax", stated as an approximation in the docstring the way `DEFENCE_FLOOR_BY_ACT` states its provenance.
- **Let the gauntlet run long enough to end on a boss.** `GAUNTLET_FIGHTS = 5` was tuned to discriminate *starter* decks and its docstring says so honestly. A scaling deck needs a horizon that includes the payoff; add an act-shaped variant — `n` normals then the act boss, scored on HP surviving the boss, 0 if dead — which is closer to the run's real objective than 5 normals at full HP.

**Gate.** On one fixed pair of decks (tempo, scaling), the same code path ranks tempo above scaling at floor 4 and scaling above tempo at floor 40. If the ordering does not invert with floor, the maturity term is not doing anything and should be deleted rather than kept as decoration.

### Phase D2 — Measure synergy instead of asserting it

This is `docs/PLAN.md` Phase 5, unchanged in intent.

**This phase is now OPTIONAL and should not be started until D-1, D0 and D1 have landed and been shown to leave a gap.** The first draft made it the centrepiece. It was mispriced by about two orders of magnitude, and the first draft contained its own refutation two paragraphs later.

- **Build the pair table offline.** `S(a,b) = V(D+a+b) - V(D+a) - V(D+b) + V(D)` over a fixed reference deck `D`, on paired seeds, both pilots.

  **Corrected cost.** The first draft budgeted 3,655 pairs × ~1.7 s ≈ 1.7 h and called it "affordable overnight". But 1.7 s buys ~90 fights, while `card_choice`'s own table shows a single-card delta of 0.005 needs ~240 fights merely to rank *correctly* — and this document states, correctly, that a pair delta is a difference of differences carrying roughly twice the noise, hence about **four times** the fights. That is ~960 fights per pair, ~18 s, so:

  ```
                              first draft      corrected
  per reference deck/pilot        1.7 h          ~18 h
  x 2 pilots, x 3 reference decks  --           ~110 h
  ```

  The Honest Limits section already asks for 3–4 reference decks and both pilots. So the plan as first written was a ~110-hour measurement, budgeted at 1.7, to produce a table its own next bullet predicts will be **mostly empty**.

  And the half it buys is the half that matters least. Per the correction in reason 3 above, *realised* synergy is already inside `V(D+c)` once the pilot can fly it; the pair table is needed only for the **prospective** term — which D3's reachability discount then deliberately shrinks toward zero as the run progresses. Spending 110 hours to improve the term that vanishes by act 3 is the wrong order.
- **Store sample size and sem per pair, and keep only what clears 2 sem.** Expect the table to be *sparse*. A pair delta is a difference of differences, so its noise is roughly twice a single-card delta's — and single-card deltas were measured at 0.005 with 240 fights. Most pairs will not clear. That is information, not failure: a handful of real synergies is exactly what an archetype is, and a table of 3,655 confident numbers would be the thing to disbelieve.
- **Report how many pairs were dropped for insufficient power.** Rule 5, no silent caps.

**Gate — the important one.** The surviving table reproduces known pairs *without being told about them*: `BODY_SLAM × BARRICADE_CARD`, `INFLAME × SWORD_BOOMERANG` (Strength × multi-hit), `FEEL_NO_PAIN_CARD ×` exhaust fodder, `CORRUPTION_CARD ×` skills. If it cannot, the pilot or the horizon is still wrong — go back to D0/D1 rather than shipping a table that will be believed.

**Then delete the keyword lists in `sts2_env/gym_env/deck_features.py`.** They are hand-written STS1 card names matched as substrings against STS2 card names, and they have already rotted: **20 of the 45 keyword strings match no Ironclad card at all.** `RETAIN_KEYWORDS` matches nothing whatsoever, so feature 16 is identically zero for every deck. `HEAL_KEYWORDS` contains `burning_blood`, which is a relic. `STRENGTH_KEYWORDS` contains `strength`, `limit_break` and `brutality`, none of which name a card in this game. Three of the 32 features the meta-policy reads are "archetype signals" built on top of these. This is the exact failure mode `card_choice` cites as its reason for keying off `base_block` instead of a card list.

### Phase D3 — Draft with a plan instead of one card at a time

The score in `_to_winrate_points` grows one term:

```
score(c) = prior(c)                                     # untapped, unconditional, leads on card quality
         + BATTERY_POINTS_PER_UNIT * (V(D+c) - V(skip)) # today's deck-fit term, unchanged
         + PLAN_POINTS_PER_UNIT * reach(c | D, floor)   # new
```

with reachability split into what is real and what is hoped for:

```
reach(c | D, floor) = Σ_{p ∈ D}   S(c, p)                       # realised: partner already held, full weight
                    + Σ_{p ∉ D}   S(c, p) · P(see p | R)         # prospective: discounted by whether it can happen
P(see p | R)        = 1 - (1 - q_p)^R
```

`R` = card rewards remaining in the run, `q_p` = probability card `p` appears in a given reward. Do not hardcode `q_p` from STS1 folklore — sample `sts2_env/run/rewards.py`'s roll 10,000 times per rarity per act and read it off. That cannot drift against the simulator, and it re-derives itself after a patch.

Properties worth noting, because they are the reason to prefer this shape:

- Late in the run `R → 0`, so the prospective term vanishes and the drafter reverts to solving immediate problems. **That is the guide's headline advice, obtained for free rather than bolted on.**
- A rare payoff card with a rare enabler is discounted twice and stays unattractive.
- Anti-synergy is expressible: `S < 0` prices the second competing win condition down, which is the failure mode of a deck that "has good cards" and no plan.
- `PLAN_POINTS_PER_UNIT` needs the same treatment `BATTERY_POINTS_PER_UNIT` got — chosen so simulation cannot outshout 27,000 real runs. Start it at half the battery's, then calibrate by measuring *how often it changes a pick*, which is a number you can look at, unlike "does it feel right".

**The plan itself should be discovered, not declared.** Do not write an `ARCHETYPES` list — that is the keyword-list mistake again. Run community detection over the pair graph `S`, and define the current plan as the community with the highest realised-plus-reachable value given the deck and relics. Log its name-by-membership ("block/Body Slam cluster: 3 held, 2 reachable"), not a hand-chosen label.

**Hysteresis, and an exit.** Once a plan is chosen, require the alternative to beat it by a margin before switching, so it does not flip every floor. And keep the generic no-plan deck as a standing candidate: if the plan's expected value falls below it — the enabler never showed up — abandon and say so in one log line. A plan you cannot abandon is the greedy-combo drafting the guide warns about.

**Gate.** A/B against the current drafter on **held-out** seeds, full runs, clearing 2 sem, or it goes into `MODELS.md` as a null result with the reason. `run_ppo_v4` is the cautionary tale and rule 6 exists for it.

### Phase D4 — Aim the whole run at the boss

This is the rest of the guide's advice, and most of it is small.

- **Weight the tiers by proximity to the boss.** `tiers_for_floor(floor)` returns `(normal, elite)` flat across a whole act. Make it `tier_weights_for_floor(floor)`, with boss weight rising as the boss approaches — on floor 3 you are drafting for hallways, on floor 15 you are drafting for the boss you are about to fight. `rest_choice` already does exactly this with its `boss_next` branch and `_boss_win_rate`; this generalises that to a weight instead of a flag, and it depends on D1's graded boss cell to be worth anything.
- **Reuse the gauntlet number as the elite-readiness trigger.** Commit 752dc96 already vetoes elites the deck cannot beat. The guide's positive form is the other half: *seek* elites once the deck can take them, because that is where relics come from. Same number, opposite sign, in `map_choice`.
- **Price upgrades through the pair table.** `rest_choice` prices upgrades against the boss already. An upgrade that turns a synergy on (Body Slam+ in a block deck) should get the `reach` term too, or the smith will keep upgrading the biggest single card.
- **Cross-check picks against `taken_pct`, free.** The scraped table already carries how often real players took each card, per act, per decision. If Cyra takes a card humans skip 95% of the time, that is not necessarily wrong — but it should appear in the log next to the score that caused it. Costs one dictionary lookup and catches a whole class of silent regression.

### Phase D5 — Make it live, and keep it patch-stable

- Freeze `data/synergy_ironclad.json` as a versioned, append-only artifact alongside the card embedding table, carrying sample sizes, sem, the pilot that produced it and the reference deck it was measured against. Live draft cost stays what it is today: a table lookup plus the existing battery call.
- Add it to `sync_content.py`'s pipeline: a new card gets no synergy row rather than a wrong one, exactly as `card_priors` treats a card untapped has never seen. **Missing must stay missing, never quietly zero** — a zero synergy row would rank an unmeasured card above everything measured as anti-synergistic.
- Distillation into a network is `docs/PLAN.md` Phase 6 and should stay there. Do not train a head on a table that has not passed D2's gate.

### What this gives Cyra

The seam is sacred and this does not touch it — publishing stays fire-and-forget, four events, degrade-to-no-op. But the gut phrase currently comes from `gap`, the margin between the top two choices, and a plan-aware drafter produces a strictly richer version of the same quantity for free: not just "I was torn" but *what it was torn between and what it is building*. `E1` in `docs/PLAN.md` is still open — the gap distribution has never been measured under whatever plays — and the A/B in D3 is the natural place to collect it, since it has to run full runs anyway.

---

## Which existing "measured" numbers this invalidates

Not a future concern. These are in the code now, labelled MEASURED in deliberate contrast to constants labelled NOT MEASURED, and quoted downstream as settled.

**`greedy_pilot` has the same blindness as `value_pilot`.** It ranks by `base_damage` alone, so every skill scores 0 and it plays them only with leftover energy, after everything damaging. It is the pilot behind the bulk of this repo's measurements.

- **`card_choice.DEFENCE_CAP_BY_ACT`** — "0 defensive cards 77.3%, 2 cards 74.7%, **3 cards 38.7% ← collapse**". Handed three block cards, greedy holds three near-dead cards it will play last or not at all. That collapse is a plausible measurement of *the pilot*, not of the deck. It is the load-bearing evidence for a cap that fires on real drafts, and it needs re-running under a fixed pilot before it is trusted further. Under the density framing above, three block cards in a 12-card deck is 25% — the bottom of the healthy band, not a collapse — which is the sort of disagreement that says the instrument was wrong.
- **The 0.005 single-card delta** quoted in `card_choice`'s docstring, in `MODELS.md`, and used as the argument that drafting is unlearnable by PPO. Measured with the same pilot. The conclusion may well survive — a single card in a ten-card deck genuinely is often not drawn — but the number should not be quoted as settled while its instrument is known blind to 44% of the pool.
- **Every claim of the form "the battery says scaling/block is worthless."** Uninterpretable until D0. This is stated in reason 1 above and repeated here because the numbers have escaped into two other documents.

**Do not re-measure these before D0 lands.** Re-running them under the same broken pilot produces the same numbers with fresh timestamps, which is worse than leaving them flagged.

---

## Honest limits, and what would make me abandon this

- **The pilot is still the ceiling.** D0 makes it able to see scaling; it does not make it good at scaling. A synergy table is a measurement of what *this pilot* can convert. If D0's gate passes but D2's gate fails, the honest conclusion is that the pilot needs to be a search, not that the table needs more seeds.
- **The effect sizes are small and the noise is not.** `card_choice`'s own table shows single-card ranking converging *from the wrong side* — at 45 fights an unplayable curse ranked better than skipping. Pair deltas are noisier still. Anything here that cannot clear 2 sem should be reported as unmeasured rather than shipped as a small effect.
- **`S` is measured against one reference deck.** Synergy is not actually a property of a pair, it is a property of a pair in a context, and the table flattens that. Measuring against 3–4 reference decks and keeping only pairs that agree across all of them is the cheap partial answer; it also quadruples the cost.
- **Abandon condition.** If D3's A/B does not clear 2 sem on held-out seeds, this goes in `MODELS.md` as a null result and the drafter stays as it is. That outcome is genuinely likely and it is not a wasted phase — D0 and D1 are worth doing on their own, because a pilot that cannot play 44% of the card pool is a defect in every measurement this repo produces, not only in drafting.

---

## If there is only a day

In this order. Each is cheap, each is worth having even if everything above is abandoned, and the first three are one-liners or nearly so.

1. **`_draw_value` reads `draw`; the variable is `cards`.** One line. Turns on 9 cards for the pilot; the term has returned 0.0 for every card in the game since it was written.
2. **Route the pilot's damage through `calculate_damage`.** `sts2_env/core/damage.py:39`. Fixes the 3 invisible `calc_base` cards including Body Slam — `card_choice`'s own worked example of the evaluator getting a pick wrong — *and* the 5 mis-priced ones, *and* makes Strength and Vulnerable show up in the pilot's numbers automatically, in one change instead of eight.
3. **Block density and upgrade density as pure functions.** An hour, no simulation, no pilot dependency, and immediately usable in `card_choice` and `rest_choice`. This is the whole of Phase D-1's core and it is the highest value-per-hour item in this document.
4. **Log `taken_pct` next to every pick.** Free — the scraped table already carries it. Tells you within one evening of live play whether the drafter is taking cards humans skip 95% of the time.

Then **flag, do not re-run**, the numbers in `card_choice`'s docstring. Several were measured with a pilot blind to 44% of the card pool, and re-measuring them with that same pilot would only restamp them. See "Which existing measured numbers this invalidates" above.

---

## Reading order, if picking this up cold

The first draft of this document argued for a synergy table as the centrepiece. It is now the least urgent thing here. The short version of what changed:

1. **D-1 (densities) is nearly free and was missing entirely.** Three of Baalorlord's four metrics need no simulation at all.
2. **The defence question was framed as a card count and should be a density.** That reframing dissolves the apparent conflict between pdgeorge's "take more block in act 2" and untapped's "act 2 block cards are -4%": a conditional quantity was being estimated by an unconditional statistic.
3. **D2 was mispriced ~100x**, and it buys the prospective term, which the reachability discount deliberately shrinks toward zero anyway.
4. **The pilot is the ceiling on everything that simulates**, which is why D-1 — the one phase that never plays a card — goes first.
