# Plan: card embeddings, deck evaluation, and what plays the game

Written 2026-08-02. This is a working plan, not a design doc — it records what is established, what is not, and what has to be answered before each next step. Read the "Known wrong" section before trusting anything remembered from the session that produced this.

The motivation: `alpha` reaches 9.7 floors and 0 wins in 200 runs, `run_ppo_v4` was a null result, and a twelve-line greedy heuristic matches the trained combat model. More PPO steps will not close that. The direction is to replace the parts that are weakest — how a card is represented, how a deck is judged, and what actually plays — while keeping the one thing that must not move.

---

## The one hard constraint

**The Cyra seam is sacred.** `sts2_env/bridge/cyra_events.py` and `sts2_env/bridge/milestones.py` define it:

- Exactly four events: run start, Ancient chosen, elite defeated, boss defeated.
- Publishing is fire-and-forget and must never block the game loop.
- A missing broker, a missing `cyra_game` checkout, or a missing `aio_pika` degrades to a no-op with one log line.
- Each milestone carries a "gut phrase" derived from `gap`, the margin between the policy's **top two choices**.

Everything else in this repo may be rewritten. Whatever plays the game must keep exposing a top-two margin, and publishing must stay off the critical path.

---

## Established this session (evidence, not recollection)

| Fact | Evidence |
|---|---|
| `copy.deepcopy` does not produce an independent `CombatState` — monster effects close over their creature, `deepcopy` copies functions by reference, so a clone's monsters act on the original's creatures | Both copies driven through identical actions diverge; fixed by `sts2_env/search/cloning.py`, 0 divergences over 40 seeds, in-run and standalone |
| Intent damage means different things in sim and bridge: bridge sends modifier-adjusted (`GetSingleDamage`), sim holds a static literal (3 stays 3 under Vulnerable) | `bridge_mod/RlCombatHandler.cs:617` vs `sts2_env/monsters/intents.py:15`; probe applied Vulnerable and re-read the intent |
| Baselines on 30 fixed seeds, Ironclad starter deck: random 40.0% ± 8.9%, greedy-damage 73.3% ± 8.1% | `scripts/eval_combat_search.py` |
| Greedy-damage ≈ trained `combat_ppo_v3` (~71%) on the same problem | `docs/MODELS.md` + the above |
| The action mask is *stricter* than the engine — index 1 is masked off for a card needing a target, but `play_card(0, None)` succeeds via auto-targeting | Probe in `tests/test_flat_mc.py::_empty_hand_slot_action` |
| HF dataset `t22000t/slay-the-spire-2-cards` joins to `CardId` at 99.8% (575/576) using the existing suffix aliases; only `FOLLOW_THROUGH` unmatched | Full id list pulled and compared against the enum |
| Qwen3 embeddings, PCA explained variance over 576 cards: 32d → 61.4%, 64d → 76.2%, 128d → 88.6%, 256d → 96.9% | SVD over the downloaded 576×1024 matrix |
| Embedding neighbours are mechanically sensible: `BODY_SLAM → IRON_WAVE, RAGE, JUGGERNAUT`; `BARRICADE → FLAME_BARRIER, DEFEND_*, IMPERVIOUS`; `WHIRLWIND → VOLLEY, SKEWER, SWORD_BOOMERANG` | Cosine nearest-neighbours on the same matrix |
| Timing: combat obs encode 14 µs, full-run obs encode 394 µs, policy forward 70 µs CPU / 55 µs CUDA (4070 Super) | Direct measurement |
| Observation encoding costs ~5.6× the network forward pass — the hashing is the bottleneck, not inference | Same measurement |
| A trained policy costs ~0.5 ms per decision: ~2 ms per turn, ~0.25 s of model time for an entire run | Same |
| Flat MC search costs 0.056–0.545 s per decision — roughly 1000× the trained policy | `scripts/eval_combat_search.py` |

---

## Known wrong — do not carry these forward

These were asserted during the session and did not survive checking. They are listed because they are the kind of thing that gets half-remembered as fact.

- **"Search is worse than greedy."** 63–67% vs 73.3% is ~0.8 sem. Not a real difference in either direction. Every measurement so far is underpowered at 30 seeds.
- **"Search turtles."** Built on a mean over a heavy-tailed distribution, almost certainly dragged by unwinnable boss fights grinding on. A traced fight played Bash → Strike → Strike → Strike → Defend → Strike and ended in 11 decisions. `MODELS.md` already warns about exactly this ("the eval mean was dominated by a heavy tail").
- **"Rollouts only win 29%."** Artifact of probing seeds 0–11 (the harder half, containing the 308 HP boss) and comparing against a 30-seed number. Random wins 50% on seeds 12–29.
- **"The battery has no elites or bosses."** False. `ALL_ACT1_ENCOUNTERS` contains 3 elites and 3 bosses out of 22.
- **"Rollouts score mid-turn so block looks worthless."** False. `turn_count` advances in `_start_player_turn`, so the existing break lands after the enemy turn.
- Any explanation of *why* search underperforms. Five were offered; none were established. The phenomenon itself is not established.

---

## Open questions

Grouped by what they block. Each needs an answer before the phase that depends on it.

### A. Representation

- **A1. ANSWERED — partial pass, with a design consequence.** Linear probe from PCA-64 to simulator metadata, held-out 20% of 544 joined cards:

  ```
  card_type   96.3% accuracy   (majority-class baseline 40.4%)
  cost        R^2 0.26
  base_damage R^2 0.34
  base_block  R^2 0.21
  ```

  Categorical mechanics are strongly recoverable — the embedding knows attack vs skill vs power vs curse vs status almost perfectly. Numeric values are only weakly recoverable. (Probes at 256 and 1024 dims returned negative R^2; that is unregularized least squares overfitting with 435 training rows against 256+ features, not a statement about the embedding. Ignore those.)

  **Consequence: embeddings replace the hashed *identity* blocks, and the existing scalar card features stay.** `observation.py` already encodes `cost`, `base_damage`, `base_block` per hand slot explicitly, so the embedding does not need to carry numbers — its job is identity and effect semantics, which hashing could not represent at all. Keep both.

  Caveat: this is a *linear* probe, so it is a lower bound. A nonlinear policy head may extract more.
- **A2. Upgraded cards — separate vector or a flag?** The dataset carries `description_upgraded`. Strike+ and Strike play differently. Options: embed both texts separately, or one vector plus an upgraded bit. *Blocks: table schema.*
- **A3. What is the OOV policy?** 24 `CardId` members have no HF entry (likely statuses/curses/patch drift), plus `FOLLOW_THROUGH`, plus any card appearing before a sync runs. Zero vector, a learned OOV row, or refuse to start?
- **A4. Relics and potions have no embeddings.** The dataset is cards only. Do they keep feature hashing, get embedded from decompiled text, or something else? Relics change deck value enormously, so this is not cosmetic.
- **A5. Frozen PCA or learned projection?** Frozen is patch-stable and is the recommendation. A learned projection might represent better but re-learns every training run and loses the zero-shot property.
- **A6. Does observation encoding actually get faster?** Predicted, since a table lookup beats hashing 500+ strings, but unverified. Current baseline is 394 µs.

### B. Deck evaluation battery

- **B1. Which encounters populate each cell?** Grid is act1/2/3 × weak/normal/elite/boss (~10 cells). Needs an explicit list per cell, at fixed seeds, with a second held-out seed set.
- **B2. Where do tier-matched decks come from?** Chicken-and-egg: judging decks needs a battery, and calibrating the battery needs decks. Bootstrap with hand-built reference decks (starter, starter+5 good cards, a deck expected to clear Act 2).
- **B3. How many seeds per cell?** Must be chosen from a target sem, not by feel. 30 seeds gives ±9% on a win rate, which is too coarse to have settled anything this session.
- **B4. Does the battery separate the reference decks in the expected order?** This is the calibration gate. If it fails, the battery is broken and nothing downstream means anything.
- **B5. What HP and relics does each cell assume?** Recommendation is full HP, no potions, deck's own relics — so deck quality and current trouble stay separable.

### C. What pilots the battery

- **C1. Is search actually better than greedy, measured with enough power on a battery that can discriminate?** Genuinely unknown. Everything measured so far is inside the noise.
- **C2. If search wins, is it affordable for bulk evaluation?** ~400 fights per deck evaluation. At greedy speed that is ~0.3 s per deck; at measured search speed it is ~75 minutes per deck. If search wins on quality it still cannot do bulk, so the shape becomes bulk-with-fast-pilot, spot-validate-with-search.
- **C3. Rollout depth, rollout policy, and leaf score are all unresolved.** The depth sweep measured nothing usable. Do not resume this without a battery that discriminates and a seed count chosen for power.

### D. What plays the game at runtime

- **D1. Trained network, or search distilled into a network, or search directly?** Speed says a trained net for live play (0.5 ms vs 0.5 s per decision). Quality is unknown.
- **D2. Does the new representation move the PPO plateau at all?** The honest prior is that representation alone will not produce a win; it removes a known tax and buys patch-stability.

### E. Cyra

- **E1. What is the gap distribution under whatever plays?** `TORN_GAP=0.15` and `OBVIOUS_GAP=0.50` are tuned for 0–1 softmax values. Search gaps ran 0.04–0.37 on a [0,2] scale. Measure the distribution, then set thresholds at sensible quantiles.
- **E2. Does milestone detection survive a run-loop rewrite?** `milestones.py` keys off room type and post-combat states. Anything that restructures the run loop must keep those observable.

### F. Operations

- **F1. What exactly does `sync_content.py` do end to end,** and what does it emit as patch notes?
- **F2. What requires a human, and how is that surfaced?** Stated requirement: minimal intervention, because mistakes will happen. Principle: make the safe choice automatically, refuse loudly when it can't.

---

## How the HuggingFace dataset is incorporated

**Decision: take the model and the technique, not the data.** The production vectors are generated here, from this repo's own authoritative sources. `t22000t/slay-the-spire-2-*` is used for validation and nothing else.

### Why not just use their vectors

This repo has already been burned by exactly this shape of mistake, and it is written down in the code:

> `sts2_env/cards/factory.py:375` — "Values come from the decompile. docs/CARDS_REFERENCE.md is prose and routing only -- it used to define these as well, which made it a third copy of every number, needing hand-edits in lockstep with the factories and the decompile. It fell behind, and since the tests read it as an oracle they stayed green against it while the simulator disagreed with the game that shipped."

> `sts2_env/cards/derived_values.py:4` — "All three had to be edited together, so of course they drifted -- and because they drifted *together* the test suite stayed green while every one of them was wrong about the live game. 4,609 passing tests meant 'the repo agrees with itself'."

Importing third-party vectors would create a fourth copy of card truth, owned by someone else, frozen at May 2026, drifting silently every patch. Worse than a stale number: a stale *vector* has no visibly wrong value, so nothing would ever look incorrect.

There is also a subtler trap. An embedding is a function of its input text. If the bootstrap vectors come from their "prettified JSON card document" and post-patch cards are embedded from our template, the two are not comparable — the space becomes inconsistent at exactly the moment it is supposed to be doing its job. **Whoever generates the text must generate all of it, forever.**

Running Qwen3-Embedding-0.6B over ~600 short texts on the 4070 takes seconds, so owning it costs essentially nothing.

### Where the text comes from

From the same authoritative sources the simulator uses, never from `CARDS_REFERENCE.md` (documented as drifted) and never from HF:

- `reference_static_metadata`: `card_type`, `rarity`, `cost`, `keywords`, `tags`, `target_type`, `has_energy_cost_x`, `max_upgrade_level`
- `derived_values` (from the decompile): damage, block, effect vars, upgrade deltas

Rendered through a **frozen template**, e.g.:

```
BASH | Attack | Basic | ironclad | cost 2 | target AnyEnemy
keywords: none
tags: none
effects: Damage 8; VulnerablePower 2
upgrade: Damage +2; Vulnerable +1
```

Structured rather than prose. That matches what the HF set did (they embedded a JSON document) and their neighbours came out mechanically sensible, so there is precedent that structured input works with this model.

Because the text is *derived* from the decompile rather than *duplicated* alongside it, it cannot drift from the simulator. A patch changes the decompile, the text regenerates, the vector regenerates.

### Artifacts, all versioned and frozen

```
data/card_embeddings/v1/
  template.py|txt      frozen text template  (changing it shifts every vector)
  mean.npy             frozen centring vector (PCA centres before projecting)
  projection.npy       frozen PCA 1024 -> 64 (never refit)
  vectors.npy          N x 64, append-only
  card_ids.txt         row order, append-only
  manifest.json        model id + pinned revision, template hash, dims, created-at
```

Four things are frozen together and must be versioned as one unit: **the model revision, the text template, the centring mean, and the PCA matrix.** Change any one and every vector moves. That is why the directory is `v1/` and not a pile of loose files.

The mean is easy to miss and fails the same way as the rest: PCA centres the data before projecting, so recomputing the mean when a card is appended shifts every existing vector by a constant. Freeze it, and appending is `(embed(text) - mean) @ projection` — a pure function that leaves every existing row bit-identical.

### What a new card costs

Adding a card is a **fine-tune, not a retrain**. The observation shape is fixed — 64 numbers per card slot, forever, whether the game ships 576 cards or 900. What grows is the lookup table (one more row), which is not part of the network. The input layer never changes, so the checkpoint loads.

This is the whole point of the change, and it is the one thing feature hashing could not give: hashing also keeps the shape fixed, but a new card lands in an occupied bucket and silently corrupts what that bucket had learned.

Switching from hashing to embeddings is itself a one-time from-scratch retrain, because the layout changes. That cost is paid once; content patches stop costing retrains afterwards.

### What HF is actually used for

1. **Validation of the template.** Embed our text, compute nearest neighbours, compare against theirs. If our `BODY_SLAM` lands near `JUGGERNAUT`/`RAGE` and our `WHIRLWIND` near `VOLLEY`/`SKEWER` as theirs did, the template is capturing mechanics. If it does not, the template is wrong and we would otherwise not have known.
2. **A cross-check on text extraction** — their `description` / `description_upgraded` against our rendered effects, as a way to catch decompile-parsing gaps.
3. **Already delivered: de-risking.** The 99.8% join, the sensible neighbours, and the PCA curve are why this approach is worth building at all.

None of that requires a runtime dependency on the dataset.

### Still open here

- **A2** upgraded variants: one vector per (card, upgrade level), or base vector plus an upgraded flag. The template already has room for `upgrade:`.
- **A3** OOV: cards with no vector yet (new content before a sync runs). Zero row, learned OOV row, or refuse to start.
- **A4** relics and potions: same treatment from their own metadata, or keep hashing. Relics move deck value enough that this needs an answer, not a default.

---

## Next session: status cards, and the exception a policy cannot learn

### 1. Statuses and curses should say what they are and what they do

Today 37 of 577 cards (6.4%) render with no mechanical line, all statuses and
curses. `WOUND` renders as little more than `unplayable`, and `BURN` renders
almost identically to it despite costing you 2 HP every turn it sits in hand.

Wanted, and deliberately simple: an explicit `status` marker plus the effect in
plain terms.

```
BURN | Status | Status | status | cost -1 | target NONE
keywords: unplayable
status: deals 2 damage at end of turn while in hand
```

The work is finding where each status's behaviour lives. It is *not* in any
`_CARD_*_HOOKS` registry -- that was checked; `BURN` appears in none of them --
so it is special-cased in the combat engine. First task is to locate those paths
and decide whether to describe them from a small explicit table (simple, needs
maintenance) or by introspecting the engine (no maintenance, more work).

Given how few cards are affected and how stable statuses are between patches, a
small explicit table is probably the right trade here -- but say so out loud in
the code, because it is the one place in this design that can drift.

### 2. The boss that rewards playing status cards

**The problem shape, which matters more than the specific boss.** "Never play
status cards" is a strong, globally-correct prior. Against this one encounter it
is wrong and fatal. A trained policy cannot learn the exception:

- the encounter appears at most once per run, and only if the agent survives
  that deep -- at a mean of 9.7 floors it essentially never sees it
- overturning a heavily-reinforced prior needs many samples, and this is the
  encounter that supplies the fewest
- being confidently wrong is the worst failure mode for the Cyra seam, because
  the top-two margin will be *wide* -- she will sound certain while dying

**IDENTIFIED: The Insatiable, the Act 2 boss.** The mechanic, read from
`sts2_env/monsters/act2.py` and `sts2_env/powers/remaining_c.py`:

1. `liquify_ground` puts `SandpitPower(4)` on the boss, targeting the player, and
   adds 3 `FRANTIC_ESCAPE` to the draw pile and 3 to the discard.
2. `SandpitPower.after_side_turn_start` decrements the counter every enemy turn.
   **At zero it calls `_kill_target` -- the player dies outright, at any HP.**
3. `FRANTIC_ESCAPE` is cost 1, `STATUS`, and -- unlike `INFECTION` and the other
   statuses -- **playable**. Playing it increments the sandpit counter by 1,
   buying one more turn.
4. Each play raises that card's own cost by 1 (`cost_increase`), so stalling gets
   progressively more expensive and the boss still has to be killed.

This is worse than "statuses are situationally good". It is **a timer you must
feed or die**, and three consequences follow:

* A policy carrying a "never play a status card" prior loses **100%** of these
  fights, at full HP, having played correctly by every rule it learned.
* **The reward function is blind to it.** Combat shaping is
  `phi = HP_WEIGHT*player_hp_frac - ENEMY_WEIGHT*enemy_hp_frac`. HP stays full
  until the instant of death, so there is no gradient warning of anything --
  neither the terminal signal nor the shaping can teach the lesson before it is
  fatal. This is not a sparse-reward problem, it is a *no-signal-at-all* problem.
* **Shallow search would miss it too.** The kill lands 4 enemy turns out, so a
  rollout depth of 1-3 rounds -- the design favoured earlier for its low noise --
  never simulates far enough to see the death. Search only solves this at depth
  >= 4-5 rounds. Concrete evidence that horizon depth is task-dependent and that
  a single fixed shallow depth has a real, lethal failure mode.

At depth 5+ with a rollout policy that sometimes plays legal cards, search solves
it immediately and without training: rollouts that play `FRANTIC_ESCAPE` survive,
rollouts that do not die. That is about as clean a signal as this project has.

**Not yet the binding constraint, and now with numbers.** The Act 1 boss sits
around floor 17 and the Act 2 boss around floor 34. `alpha` averages 9.7 floors
with a best single run of 29, so **no run has ever reached The Insatiable**. It
cannot be costing anything today. Revisit when runs clear Act 1 reliably.

**Options, honestly assessed.**

*Mod it out.* Simple, and legitimate as an explicit scoping decision -- but only
if it is modded in the same place the agent plays. Removing it from the simulator
alone is worse than doing nothing: the agent then meets it live having never
trained against it. Removing it from the real game via the bridge mod is
consistent, but means no run is a legitimate clear, which needs to be said in
`MODELS.md` rather than quietly assumed.

*Make the mechanic visible.* The embedding work already planned for relics and
powers applies here: if the boss's power is rendered as text and embedded, the
observation can carry "this enemy rewards status plays" instead of requiring the
agent to discover it from data it will never have. This does not guarantee the
agent uses it, but it moves the problem from impossible to learnable.

*Let search handle it.* A rollout policy simulates playing the status, sees the
result, and plays it correctly with **zero training and no rare-data problem**.
This is the strongest concrete argument yet for keeping search in the toolkit,
and it reframes what search is for: not a better everyday player, but the thing
that covers exceptions a trained policy structurally cannot. It is also exactly
the gap-gate case -- the trained policy is confident, search disagrees, and the
disagreement is the signal worth surfacing.

*Targeted curriculum.* Oversample that encounter during training. Cheap,
standard, testable, and worth trying before anything drastic.

**Recommendation for MVP: do not mod it out yet.** No run has reached floor 34,
so this costs nothing today, and modding would introduce exactly the kind of
simulator/game divergence this project has otherwise worked hard to avoid. Log it
as a known loss, measure how often runs reach it, revisit when they do. Keep the
mod as the fallback if it ever becomes the last thing between the agent and a
first win.

The cheap partial fix, worth doing far earlier than any of this: make
`FRANTIC_ESCAPE` render as *playable* and say what playing it does. It is already
distinguishable from every other status by being the only one with a cost and no
`unplayable` keyword -- the card text template just has to say so.

---

## Rules that must not be broken

1. **Never refit the PCA.** Fit once, commit the projection matrix, project all future cards through it. Refitting rotates the space, silently changes every existing vector, and breaks every checkpoint with no error. Same failure shape as the enum-index bug that started this.
2. **The embedding table is append-only.** New card, new row. Never reorder, never renumber.
3. **`OBS_LAYOUT_VERSION` is stamped into every checkpoint and checked on load.** Mismatch refuses to load rather than misreading.
4. **Never `copy.deepcopy` a `CombatState`.** Use `clone_combat`.
5. **No silent caps.** If a sweep drops a config or truncates a battery, it says so.
6. **Selection requires clearing 2 sem on a held-out seed set.** `run_ppo_v4` was chosen as the max of 41 noisy draws; this is the rule that prevents a repeat.

---

## Phases and gates

Each phase ends with a gate. If the gate fails, stop rather than build on it.

### Phase 0 — Cheap insurance (independent of everything else)

- `OBS_LAYOUT_VERSION` constant, stamped into checkpoints, refused on mismatch.
- Layout-pinning test asserting every block's offset and size as literals.
- Delete `sts2_env/run/rewards.md` (an accidental byte-identical copy of `rewards.py`). `rewards.png` is already gone.
- Decide and log: intent damage — fix the simulator to match the game, or the bridge to match the simulator.

Deferred by choice: the README's 92% combat claim (`MODELS.md` says ~71%). Left until the numbers are worth publishing.

**Gate:** tests pass; a deliberately shifted column fails CI.

### Phase 1 — Card embedding table

- A1 is answered (see above): proceed, keeping the scalar card features alongside the embeddings.
- Answer A2, A3, A5.
- Dimension: **64**. Best type accuracy of the non-overfit probes, numeric R^2 comparable to 32 and 128, and 76.2% explained variance.
- Offline build script: fetch → join via suffix aliases → embed anything missing locally → PCA-64 → save vectors **and the frozen projection matrix**.
- Commit the table and the matrix as versioned artifacts.

**Gate:** A1 probe succeeds; 575/576 join reproduces; a held-out card projected through the frozen matrix lands near its known neighbours.

### Phase 2 — Representation swap

- Replace the hashed card blocks in `entity_encoding.py` with embedding lookups.
- Bump `OBS_LAYOUT_VERSION`. Existing checkpoints die; that is accepted.
- Answer A4 (relics/potions) and A6 (encoding speed).

**Gate:** observation encode time ≤ 394 µs; parity tests pass; sim and bridge produce identical vectors for identical states.

### Phase 3 — Battery

- Answer B1–B5.
- Build reference decks, then the grid, then calibrate.

**Gate:** the battery orders the reference decks correctly, and the sem per cell is small enough to detect the differences it exists to detect.

### Phase 4 — Pilot decision

- Answer C1 and C2 on the calibrated battery, with seed counts chosen for power.

**Gate:** a pilot is chosen with a measured margin clearing 2 sem, or the honest conclusion is recorded that no pilot separates and the battery is what needs work.

### Phase 5 — Deck knowledge

- MAP-Elites archive over the battery. Measured synergy table. Delete the hardcoded keyword lists in `deck_features.py`.

**Gate:** the measured synergy table reproduces known-good pairs without being told about them.

### Phase 6 — Runtime policy

- Answer D1, D2, E1.
- Train, evaluate on held-out seeds, record in `MODELS.md` — including null results, with reasons.

**Gate:** clears the incumbent by 2 sem on held-out seeds, or is logged as a null result.

---

## `sync_content.py` — the one command

```
sync_content.py
  1. diff decompiled content against implemented content
  2. new/changed cards -> extract text -> embed locally (Qwen3-Embedding-0.6B, GPU)
  3. project through the FROZEN PCA matrix -> append rows
  4. bump CONTENT_VERSION; bump OBS_LAYOUT_VERSION only if the layout actually changed
  5. run parity + layout-pinning tests
  6. write patch notes: new cards, changed values, unimplemented effects, unmatched ids
  7. exit non-zero if anything needs a human
```

Everything append-only, versioned, and checked, so a mistake surfaces as a failed test rather than a model that quietly misreads the game.

---

## Out of scope for now

- Characters other than Ironclad. The architecture must not *prevent* them: character one-hot in the observation, fixed-width reserved blocks for orbs/stars/pets zeroed when unused. Note that any character can acquire another's mechanic (Ironclad can hold an Osty), so these blocks must always exist rather than being swapped per character.
- Ascension above 0.
- Multiplayer.
- Rewriting the ~121 monster factories to be clone-safe by construction. `clone_combat` contains the problem; the deeper fix is recorded in `KNOWN_ISSUES.md`.
