# Model ledger

What was trained, what it measured, and what game build it learned. The models
themselves live under `output/`, which is gitignored -- this file is the part that
survives, so a checkpoint on disk can still be identified a month later.

Add a row when a model is worth keeping. A model with no row is a scratch run.

---

## The act 1 combat benchmark — 2026-08-05

`tests/fixtures/act1_combat_benchmark.json`, scored with
`scripts/score_combat_benchmark.py`.

Every combat number above this line was measured on `STS2CombatEnv`: the Ironclad
starter deck, at full HP, against a random act 1 encounter. No run is in that
state after floor 1, which is why a model reported at 92% there dies on floor 8
of a live run. The benchmark is 200 fights harvested from real runs -- the deck as
it had grown, the HP as it had been spent, the relics collected, against the
encounter the map actually rolled.

200 situations: 50 per floor band, 159 monster / 26 elite / 15 boss, decks of
10-20 cards, HP from 16% to full, relics up to 3.

**`combat_v3_overnight/final_model.zip`, the bar Phase 1 has to beat:**

```
             win rate      hp lost
overall      74.0% +/-3.1     24.5
MONSTER      85.5%           19.1
ELITE        42.3%           41.0
BOSS          6.7%           53.1
floors 13-16 54.0%           32.9

random baseline 32.0%        44.5
```

**Read the elite and boss rows.** This model wins hallway fights and loses 58% of
elites and 93% of bosses -- 1 boss fight in 15. That is exactly where live runs
end: `output/live_eval_*.jsonl` shows deaths at elites and at floor 17, the act 1
boss. The headline 74% is not the problem; the 6.7% is.

Not comparable to the 71%/92% recorded below for `combat_ppo_v3`, which are
starter-deck numbers measuring a different thing.

---

## Everything trained before 2026-08-05 learned a game with one relic in it

Not a model row. It applies to every row below, so it goes above them.

`RunState.__init__` aliased `self.relics = self.player.relics`, and
`CombatState._build_player_state` rebound the player's list on every combat.
After the first fight the two names pointed at different lists: relics obtained
later went to the player's, `RunManager._enter_combat` fed the stale one into the
next combat, and that overwrote the player's. **No relic obtained after the first
combat survived into the next one.** No error, no log line.

So the simulator held exactly one relic for a whole run while the real game hands
out five or six by the end of act 1. Every model below trained against that, which
makes the sim-to-live gap partly an artefact rather than a mystery -- relics are
most of what makes a real act 1 survivable, and the deckbuilding and routing
policies were learning without them.

Fixed 2026-08-05; `RunState.relics` is now a property over the player's list and
cannot come apart. Pinned by `tests/test_relic_persistence.py`, four of whose
five cases fail on the old code. **Every number below predates the fix.**

---

## meta_ppo_v1 .. v7 — 2026-08-01/02 — trained with almost no reward signal

`output/meta_ppo_v{1..7}/`. Hierarchical meta-policy: non-combat decisions only,
combat fast-forwarded by a solver.

```
        evals   steps      first    last     best
v1       46      920,000    3.37     4.20    4.55
v2       19      380,000    3.34     3.40    4.34
v3       15      300,000    3.02     3.57    4.18
v4        2       39,984    2.78     2.78    2.93
v5        3      299,988    3.24     3.24    4.28
v6        7      999,960    3.24     3.55    4.41
v7        2    1,207,608    3.92     3.92    3.99
```

**Why they went nowhere**, found 2026-08-05 by reading rather than by training:
`HierarchicalRunEnv.step` fast-forwards combat by calling `run_env._step_combat`
directly, which bypasses the reward block in `run_env.step`. `CombatSolver.solve`
returns a hardcoded `0.0`, and floors gained inside a fast-forwarded combat are
never credited either, because the next step's `floor_before` has already moved.

The meta-policy's entire reward was therefore the terminal +/-  payout and the
card-reward shaping term. No combat win, no elite, no boss, no floor. It was
asked to learn deckbuilding and routing from a signal that arrives once, at the
end, hundreds of steps later.

Compounding it, `HeuristicCombatSolver` picks the highest-damage card and **never
plays block**, so the world these policies were optimising for was one where
fights go badly for reasons no meta decision could affect.

Do not resume any of these. The environment they trained in has to be repaired
first.

---

## combat_ppo_v4, v5, combat_v3_overnight — 2026-08-01/04

```
                     evals   steps         first    last    best
combat_ppo_v4         201     2,010,000    -0.68   -0.22    0.83
combat_ppo_v5          11       219,912    -0.42    0.07    0.67
combat_v3_overnight   160    40,000,000    -0.01   -0.11    0.55
```

`combat_v3_overnight` is the one still in use, and the one benchmarked above.
**40M steps with a flat evaluation curve** -- first three evals -0.01, last three
-0.11. Whatever it knows, it knew early; the remaining 39M steps bought nothing
measurable. That is the same shape as `run_ppo_v4` below, and the same lesson:
more steps at the same strength do not help.

Trained on starter-deck-at-full-HP only (`STS2CombatEnv.reset`), which is why its
benchmark numbers on real situations are so uneven -- 85.5% on hallway fights,
6.7% on bosses.

---

## run_ppo_v2_6m — 2026-08-03 — unloadable against the current build

`output/run_ppo_v2_6m/`. 6M steps, eval 4.77 -> 4.17, best 5.65.

Kept only as the record of a failure mode worth not repeating: it was trained
against observation layout v2 (2586 dims) and the current build is v3. Loading it
raises `ObservationLayoutMismatch` -- see `output/crash_log.json`. The layout
fingerprint doing that is working exactly as intended; the model is simply gone
unless the tree is checked out at the revision that produced it.

This is the cost a learned policy carries and a searcher does not.

---

## teacher / student — 2026-08-05 — flat MC teacher, distilled

`output/teacher_data.npz`, `output/teacher_16.npz`, `output/student_v{1,2}.pt`.

Flat Monte Carlo search used as a teacher, then distilled into a network.
46,460 decisions from 150 runs, 309.7 per run, **44,067 of them in combat**.
Teacher mean floor **9.5**.

That number is the interesting one: it is the same wall `alpha` (9.7) and
`run_ppo_v4` (9.5) hit by a completely different method. Flat MC with random
rollouts is a weak evaluator for a card game -- a random policy discards its own
block and misplays its own hand, so every branch scores about equally badly -- and
flat MC has no tree, so it cannot find *sequences*, which is where most of the
value in a turn lives. It is evidence about that specific search, not about
search.

---

## run_ppo_v4 — 2026-07-30 — a null result, kept as one

`output/run_ppo_v4/best_model/best_model.zip`

**Learned nothing.** Recorded so the same resume is not attempted again.

| | |
|---|---|
| training | 20,499,672 steps, ~3 h, resumed from `output/alpha` |
| observation | 277 dims |
| actions | 157 |
| selection | `best_model`, saved at ~4M steps |

Eval reward across 41 evaluations, 100 episodes each:

```
first 8 evals   4.94
last 8 evals    4.90        change -0.03  (-0.1 sem)
range      3.68 - 5.53      spread 3.1 sem -- entirely eval noise
linear trend    -0.05 reward over the whole 20.5M steps
```

200 deterministic runs, against alpha on the same measure:

```
             mean   median   max   wins
alpha         9.7      8      29     0
v4 best       9.5     10      19     0
```

**Why it stalled.** `--lr` was silently ignored on `--resume-from`: `load()`
restores the checkpoint's own rate, so this ran at alpha's original 3e-4 rather
than a fine-tuning rate. Over the whole run `approx_kl` held at 0.037 and
`clip_fraction` at 0.20 -- both high -- while entropy stayed flat at -0.79 and
`explained_variance` sat near 0.87. Large updates every step, landing back where
they started: thrashing at a plateau rather than refining it. Fixed in
`f43a45f`; `--lr` now applies on resume.

**`best_model` here is a noise artifact.** It was selected as the max of 41 noisy
evaluations whose sem is 0.59, and the max of 41 such draws sits ~1.8 sem above
the mean by construction. 5.53 against a 4.9 baseline is exactly that. It is not
a better policy, which is why its floors are no better than alpha's.

**The one useful number.** A win is 3 acts x 15 rooms + 3 bosses, ~48 floors. A
mean of 9.5 floors implies ~0.895 per-floor survival, so P(win) is about 0.5% --
one win per ~200 runs. Reaching 1-in-20 needs 0.939, and a coin flip needs 0.986.
The gap is multiplicative over 48 floors, which is why small per-floor gains
matter far more than they look, and why more steps at the same strength do not
help.

---

## alpha — 2026-07-30

`output/alpha/alpha_model.zip`

**First model that decides card rewards, map routes and rest sites itself**, and
the first to clear act 1. Kept because it is the first, not because it is good.

| | |
|---|---|
| training | 14,721,024 steps, ~11 h, from `output/run_ppo_v3` |
| observation | 277 dims (combat 131 + run-level 20 + choices 126) |
| actions | 157 |
| game build | `6d971d7b` / `0.1.0+c8c577f6` |
| selection | `best_model` over 100 eval episodes |

Measured over 200 deterministic simulator runs:

```
floors   mean 9.7   median 8   max 29      (blind predecessor: 8.9, max 16)
died 182, timeout 18

  1-5     57 | 6-10  58 | 11-15  28 | 16-20  56 | 21-30  1
```

**Read the shape, not the mean.** 9.7 against 8.9 looks like noise. The cluster of
56 runs at floors 16-20 is the result: that is the act 1 boss and the act 2
boundary, and the blind model's best single run was floor 16. Roughly a quarter of
runs now clear act 1.

Still dies in act 1 or early act 2 four times in five. A win is 3 acts, ~50 floors.

**Requires the mod from 2026-07-29 or later.** It reads `act_floor`, `ascension`,
`max_potion_slots` and `room_type`; an older mod leaves them zero and the model
misreads the run without any error.

**Inefficiently trained**, worth recording so it is not repeated: those 11 hours
produced 14.7M training steps and 34.2M *evaluation* steps. `--eval-freq 20000`
with `--eval-episodes 100` at ~465 steps an episode means each evaluation costs
46,500 env steps against 20,000 of training -- more compute spent measuring than
learning. Use `--eval-freq 500000` for full-run training; combat episodes are ~25
steps so the default is fine there.

---

## combat_ppo_v3 — 2026-07-29

`output/combat_ppo_v3/final_model.zip`

Combat only. Plays fights; card rewards and map are left to the heuristics in
`agent_runner`, so a live run stalls around floor 8 regardless of how well it
fights.

| | |
|---|---|
| training | 500,000 steps, 176 s |
| observation | 131 dims |
| actions | 115 |
| game build | `6d971d7b` |

~71% win rate on act 1 encounters in the simulator; 82% on the script's own final
evaluation, which uses one fixed seed range and reads optimistically. Use
`final_model`, not `best_model`: selection at that point was noise-limited and the
two are indistinguishable over 300 episodes.
