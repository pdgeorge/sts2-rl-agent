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

## LIVE — 2026-08-06 — turn search transfers to the real game

The first 20-run live session to finish all 20, and the first live evidence that
the searcher does what the benchmark says it does. Raw data and the full writeup
are kept in `docs/milestones/2026-08-06-live-search-works/` -- `output/` is
gitignored and this session is not reproducible.

Same model both sides (`combat_v3_overnight`), 20 live runs each:

```
                            baseline      --live-search
mean floor                     9.1            14.7
median floor                     8              17
reached the act 1 boss     2/20  10%      10/20  50%     +40 pts +/-13.0  (~3 se)
cleared act 1              0/20   0%       2/20  10%     +/-6.7 -- two runs
boss win rate               0/2            2/10  20%     +/-12.6
```

**The benchmark predicted the live result.** Turn search v2 scores 20% boss win
rate on the offline 200-fight benchmark (below); live it came in at 2/10 = 20%.
A change can therefore be pre-screened offline in minutes rather than in an hour
of live runs. One corroboration at n=10, not a law -- but it is the first time
an offline number here has been checked against the game and held.

**The wall moved.** 8 of the 20 died *to the act 1 boss* (Waterfall Giant x3,
Vantom x2, Kin Follower, Lagavulin Matriarch); only 6 died in hallways. Arriving
is close to solved; surviving is not.

Read the clear rate as "not zero", not as 10%. It is two runs.

---

## RNG corrected — 2026-08-06 — every number above this line is a different measurement

`sts2_env/core/rng.py` now uses the game's generator: xoshiro256\*\* seeded by
Splitmix64 from the full 64-bit seed, with XxHash64 for named streams. It
previously ran a `System.Random` clone from a 32-bit-truncated seed with the
game's deprecated string hash. See `docs/PARITY_GAPS.md`.

Enemy HP rolls therefore resolve differently for every fixture seed, so the
benchmark presents *different fights* than it did before. The fixture is the
same 200 situations -- same decks, same HP, same relics, same encounters -- but
the monsters inside them roll different HP.

**`combat_v3_overnight/final_model.zip`, re-measured on the same fixture:**

```
             win rate      hp lost        (was, pre-RNG-fix)
overall      71.0% +/-3.2     24.7        74.0%   24.5
MONSTER      84.9%           18.2         85.5%   19.1
ELITE        23.1%           47.4         42.3%   41.0
BOSS          6.7%           53.7          6.7%   53.1
floors 13-16 50.0%           34.1         54.0%   32.9
```

The headline moves 3 points and the boss row does not move at all. The elite
row halves, 42.3% -> 23.1%, which is the one worth noting: 26 fights at
roughly +/-10%, so a 19-point drop is around two standard errors and probably
real rather than noise. The likely cause is not the generator as such but
`SetUniqueMonsterHpValue` (`CombatState.cs:499`), which excludes HP values
already taken by siblings on the same side -- a rule this simulator does not
implement, and which matters most in the multi-monster fights elites tend to
be.

**Use these numbers, not the ones above them,** for anything measured after
2026-08-06. The pre-fix figures are kept because they are what the earlier
entries were written against, not because they are comparable.

---

## combat_real_situations — 2026-08-06 — trained on real situations, a null result

The Phase 3.3 experiment from `docs/GLM_ROADMAP_50P_ACT1.md`: fine-tune
`combat_v3_overnight` on 2000 harvested real situations rather than the starter
deck, on the diagnosis that the model "literally couldn't learn" Powers and
blocking because no training situation ever made them the right call.

`--situation-set act1_combat_train_2000.json` (2000 fights, disjoint from the
benchmark), `--eval-situation-set act1_combat_benchmark.json` (the held-out
200), resumed from `combat_v3_overnight`, 8 envs. Reached 1.19M of 2M steps
before the run was killed; `best_model.zip` is the checkpoint the held-out eval
selected.

Paired over the same 200 fights:

```
                  v3_overnight   real_situations
win rate             74.0%          72.5%     -1.5% +/- 1.5%   inside the noise
hp lost               24.5           24.7     +0.2  +/- 0.4    inside the noise
MONSTER              85.5%          83.0%
ELITE                42.3%          46.2%     (26 fights, +/-10%)
BOSS                  6.7%           6.7%     (15 fights -- 1 fight either way)
won only by real_situations: 3,  only by v3_overnight: 6
```

**The distribution hypothesis did not survive contact with the measurement.**
Every headline number is inside the noise, and the boss row -- the one the
retrain was for -- did not move at all. It is the same single won fight out of
fifteen.

Nor was it a matter of steps. The held-out eval reward is **flat across the
whole 1.19M**: first quarter mean +0.397, last quarter +0.428, block means
oscillating around +0.45 from the first evaluation onward. The curve never
climbed, so the missing 810k steps are not where the improvement was hiding.
The fine-tune began at its ceiling and stayed there.

**What this rules in.** Compare the turn-search rows below: 83% overall and
20-26.7% boss, against this model's 72.5% / 6.7%, on the same 200 fights.
Lookahead moves bosses; more combat training on a better distribution does not.
That points the remaining Act 1 gap at search and at deckbuilding rather than at
the combat policy's training data, which is the Phase 2.4 decision point the
roadmap already wrote down -- reached here from the training side instead.

**A caveat on the pass bar.** Phase 3.3 set "≥30% boss" as its gate. The
benchmark has 15 boss fights, so a proportion near 30% carries a standard error
of about 12 points -- the bar and the current 6.7% are roughly one standard
error apart, and no run can be said to have cleared it on this fixture. A
boss-weighted held-out set is a prerequisite for asking the boss question at
all, not a refinement of it.

---

## turn search v3 — 2026-08-05 — terminal rollouts, a null result, kept off

`--top-k 5 --rollout-samples 3`. Off by default; the code stays because the
diagnosis below is worth acting on later.

The top five lines of each turn played to the END of the fight, three sampled
futures each, scored `immediate + 0.5 * mean(terminal)`. Built to fix two things
-- bosses, and Powers never being played -- on the reasoning that both are
multi-turn and the horizon was truncating exactly where their value lives.

Paired over the same 200 fights:

```
                     v2        v3
win rate           83.0%     83.5%     +0.5% +/- 1.1%   inside the noise
hp lost             16.6      15.7     -1.0  +/- 0.3    clear
ELITE              53.8%     61.5%     (26 fights, +/-10%)
BOSS               20.0%     26.7%     (15 fights, +/-11%)
won only by v3: 3,  only by v2: 2
```

**It did not do what it was built for.** The win rate is a coin flip. The elite
and boss rows moved the right way and cannot be claimed at those sample sizes.
Cost: five times the compute, 0.57 s a turn against v2's 0.11.

**And the power test it was designed against fails outright:**

```
                  turns/fight    v2      v3
BOSS                  9.3       3.2%   4.2%
ELITE                 5.6       2.3%   3.3%
MONSTER              23.4       4.6%   4.5%
```

The rate is still flat in fight length -- the longest fights show no more scaling
than the shortest, which is the signature of not seeing the payoff at all. The
bar was "sensitive to fight length", not "a bigger number", precisely so a bias
shift could not be mistaken for a fix. It is a bias shift.

**It also lost something correct.** With rollouts on the searcher plays Strike
before Bash, throwing away the Vulnerable multiplier, because three samples
cannot resolve a three-damage difference and the noise decides which line wins.
`tests/test_search_turn.py` catches this; the test flipped with the sampling
seed, which is what a decision made by noise looks like.

**The diagnosis, which is the useful part.** Depth was never the problem. A
rollout inherits every blind spot of the policy that plays it, and the playout
ranks Powers last and plays them only when nothing else is legal -- so playing a
fight to its conclusion still never shows a Power being *used*. The same
limitation was measured independently from the deckbuilding side: with the
trained model as playout, adding Inflame to a starter deck scored +1.5 HP, i.e.
worse than not taking it.

So a competent playout policy is the prerequisite for three separate things:
terminal rollouts paying for themselves, deck evaluation by simulation being
honest about scaling cards, and Powers being played at all. That is the next
piece of work, and it was already visible in the flat-MC teacher's 9.5 floors.

The noise floor was measurable before any of this was built -- two identical decks
differ by +/-8.8 HP over 16 fights, +/-3.3 over 128 -- and it was read as a caveat
rather than as the prediction it turned out to be.

---

## turn search v2 — 2026-08-05 — two turns of lookahead

Same searcher, plus a cheap playout of two further turns blended into the score
at half weight. Against `combat_v3_overnight` on the same 200 fights:

```
                    model     search v1   search v2
win rate            74.0%       79.5%       83.0%
hp lost overall      24.5        17.3        16.7
MONSTER             85.5%       90.6%       93.7%
ELITE               42.3%       53.8%       53.8%
BOSS                 6.7%        6.7%       20.0%

paired against the model:  win +9.0% +/- 2.1% clear, hp -7.8 +/- 0.7 clear
won only by search 19, only by the model 1
```

**The boss row is the point.** v1 matched the model exactly at 6.7% because its
horizon ended with the enemies' reply, and a boss is not a one-turn problem.

**Half weight, not full, and it cost boss wins to do it.** Scored on the playout
alone the boss rate was 33.3%, and the searcher also played Strike into 12
telegraphed damage at 12 HP and died, where v1 played Defend and lived: when both
lines end in death somewhere inside a crude playout, dying now and dying in two
turns score the same, so surviving the turn stopped counting. Keeping the
immediate term at full weight makes dying now strictly worse than dying later --
which is true, and truer still given the playout is a rough policy whose
predicted deaths are not to be trusted. 20% with that property beats 33% without.

**Deeper is not better.** Four turns of lookahead scored *worse* than two (boss
13.3% against 33.3% on the elite and boss subset). The playout compounds its own
errors the further it runs, which is the argument for using `combat_v3_overnight`
as the rollout policy rather than a hand-written one -- the next thing to try.

**It still does not play Powers.** 3.6% of plays against 3.2% before, on decks
where 73 of 200 hold one. Two turns is not long enough for a scaling card to pay
off, so the searcher declines them and is arguably right to in a short fight and
wrong in a boss. Unsolved.

---

## turn search v1 — 2026-08-05 — the Phase 1 gate, passed

`sts2_env/search/`, scored with `scripts/score_combat_benchmark.py --search`.

No training. Every turn, clone the fight, play every legal ordering of the cards
in hand, end the turn on each copy, let the enemies actually act, and keep the
line that came out best. 20,000 nodes or 3 s a turn, whichever comes first.

Against `combat_v3_overnight` on the same 200 fights:

```
                    model     search
win rate            74.0%      79.5%
hp lost overall      24.5       17.3
hp lost | won        17.4        8.7
turns                10.7       13.4

MONSTER             85.5%      90.6%
ELITE               42.3%      53.8%
BOSS                 6.7%       6.7%
```

**Paired over the same fights**, which is the test that matters -- both agents
face identical situations, so pairing removes the fixture's own variance:

```
win rate   +5.5% +/- 2.4%   clear
hp lost     -7.2 +/- 0.9    clear
won only by search 17, only by the model 6
```

A 29% reduction in HP lost per fight is the result, more than the win rate. Over
sixteen floors that is the difference between arriving at the boss healthy and
arriving at it dead, which is how act 1 runs actually end.

**Bosses did not move: 6.7% for both, one win in fifteen.** The search horizon is
one turn plus the enemies' reply, and a boss is a multi-turn problem -- there is
no line inside a single turn that answers it. Honest limit of v1, and the next
thing to work on.

Two bugs were found by building this, both of which made every earlier number
worse than it needed to be. They are recorded below rather than here, because
they were never properties of a model.

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
