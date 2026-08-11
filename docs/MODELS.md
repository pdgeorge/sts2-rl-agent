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

## Archetype card picking, with search — 2026-08-07 — suggestive, not proven

The same paired A/B as below, but with `SearchAgent` playing the fights instead
of the frozen PPO. 100 pairs, identical seeds within a pair.

```
control    mean floor 12.38   median 11
archetype  mean floor 13.42   median 12

difference +1.04 +/- 0.69   =  +1.5 se
won 23, lost 16, tied 61
```

**+1.5 se does not establish the effect.** Roughly a one-in-seven chance of
landing this way by luck. Do not quote "+1 floor" as a result yet.

**The regime comparison is what this actually shows:**

```
                effect      se      tied
  model arm    +0.08     +/-0.18   208/300  (69%)
  search arm   +1.04     +/-0.69    61/100  (61%)
```

Thirteen times the effect, measured with the agent that actually runs. That is
strong evidence the first null result was measuring a regime where decks barely
exist rather than measuring archetypes. Runs reach floor 12-13 instead of 8,
non-committing runs fall from 44% to 37%, and the win/loss split goes 47-45 to
23-16. All consistent, none conclusive alone.

**To resolve:** ~280 total pairs would take this to ~2.5 se at the same effect
size. Pool a second batch from a different seed offset rather than restarting:

```bash
.venv/bin/python scripts/ab_archetype_picking.py \
    --runs 200 --combat search --time-budget 0.5 --seed 1000 \
    > output/ab_search_seed1000.log 2>&1
```

Roughly two hours. `output/` rather than a scratchpad because /tmp here is a
tmpfs and the first run's log was nearly lost to a reboot.

---

## Archetype card picking — 2026-08-07 — a null result the experiment could not have avoided

Step 10 of the Phase 5 build plan. 300 paired simulated runs, identical seeds,
differing in exactly one place: `rank_cards(offered, deck)` against
`rank_cards(offered, deck, direction)`.

```
control    mean floor 8.19   median 7
archetype  mean floor 8.28   median 7

difference +0.08 +/- 0.18   =  +0.5 se   INSIDE THE NOISE
won 47, lost 45, tied 208
```

**Read the diagnostics, not the headline.** This experiment could not have
detected an effect:

- **44% of runs committed to no archetype at all.** The logic engages after
  three non-starter cards with a clear margin, and most runs died first.
- **208 of the 300 pairs tied** -- the two arms picked the same card throughout.
  At `ARCHETYPE_WEIGHT = 0.6` only cards of similar quality reorder, which is
  ~3% of pairs across the ironclad pool.
- **The median run died on floor 7**, having made three picks.

So this shows archetype picking does not help *in a regime where decks barely
matter*. It is not evidence that it does not help.

**The cause is the combat solver, the same one that caps the meta-policy.** The
simulator fights with `combat_v3_overnight` -- 74% overall, 6.7% boss -- so runs
end around floor 8. The live agent using turn search reaches a median of **17**.
Deckbuilding decides runs at floor 17; at floor 7 you have made three picks and
died.

Pointedly, the first live run with archetype picking committed to strike-synergy
on floor 3, drafted coherently for seven picks, and beat the act 1 boss at 18
HP. That is precisely the regime this A/B has almost no data in.

**What would answer it:** the same paired design with `SearchAgent` playing
combat, so runs reach the depth where a deck exists. Roughly 50 pairs overnight
rather than 300 in an hour, and measuring the agent that actually runs.

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

---

## sim vs live — 2026-08-10 — the simulator now predicts the live game

The question the whole parity effort existed to answer, run with
`scripts/sim_vs_live.py --runs 60`: does an offline run predict a live one? If
it does, every question stops costing an hour.

```
                    SIMULATED (n=60)     LIVE (n=13)
mean floor              14.8                15.6
median                  16                  17     <- boss on 16 here, 17 live
cleared act 1        32% +/- 6%          23% +/- 12%
```

**32% against 23% is 0.7 standard errors.** Statistically indistinguishable, on
the metric that matters rather than only on mean floor.

The shape matches too, which is the part that could have failed while the mean
agreed: a heavy cluster at floor 16 (runs dying to the act 1 boss), an
early-death tail at 4-8, and clears spread out to 45. Same distribution, not a
different one with a coincidentally similar centre.

**What it cost to learn.** 157 -> 2 simulator disparities over one day: nine
wrong monster HP constants, eight wrong attack damages, Skulking Colony rebuilt
(its Inertia was modelled as a block gain where the game attacks for 9), twenty
misnamed move ids, and two dynamically-created states -- STUNNED and
REVIVE_MOVE -- that no lookup against a monster's declared states could ever
have found.

**What it buys.**

```
live session      ~1 hour     +/- 10-12%
simulated n=60    39 min      +/- 6%
simulated n=500   parallel    +/- 2%
```

This is the difference between guessing and measuring, and the cost of guessing
was demonstrated repeatedly on the day this was written: a per-room search
budget proposed and then killed by measurement (the search uses 3% of its
budget), a rollout policy called "harmful" off one session and then scoring the
best result yet in the next, and a crash blamed on our own animation patch that
reproduced at 1x. Every confident call from a single 20-run session was wrong.
Every call from a direct count -- disparities, decompile values -- held.

**Caveats.** The live arm is n=13 at +/- 12%, so this shows consistency rather
than equivalence, and the simulator reads 9 points optimistic -- inside noise,
but worth rechecking against a larger live sample. Its deep tail may also be
generous: 4 of 60 reached act 3, which the live agent has managed twice ever. A
large offline effect should still be confirmed live before it is believed. The
point is to stop spending an hour finding out there was no effect.

---

## eval weight sweep — 2026-08-10 — a negative result, and a lesson about n

> **VOID — read this first.** Both sweeps below ran with a WALL-CLOCK search
> budget of 1.0s across 14 workers, and that budget binds under contention: the
> search was truncated 19-36 times per run, and one seed finished on floor 16 or
> 27 depending on what else the machine was doing. Sweep 1 also ran alongside a
> live session and sweep 2 did not, so the two were shaped by different machine
> load and pooling them was wrong.
>
> The conclusion may still hold — nothing came close to significance, and
> truncation is more likely to add noise than to manufacture a null — but it is
> not supported by these numbers. `887c12c` sets offline budgets to 60s, which
> cannot bind. Rerun before citing anything here.
>
> Kept rather than deleted because the METHOD lessons stand and were dearly
> bought: absolute rates only compare within a seed set, paired differences are
> the comparison that survives, and n=60 cannot resolve a 5-point effect.

First use of the offline harness for a real question. Five arms, one `EvalWeights`
field each, paired on seeds, run TWICE on different seed sets because a single
offline number is no more trustworthy than a single live session.

```
arm              seeds 9000   seeds 5000     POOLED (n=120)
enemy_hp 0.35      +5.0%        +1.7%       +3.4% +/- 3.3%  (+1.0 se)
enemy_hp 0.50      -5.0%         0.0%       -2.5% +/- 3.4%  (-0.7 se)
block_unused 0     +1.7%        +1.7%       +1.7% +/- 2.4%  (+0.7 se)
turn -0.005        +3.3%        -5.0%       -0.9% +/- 2.2%  (-0.4 se)
```

**Nothing is supported.** Pooled at n=120, no arm reaches 1.1 se. The evaluation
weights are approximately right, and further tuning is not the lever -- which
redirects the boss-race work to deckbuilding, where the ceiling lives rather
than the play quality.

**`turn -0.005` is the reason the replication happened.** +3.3% on one seed set,
-5.0% (-1.8 se) on the other. Reporting either alone would have claimed an
effect with a sign chosen by which seeds were drawn.

TWO NUMBERS THAT LOOK ALIKE AND ARE NOT
---------------------------------------
**Absolute clear rate wanders about 10 points between seed sets of 60.** Baseline
is 20% on seeds 9000 and 30% on seeds 5000 -- same code, same everything. So an
absolute offline rate may only be compared against another run on the SAME
seeds. It reproduces there: `sim_vs_live` gave 32% / mean floor 14.8 on seeds
5000, and this sweep's baseline gave 30% / 14.2 on the same seeds, which is what
ruled out a harness difference between the two scripts.

**Paired differences are the comparison that survives**, because pairing removes
seed difficulty, and they are what this script reports.

WHAT n HAS TO BE
----------------
At n=60 paired the difference resolves to about +/-4%, while arms move +/-5%
between replications. A 5-point effect is therefore not detectable at n=60, and
the earlier claim that offline resolves to +/-6% was too optimistic: that figure
was the error on an absolute rate within one seed set, not the reproducibility
across seed sets.

Detecting a 5-point effect needs roughly n=240 an arm -- about 80 minutes at 14
workers, still far cheaper than the four live hours it would take to say less.

This does not overturn `sim vs live`. Both offline baselines (20% and 30%) sit
inside the live 23% +/- 12%. It does mean the agreement looked tighter than it
was, because seeds 5000 happened to be drawn first.

---

## potion rules + waterfall giant siphon — 2026-08-11 — a null result, kept as one

Live, 40 runs, `--live-search`, `output/live_journal_potionrules.jsonl`.

| | before | after |
|---|---|---|
| act 1 clear | 14.7% +/- 6.1% (n=34) | **15.0% +/- 5.6% (n=40)** |
| reached the boss | 59% | 60% |
| won the boss | 25% | 28% +/- 9% |

Two changes shipped together: Waterfall Giant's Siphon corrected from a flat 15
to the ascension-gated 10, and the potion rules of thumb (drink the rock on
sight, card generators on turn 1 of an elite or boss).

THE MECHANISM WORKED. That is what makes this worth keeping.

| | before | after |
|---|---|---|
| lost boss fights holding a potion, used nothing | 49% | **8%** |
| non-automatic potions dead in the belt | 49 | **1** |

Card generators were drunk 12 times in boss rooms and 7 in elites, the rock
twice. The hoarding the live journals showed -- Skill 10, Attack 10, Colorless
7, Power 6, Duplicator 6 dying unused across lost boss fights -- is gone.

And the clear rate did not move. So the potions were not the constraint. A
behavioural change that large producing no outcome change is stronger evidence
than the flat clear rate alone: it is not "we could not tell", it is "we moved
the thing and the number stayed".

n=40 cannot see an effect smaller than about 11 points, so a small gain is not
excluded. It is excluded as the answer to 50%.

WHAT THIS SESSION ALSO SETTLED

- The `intent_damage WATERFALL_GIANT.PRESSURE_GUN_MOVE sim=20 game=25/30` reports
  (12 of them) are NOT a bug. `_override_enemy_intent` logs the difference and
  then assigns `existing.intents = [intent]`, so the bridge's number replaces the
  simulator's before the search plans, and the rollout escalates from the
  corrected value. It is the expected noise of rebuilding the monster from base
  on every decision.
- The AnimationSpeedPatch crash test is INCONCLUSIVE, not passed. The patch was
  correctly disabled (the log lists only IsReleaseGamePatch and WaitSpeedPatch)
  and 40 runs did not crash -- but Punch Off, the trigger, appeared zero times,
  despite Underdocks being 25 of 49 boss fights. The test has not run yet.

THE GAP THAT IS LEFT

Offline, same agent, 399 runs: reach 64%, boss win **71%**, clear 46%.
Live, this session: reach 60%, boss win **28%**, clear 15%.

Reach agrees to within 4 points. Boss win is 43 points apart. Offline also
searches with max_nodes=2000 against live's 20,000, so the weaker searcher is
the one winning -- budget cannot explain it, and the environmental difference
must be larger than the raw gap suggests.

---

## siphon fix + potion rules, offline — 2026-08-11 — null, paired on 399 seeds

`scripts/measure_funnel.py`, 400 seeds each arm, act 1 variant random.

| | baseline | after |
|---|---|---|
| reach boss | 64% +/- 2.4% | 65% +/- 2.4% |
| win boss | 71% +/- 2.8% | 72% +/- 2.8% |
| clear act 1 | 45.6% | 46.9% |

**Paired difference +1.3% +/- 1.2% (1.0 se).** 376 of 399 seeds ended
identically. Waterfall Giant, the boss the siphon fix belongs to, went 62% ->
66% offline, which is the right direction and far inside the noise.

Offline agrees with live, which measured the same two changes at 15.0% +/- 5.6%
against 14.7%. Two independent measurements, one of them paired and an order of
magnitude tighter, both null. The changes were correct -- the siphon really was
wrong and the hoarding really did stop -- and neither is worth anything toward
50%.

## the sim/live boss gap, sharpened

Same agent, same act 1, both arms of both funnels:

| | offline | live |
|---|---|---|
| reach the boss | 65% | 60% |
| win the boss | **72%** | **28%** |
| clear act 1 | 47% | 15% |

Reach agrees within 5 points. Boss win is 44 points apart, and offline searches
with max_nodes=2000 against live's 20,000 -- the WEAKER searcher is the one
winning, so budget cannot explain it and the real environmental difference is
larger than the raw gap.

Every boss is over-predicted, and the three audited against the decompile today
(The Kin, Soul Fysh, Lagavulin Matriarch) were correct in constants,
transitions and move bodies. This is now the largest unexplained thing in the
project and the only lead left with 30+ points in it.

---

## search truncation — 2026-08-12 — measured, and it is not the problem

`searches_truncated` over 9 live runs: **37 of 2,379 searches, 1.6%**, hit the
3-second wall clock. Per run it ranges 0.0% to 1.8%.

That kills one of the two leading explanations for the live/offline boss gap.
Live searches under a wall clock and offline under a hard node cap, so the
suspicion was that the wide boss turns were being cut short live while offline
never was. They are not.

It also deepens the remaining puzzle rather than resolving it: live searches
essentially uninterrupted, with max_nodes 20,000 against offline's 2,000, and
still wins 28% of act 1 boss fights to offline's 72%.

What is left on the list: the same-seed diff, offline against live, on identical
maps and decks. Seeding was fixed on 2026-08-11 and has never been used for this.
