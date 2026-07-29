# Model ledger

What was trained, what it measured, and what game build it learned. The models
themselves live under `output/`, which is gitignored -- this file is the part that
survives, so a checkpoint on disk can still be identified a month later.

Add a row when a model is worth keeping. A model with no row is a scratch run.

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
