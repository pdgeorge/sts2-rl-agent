# SCOREBOARD

One number decides whether work counted: **pooled live clear rate, with n and a 95% confidence interval.**

Not offline. Not "explained a gap". Not "shipped a change". Those can all be true while this number sits still, which is exactly what happened in the week of 2026-08-10.

## The number

| date | n | reach boss | win \| reach | **clear** | what changed |
|---|---|---|---|---|---|
| 2026-08-13 | 496 | 48.0% +/- 4.4 | 26.5% +/- 5.6 | **12.7% +/- 2.9** | baseline, every live journal to date |
| 2026-08-14 | 89 | 58.4% +/-10.2 | 25.0% +/-11.8 | **14.6% +/- 7.3** | the subset that actually ran the SEARCH |
| 2026-08-14 | 410 | 45.4% +/- 4.8 | 26.9% +/- 6.4 | **12.2% +/- 3.2** | the subset that ran the v3 TRAINED MODEL |
| 2026-08-15 | 100 | 66.0% +/- 9.3 | 30.3% +/-11.1 | **20.0% +/- 7.8** | `live_journal_overnight.jsonl`, the standing baseline |
| 2026-08-16 | 89 | 59.6% +/-10.2 | 32.1% +/-12.6 | **19.1% +/- 8.2** | `run100_2`, after the playout rewrite and the Endless Conveyor fix |
| 2026-08-16 | **189** | 63.0% +/- 6.9 | 31.1% +/- 8.3 | **19.6% +/- 5.7** | **pooled, both 100-run sessions of the current agent** |

| 2026-08-17 | 100 | 68.0% +/- 9.1 | 45.6% +/-11.8 | **31.0% +/- 9.1** | `postfix`, the bridge power-hook fix (`policy_version` v001, git sha 6b2c141 + uncommitted tree) |
| 2026-08-17 | 100 | 75.0% +/- 8.5 | 58.7% +/-11.1 | **44.0% +/- 9.7** | `boss_telemetry`, sha 2e4cc4a. Same act 1 play as `postfix` |
| 2026-08-17 | **200** | 71.5% +/- 6.3 | 52.4% +/- 8.2 | **37.5% +/- 6.7** | **POOLED, the agent since the power-hook fix** |

### The power-hook fix is confirmed, and it is the biggest move this project has measured

| | n | reach | win \| reach | clear |
|---|---|---|---|---|
| before the fix (`overnight` + `run100_2`) | 189 | 63.0% +/- 6.9 | 31.1% +/- 8.3 | **19.6% +/- 5.7** |
| after (`postfix` + `boss_telemetry`) | 200 | 71.5% +/- 6.3 | 52.4% +/- 8.2 | **37.5% +/- 6.7** |
| | | z=1.79, p=0.073 | **z=3.48, p=0.0005** | **z=3.90, p=0.0001** |

**Win-given-reach is what moved, and reach is not resolvable.** That is precisely the signature a boss-execution fix should leave, and it is the shape prediction 9 was written against. Prediction 9 asked for +2 to +5 clear points and got **+17.9**.

The two post-fix sessions differ by 13 points (31.0% against 44.0%, z=1.90, p=0.058) on **identical act 1 decision code** -- the only changes between them were act 3 parity constants and instrumentation. That difference is variance, and it is a standing reminder that a single 100-run session still carries +/-9: neither 31% nor 44% is the number, 37.5% +/- 6.7 is.

**The Waterfall gate, re-tested at n=200 and no longer flat**, though still not resolved on its own:

| | pre-fix (44 fights) | post-fix (23 fights) |
|---|---|---|
| Waterfall win rate | 32% | **48%** (z=1.29, p=0.198) |
| eruption reached -> run died | 58% | **42%** |
| zero-block eruption turns | 23/34 = 68% | 11/19 = 58% |

All three move together and none resolves alone, which is what an under-powered but real effect looks like. The headline result is what carries the fix; the per-boss attribution stays provisional.

`run100_2` is not distinguishable from the baseline it followed: clear z=0.16 (p=0.88), reach z=0.92 (p=0.36). It stopped at 89 of 100 on a manual interrupt at 19:28, not on exhausted restarts.

**`postfix` is the first session since 2026-08-15 that moved the number.** Against the pooled 189-run baseline: clear +11.4 points, **z=2.18, p=0.029**; win-given-reach +14.5 points, **z=1.98, p=0.047**; reach flat at +5.0, z=0.85, p=0.39. The shape is the one a boss-execution fix should have -- the win half moves and reach does not.

Two cautions on it, both required by the rules above. The session summary quotes `31.0% +/- 4.6%`, which is **one standard error**; the 95% interval is **+/- 9.1** (Wilson 22.8 to 40.6) and that is the number this file takes. And p=0.029 is one session against pooled history, at an n that resolves nothing finer than ~9 points -- it clears the n>=100 bar and does not clear the ~290 needed to call 31% to +/-5. The pooled row is the honest headline.

## Target

**25% clear by Sunday 2026-08-16.**

`clear = reach x win`, so 25% needs roughly **62% reach x 40% boss win**. Neither half alone reaches it: routing at 62% reach with today's 26.5% win is 16%; today's 48% reach with a 40% win is 19%. Both have to move.

## The rule

Every change gets a prediction written **before it is built**, in the table below. After measurement the row is marked HIT or MISS, and a MISS stays in the table.

This exists because the failure mode is mine and it is documented: quoting a favourable subset as the rate, and calling a null a win after the fact. A prediction written afterwards is not a prediction.

| # | change | predicted effect | measured | verdict |
|---|---|---|---|---|
| 1 | shop card removal, curses -> worst-card | +2 clear points | pending | — |
| 2 | reconstruction audit fixes | unknown until the audit runs; predict before fixing | audit clean at the boss (0/60 wrong); no fix needed | **CLOSED, no change** |
| 4 | `--live-search` on by default | reach 45.4% -> 58.4% (the rate search already achieves on the 89 runs that used it) | pending | — |
| 3 | map routing with lookahead (`map_planning.py`) | reach 48% -> 62% live; offline reach 54% -> 65% | offline reach 60.7% -> 56.0%; paired net -2, p=0.89; best of 5 variants +0, p=1.00 | **MISS** |
| 5 | demote `elite` below `monster` in `ROOM_PRIORITY_HEALTHY` | reach 66% -> 72%, clear 20% -> 25% | never measured -- **withdrawn and reverted before it ran** | **WITHDRAWN, wrong direction** |
| 6 | playout policy priced in HP (`_heuristic_playout_action`) | power-play rate stops being flat in fight length and rises with it; damage per elite fight falls; live clear 20% -> 28% | offline paired A/B, 150 situations: wins 134 -> 134, damage/fight 13.03 -> 12.69, power rate 0.82/1.21/1.10/1.71% -> 1.30/1.62/1.01/1.23% by fight length. Power-holding decks only (n=73): wins 54 -> 54, damage/fight 19.79 -> 19.23, power rate 7.25/4.28/4.11/5.41% -> 10.96/4.66/4.39/4.62% | **MISS on the behavioural gate** |

| 7 | multi-enemy target selection: does the searcher take an available kill on the enemy telegraphing the most damage? | **behavioural gate, no change yet.** Predict the kill is MISSED in >= 30% of positions where it is available and affordable, against 8/8 taken in the verified single-enemy case; and predict the miss rate RISES with enemy count, because `evaluate` scores `kill` as `0.10 * dead/len(killable)` so one corpse out of three is worth a third of one corpse out of one. If the kill is taken >= 90% of the time and shows no trend in enemy count, the hypothesis is dead and nothing gets built. | `scripts/probe_multi_enemy_kill.py`, 124 positions over 31 encounters: kill taken **92.7%** with several bodies on the table against 96.0% for the same victim alone. By enemy count 92.2% / 91.7% / 100.0% at 2 / 3 / 4 -- no trend, and the 4-enemy cell is perfect | **MISS, on both halves** |
| 8 | the same question with the kill and the block made mutually exclusive: 1 energy, so Strike-the-threat and Defend cannot both be played | 7 gave the searcher 3 energy against a 5 HP victim, so it killed AND blocked and never had to choose -- it measures "does it notice a free kill", not the valuation. Predict the kill is taken in < 60% of positions once it costs the block, because `evaluate` is the only judge of the leaf and it prices a kill's future through `player_hp` over a 3-turn window while a block is banked immediately and certainly | same 124 positions at 1 energy: kill taken **91.1%**, against 93.5% solo. By enemy count 93.8% / 85.4% / 100.0% | **MISS** |
| 9 | bridge power reconstruction returns the registry class (`situation._bridge_power_instance`), so hooked powers -- first among them Waterfall Giant's STEAM_ERUPTION detonation -- exist in every live-search lookahead | the mid-fight clone installed bare `PowerInstance`s, so the eruption never fired in search and every kill on the giant scored a clean win. Predict: **zero further live deaths where the killing blow on Waterfall Giant is followed by lethal eruption damage** (run 7 sunday died exactly this way at 37 HP against a ~37 bank); Waterfall's live boss record lifts off 4/28; pooled clear +2 to +5 points depending on boss incidence. Prediction written before the fix is measured live; the offline behavioural gates (search holds a fatal kill, takes a survivable one) already pass | **clear 19.6% -> 31.0% +/- 9.1 (z=2.18, p=0.029), inside the predicted direction but 2x the predicted size.** Both behavioural gates FAIL: 5 further eruption-phase deaths, not zero (56% of eruptions against 58% pre-fix), and Waterfall's record is 4/12 = 33% against a pooled pre-fix 14/44 = **32%**, z=0.10, p=0.92 -- the "4/28 = 14%" baseline the prediction was written against does not reconcile with the journals | **MISS on both gates; the clear rate moved and the named mechanism did not** |

| 10 | boss counterfactuals (`scripts/boss_counterfactuals.py`): replay the lost act 1 boss arrivals under arms that change either how she PLAYS the fight or what she ARRIVED with | **Written before the run.** Predict `think_10x` converts **fewer than 20%** of the positions the baseline loses on every reshuffle -- i.e. most losses are arrival, not search depth, and the 3s budget is not what is costing them. Predict the arrival arms (`hp_plus_15`, `minus_2_basics`) each convert **more** positions than any play arm. If `think_10x` clears 20%+, search budget becomes a live lever and the measured 42%-at-full-HP is a play ceiling we can actually reach | 84 arrivals x 7 arms x 6 reshuffles, 3528 replays. `think_10x` rescued **1 of the 11** positions the baseline loses outright = **9%**, and moved the pooled rate **+1.8 +/- 2.4 (not resolvable)**. Arrival arms did NOT each beat every play arm: `hp_plus_15` rescued 2 and `minus_2_basics` 1, against `lookahead_4`'s 2 | **HIT on the ceiling, MISS on the ordering** |

| 11 | HOLD the fight-ending potions for elites and bosses (pd's call): `PowderedDemise`, `DistilledChaos`, `GigantificationPotion`, `OrobicAcid`, and `Duplicator` | **Written before the build.** Measured over the 200 post-fix runs, those five are drunk on hallway trash **82%** of the time (50 of 61) -- `PowderedDemise` 89% and never once on a boss -- against **12%** for the four already forced into big fights. Behavioural gate: trash use of the held group falls below **25%**, and potions held entering the act 1 boss rise from 0.9 to above 1.3. Outcome: clear **+0 to +3 points**, and I am explicitly allowing this to come back NEGATIVE -- holding a fight-ender through a hallway means eating the chip damage it would have prevented, and prediction 10 priced 15 HP at +4.6 boss-win points. If the behavioural gate passes and clear does not move, the potions were worth more as chip prevention than as boss damage, and that is a real answer | `potion_fix_2`, n=91: trash use **85% -> 12%** (gate 1 PASSED hard), potions held entering the boss **0.99 -> 0.97** (gate 2 FAILED, unmoved). Clear 37.5% -> **25.3% +/- 8.9**, z=-2.05 p=0.041 | **SPLIT live; NULL offline. Closed.** |

**Settled offline, 400 paired seeds:** the hold is worth **+0.75 points, 95% CI -1.0 to +2.5**. A clean null -- and the interval **excludes the -12.2 the live session suggested**, so that drop was session variance, exactly as the identical-code 31%/44% pair warned it would be.

| | v001 (no hold) | v002 (hold) |
|---|---|---|
| clear | 45.0% +/- 4.9 | 45.8% +/- 4.9 |
| reach | 63.2% +/- 4.7 | 64.2% +/- 4.7 |
| potions at boss | 0.87 | 0.93 |

Paired: gains 8, loses 5, **13 discordant pairs of 400** -- and the discordant count, not n, is what sets the resolution.

**Why it is a null is the useful part.** The arm altered **43 of 1467 potion decisions, 2.9%, or 0.11 per run**, and 86% of seeds ended on the identical floor. There was never enough of it happening to move a rate. Live, the same change moved 0.34 decisions per run -- three times the offline exposure -- which is worth knowing before offline is trusted on a potion question again.

**A flaw in my own instrument, stated because it limits the claim.** `ab_potion_hold.py` counts potions leaving the belt, so its where-they-are-drunk columns cover EVERY potion, not the five the arm holds. Live measured only the five (85% -> 12%); offline's 68% -> 65% is that signal diluted across every potion in the game and is not the same quantity. The OUTCOME is sound -- paired seeds, one difference between arms -- but this run cannot confirm the behavioural gate the way the live session did.

**Verdict.** The revert stands, and on gate 2 rather than on the rate: a hold that moves potions from trash to elites without changing how many reach the boss has not tested its own hypothesis. Offline now says it costs nothing either. Both readings agree it is not a lever.

| 13 | break EXACT evaluator ties toward concentrating damage instead of toward enumeration order (`v003_tiebreak_focus`) | **Written before the build.** 32.7% of live combat decisions (3,457 of 10,556, `wednesday`) have the chosen line exactly tied with another, and **100%** of those take whichever line the enumerator emitted first -- a third of every combat decision settled by iteration order. Offline reproduces the rate at 28.0% (854 of 3,048), which is the first thing that makes this measurable. **Correction, recorded because it limits the offline half of this row:** that 28.0% and the split below were measured by a probe that passed `None` as the agent to `_search_combat_action`, which raises, returns `None`, and falls through to `rng.choice(valid)` -- so the positions were reached by a RANDOM combat player, not by the searcher. The live 32.7% is real agent play and is unaffected, and the two rates being close is mild reassurance, but the 89/11 split is being re-measured under agent play by `ab_tie_break.py --gate`. The A/B itself passes a real agent and is not affected. But the leaf snapshot says most of that is not a defect: **89.0% of ties end in a genuinely IDENTICAL position** and scoring them equal is correct. Only **11.0%** are different boards, and 89 of those 94 are the same POOLED enemy HP with the damage spread differently -- `evaluate.py` scores enemy HP as one pooled fraction, so focusing a kill and smearing the same damage are worth the same to it. **0** of the 854 differ in enemies killed and **0** differ in player HP by 3 or more within the horizon. So the exposure is ~3% of combat decisions and the payoff, if any, lands BEYOND the 2-turn horizon as a kill arriving a turn sooner. Predict clear **+0 to +2 points, and I expect a NULL** -- prediction 11 taught that a 2.9%-exposure arm cannot move a rate. Behavioural gate: the arm changes the chosen line in **2-5%** of combat decisions, and mean turns-to-first-kill in 3+ enemy fights falls. If the gate passes and clear does not move, tie-breaking is confirmed dead and the pooled `enemy_hp` term is the thing to argue about instead | pending | — |

| 12 | `tuesday`: the potion hold reverted, plus the Lagavulin wake fix | **Predict NOTHING MOVES.** Reach and clear both unchanged within error against the pooled 34.5% +/- 5.0. The hold is measured null offline (+0.75, CI -1.0 to +2.5) and the Lagavulin fix is worth ~0.5-2 clear points -- SLASH is 19 damage against DISEMBOWEL's 9, so the search under-blocked by 10 on the wake turn, but that boss is only 23% of act 1 boss fights. A 100-run session resolves +/-9 and needs +15 to see reliably, so it CANNOT see this fix. The session is run for pooled n and for clean transcripts, not to test anything. If clear lands outside 25-44% something unexpected happened and it is worth chasing | **reach 86.0% +/- 6.8, win|reach 57.0% +/-10.5, clear 49.0% +/- 9.8.** Clear vs pooled z=+2.63 p=0.008; reach vs pooled z=+2.92 p=0.003 | **MISS -- and by its own terms, worth chasing** |

### 12: the highest session on record, with no mechanism to explain it

`tuesday` (sha 5eeec36, policy v001) came in at **49.0% clear and 86.0% reach**, well outside the 25-44% band this prediction named as the surprise threshold. Verified rather than assumed: 100 eval rows, and an independent recount straight from the journal's `run_end` events also gives 49/100.

**Nothing shipped explains it, and reach is the reason to say so.** The Lagavulin wake fix is boss-only and cannot touch whether a run survives to floor 17, yet reach moved MORE than clear did (+14.4 against +14.5, and reach is the more significant of the two at p=0.003). The potion hold was already reverted, and the two earlier `v001` sessions reached 68% and 75%. Search failures were 0, but they were also 0 in `postfix`, which cleared 31%. The game build is unchanged since 2026-08-14 and no update was detected.

**So it is either a real jump with an unfound cause, or the high excursion again.** The series now reads **31.0, 44.0, 38.9, 25.3, 49.0** with no trend, and this file has already established that two sessions on identical act 1 code differ by 13 points. A single session cannot separate those, which is exactly the rule that made prediction 12 a null in the first place.

Pooled over all five: **clear 37.8% +/- 4.5, reach 74.8% +/- 4.0 (n=445)** -- and that is the number, not 49%.

**What settles it:** a second session at the same sha. If `tuesday` was an excursion the next one regresses toward 38%; if something real landed, it holds near 49% and the pooled figure climbs. Nothing should be built on 49% until then.

### 11: the hold works, the potions still do not reach the boss, and the session cannot settle it

The mechanism did exactly what it was built to do. Trash use of the held five went **85% to 12%** -- Powdered Demise 89% -> 14%, Duplicator 94% -> 0%.

And it bought nothing at the boss. **Potions held entering the act 1 boss: 0.99 before, 0.97 after.** They were not saved for the boss; they were spent one room earlier, on elites. Elite use of the group went from 7 to 13 in half as many runs.

The clear rate fell 12 points at p=0.041, and every link of the chain that would explain it moves the right way and **not one of them resolves**:

| | before | after | t |
|---|---|---|---|
| monster damage per fight | 5.17 | 5.69 | +1.09 |
| elite damage per fight | 26.59 | 28.24 | +0.70 |
| boss arrival HP | 86.0% | 82.6% | -1.44 |

Elite damage went UP while elites received more potions, which is the opposite of the story that would make this a clean causal finding.

**Why this session cannot decide it.** n=91 resolves +/-8.9. The effect is 12 points. And this project has already measured **31.0% against 44.0% on identical act 1 code** (`postfix` vs `boss_telemetry`, p=0.058) -- a 13-point swing with nothing changed at all. The observed session-to-session variance is the same size as the effect being tested, so a single live session is the wrong instrument for a question this fine, whichever way it lands.

Pooled over all three post-power-fix sessions: **33.7% +/- 5.4 (n=291)**, which is the number that survives.

**The honest reading is that gate 2 is the finding.** A hold that moves potions from trash to elites without moving the count at the boss has not tested the hypothesis it was built for. Whether holding them *through* the elite is better is a different change and a different prediction.

### 10: MISPLAYS ARE NOT THE PROBLEM. This is the most useful null this project has bought.

| arm | offline rate | paired vs baseline | better / worse / tied (of 84) |
|---|---|---|---|
| baseline | 69.8% | — | — |
| `think_10x` | 71.6% | +1.8 +/- 2.4 | 11 / 5 / 68 |
| `lookahead_4` | 68.7% | -1.2 +/- 3.3 | 15 / 14 / 55 |
| `rollouts_on` | 72.1% | +1.4 +/- 3.1 | 16 / 13 / 54 |
| **`no_potions`** | 57.7% | **-12.1 +/- 4.8** | **1 / 25 / 58** |
| **`hp_plus_15`** | 74.4% | **+4.6 +/- 2.4** | **17 / 2 / 65** |
| `minus_2_basics` | 71.6% | +1.8 +/- 3.5 | 19 / 14 / 51 |

**Not one play arm is resolvable.** Ten times the thinking, twice the horizon, and rollouts switched back on all sit inside their own error bars. If the 42%-at-full-HP losses were misplays, more search would convert them, and it does not -- `think_10x` changes 16 positions of 84 and ties on 68. **The remaining boss losses are not a search problem**, which retires search depth, `lookahead_turns` and `DEFAULT_TOP_K` as levers for 50% and means the 49%-of-decisions-are-ties finding is a curiosity rather than a lead.

**Two arms did resolve, and both are arrival, not play:**

- **Potions are worth 12.1 points.** Switching them off is the single largest effect in the grid -- 1 position better, **25 worse**. The potion rules are load-bearing, which is the opposite of the `WEEKEND_DECISIONS.md` section 1 reading that potion use was "currently backwards". They are carrying the boss fight.
- **15 HP is worth +4.6 +/- 2.4**, 17 better against 2 worse. pd's chip-damage thesis, priced on real lost positions rather than on a correlation, and it is clean.

**And 6 of the 11 outright-lost positions are rescued by nothing at all.** Those were decided before the fight began.

Read together: the boss fight is played about as well as this searcher can play it, and the remaining act 1 gap is in what she brings to it -- HP, potions, deck. `PHASE_TWO.md` section 2.1's pattern holds once more, from the other side: tuning the search is the sixth, seventh and eighth null, and the two things that moved are both resources.

One data hygiene note: `TEST_SUBJECT` appears as a boss lineup in 28 captured states. A test monster reaching a live act 1 boss room is either an event encounter mislabelled `Boss` or real content named like a fixture, and it is worth a look before it contaminates a per-boss table.

### 9: the fix is real, the gate is flat, and the attribution is unproven

The fix itself is correct end to end, verified rather than assumed. The bridge really does report the power (`STEAM_ERUPTION_POWER`, 38 of 50 captured Waterfall states), `_coerce_power_id` maps it to `PowerId.STEAM_ERUPTION`, and `_bridge_power_instance` returns a `SteamEruptionPower` with `after_death` overridden and the amount verbatim. The pre-fix death is in the journal exactly as described -- sunday, giant on 2 HP at round 11, killed, round 12 opens with the giant at 999999991, the agent plays UNRELENTING into it, ends the turn on **zero block**, and dies 80 -> 0.

What did not happen is the change that was predicted to follow:

| | pre-fix (pooled, 44 fights) | postfix (12 fights) |
|---|---|---|
| Waterfall act 1 record | 14/44 = **32%** | 4/12 = **33%** |
| fights reaching the eruption | 33 | 9 |
| ... that killed the run | 19 = **58%** | 5 = **56%** |
| zero-block eruption turns | 23/34 = **68%** | 6/9 = **67%** |

Every column is flat. So the +11.4 clear points are **not attributable to the Waterfall mechanism**, and this is the shape of prediction 6 -- a plausible story with a dead behavioural gate. Three readings survive and they are not separated yet: the fix pays through *all* hooked powers in every fight rather than through this one boss (the change restores every `after_death` in `powers/monster.py` and every hooked player power, so it would show broadly and not in the Waterfall column); or the gain rides on something else in a large instrumentation diff; or p=0.029 on one session is the false positive this file keeps buying. n=9 eruptions cannot resolve any of it.

**The instrument that would resolve it broke in the same session.** `postfix` captured **0 Boss states** (203 total) against `sunday`'s **324** (3931 total). `RawCapture` opens its file with `"w"` and is reconstructed inside `run_agent`, so **every `--restart-on-crash` relaunch truncates the capture and only the final segment survives** -- `postfix` restarted 4 times and its last segment was one run. The journal and eval log append across restarts; the capture does not. `sunday` only worked because it ran crash-free in a single segment. At the measured ~1-crash-in-9 rate this destroys the Track A capture on essentially every 100-run session, silently.

### The `postfix` disparities: four real parity bugs, and none of them touch act 1

Fixed 2026-08-17, every value quoted from `decompiled/MegaCrit.Sts2.Core.Models.Monsters/` rather than fitted to the observed telegraph:

| subject | was | game | what it was |
|---|---|---|---|
| `NOISEBOT` / `STABBOT` / `ZAPBOT` HP | 23-28, tough 24-29 | **18-23, tough 19-24** | every bound +5, with our MIN sitting on the game's MAX -- a window transcribed one step up |
| `TORCH_HEAD_AMALGAM` opening move | `TACKLE_1_MOVE`, 18 damage | **`STRONG_TACKLE_MOVE`, 26** | the state id the bridge sent did not exist here, and the act 3 boss's opening hit was under-modelled by 8 |
| `TORCH_HEAD_AMALGAM` deadly damages | tackle 19, weak 15 | **22, 16** | guessed as base+1 |
| `THE_FORGOTTEN` Dread | flat 15 | **13 + its own Dexterity** | `DreadDamage => base + GetPowerAmount<DexterityPower>()`. Miasma self-applies +2 Dex a cycle, so Dread grows all fight; the flat 15 was the base with two Miasmas silently baked in, which matches one mid-fight telegraph and nothing else |

**Honest magnitude for the 50% goal: approximately zero.** Every one is act 3 content. Of the 100 `postfix` runs, 69 ended in act 1, 30 in act 2 and **1 in act 3** — the deepest run this project has recorded, floor 48, which died on the Queen + Torch Head Amalgam fight whose opening hit we had modelled 8 low. These are correctness fixes and they belong to the act 3 plan, not to the act 1 number.

That the disparity channel is now surfacing only act 3 is itself the finding, and it is the good kind: act 1 went 157 disparities to 2 under sustained audit, and `PHASE_TWO.md` §7 records that act 2/3 "had almost none of the parity attention act 1 received." The instrument is working and it is pointed at unaudited content. **It is not currently a lever for 50%.**

One thing worth keeping: `tests/test_monster_ai_state_machine_parity.py` asserted the wrong values too — `*_A8_HP_RANGE = (24, 29)`, `THE_FORGOTTEN_DREAD_DAMAGE_A9 = 17`, `TORCH_HEAD_AMALGAM_TACKLE_DAMAGE_A9 = 19` — under test names ending `_matches_csharp`. The suite encoded the same transcription error it existed to catch, so it locked each bug in rather than finding it. A parity test written from the same source as the code it checks measures itself; this is `WEEKEND_DECISIONS.md` §5's rule turning up again in the test suite.

### 7 and 8: multi-enemy targeting is CLEAN, and that closes it

`WEEKEND_DECISIONS.md` §1 named the multi-enemy case "NOT verified, and the likely site of the observed failures... the first thing to test on Monday". It is now tested, by `scripts/probe_multi_enemy_kill.py`, and it is not the site of anything. The searcher takes the kill on the enemy telegraphing the most damage in **91-93%** of positions where one is available, whether or not taking it costs the block, and the small residual is the same in the one-enemy control (93-96%). It is not a multi-enemy effect.

Both halves of my prediction were wrong and the reason is worth keeping. `evaluate` runs **after** `end_player_turn()` and after a 2-turn playout, so an enemy killed this turn stops attacking for the whole scored window and that value lands in `player_hp`. Killing a 5-damage enemy is therefore worth roughly 15 HP over the window against a block's 5 once -- the kill wins by a wide margin without the `kill` term contributing much at all, which is why dividing that term by `len(killable)` never mattered. **I reasoned about the weights and not about when the evaluation is called.** The 4:1 block-over-damage arithmetic in the original lead has the same hole.

The three encounters that account for most of the residual (`setup_turret_operator_weak`, `setup_inklets_normal`, `setup_corpse_slugs_normal`) miss the kill in the solo control too, so they are not a targeting question either.

### Verified against the decompile and DECLINED: the Pact's End play-gate

`PACTS_END` was taken **27 times across the 189 pooled runs and played 0 times** in 25,043 card plays -- the exact signature of the 68 unplayable cards. It is a real disparity: `decompiled/MegaCrit.Sts2.Core.Models.Cards/PactsEnd.cs` declares only `ShouldGlowGoldInternal => CanDealDamage`, which is the gold-glow **display hint**, and overrides `IsPlayable` nowhere -- so the game always lets you play the card and `OnPlay` simply deals no damage when the exhaust pile is short. We implement that hint as a hard `register_playability_hook`, which masks the card out of the action space.

The audit generalises and comes back otherwise clean. All four of our playability hooks, against the decompile:

| card | game | ours | verdict |
|---|---|---|---|
| `CLASH` | `IsPlayable => all hand cards are Attack` | same | correct |
| `GRAND_FINALE` | `IsPlayable => draw pile empty` | same | correct |
| `HIGH_FIVE` | `IsPlayable => !Owner.IsOstyMissing` | osty alive | correct |
| `PACTS_END` | **no `IsPlayable`; `ShouldGlowGold` only** | hard gate | **wrong** |

**Not fixing it, because fixing it cannot help.** Only **3.0% of fights (57 of 1897)** ever reach the 3 exhausted cards the damage needs; our sim already permits the play in those. In the other 97% the game's own `OnPlay` deals zero, so removing the gate buys the searcher the option to waste a card. Correct-in-effect, wrong-in-mechanism, and worth nothing at the number.

**What it does expose is a card-selection defect**, and that one is real: 26 runs spent a reward on a 0-cost 18-damage AoE rare whose enabling condition was met in **8 fights out of 1897**. The scorer reads damage and cost and cannot see that a ~4%-exhaust-density Ironclad deck never turns it on -- the same blindness as `docs/KNOWN_ISSUES.md` TODO 7. Honest magnitude: 26 of 1305 real cards taken, **2.0% of picks**. A fraction of a clear point, not a lever.

### 8 is not a lead: `enemy_hp` is already measured, twice, and it hurts

The standing "EvalWeights is biased 4:1 toward blocking, sweep `enemy_hp`" lead **has already been run** and is in `output/sweep_eval_weights_v2.txt`, 120 paired seeds an arm:

| arm | clear | vs baseline (paired) |
|---|---|---|
| baseline | 33% | — |
| `enemy_hp` 0.35 | 29% | **-4.2% +/- 2.5 (-1.7 se)** |
| `enemy_hp` 0.50 | 26% | **-7.5% +/- 2.9 (-2.6 se)** |
| `block_unused` 0 | 34% | +0.8% +/- 0.8 (+1.0 se) |
| `turn` -0.005 | 32% | -0.8% +/- 1.4 (-0.6 se) |

Monotone in the wrong direction, and the earlier n=60 sweeps that read positive (`output/sweep_eval_weights.txt`, +5.0% then +1.7%) are the small-n noise this project keeps buying. `PHASE_TWO.md:68` already logs it as a null. **Do not re-run it.**

The 4:1 arithmetic behind the lead also omits the horizon, which is where the kill value actually lives. `evaluate` is called *after* `end_player_turn()` and after a 2-turn playout, so an enemy killed this turn stops attacking for the whole scored window and that lands in `player_hp`, not in `enemy_hp`. Blocking 5 scores 0.0625 once; killing a 5-damage enemy scores roughly 15 HP of prevented damage across the window. Damage-vs-block is priced. The remaining suspicion was that **which enemy dies** is not — `kill` is a flat count and `per_enemy` damage appears nowhere in `evaluate`, only in the playout policy rewritten under prediction 6. Predictions 7 and 8 tested exactly that and measured it clean, so this is closed too; see the section below.

### Why 6 missed, and what the diagnosis got wrong

The claim was that the live power-play rate is flat in fight length (2.32% at 1-2 turns against 2.25% at 8+) BECAUSE the playout scoring the lookahead priced every Power at a flat 0.5 and so never played one. Rewriting the playout to price everything in HP did not move the rate, in either arm, and did not change a single win.

The measurement that should have come first: **live decks hold about 1.4 Powers** (138 taken across the 100 runs; a Power is on offer at 30.2% of card rewards and taken 43.0% of the time it is). In a ~20-card deck over a 5-turn fight the agent draws roughly 1.3 Powers and plays about 0.5. The flat 2-3% rate is mostly **deck composition, not play** -- there are barely any Powers in the deck to play. A rate that low cannot rise with fight length no matter how the playout scores.

The offline benchmark is worse than useless for this question in its default form: **110 of the first 150 decks hold no Power at all**. `--powers-only` exists now so this cannot be diluted to nothing again.

Damage per fight fell ~3% in BOTH arms, which is the sort of thing that gets written up as a small win. Paired sign test over the 73 fights: **mean -0.99 HP, better 16, worse 12, tied 45, z=0.76, p=0.450 -- not resolvable.** 45 of 73 fights come out identical. It is noise, and it would have been the sixth false positive of the same shape as the five in `WEEKEND_DECISIONS.md` section 5.

What survives: the playout had a genuine unit error (block and damage compared as one quantity, so a 6-damage Strike beat a 5-block Defend) and no term separating a kill from a chip. Fixing those is defensible on its own and measured null. **Null is not a reason to take it live.**

Note for anyone rerunning `ab_playout_policy.py`: the searcher is time-budgeted, so it is not bit-deterministic and the per-bucket percentages move a little between runs. That is another reason to read the paired test rather than the summary rows.

### 5 was wrong, and the reason is worth keeping

The reasoning was: elite rooms end 23% of the runs that enter them against 2.0% for monsters, so take fewer. pd rejected it and the [map guide](https://sts2.untapped.gg/en/guides/how-to-make-the-best-map-choices-in-slay-the-spire-2) says why -- **target 2-3 elites in act 1**, and "avoiding elites purely to preserve HP is a trap" because relics are what carries a run past act 1. We already take **0.87 elites a run**. We are under-fighting them, not over-fighting them.

The 23% is real. It is a statement about how badly the agent PLAYS elites, not about whether to enter them, and reading it the other way would have optimised act 1 by giving up acts 2 and 3. Prediction 6 is the same evidence pointed at the fight instead of the map.

Correction to the numbers used in 5: `damage_taken` in the journal is computed as `hp_before - hp_after`, and `hp_after` is captured AFTER Burning Blood's +6 refund, so every chip figure in this repo dated before 2026-08-16 is net of that heal. Gross, act 1 pre-boss: elite **37.9** damage over 5.2 turns (7.2/turn), monster **11.6** over 3.1 (3.8/turn). The elite is 1.9x worse per turn, not the 3x first reported.

## Measurement rules

- **Live only.** Offline is a bug detector and a cheap filter; it has never predicted live on boss questions, and the gap is 43 points.
- **Pooled**, never a favourable session. A 20-run session resolves nothing finer than ~25 points.
- **n >= 100** for any claim, and n ~290 to resolve 25% at +/- 5.
- Report the interval every time. A number without one is not a result.

## What does NOT count as progress

- an offline improvement that has not been confirmed live
- a hypothesis eliminated (useful, but it is not the number)
- a change shipped whose effect has not been measured
- the best session out of several
