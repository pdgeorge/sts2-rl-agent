# WEEKEND DECISIONS

Written 2026-08-14 with credits nearly exhausted. Ordered by what should be worked on first, not by when it was discovered. Every number here is pooled with n; anything I could not resolve is marked as unresolved rather than rounded up.

---

## 1. CHIP DAMAGE IS THE TARGET (pd's thesis, and the data agrees)

**The boss fight is decided before it starts.** Boss win by HP fraction on arrival, all search-era runs that reached the boss:

| HP on arrival | n | boss win |
|---|---|---|
| under 70% | 20 | **5%** (1/20) |
| 70–79% | 7 | 14% (1/7) |
| 80–89% | 16 | 56% (9/16) |
| 90–100% | 33 | 55% (18/33) |

Below 80%: **2 wins in 27**. At or above: **27 wins in 49**. That is a cliff, not a gradient, and **36% of runs arrive already beaten**.

Mean arrival is ~82% and has been flat across every session all day. Winners take **3.6** chip damage per hallway fight, losers **4.5**. Winners reach the boss having used **3.3** potions, losers **5.7** — losers burn potions surviving the corridor and arrive with nothing for the boss.

### The three ways to avoid chip damage

1. **Good blocking.**
2. **Using potions appropriately** — the potion data above says this is real and currently backwards.
3. **Ending fights sooner, specifically by killing the thing that is about to hit you.** An enemy on 6 HP intending 5 damage: striking it removes 5 damage *this* turn and every turn after, where blocking removes it once. pd has watched this be skipped and pay chip damage the following turn.

**What is verified:** the search takes the kill in the simple single-enemy case, 8 positions out of 8, finishing on 50→50 HP with zero damage taken.

### 2026-08-16: the mechanism, found. It was the playout, not the searcher.

The n=100 session confirmed the cliff (below 80% HP on boss arrival: **1 win in 31**; at or above: **19 in 35**) and then named the cause. Across all 13,251 live card plays:

| fight length | plays | power | attack | skill |
|---|---|---|---|---|
| 1–2 turns | 2416 | **2.32%** | 77.2% | 20.4% |
| 3–4 turns | 5456 | **3.06%** | 66.0% | 30.2% |
| 5–7 turns | 3247 | **3.60%** | 58.6% | 36.6% |
| 8+ turns | 2132 | **2.25%** | 54.6% | 41.0% |

The power-play rate is **flat in fight length**. In an 8-turn fight a Power should go down on turn 1 nearly every time. Instead the agent's entire answer to a long fight is to block more — skills 20% → 41% — so it defends its way through 5.2-turn elite fights at 7.2 gross damage a turn instead of closing them.

`turn_search.py` already contained the diagnosis: the playout that scores the 2-turn lookahead ranked every card as `block if a hit is coming else damage` and gave a Power a flat **0.5**, below the worst attack in the deck. So the rollout whose entire purpose was to reveal a scaling card's payoff never played one. It also compared block and damage as though they shared a unit — a 6-damage Strike beat a 5-block Defend on a turn needing block, because 6 > 5 — and had no term at all distinguishing a kill from a chip. `DEFAULT_TOP_K` records the earlier attempt to fix this with depth: +0.5% ± 1.1%, power rate just as flat, because a deeper rollout of a policy that never plays a Power still never plays a Power.

The replacement prices every card in one unit, HP: block at `min(block, unblocked)`, a kill at the dead enemy's damage for every remaining turn, any other attack at `damage × D/P` (derived from the board — enemies dealing D a turn against a player dealing P), and a Power at its declared per-turn value times the turns left to collect it. Pinned in `tests/test_playout_policy.py`.

### Two corrections to the numbers above

**`damage_taken` is net of Burning Blood.** The journal computes it as `hp_before - hp_after`, and `hp_after` is read after the relic's +6 fires (visible directly: `hp 55→61, dmg -6`, and `78→80, dmg -2` where it caps at max). Burning Blood works — `decompiled/MegaCrit.Sts2.Core.Models.Relics/BurningBlood.cs` heals 6 on `AfterCombatVictory` and the live game does it. Every chip figure in this repo dated before 2026-08-16 is therefore 6 HP too low per won fight. Gross act 1 pre-boss: elite **37.9** over 5.2 turns, monster **11.6** over 3.1.

**Fewer elites is the wrong lever.** Elite rooms end 23% of the runs entering them against 2.0% for monsters, and the first reading of that was to route around them. The [map guide](https://sts2.untapped.gg/en/guides/how-to-make-the-best-map-choices-in-slay-the-spire-2) says target **2–3 elites in act 1** and that avoiding them to preserve HP is a trap, because relics are what carries a run past act 1 — and we take **0.87 a run** already. The 23% measures how badly we play elites, not whether to enter them. Logged as WITHDRAWN in `SCOREBOARD.md`.

**What is NOT verified, and is the likely site of the observed failures:** the multi-enemy case, where the agent must choose *which* enemy to kill and whether to split damage. That has never been tested. It is the first thing to test on Monday, and it is cheap — the same harness shape as the Bash and kill tests already written.

### Why this is not a fool's errand

Less chip damage → fewer rest sites spent healing → rest sites free to smith → upgrades, which are measured at **~5 boss-win points each** — and higher HP at the boss, which is the cliff above. The mechanisms compound rather than compete. Every other lever tried this week was flat.

**Pre-registered prediction: raising mean HP-at-boss from 82% to 90% lifts win-given-reach from 38% to at least 50%.**

---

## 2. The one solid win: the search was switched off for 82% of all runs

`--live-search` was opt-in. Only 93 of 508 recorded runs ever used it; the rest were played by the v3 trained model.

| | n | reach | clear |
|---|---|---|---|
| v3 model era | 410 | 45.4% +/- 4.8 | **12.2% +/- 3.2** |
| search era | 115 | 66.1% +/- 8.7 | **25.2% +/- 7.9** |

`z=3.45, p=0.0006`. The clear rate doubled. This is the only change all week that moved the number, and it was a **measurement error**, not an improvement — the agent under development had barely been measured.

It also explains pd's observation of Vulnerable being played after damage cards: that was the model. The search sequences correctly (verified: Bash before Strike, 38→21, the optimal 17).

**The 48% final session (n=25) is NOT resolvable** — `p=0.050` against the previous session, exactly on the line. Do not build on it.

---

## 3. Confirmed dead ends — do not spend more time here

Each cost at least half a day. All were measured, none moved the number.

| lever | result |
|---|---|
| rest-site smith gate | heal is worth ~30 boss-win points, a smith ~5. Loosening it **loses** runs. Measured three separate ways. |
| map routing with lookahead | reach 60.7% → **56.0%**, paired net −2, p=0.89. Five parameter settings, best +0. |
| elite HP gate | halving it bought **0.22 elites per run**. The gate was never the constraint. |
| relic acquisition | offline carries **fewer** relics (2.8 vs 4.7) and wins more. Not the gap. |
| deck size / quality bar | stricter skipping came back **−5.7%**; deck at boss is 19.7 for winners against 20.2 for losers — **flat**. |

Deck size, relics and max HP are all flat between winners and losers once measured **at the boss** rather than at run end. That correction matters: see §5.

---

## 4. Bugs fixed today, and the one still open

**Fixed — these were killing runs:**

- **`CloneError` / the agent standing still.** A fight opening on a pending choice ("choose cards to discard") could not be cloned, `SearchAgent` caught the refusal and ended the turn. Floor 11, Punch Construct: **8 turns, 0 cards played, 61 damage, died** — while the journal recorded `searches: 9, search_failures: 0`. It reported success while doing nothing. Fixed by making the pending callback rebuildable data.
- **Card-reward loop.** `can_skip: true` means the game renders a Skip button, not that clicking it consumes the reward. The policy skipped correctly and forever. Now skips once, then takes the best card and logs at ERROR.
- **Endless Conveyor gold drain.** Every option is "Pay 40 Gold… Continue feasting!" and the safety check only ever parsed **HP**. Fed the belt 120 gold in one room. Gold is now parsed and is the second sort key.
- **Phantom `combat_end`.** A mid-fight `card_select` counted as the fight ending: **156 of 511 recorded boss "fights"** were this artefact, corrupting every per-fight statistic.

**Still open:**

- **Unmodelled content, found by the identifier audit on the newest capture.** `BATTLEWORN_DUMMY_EVENT_V1_ENCOUNTER` and `BATTLE_FRIEND_V1` do not resolve in this build. If a run enters that event the reconstruction raises, the search fails, and the fight falls back to the v3 model -- the exact failure this week was spent eliminating. Two red tests in `test_bridge_identifiers_resolve.py` mark it; they appeared because more was played, which is the audit working as designed.
- **The mod's skip click does not consume the card reward.** Python works around it; the mod bug is real and unfixed.
- **Endless Conveyor may still hang.** Three fixes in, and the two observed failures had different shapes (83 choices vs 3 choices then `run_complete` at full HP). The gold drain is certainly fixed; whether that was *the* hang is unknown.

---

## 5. The discipline that actually mattered

**Six false positives came from my own test harnesses in one day.** Every one looked like a major discovery:

- reconstruction "17% faithful" → actually **96%**, and **0 of 60 boss states wrong**. Three normalisation bugs: player fields nested under `player`, intent a bare string, and card ids carrying `_CARD` (`SETUP_STRIKE` vs `SETUP_STRIKE_CARD`).
- "monster HP mismatch, 38 vs 40" → both legal rolls from a declared 38–40 range.
- "Rage is a dead card" → my test mistook END_TURN discarding the hand for the card being played.
- "bigger decks win, 100% above 24 cards" → `run_end` deck size includes **act 2**; winners keep taking cards after clearing. Measured at the boss it is flat.

**The rule that caught all of them: check the number, not the story.** An audit that reimplements the thing it audits measures itself. Both sides must go through the real resolvers (`resolve_card_id`, `_coerce_power_id`), and anything read from `run_end` is contaminated by what happened *after* the event of interest.

**`SCOREBOARD.md` holds the pre-registered predictions.** Routing is logged as a MISS. Predictions go in before the work, and a miss stays in the table.

---

## 6. Where 50% actually stands

```
now      66.1% reach  x  38% win|reach  =  25.2% clear
target   ~70%         x  ~71%           =  50%
```

Reach is close to target. **Boss win must roughly double**, and §1 is the only lever with evidence behind it — the HP cliff is worth ~17 points of win-given-reach on its own if the sub-80% arrivals can be moved above it.

Run 100+ runs before believing any of it. At n=25 the confidence interval is ±20 points and proves nothing.
