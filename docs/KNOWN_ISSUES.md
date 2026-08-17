# Known Issues and Limitations

Current known issues, bugs, and limitations of the STS2 RL Agent project.

---

## STOP. Read this before investigating a crash.

### The Punch Off crash — KNOWN, NOT OURS, DO NOT RE-INVESTIGATE

This has now been diagnosed from scratch at least twice, and the second time cost most of a session. **It is closed. If you see this signature, the answer is "known Punch Off crash" and nothing else needs looking at.**

```
[RlMap] Agent chose node N: Unknown
Creating NCombatRoom with mode=VisualOnly encounter=PUNCH_OFF_EVENT_ENCOUNTER
EventRoom.EnterInternal -> PunchOff.AfterEventStarted -> PunchOff.PunchEachOther
  -> CreatureCmd.TriggerAnim -> NCreature.SetAnimationTrigger_Patch1
  -> MegaSpineBinding.Call
ERROR: Signal '_internal_spine_objects_invalidated' is already connected
```

Python then reports `Connection lost` / `Connection refused` — the game process is gone.

**What it is.** `_Patch1` is **BaseLib's** Harmony patch on `NCreature.SetAnimationTrigger`, against a game build BaseLib predates (BaseLib.dll Jul 31 09:59, game `.pck` Jul 31 19:28; 3.4.0 is the newest published, and it already throws `MissingFieldException` on `NTreasureRoom._chestNode` in every treasure room).

**It is not our mod.** Cleared on 2026-08-11 by removing `AnimationSpeedPatch` from the patch list entirely and reproducing the crash anyway on forced seed `6D038P4FSM2F`. See the comment at `bridge_mod/MainFile.cs:53`. Note that `--speed normal` is **not** that test: it sets the multiplier to 1.0 while the prefix stays installed.

**There is nothing the agent can do.** `PunchOff` calls `PunchEachOther` from `AfterEventStarted`, so the crash happens on room entry with no option presented, and a `?` room's contents cannot be read from the map beforehand.

**The only lever is restarting.** Measured rate: **4 crashes in 36 runs, one per nine** (2026-08-16). `scripts/run100.sh` therefore passes `--restart-on-crash 30`, sized from that rate. If a session is ending early, check that number before checking anything else.

Repro seed for the same crash: `VHHTGKTPEZWF`.

---

## The shop purchase deadlock — FIXED 2026-08-18

Two sessions soft-locked on it, and the second lost 46 of 100 runs sitting on
one screen for six hours. Worth keeping because the shape is not obvious and
the shop handler *looked* like it already handled the case.

**What you see.** The game stops. Nothing crashes, nothing logs an error.
Python repeats `Timeout waiting for state. Sending ping...` once a minute
forever. The game's own state dump says:

```
Room: Shop
Overlay Stack: 1 screens
  Current: NRewardsScreen
Watchdog timeout: No progress for 21389.0s. Last activity: Entering Shop room
```

The journal's last event is a shop purchase — `ORRERY` the first time,
`CAULDRON` the second — and nothing after it.

**What it is.** A three-way deadlock:

```
RlShopRoomHandler:  await OnTryPurchaseWrapper(...)      never returns
  -> Orrery.AfterObtained()  ->  RewardsCmd.OfferCustom
       -> RewardsSet.Offer() ->  await task              completes only when the
                                                         rewards screen is dismissed
                                                         ...which only
                                                         DrainOverlayScreensAsync does
                                                         ...which only runs after
                                                         HandleRoomAsync returns
```

The purchase waits on the screen, the screen waits on the drain loop, the drain
loop waits on the purchase. The shop handler's own overlay check sat on the
line *after* the await and was therefore unreachable.

**Why exactly these two relics.** Six relics call `RewardsCmd` from
`AfterObtained` — Orrery, Cauldron, CallingBell, Kaleidoscope, LostCoffer,
ToyBox — and **only Orrery and Cauldron are `RelicRarity.Shop`**. The other
four are Ancient, taken from the post-boss event, and that handler uses
`UiHelper.Click` plus a delay rather than awaiting a game task, so it cannot
deadlock. The shop is the only handler that awaits a real game API instead of
clicking a button, which is why it is the only one that hangs.

**The fix.** `RlNonCombatRoomHandlers.RlShopRoomHandler` starts the purchase
without awaiting it, then races it against the overlay stack. If a screen opens
it leaves immediately and lets the drain loop close it, which is what completes
the still-pending purchase task. A fault on that task is logged rather than
swallowed, and a purchase that neither completes nor opens an overlay inside 5s
leaves the shop rather than blocking the run.

**If you see it again**, check the game log's state dump first: `Overlay Stack`
naming a screen while `Last activity` still names a room is this shape,
whatever the screen turns out to be.

## Live-run failure modes

How a live run actually stops. Kept as a list because twice now a run was
reported as "crashed" when nothing crashed -- it stopped making progress and the
log stayed clean. That distinction is the whole diagnostic value here.

| What you see | What it is | Status |
|---|---|---|
| Python exits after the player dies and the game returns to the main menu | Expected. The runner has no menu state, so it stops. | By design |
| Same action chosen over and over inside one screen | The mod advertised an action it cannot execute. The screen never closes, so it re-presents, and a deterministic policy makes the same choice again. | Fixed for card rewards |
| Run sits on a map after beating a boss, and every subsequent run logs "Main menu not visible" | A transition timed out mid-run, leaving the game inside the run. The loop then waits for a menu that will never appear. | Logged plainly once; needs manual return to menu |
| Everything looks fine but a decision is silently mistranslated | Simulator and bridge agree on the observation but disagree on what an action index means. No error either side. | Card-reward slot fixed; the parity suites exist for this class |
| A healthy run is recorded `terminated`, or End Turn is refused forever | The combat loop asked for the next action before the previous card finished resolving. | Fixed 2026-08-09, see below |

**The mid-resolution race, found 2026-08-08 across five live sessions.** One root
cause, wearing two very different faces, and the second face was mistaken for an
unrelated old bug for days.

`PlayCardAndWaitAsync` waited for energy or hand size to *change*. That fires when
the card leaves the hand -- when the play has **started**. For any card that asks
a question mid-resolution (Armaments, Battle Trance, Burning Pact, Acrobatics,
Headbutt) the play is then only half done, and the loop had already serialised the
next state and sent it.

*Face one -- the run ends while the player is healthy.* Only one bridge request may
be outstanding, so the card-select's request superseded the combat request still in
flight. `SendStateAndWaitForActionAsync` cancelled the old one, the caller received
a bare `null`, and `RlCombatHandler` reads null as "the agent is unreachable" and
stops -- `RlAutoSlayer` then waits for a rewards screen that cannot come, because
the fight is still going. Thirty seconds later the AutoSlay watchdog throws, and
`PlayRunAsync`'s catch-all reports `run_complete` / `terminated`. Two of the five
sessions ended exactly here, one on the act 1 boss at 53/90 HP, one on an elite at
54/80.

*Face two -- "no cards playable, End Turn greyed out".* Same gap, worse outcome. The
next play was dispatched **into the open select** and never resolved: the log shows
`Playing card: DEFEND_IRONCLAD` with no matching `Player 1 playing card
DEFEND_IRONCLAD` after it. The game then refuses to end the turn because it is
still waiting on that card, forever. This had been carried as its own mysterious
issue; it was never separate.

Fixed in three parts. `WaitForQuiescenceAsync` blocks until
`RunManager.Instance.ActionQueueSet.IsEmpty` **and** `RlCardSelector.SelectionPending`
is clear -- both, because the queue parks while a choice is pending and drains
after it. Pre-emption now raises `RequestPreemptedException` instead of arriving as
`null`, so the combat loop can tell "superseded, ask again" from "the agent is
gone" and retries rather than ending the run. The watchdog is fed while waiting,
since a card-select round trip is a legitimate pause the watchdog would otherwise
read as a freeze.

**Still open, same screen.** `RlCardSelector` implements a generic hook, so the
payload cannot say *why* the game is asking. The agent applies one heuristic
(non-basic first, never a curse) to every prompt, which is right for Armaments and
backwards for Burning Pact -- it exhausted Shrug It Off when a Strike was sitting
there. Fixing this needs the prompt's source card in the payload, which
`ICardSelector.GetSelectedCards(options, minSelect, maxSelect)` does not receive.


**Confirmed at the act 1 -> act 2 boundary.** `FindAll<NMapPoint>` returned two
candidates reporting the same coord (row 0, col 3), which a map cannot contain,
so one was stale or duplicated across the transition and could never be clicked.
The agent chose it, the 10s "Map point not enabled" wait threw, and a throw in
the map handler reaches `PlayRunAsync`'s catch-all whose finally reports
`run_complete` / `terminated`. The run was alive at 26 HP. Fixed in `4fc36c0`.

**The loop is the one to watch for.** It produces no exception, no timeout and no
error line -- just a run that stops advancing. It is caused by claiming an action
the game cannot perform, so the fix is always the same: do not advertise it, and
never return from a handler leaving the screen open.

Concretely, on card rewards: `NCardRewardSelectionScreen` has no skip control and
the game's own `CardRewardScreenHandler` always picks a card, so a card reward on
the screen-driven path cannot be skipped at all. `RlCardSelector` is a different
path that hooks the game's own selection API, where a skip *is* real
(`SkipReward() => default`, meaning no card taken). The two disagree; only the
selector may claim `can_skip`.

---

## Fixed Issues

### 1. Energy always displayed as 3 with CardCmd.AutoPlay

**Status:** Fixed

**Problem:** The C# bridge mod initially used `CardCmd.AutoPlay()` to execute card plays. This method bypasses the normal energy deduction, so the player's energy always stayed at 3 (max) regardless of cards played. The agent could play unlimited cards per turn.

**Fix:** Switched to `PlayCardAction` which properly spends energy:
```csharp
var playAction = new PlayCardAction(card, target);
RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(playAction);
```

**Location:** `bridge_mod/RlCombatHandler.cs` line 187-188

### 2. EchoForm / modify_card_play_count was missing

**Status:** Fixed

**Problem:** The hook for modifying how many times a card is played was not implemented. Powers like EchoForm (play each card twice) had no effect.

**Fix:** Added `modify_card_play_count` to `core/hooks.py` and wired it into `CombatState.play_card()`.

**Location:** `sts2_env/core/hooks.py` lines 189-200, `sts2_env/core/combat.py` line 255

### 3. Enemy round-1 block not cleared

**Status:** Fixed

**Problem:** Enemies that gained block before their first turn (from combat-start effects) were not having their block cleared at the start of the enemy turn on round 1.

**Fix:** The enemy turn now always clears block for each alive enemy, regardless of round number.

**Location:** `sts2_env/core/combat.py` `_execute_enemy_turn()`

### 4. State adapter and action mask protocol mismatches

**Status:** Fixed

**Problem:** The Python `StateAdapter` was expecting different field names and formats than what the C# mod was actually sending. For example, target type strings like `"AnyEnemy"` vs `"ANY_ENEMY"`, and power list format differences.

**Fix:** Updated `state_adapter.py` to handle both formats:
```python
_UNTARGETED_TYPES = {TargetTypeName.SELF, TargetTypeName.NONE, TargetTypeName.ALL_ENEMIES,
                     "SELF", "NONE", "ALL_ENEMIES", "Self", "None", "AllEnemies"}
```

**Location:** `sts2_env/bridge/state_adapter.py` lines 69-71

---

## Open Issues

### 5. AnimationSpeedPatch fails to apply

**Severity:** Low (affects real-game speed only)

**Problem:** The Harmony patch targeting `MegaAnimationState.SetTimeScale` fails on some game versions because the method signature changed between updates. The patch is skipped with a log message.

**Impact:** The game runs at normal animation speed instead of 5x. The `WaitSpeedPatch` (which reduces timed delays by 10x) still applies successfully, providing some speedup.

**Workaround:** None currently. The animation patch needs to be updated when the game's `MegaAnimationState` API changes.

**Location:** `bridge_mod/MainFile.cs` `AnimationSpeedPatch` class

### 6. Mod abandon-run popup path may not match all versions

**Severity:** Low

**Problem:** The Godot scene tree paths used to find the abandon-run confirmation popup (`VerticalPopup/YesButton`) may not match all game versions. If the path is wrong, the mod cannot automatically abandon an existing run before starting a new one.

**Impact:** If there is already a run in progress when the mod starts, it may fail to abandon it cleanly.

**Workaround:** Manually abandon the run from the main menu before starting the agent.

**Location:** `bridge_mod/RlAutoSlayer.cs` `PlayMainMenuAsync()` lines 455-472

### 7. Full-run training needs significantly more steps and better reward shaping

**Severity:** High (fundamental training challenge)

**Problem:** The full-run environment produces 0% win rate even after 1M training steps. The agent learns to progress further through Act 1 (avg 8.9 floors vs 3.9 for random) but cannot complete a run.

**Root causes:**
- Sparse reward: only +1 at run victory, -1 at death. No intermediate signal.
- Long episodes: a full run spans thousands of steps.
- Multi-phase action space: `Discrete(157)` across combat, map, rewards, shop, rest, event, treasure, and player-selection slices.
- Compounding decisions: bad deck choices early doom later combats.

**Mitigation:** Reward shaping is available (`--reward-shaping` flag) but only provides small floor-progression bonuses. A fundamental redesign of the reward function or training approach (hierarchical RL, curriculum learning) is needed.

### 8. Only Ironclad combat model trained

**Severity:** Medium

**Problem:** The combat training pipeline only creates Ironclad starter decks. All training and evaluation use the Ironclad character.

**Impact:** The trained model is specific to Ironclad. It cannot play Silent, Defect, Necrobinder, or Regent effectively because:
- Different starter decks and starting HP
- Character-specific mechanics (orbs, stars, pets)
- Different card pools with different effect distributions

**Workaround:** The simulator supports all 5 characters (cards, powers, monsters are all implemented). Training scripts need to be extended to support character selection.

### 9. Combat potion actions were missing from the RL action space

**Status:** Fixed

**Problem:** The combat action space originally only covered card plays and end turn, so the agent could not use potions strategically during combat.

**Fix:** The combat action space now includes fixed-width potion actions, `CombatState` can execute potion uses directly, and the bridge path serializes and decodes potion actions as explicit `POTION` commands.

**Location:** `sts2_env/core/constants.py`, `sts2_env/core/combat.py`, `sts2_env/gym_env/action_space.py`, `sts2_env/gym_env/combat_env.py`, `sts2_env/bridge/state_adapter.py`, `bridge_mod/RlCombatHandler.cs`

### 10. Some card effects may not match the real game exactly

**Severity:** Medium (simulator fidelity)

**Problem:** The headless simulator reimplements card effects based on the decompiled C# source, but exact parity is still broader than the currently audited test surface. The earlier helper-level gaps are fixed, but some card and relic interactions still need direct decompiled-backed regression tests before they should be treated as exact.

**Examples of still-audited-not-proven-exact areas:**
- selected colorless/event cards such as `Alchemize`, `BeatDown`, and `HandOfGreed`
- selected Defect and Silent follow-up effects such as `Compact`, `WhiteNoise`, and `TheHunt`
- wider relic-hook interactions outside the targeted parity suites

**Impact:** The trained model may develop strategies that exploit simulator inaccuracies and fail to transfer to the real game. The bridge mod's real-game evaluation is the ground truth.

### 11. Reconnection timing issues

**Severity:** Low

**Problem:** If the Python agent connects before the game has finished loading and the AutoSlayer has started, there can be a race condition where the first state message arrives before the agent is ready.

**Workaround:** Start the game first, wait for the main menu to appear, then start the Python agent. The agent runner has reconnection retry logic (`_reconnect_with_retry` with 10 attempts, 3s delay).

**Location:** `sts2_env/bridge/agent_runner.py` lines 288-309

### 12. `inspect.signature` on hot path

**Status:** Fixed

**Severity:** Low (performance)

**Problem:** `fire_after_card_drawn` used to call `inspect.signature(method).parameters` for every card draw to determine the parameter count of each power's `on_card_drawn` method. This was slow.

**Fix:** All power `on_card_drawn` implementations now use `(owner, card, from_hand_draw, combat)`, and the dispatcher calls that signature directly.

**Location:** `sts2_env/core/hooks.py`

### 13. `run_env` exception handling used to hide simulation bugs

**Status:** Fixed

**Problem:** `STS2RunEnv.step()` used to convert internal simulation exceptions into silent losses, which made debugging difficult.

```python
try:
    if phase == RunManager.PHASE_COMBAT:
        self._step_combat(action)
    # ...
except Exception:
    if not self._mgr.is_over:
        self._mgr.run_state.lose_run()
```

**Fix:** `STS2RunEnv.step()` now logs the exception before forcing the run to end, so failures are visible in logs instead of disappearing into episode outcomes.

**Location:** `sts2_env/gym_env/run_env.py`

### 14. Pile-summary distribution shift between simulator and bridge

**Status:** Fixed

**Problem:** The observation vector used to encode pile-composition features in simulator mode even though bridge mode could not provide them.

**Fix:** The simulator now keeps those three pile-composition slots zeroed as well, so simulator and bridge observations match on that segment without changing observation size.

**Location:** `sts2_env/gym_env/observation.py`, `sts2_env/bridge/state_adapter.py`

---

## TODO — found 2026-08-07, from the `meta_ppo_v8_rewarded` training run

Ordered by how much they cost, not by effort.

### 1. Multi-select selection state is not in the run observation — BLOCKING

**The one that matters.** `PendingCardChoice` with `is_multi=True` (a
`TransformCardsReward`, for instance) presents N cards to toggle before
confirming. Toggling works — `selected` flips 0→1→0 — but **the observation is
byte-identical whether a card is selected or not.** Verified: unchanged across 8
consecutive steps while selection flipped on and off.

So the policy cannot see the effect of its own action. A deterministic policy
picks the same option every step, toggling one card on, off, on, off, until the
episode hits a step cap.

**Cost, measured on the v8 eval logs:** 61 of 2350 episodes (2.6%) ran to
~1950-1988 steps against a median of 19. Those 61 consumed **~73% of all
evaluation steps**.

**Worse than the 2.6% suggests: it strikes the good runs.** The captured case
was at **floor 16** — the run had reached the Act 1 boss. `TransformCardsReward`
comes from later rewards, so this systematically burns budget on, and truncates,
precisely the episodes worth learning from.

**Fix:** encode selection state in the run observation. Probably a handful of
dims. Reproduce with seed 9004 against `meta_ppo_v8_rewarded/best_model`.

### 2. `can_confirm()` never becomes true on that screen

Related and possibly the same root cause. With one card selected,
`can_confirm()` stays `False`, so action 0 (confirm) is never unmasked and there
is no legal way to finish the screen. If the transform requires exactly N cards,
the policy must be able to *reach* N — which it cannot while blind to what it
has already selected. Worth checking whether the min is satisfiable at all.

### 3. Floor-scaled step cap, as a backstop

Not the fix for 1 and 2 — with those fixed these episodes end in one step rather
than two thousand. But a fixed cap should not be what kills the deepest run of a
session, which is exactly what happened here.

Proposal (user's): scale the cap with progress rather than fixing it, e.g.
`max_steps = MULTIPLIER * floors_reached`, so a floor-1 loop dies quickly and a
floor-20 run gets room. Same reasoning as the live runner's end-turn escalation:
the *next* unmodelled loop should cost a screen, not a run.

### 4. Stalling must be punished, not merely discounted

```
COMBAT_TURN_COST = 0.005      COMBAT_WON = 1.0
0.005 * 200 turns = 1.0
```

A 200-turn win nets **zero**. There is no gradient preferring an 8-turn win to
an 80-turn one until the extreme, so stalling is close to free.

**Requirement (user):** stalling for even ~20 turns should score worse than
taking damage. So the turn cost has to be steep enough that a long fight is a
real loss, not a small tax — which means an order of magnitude or two above
0.005, and worth checking against `COMBAT_HP_WEIGHT` so the two are comparable
at the intended crossover.

**This is not only a training concern.** Cyra is watched. Nobody wants to sit
through 200 turns of a Defend loop, so a policy that stalls is a product
failure as much as a scoring one — which is the argument for making it
genuinely negative rather than merely unrewarding.

Retuning, not a new mechanism, but it means retraining combat.

### 5. `PHASE_BOSS_RELIC` is a mechanic STS2 does not have — CONFIRMED

Checked against the decompile. **There is no boss-relic concept in the game.**
No `BossRelic` type exists anywhere in `decompiled/`. What follows a boss is an
**Ancient**, and `AncientEventModel : EventModel`
(`MegaCrit.Sts2.Core.Models/AncientEventModel.cs:24`) — it is an *event*, run by
`EventRoom`, with a dialogue set and generated options. Per the user it sits on
the floor *after* the boss: you pick the next node, take the reward, then move
on again.

The simulator models the Ancients correctly — `Darv`, `Orobas`, `Tezcatara` and
the rest are `EventModel` subclasses in `sts2_env/events/shared.py`.

**And it also has a fabricated Slay the Spire 1 boss-relic phase**:

```python
# run_manager.py:731
def _enter_boss_relic(self) -> None:
    """Offer three boss relics after defeating a boss."""
```

`_BOSS_RELIC_POOL` (`run_manager.py:187`) holds 11 relics — `ASTROLABE`,
`BLACK_STAR`, `CALLING_BELL`, `ECTOPLASM`, `PANDORAS_BOX`,
`PHILOSOPHERS_STONE` … which are **StS1 boss relics**.

So after every boss the simulator enters a phase the game does not have and
awards relics that do not exist in STS2. `meta_ppo_v8_rewarded` spent 17 hours
training partly on deciding it, and `PHASE_BOSS_RELIC` is one of the phases the
meta-policy nominally learns.

**Also worth investigating (user):** floor 16 has historically been slow, and
the Ancient screen is much longer than a standard card reward. Whether that
interacts with TODO 1 — the observed floor-16 stall was a
`TransformCardsReward` multi-select — is unconfirmed but suspicious, and the
two should be looked at together.

### 6. The meta-policy is capped by the combat solver beneath it

`meta_ppo_v8_rewarded` moved `mean_floor` 8.8 → 9.7 across 4.28M steps and
plateaued after ~860k. It trains with combat fast-forwarded by
`FrozenRLCombatSolver(combat_v3_overnight)` — 74% overall, 6.7% boss. Dying
around floor 9-10 means losing hallway and elite fights, well before the boss at
16. No routing decision saves a run from losing its fights.

The floor distribution is **bimodal, not a bell curve** — 28% die by floor 5,
25% reach the boss, 7% clear Act 1 — so the mean describes almost no run, and
the two humps are different problems.

Roadmap Phase 4.3 (`FrozenSearchCombatSolver`) was written as *"skip unless
4.1+4.2 alone reach the milestone"*. They did not. The cost is that search is
~3s/turn inside a fast-forward loop, so it is far slower than the 17 hours this
run took.

**Also affects any deckbuilding A/B:** whichever solver runs underneath defines
what "a better card" means. Comparing card-pickers under a solver that loses 93%
of bosses may pick a different winner than the live stack, which plays combat
with search.

### 7. Card quality reads only damage and block — LARGELY FIXED 2026-08-07

`card_quality.score_card` values a card by `(damage + block) / cost` plus
rarity. It cannot see **draw, energy, debuff duration, or any conditional
effect**, which shows up in three places found so far:

- **Logic-only cards score 0.0.** Body Slam ("damage equal to your Block") and
  Entrench ("double your Block") have no base damage or block. Worked around in
  `score_card_for_deck`, where a zero is treated as *no opinion* and archetype
  fit decides instead — but the underlying scorer is still blind.
- **Upgrade gains are understated.** Pommel Strike gains a card of draw and
  scores `+0.10`; Uppercut doubles Weak and Vulnerable duration and scores
  `+0.00`. Both are cards the user cited as going from fine to excellent when
  upgraded. The `upgrade_targets` delta is correct; what it differences is not.
- **The card text generator had the same hole**, for the same reason, and was
  fixed separately by reading effects out of the decompile.

The pattern is worth stating on its own: **anything that reads base damage and
block is blind to roughly a third of the Ironclad's cards.** Three components
have now hit it independently.

**Fixed** by scoring `effect_vars` — the numbers the simulator already derives
from the decompile — with weights grounded in a survey of all 577 cards, plus
valuing energy cost on its own. Every card named as a fine-to-excellent upgrade
now registers:

```
                originally    now    what the upgrade does
POMMEL_STRIKE      +0.10     +0.70   +1 damage, +1 card drawn
UPPERCUT           +0.00     +0.25   doubles Weak & Vulnerable duration
BARRICADE          +0.00     +0.15   cost 3 -> 2
ARMAMENTS          +1.50     +1.50   upgrades your whole hand (behavioural)
BLUDGEON           +0.33     +0.33   32 -> 42 damage
```

And Pommel Strike now outranks Bludgeon as an upgrade target, which is the rule
this was for: biggest benefit, not biggest card.

**Still missing.** Anger's upgrade duplicates the upgraded card, which lives in
`OnPlay` logic and appears in neither `effect_vars` nor an `IsUpgraded` branch,
so it scores its damage bump and nothing more. Any upgrade of that shape is
invisible. And a card whose damage is computed rather than declared — Body Slam
is "damage equal to your Block" — is still valued at its `extra_damage` var
rather than what it would actually hit for.

`quality_is_uninformative` now marks the cards the scorer genuinely cannot read,
so archetype fit decides for those instead of a rarity-and-cost number
masquerading as an assessment. `CARD_RATINGS` remains the override hook for
anything needing a hand-set value, and is still empty.

---

### 8. The HP economy governed every room with one 50% threshold — FIXED 2026-08-09

Found by asking why the deepest live run (floor 45, act 3) died, and the answer
was not combat, deckbuilding or luck. It was three map decisions.

`_pick_map_node` picked `ROOM_PRIORITY_HEALTHY` whenever `hp > 0.5 * max_hp`, and
that table ranks elite second and restsite **last**. So the threshold authorised
exactly the rooms it could not pay for. Measured over 1119 live fights:

| Entering HP | Elite death rate | Monster | Boss |
|---|---|---|---|
| 20–29 | 100% | 35% | — |
| 30–39 | 43% | 17% | — |
| 40–49 | 18% | 6% | 88% |
| 50–59 | 29% | 4% | 73% |
| 60–69 | 20% | 2% | 60% |
| 70–79 | **0%** | 0% | never happened |

Two findings sit in that table. **32 of the 56 recorded elite choices were made
between 40 and 59 HP**, in the 18–29% band, because 45/80 reads as healthy. And
the agent arrives at act bosses worn down: **median entry 54 over 45 act 1 boss
fights, with only 3 of 47 boss fights overall entered above 69 HP**, so nearly
every boss is fought from a position that loses 60–88% of the time. The rest site
inherited the same threshold and chose SMITH over HEAL 17 times on the floor
immediately before an act boss, at a median 49 HP.

> **Correction.** This first read "median 47 over 89 attempts, never above 69".
> Both numbers were wrong, and wrong for an instructive reason: they counted
> `combat_end` *segments* rather than fights, and a fight is split into several
> segments by exactly the mid-resolution card-select bug documented above. The
> metric was inflated by the bug it sat next to. Deduplicating by
> (session, run, floor) gives 47 boss fights, median act 1 entry 54, and three
> entries above 69 (73, 73, 80). Every other figure here — the death-rate table,
> the elite fit, the 32-of-56 and 17-times counts — was computed on deduplicated
> fights or on one-per-decision `choice` events and is unaffected.

The floor-45 run died of exactly this. Floor 42, 76/97 — 78%, "healthy" — took an
act 3 elite worth 58 HP and reached floor 45 at 21 with no rest between. It had
survived the elite. It could not afford what came after.

**What made that run good** was not its deck. It entered the act 1 boss at 73 HP
against a median of 54 — joint second-highest of 47 recorded boss fights, and
inside the band where elites kill nobody. Everything downstream followed from
that, which makes the target a single number: **median HP entering the act 1
boss, 54 → 73**.

Fixed by gating each room on a fraction of max HP (`ROOM_MIN_HP_FRACTION`), fitted
against the 116 elite fights with a known max HP rather than chosen:

```
hp > 0.50 * max  (old)    85 taken   21% died
hp >= 0.75 * max          35 taken   17% died
hp >= 0.80 * max          21 taken   10% died   <- the bend
hp >= 0.85 * max          18 taken   11% died
```

Elites are still ranked above monsters, so a healthy run still farms them; it
just takes them at 80% instead of 56%. A fraction rather than an absolute HP
figure because Max HP rewards push the ceiling to 103, and an absolute threshold
silently loosens as the character grows — backwards, since the rooms get harder.

**Two things this deliberately does not do.**

*No per-act scaling.* The obvious move, and the data refuses it: act 2 elites
measure a p90 of 24 against act 1's 54, which reads as "act 2 is easier" and is
really survivorship at n=4. The version that fitted a multiplier to that made act
3 elites unaffordable at any HP and stopped the agent upgrading a card ever
again. Revisit when deep runs are common enough to measure — which is what this
change exists to produce.

*No lookahead.* The genuinely correct question at floor 42 was "can I recover
before the next boss", and it is unanswerable here: `map_select` sends only the
immediately reachable nodes with their row/col, never the graph. Reaching the
right answer needs the mod to send the map. The 0.80 gate catches floor 42
(76/97 = 0.78) but catches it for the wrong reason, and a map where the rest site
sits two rows further on would still fool it.
