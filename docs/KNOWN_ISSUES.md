# Known Issues and Limitations

Current known issues, bugs, and limitations of the STS2 RL Agent project.

---

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

## Intent damage means different things in the simulator and the bridge

**Severity:** High (silent observation mismatch; affects every model trained so far)

**Status:** Open

The same observation slot carries a **base** number in training and a
**modifier-adjusted** number in live play.

Bridge, `bridge_mod/RlCombatHandler.cs:617`:

```csharp
data["intent_damage"] = attackIntent.GetSingleDamage(cs.PlayerCreatures, enemy);
```

That is the game's own calculation, evaluated *against the player creatures*, so
it applies Vulnerable on the player, Strength and Weak on the attacker -- the
number a human sees on the intent.

Simulator, `sts2_env/monsters/intents.py:15`:

```python
damage: int = 0
```

A static literal fixed when the move is defined. Applying 3 Vulnerable to the
player and re-reading the intent gives **3 -> 3, unchanged**.

**Consequence.** `observation.py` writes `intent.damage / 30.0` and
`state_adapter.py` writes `enemy["intent_damage"] / 30.0` into the same column.
A model learns that column as raw base damage and then reads adjusted damage
from the real game. Under Vulnerable the live value is 1.5x what training ever
associated with the same situation. No error is raised on either side -- the
"silently mistranslated" row in the table above, in a new place.

**Also note** that summing visible intents is not a safe estimate of incoming
damage even within the simulator: a probe on seed 0 took 20 HP in one enemy turn
while the individual intent read 3, because the total spans every enemy and all
modifiers.

**Implications for any incoming-damage calculation.** Do not read the intent
number to decide how much block is needed. It is correct live and wrong in the
simulator, which is the worst arrangement -- it would behave when tested by hand
and be silently wrong in the environment that generates all training data.
Simulate the enemy turn on a clone (`clone_combat`, then `end_player_turn`) and
read the HP actually lost. That is exact on both sides and needs no damage
formula to maintain across patches.

**Fix options, none taken yet.** Either compute the intent through the damage
pipeline in the simulator so it matches the game, or have the bridge send the
unmodified base. Matching the game is the better target, since the intent is
what a player sees, but it is the larger change.

---

## `copy.deepcopy` does not produce an independent CombatState

**Severity:** High (silent state corruption for anything that duplicates a combat)

**Status:** Contained by `sts2_env.search.cloning.clone_combat`; the underlying
monster pattern is unchanged.

Monster factories create a `Creature` and close over it:

```python
creature = Creature(max_hp=hp, monster_id=AXE_RUBY_RAIDER_ID)

def swing(combat: CombatState) -> None:
    _deal_damage_to_player(combat, creature, swing_dmg)
    _gain_block(creature, swing_block, combat)   # closure variable
```

`copy.deepcopy` treats a function as atomic and returns *the same function
object*, closure cells included. So a deep-copied combat's `MoveState.effect_fn`
is still bound to the **original** combat's creature. The clone's monster reads
its strength from the original, gains block on the original, and fires on-attack
hooks against the original. Both combats corrupt each other.

**How it presents.** The copy compares equal at copy time. Drive the original
and the copy through an identical action sequence and they diverge several
actions later -- observed at action 12 on seed 7, where one enemy gains 10 block
in the original and 0 in the clone. The amount changes depending on which copy
you step first, because they are fighting over one creature. No exception, no
log line. Exactly the "everything looks fine but a decision is silently
mistranslated" shape as the row in the table above.

**Why it matters beyond search.** Any deck-strength or route evaluation built on
cloned states measures a fight whose monsters are partly somewhere else, and the
error reads as ordinary variance. It would have quietly invalidated the results
rather than failing.

**Containment.** `clone_combat()` deep-copies with an explicit memo, then walks
every copied object and rebuilds any function whose closure cells or defaults
still point at something that was copied. Rebuilding rather than mutating is
required: `cell_contents` is writable, and writing to it would rebind the
original monster's move too. Costs ~0.6 ms standalone and ~3.2 ms inside a run,
against ~0.4 ms and ~2.4 ms for the unsafe `deepcopy`.

**The deeper fix, not taken.** Rewriting the ~121 monster factories to resolve
their creature from `combat` instead of closing over it would make the simulator
clone-safe by construction. That is a large change across the most
parity-sensitive code in the project, so it is recorded here rather than
attempted alongside the feature that found it.

**Use `clone_combat`, never `copy.deepcopy`, on a `CombatState`.**

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
