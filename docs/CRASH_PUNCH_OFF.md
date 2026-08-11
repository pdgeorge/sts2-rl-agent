# Punch Off crashes the game on room entry

Reproducible, deterministic, and not caused by anything we can switch off.

## Reproduction

Game seed **`6D038P4FSM2F`**, Ironclad, ascension 0. Route to the Underdocks
`?` node that resolves to the Punch Off event. The game dies entering the room.

Reproduced 2026-08-11 on that seed under two different mod configurations, and
twice more on unseeded sessions earlier the same day.

## What dies

The event builds a visual-only combat room and the process ends there:

```
[INFO] Creating NCombatRoom with mode=VisualOnly encounter=PUNCH_OFF_EVENT_ENCOUNTER.
[WARN] Asset not cached: res://scenes/creature_visuals/punch_construct.tscn
<log ends -- no exception, no shutdown>
```

The asset warnings are routine (54 of them across 9 rooms in the same session,
including rooms that loaded fine) and are not the signal.

One occurrence got further and left a managed trace, which is the useful one:

```
PunchOff.AfterEventStarted() -> PunchOff.PunchEachOther()
  -> CreatureCmd.TriggerAnim -> NCreature.SetAnimationTrigger_Patch1
  -> CreatureAnimator.SetNextState -> MegaAnimationState.SetAnimation
  -> MegaSpineBinding.Call

ERROR: Signal '_internal_spine_objects_invalidated' is already connected to
       given callable '<CallableCustom>' in that object.
       at: connect (core/object/object.cpp:1538)

ERROR: Attempt to disconnect a nonexistent connection from
       '<SpineSkeletonDataResource#-9223314039579337134>'.
       Signal: '_internal_spine_objects_invalidated'.
       at: _disconnect (core/object/object.cpp:1621)
       C# backtrace: [0] Godot.GodotObject.Finalize()
                     [1] uint System.GC.RunFinalizers()

ERROR: Condition "p_I->data != this" is true. Returning: false
       at: erase (./core/templates/list.h:230)
       C# backtrace: [0] Godot.GodotObject.Finalize()
                     [1] uint System.GC.RunFinalizers()
```

Read in order: the Spine signal is connected twice; the resource is later
finalized by the GC, which tries to remove a registration that was never
properly made; the intrusive list erase then fails its own invariant. The
process dies without unwinding, which is why the second reproduction has no
trace at all -- it died before reaching the managed frame.

## What has been ruled out

- **Our AnimationSpeedPatch.** It prefixes `MegaAnimationState.SetTimeScale`,
  one call from `SetAnimation`, so it was the obvious suspect. Removed from the
  patch list entirely -- not set to 1.0, which is NOT the same test and is why
  an earlier `--speed normal` run was wrongly read as evidence -- and seed
  `6D038P4FSM2F` still crashed. Restored.
- **A stale BaseLib.** We were on 3.4.0 (NuGet's newest) against a game .pck
  from 2026-07-31 19:28. 3.4.0 is genuinely broken against that build: it does
  `FieldRefAccess<NTreasureRoom, Node2D>("_chestNode")` and the field is now
  `_chestButton`, throwing `MissingFieldException` in every treasure room.
  Updated to 3.4.4 from the GitHub releases page, which NuGet does not carry.
  That fixed the treasure rooms -- zero `_chestNode` exceptions since -- and did
  NOT fix this crash.
- **The asset warnings**, as above.

## What has not been ruled out

Whether vanilla crashes. `NCreature.SetAnimationTrigger_Patch1` is a Harmony
patch and it is BaseLib's -- our mod patches only `Cmd`, `MegaAnimationState`
and `NGame`. But we cannot test without BaseLib, because our mod is built on it.

The test that would separate "game bug" from "BaseLib bug" is a manual,
mods-disabled run on seed `6D038P4FSM2F` played to that event. Until someone
does it, "BaseLib is in the trace" is a hypothesis and the double-connect could
equally be the game's own.

## Cost

Low but not nil. Punch Off appeared zero times in a 40-run session, twice in one
earlier evening. Each occurrence ends the session: the game dies, and the client
gives up after ten reconnection attempts with
`ConnectionError: Could not reconnect to STS2 bridge`.

It cannot be routed around. The crash is in `AfterEventStarted()`, before any
option is offered, so the option-label workaround used for Nutritious Soup does
not apply, and the map shows only `Unknown` until entered.
