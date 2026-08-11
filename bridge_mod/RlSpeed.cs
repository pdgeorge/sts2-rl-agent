// RlSpeed.cs -- how fast the game plays itself.
//
// Turbo is unwatchable on a stream and normal is unusable for gathering traces,
// so this is a preset rather than a constant. Set from Python at connect time,
// so it can change without restarting the game.
//
// IMPORTANT: the slow presets are an explicit delay, NOT the old behaviour.
// Combat used to take ~3 seconds a card because PlayCardAndWaitAsync sampled its
// "before" state after enqueuing the action, missed the change, and burned a
// 3000ms timeout every play. That was a bug, and it was also unreliable -- the
// wait ended on a timeout rather than on the game being ready. Reintroducing it
// as "slow mode" would make slow mode flaky too. A deliberate delay after a
// completed action is slow in the same way and correct.

using System;
using System.Collections.Generic;

namespace STS2BridgeMod;

internal sealed class SpeedPreset
{
    public string Name = "turbo";

    /// <summary>Multiplier on Cmd.CustomScaledWait -- the game's own timed pauses.</summary>
    public float WaitMultiplier = 0.1f;

    /// <summary>Multiplier on Spine animation playback.</summary>
    public float AnimMultiplier = 5.0f;

    /// <summary>Deliberate pause after each completed action, in milliseconds.</summary>
    public int ActionDelayMs = 0;
}

internal static class RlSpeed
{
    // 1.0 multipliers mean "leave the game alone", which is what the game does
    // with its own fast-mode setting on.
    private static readonly Dictionary<string, SpeedPreset> Presets =
        new(StringComparer.OrdinalIgnoreCase)
    {
        // TURBO IS THE FLOOR WORTH HAVING. A `hyper` preset (no waits, 10x
        // animations) was added and removed the same night: measured on a live
        // session at turbo, the median gap between agent actions is 0.23s, and
        // the search alone costs 0.08-0.52s per decision. Animation is already
        // nearly free at 5x, so the remaining wall clock is search and the
        // game's own processing, neither of which a multiplier touches. The p99
        // of 7.7s is act and boss transitions -- also untouched.
        //
        // Against that, WaitMultiplier 0.0 removes every wait, and the waits are
        // where the quiescence and preemption bugs live. Buying ~nothing for
        // that risk is a bad trade, and it would have split the record: every
        // number this project has is on turbo.
        //
        // Animation speed is NOT the Punch Off crash -- that reproduced on seed
        // 6D038P4FSM2F with AnimationSpeedPatch removed entirely. See
        // docs/CRASH_PUNCH_OFF.md before blaming this again.
        ["turbo"] = new SpeedPreset
        {
            Name = "turbo", WaitMultiplier = 0.1f, AnimMultiplier = 5.0f, ActionDelayMs = 0,
        },
        ["fast"] = new SpeedPreset
        {
            Name = "fast", WaitMultiplier = 0.25f, AnimMultiplier = 3.0f, ActionDelayMs = 50,
        },
        ["normal"] = new SpeedPreset
        {
            Name = "normal", WaitMultiplier = 1.0f, AnimMultiplier = 1.0f, ActionDelayMs = 150,
        },
        ["slow"] = new SpeedPreset
        {
            Name = "slow", WaitMultiplier = 1.0f, AnimMultiplier = 1.0f, ActionDelayMs = 2500,
        },
    };

    public static SpeedPreset Current { get; private set; } = Presets["turbo"];

    /// <summary>
    /// Whether combat may play randomly when the agent does not answer.
    ///
    /// Off by default. The fallback logged only to the game's log, so a Python
    /// console stayed silent while the mod played on by itself, and the recorded
    /// trace attributed those actions to the model. Turn it on deliberately if you
    /// want unattended random play; leave it off if you want to trust a trace.
    /// </summary>
    public static bool AllowRandomFallback { get; set; } = false;

    public static IEnumerable<string> Names => Presets.Keys;

    /// <summary>
    /// Switch preset by name. Unknown names are ignored with a log rather than
    /// throwing: a typo in a speed setting should not end a run.
    /// </summary>
    public static bool Set(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return false;
        }
        if (!Presets.TryGetValue(name, out SpeedPreset preset))
        {
            Logger.Log($"[RlSpeed] Unknown speed '{name}'; staying on '{Current.Name}'. "
                       + $"Known: {string.Join(", ", Presets.Keys)}");
            return false;
        }

        Current = preset;
        WaitSpeedPatch.WaitMultiplier = preset.WaitMultiplier;
        AnimationSpeedPatch.AnimMultiplier = preset.AnimMultiplier;
        Logger.Log($"[RlSpeed] Speed set to '{preset.Name}' "
                   + $"(waits x{preset.WaitMultiplier}, anims x{preset.AnimMultiplier}, "
                   + $"pause {preset.ActionDelayMs}ms)");
        return true;
    }
}
