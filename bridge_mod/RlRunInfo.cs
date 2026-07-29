// RlRunInfo.cs -- run-level scalars attached to every state the bridge sends.
//
// The agent's observation carries HP, gold, deck size, relic count and potions,
// but the bridge only ever sent act, floor and combat HP. So a policy trained in
// the simulator reads five fields that the live game never fills in, and plays
// against zeros without anything reporting it.
//
// These are added here rather than in each handler because there are eleven state
// builders across six files. Doing it once means a new handler cannot forget, and
// there is one place to change when the observation grows.

using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

internal static class RlRunInfo
{
    /// <summary>
    /// Merge run-level fields into a state message, in place.
    ///
    /// Never throws: a state that reaches the agent missing a field is far better
    /// than a handler that dies mid-run because the run state was not ready. The
    /// fields are simply absent, and the Python side treats absent as zero.
    /// </summary>
    /// <summary>
    /// Attach the run-level fields and serialize, in one call.
    ///
    /// Handlers build their state either into a variable or inline inside
    /// JsonSerializer.Serialize(...). Routing both through here means adding a
    /// field later is one edit rather than eleven, and a handler cannot send a
    /// state that quietly omits them.
    /// </summary>
    public static string Serialize(Dictionary<string, object> state)
    {
        Attach(state);
        return System.Text.Json.JsonSerializer.Serialize(state);
    }

    /// <summary>
    /// Set a field only if the handler has not already provided one.
    ///
    /// The first version of this assigned unconditionally and clobbered the
    /// combat state's "potions", which is a *list* of potion objects, with a
    /// count. compute_action_mask then tried to slice an int and the run died.
    /// A handler's own value is always the more specific one, so it wins.
    /// </summary>
    private static readonly HashSet<string> _logged = new HashSet<string>();

    /// <summary>Log a reason once per session. Attach runs on every state, so an
    /// unconditional log would bury the game's own output.</summary>
    private static void LogOnce(string message)
    {
        if (_logged.Add(message))
        {
            Logger.Log($"[RlRunInfo] {message}");
        }
    }

    private static void SetIfAbsent(Dictionary<string, object> state, string key, object value)
    {
        if (!state.ContainsKey(key))
        {
            state[key] = value;
        }
    }

    public static void Attach(Dictionary<string, object> state)
    {
        // Confirms Attach is reached at all. Two rebuilds were spent on whether
        // this ran or returned early, and the log could not tell them apart.
        LogOnce("attach reached");
        try
        {
            RunState runState = RunManager.Instance?.DebugOnlyGetState();
            if (runState == null)
            {
                // Logged, not silent. The first version returned here without a
                // word, so a recorded run showed every player field absent from
                // all 63 states with nothing in the log to say why -- which cost
                // two rebuilds of guessing.
                LogOnce("run state unavailable; run fields omitted");
                return;
            }

            SetIfAbsent(state, "act", runState.CurrentActIndex + 1);
            SetIfAbsent(state, "floor", runState.TotalFloor);

            // LocalContext.GetMe is how the game itself resolves "the player this
            // client is". Walking runState.Players and taking the first non-null
            // looked equivalent and was not: it produced nothing, so act and floor
            // arrived and every player-derived field was silently absent. No
            // exception, because returning early is not an error -- which is
            // exactly why it needed a trace to catch.
            Player player = LocalContext.GetMe(runState);
            if (player == null)
            {
                LogOnce("LocalContext.GetMe returned null; player fields omitted");
                return;
            }

            SetIfAbsent(state, "gold", player.Gold);
            SetIfAbsent(state, "run_hp", player.Creature?.CurrentHp ?? 0);
            SetIfAbsent(state, "run_max_hp", player.Creature?.MaxHp ?? 0);
            SetIfAbsent(state, "deck_size", player.Deck.Cards.Count);
            // Melted relics are spent, so they should not count toward what she has.
            SetIfAbsent(state, "relic_count", player.Relics.Count(r => !r.IsMelted));
            // Named potion_count, not potions: "potions" is already the combat
            // state's list of usable potions, and a count is a different thing.
            SetIfAbsent(state, "potion_count", player.Potions.Count());
        }
        catch (System.Exception ex)
        {
            LogOnce($"could not attach run info: {ex.GetType().Name}: {ex.Message}");
        }
    }
}
