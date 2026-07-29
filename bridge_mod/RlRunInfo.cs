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

    public static void Attach(Dictionary<string, object> state)
    {
        try
        {
            RunState runState = RunManager.Instance?.DebugOnlyGetState();
            if (runState == null)
            {
                return;
            }

            state["act"] = runState.CurrentActIndex + 1;
            state["floor"] = runState.TotalFloor;

            // The run holds several players in multiplayer; the agent plays one.
            Player player = null;
            foreach (Player candidate in runState.Players)
            {
                if (candidate != null)
                {
                    player = candidate;
                    break;
                }
            }
            if (player == null)
            {
                return;
            }

            state["gold"] = player.Gold;
            state["hp"] = player.Creature?.CurrentHp ?? 0;
            state["max_hp"] = player.Creature?.MaxHp ?? 0;

            state["deck_size"] = player.Deck.Cards.Count;
            // Melted relics are spent, so they should not count toward what she has.
            state["relics"] = player.Relics.Count(r => !r.IsMelted);
            state["potions"] = player.Potions.Count();
        }
        catch (System.Exception ex)
        {
            Logger.Log($"[RlRunInfo] Could not attach run info: {ex.Message}");
        }
    }
}
