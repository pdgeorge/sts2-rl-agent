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
using MegaCrit.Sts2.Core.Map;
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

            // act is 1-based here for readability; the Python side subtracts one
            // to match RunState.current_act_index, which the observation uses.
            SetIfAbsent(state, "act", runState.CurrentActIndex + 1);
            SetIfAbsent(state, "floor", runState.TotalFloor);
            SetIfAbsent(state, "act_floor", runState.ActFloor);
            SetIfAbsent(state, "ascension", runState.AscensionLevel);

            // The full-run observation carries is_elite and is_boss for the room
            // she is standing in. Without them the bridge would leave two dims
            // zero that training filled in, which is a train/deploy mismatch that
            // shows up only as worse play.
            MapPointType pointType =
                runState.CurrentMapPointHistoryEntry?.MapPointType ?? MapPointType.Unassigned;
            SetIfAbsent(state, "room_type", pointType.ToString());

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
            SetIfAbsent(state, "max_potion_slots", player.MaxPotionCount);

            // WHICH relics, not just how many. A count cannot tell Snecko Eye from
            // Ice Cream, and the agent had nothing else -- so every fight was played
            // relic-blind and near-perfect play was impossible however long it
            // trained. Melted relics are spent and excluded, matching relic_count.
            var relicIds = new List<object>();
            foreach (var relic in player.Relics)
            {
                if (relic.IsMelted) continue;
                try { relicIds.Add(relic.Id.Entry); }
                catch (System.Exception ex) { LogOnce($"relic id unreadable: {ex.GetType().Name}"); }
            }
            SetIfAbsent(state, "relics", relicIds);

            // Positional: one entry per slot, null where empty. NOT the combat
            // "potions" list, which is only the usable ones and is not slot
            // indexed -- reading that would put a potion in the wrong column
            // whenever an earlier slot was empty, and the agent picks by slot.
            // An unreadable potion is NOT an empty slot. Both used to serialise
            // as null, so a potion the mod could not name vanished from the
            // state and read downstream as "she was holding nothing" -- which
            // is indistinguishable from the real thing and silently biases any
            // question about what she held versus what she used. It has not
            // fired yet (2,635 recorded states agree with potion_count exactly)
            // and this is here so that if it ever does, it says so instead of
            // quietly deleting the potion.
            var potionSlots = new List<object>();
            try
            {
                foreach (dynamic potion in player.PotionSlots)
                {
                    if (potion == null) { potionSlots.Add(null); continue; }
                    try { potionSlots.Add((string)potion.Id.Entry); }
                    catch (System.Exception ex)
                    {
                        LogOnce($"potion id unreadable: {ex.GetType().Name}");
                        potionSlots.Add("UNREADABLE_POTION");
                    }
                }
            }
            catch (System.Exception ex)
            {
                // Partial list: say so rather than shipping a truncated one that
                // reads as a shorter potion belt than she actually has.
                LogOnce($"potion slots unreadable: {ex.GetType().Name}");
                potionSlots.Add("UNREADABLE_POTION");
            }
            SetIfAbsent(state, "potion_slots", potionSlots);

            // The whole deck by card id, with the upgraded flag. deck_size alone meant
            // every card reward was decided blind to what was being built -- the
            // agent could not tell a deck that already held four Strikes from
            // one that did not. The previous version of this sent bare id
            // strings, which also lost the upgraded flag: a Bash+ read as a
            // base Bash, and the SearchAgent's clone then featured a Vulnerable
            // for 2 turns instead of 3 -- a different fight from the one on
            // screen.
            //
            // The dict form {id, upgraded} is what sts2_env/search/situation.py
            // CombatSituation.from_bridge_state now parses; bare strings still
            // work as a fallback (Python defaults upgraded=False), used by
            // tests and any mod that has not been rebuilt against this change.
            var deckIds = new List<object>();
            try
            {
                foreach (var card in player.Deck.Cards)
                {
                    try
                    {
                        deckIds.Add(new Dictionary<string, object>
                        {
                            ["id"] = card.Id.Entry,
                            ["upgraded"] = card.IsUpgraded,
                        });
                    }
                    catch { }
                }
            }
            catch (System.Exception ex)
            {
                LogOnce($"deck unreadable: {ex.GetType().Name}");
            }
            SetIfAbsent(state, "deck", deckIds);
        }
        catch (System.Exception ex)
        {
            LogOnce($"could not attach run info: {ex.GetType().Name}: {ex.Message}");
        }
    }
}
