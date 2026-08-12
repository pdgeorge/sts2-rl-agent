// RlCardRewardScreenHandler.cs -- RL-agent-driven card reward screen handler.
//
// This handles the NCardRewardSelectionScreen overlay. When the card selector
// (RlCardSelector) is active, this screen may not appear because CardSelectCmd
// bypasses it. But if it does appear (e.g. due to some code path not using
// Selector), this handler sends the options to Python and clicks the chosen card.
//
// Falls back to random selection if Python is disconnected or times out.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;

namespace STS2BridgeMod;

public class RlCardRewardScreenHandler : IScreenHandler, IHandler
{
    private const int AgentTimeoutSeconds = 30;
    private const int HandlerTimeoutSeconds = 30;
    private const int InitialSettleDelayMs = 400;
    // UI-transition waits are backstops against a hung game, not pacing.
    // Godot throttles when the window is unfocused, so their wall-clock cost
    // depends on whether anyone is watching -- a 10s one killed a run that had
    // just beaten a boss. The two that fire after every combat are the risky ones.
    private const int CloseTimeoutSeconds = 60;
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(AgentTimeoutSeconds);

    public Type ScreenType => typeof(NCardRewardSelectionScreen);
    public TimeSpan Timeout => TimeSpan.FromSeconds(HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[RlCardReward] Card reward screen appeared");
        NCardRewardSelectionScreen screen =
            (NCardRewardSelectionScreen)NOverlayStack.Instance.Peek();
        await Task.Delay(InitialSettleDelayMs, ct);

        List<NCardHolder> holders = UiHelper.FindAll<NCardHolder>(screen);
        if (holders.Count == 0)
        {
            Logger.Log("[RlCardReward] No card holders found");
            return;
        }

        // Build state message
        var cards = new List<Dictionary<string, object>>();
        for (int i = 0; i < holders.Count; i++)
        {
            var cardData = new Dictionary<string, object>
            {
                ["index"] = i,
            };
            var card = holders[i].CardModel;
            if (card != null)
            {
                cardData["id"] = card.Id.Entry;
                cardData["type"] = card.Type.ToString();
                cardData["cost"] = card.EnergyCost.Canonical;
                if (card.IsUpgraded)
                    cardData["upgraded"] = true;
            }
            cards.Add(cardData);
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = NonCombatBridgeProtocol.CardRewardState,
            ["cards"] = cards,
            // ASK THE SCREEN. The game adds a Skip alternative exactly when
            // `cardReward.CanSkip`, and renders it under "UI/RewardAlternatives",
            // so the button's presence IS the answer. This used to be hardcoded
            // false because a reflection guess at a skip API did not resolve, and
            // the conclusion drawn was that the game has no skip -- so the agent
            // was forced to take every card reward for its entire existence, and
            // arrives at floor 17 with 21-22 cards, nine of them still basic.
            //
            // The old hazard is still real and still handled: claiming an action
            // the game cannot perform hangs the run, because the screen
            // re-presents and a deterministic policy asks for it again. That is
            // why this reports the button rather than an assumption, and why the
            // skip path falls through to taking a card if the button vanishes
            // between the report and the click.
            ["can_skip"] = FindSkipButton(screen) != null,
        };

        NCardHolder chosenHolder = null;

        if (BridgeServer.Instance.IsClientConnected)
        {
            try
            {
                string stateJson = RlRunInfo.Serialize(stateMsg);
                string responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                    stateJson,
                    AgentTimeout, ct);

                if (responseJson != null)
                {
                    using var doc = JsonDocument.Parse(responseJson);
                    var root = doc.RootElement;
                    string action = root.GetProperty("action").GetString() ?? "";

                    if (action == NonCombatBridgeProtocol.SkipAction)
                    {
                        Logger.Log("[RlCardReward] Agent chose to skip");
                        NButton skipButton = FindSkipButton(screen);
                        if (skipButton != null)
                        {
                            await UiHelper.Click(skipButton);
                            Logger.Log("[RlCardReward] Skipped the card reward.");
                            return;
                        }
                        // Falling through to a pick rather than returning. Returning
                        // left the screen open, so it re-presented, and a
                        // deterministic policy asked to skip again -- forever. Taking
                        // a card is a worse decision than skipping; hanging the run
                        // is worse than both.
                        Logger.Log("[RlCardReward] Skip could not be executed; "
                                   + "taking the first card instead so the run continues.");
                    }

                    if (action == NonCombatBridgeProtocol.ChooseAction &&
                        root.TryGetProperty("index", out var idxProp))
                    {
                        int idx = idxProp.GetInt32();
                        if (idx >= holders.Count)
                        {
                            Logger.Log("[RlCardReward] Agent chose to skip via out-of-range choose");
                            NButton outOfRangeSkip = FindSkipButton(screen);
                            if (outOfRangeSkip != null)
                            {
                                await UiHelper.Click(outOfRangeSkip);
                                return;
                            }
                            Logger.Log("[RlCardReward] No skip control; taking the first card.");
                            chosenHolder = holders.Count > 0 ? holders[0] : null;
                        }
                        if (idx >= 0 && idx < holders.Count)
                        {
                            chosenHolder = holders[idx];
                            Logger.Log($"[RlCardReward] Agent chose card at index {idx}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[RlCardReward] Agent error: {ex.Message}");
            }
        }

        // Fallback to random
        if (chosenHolder == null)
        {
            Logger.Log("[RlCardReward] Falling back to random selection");
            chosenHolder = random.NextItem(holders);
        }

        chosenHolder.EmitSignal(NCardHolder.SignalName.Pressed, chosenHolder);
        await WaitHelper.Until(
            () => !GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree(),
            ct, TimeSpan.FromSeconds(CloseTimeoutSeconds),
            "Card reward screen did not close after selection");
        Logger.Log("[RlCardReward] Card reward screen handled");
    }

    /// <summary>The screen's own Skip button, or null when the reward cannot be skipped.</summary>
    ///
    /// THE SKIP WAS ALWAYS THERE. This used to hunt for a "RewardScreen" type by
    /// name and call .Skip()/.Dismiss() on it by reflection; neither resolves, so
    /// the mod concluded the game has no skip control and advertised
    /// can_skip:false forever. The agent was therefore FORCED to take every card
    /// reward for its entire existence, which is why it reaches floor 17 with 21
    /// to 22 cards, nine of them still basic Strike/Defend, and cannot kill a
    /// 173-222 HP boss.
    ///
    /// The game builds it as an ordinary button:
    ///
    ///     if (cardReward.CanSkip)
    ///         list.Add(new CardRewardAlternative("Skip", EndSelectionAndDoNotCompleteReward));
    ///
    /// and NCardRewardSelectionScreen renders each alternative as an
    /// NCardRewardAlternativeButton under "UI/RewardAlternatives", wired to
    /// NClickableControl.Released. So it is found by walking that container, and
    /// its ABSENCE is what "cannot skip" actually means -- the game omits the
    /// button when CanSkip is false, which is the authority we should have been
    /// reading all along.
    private static NButton FindSkipButton(NCardRewardSelectionScreen screen)
    {
        try
        {
            Control container = screen?.GetNodeOrNull<Control>("UI/RewardAlternatives");
            if (container == null)
                return null;

            foreach (Node child in container.GetChildren())
            {
                if (child is not NButton button)
                    continue;
                // Match on the localised title's option id rather than the
                // rendered text, so this does not depend on the client language.
                string name = child.Name.ToString();
                if (name.IndexOf("skip", StringComparison.OrdinalIgnoreCase) >= 0)
                    return button;
            }

            // Fall back to the single alternative when only one exists and it is
            // the skip -- Reroll and Sacrifice only appear with the relics that
            // add them, so a lone alternative on a skippable reward is the skip.
            var buttons = container.GetChildren().OfType<NButton>().ToList();
            if (buttons.Count == 1)
                return buttons[0];
        }
        catch (Exception ex)
        {
            Logger.Log($"[RlCardReward] Skip button lookup failed: {ex.Message}");
        }
        return null;
    }

    private static Type? FindGameType(string typeName)
    {
        foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type? found = null;
            try
            {
                found = assembly.GetTypes().FirstOrDefault(t => t.Name == typeName);
            }
            catch (ReflectionTypeLoadException ex)
            {
                found = ex.Types.FirstOrDefault(t => t?.Name == typeName);
            }

            if (found != null)
                return found;
        }

        return null;
    }
}
