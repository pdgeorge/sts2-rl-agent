// RlMapHandler.cs -- RL-agent-driven map navigation handler.
//
// Replaces AutoSlay's MapScreenHandler. Instead of picking the first child
// of the current map point, this handler:
//   1. Enumerates available map nodes (the reachable next nodes)
//   2. Sends them to Python with their types (Monster, Elite, Shop, etc.)
//   3. Waits for the agent to choose a node index
//   4. Clicks the chosen node
//
// Falls back to random selection if Python is disconnected or times out.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public class RlMapHandler : IScreenHandler, IHandler
{
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(30);

    private TaskCompletionSource? _roomEnteredTcs;

    public Type ScreenType => typeof(NMapScreen);
    public TimeSpan Timeout => TimeSpan.FromSeconds(60);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        Logger.Log("[RlMap] Handling map screen");
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;

        // GetNode<T> throws when the node is gone, and this handler can be reached
        // after a run has already ended -- the run loop had no death check, so a
        // death was followed by one more iteration that called straight into here
        // with the game already back at the main menu. The throw propagated out
        // through a Godot synchronisation-context continuation and took the whole
        // GAME down: two SIGABRT coredumps, 16:05 and 18:26.
        NRun runNode = root.GetNodeOrNull<NRun>("/root/Game/RootSceneContainer/Run");
        if (runNode == null)
        {
            Logger.Log("[RlMap] No Run node -- the run is over and the game has left "
                       + "the map. Nothing to navigate.");
            return;
        }

        await WaitHelper.Until(
            () => runNode.GlobalUi.MapScreen.IsVisibleInTree(), ct,
            AutoSlayConfig.mapScreenTimeout, "Map screen not visible");

        List<NMapPoint> allPoints = UiHelper.FindAll<NMapPoint>(runNode.GlobalUi.MapScreen);
        RunState runState = RunManager.Instance.DebugOnlyGetState();

        // Determine available next nodes
        List<NMapPoint> availableNodes;
        if (runState.VisitedMapCoords.Count == 0)
        {
            // First room selection: all nodes in row 0
            availableNodes = allPoints
                .Where(mp => mp.Point.coord.row == 0)
                .ToList();
        }
        else
        {
            // Get the children of the last visited node
            IReadOnlyList<MapCoord> visited = runState.VisitedMapCoords;
            MapCoord lastCoord = visited[visited.Count - 1];
            NMapPoint lastNode = allPoints.First(
                mp => mp.Point.coord.Equals(lastCoord));
            HashSet<MapCoord> childCoords = new HashSet<MapCoord>(
                lastNode.Point.Children.Select(c => c.coord));
            availableNodes = allPoints
                .Where(mp => childCoords.Contains(mp.Point.coord))
                .ToList();
        }

        if (availableNodes.Count == 0)
        {
            Logger.Log("[RlMap] No available nodes found!");
            return;
        }

        // Offer only nodes that can actually be clicked.
        //
        // FindAll returns NMapPoints that are parented but not interactable --
        // observed at the act 1 -> act 2 boundary as two candidates reporting the
        // SAME coord (row 0, col 3), which a map cannot contain. One of them was
        // never going to enable. The agent picked index 0, the wait below timed
        // out after 10s, and PlayRunAsync's catch-all reported the run terminated
        // while the player was alive at 26 HP standing on the act 2 map.
        //
        // Wait for the map to become interactive first, or on a slow transition
        // everything looks disabled and every node gets filtered out.
        try
        {
            await WaitHelper.Until(() => availableNodes.Any(mp => mp.IsEnabled), ct,
                TimeSpan.FromSeconds(10), "No map point became enabled");
        }
        catch (Exception ex)
        {
            // Best effort: fall through with the unfiltered list rather than
            // ending a live run over a UI readiness check.
            Logger.Log($"[RlMap] No node reported enabled ({ex.GetType().Name}); "
                       + "offering all candidates anyway.");
        }

        List<NMapPoint> clickable = availableNodes.Where(mp => mp.IsEnabled).ToList();
        if (clickable.Count > 0 && clickable.Count < availableNodes.Count)
        {
            Logger.Log($"[RlMap] {availableNodes.Count - clickable.Count} of "
                       + $"{availableNodes.Count} candidates are not clickable; dropping them.");
        }
        if (clickable.Count > 0)
        {
            availableNodes = clickable;
        }

        // Build the state message for Python
        var nodes = new List<Dictionary<string, object>>();
        for (int i = 0; i < availableNodes.Count; i++)
        {
            NMapPoint mp = availableNodes[i];
            nodes.Add(new Dictionary<string, object>
            {
                ["index"] = i,
                ["type"] = mp.Point.PointType.ToString(),
                ["row"] = mp.Point.coord.row,
                ["col"] = mp.Point.coord.col,
            });
        }

        var stateMsg = new Dictionary<string, object>
        {
            ["type"] = "map_select",
            ["nodes"] = nodes,
            ["floor"] = runState.TotalFloor,
            ["act"] = runState.CurrentActIndex + 1,
        };

        NMapPoint chosenNode;

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
                    var rRoot = doc.RootElement;
                    int chosenIndex = rRoot.GetProperty("index").GetInt32();

                    if (chosenIndex >= 0 && chosenIndex < availableNodes.Count)
                    {
                        chosenNode = availableNodes[chosenIndex];
                        Logger.Log(
                            $"[RlMap] Agent chose node {chosenIndex}: {chosenNode.Point.PointType} at ({chosenNode.Point.coord.row},{chosenNode.Point.coord.col})");
                    }
                    else
                    {
                        Logger.Log($"[RlMap] Invalid index {chosenIndex}, falling back to random");
                        chosenNode = random.NextItem(availableNodes);
                    }
                }
                else
                {
                    Logger.Log("[RlMap] No response from agent, falling back to random");
                    chosenNode = random.NextItem(availableNodes);
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[RlMap] Agent error: {ex.Message}, falling back to random");
                chosenNode = random.NextItem(availableNodes);
            }
        }
        else
        {
            Logger.Log("[RlMap] No agent connected, selecting random node");
            chosenNode = random.NextItem(availableNodes);
        }

        // Try the agent's choice first, then any other clickable node. A map click
        // that does not land used to throw, and a throw here ends the whole run --
        // so one unclickable node discarded a run that had just beaten a boss.
        // Entering a different room is a worse decision than the agent's; losing
        // the run is worse than both.
        var order = new List<NMapPoint> { chosenNode };
        order.AddRange(availableNodes.Where(mp => mp != chosenNode && mp.IsEnabled));

        for (int attempt = 0; attempt < order.Count; attempt++)
        {
            NMapPoint node = order[attempt];
            if (attempt > 0)
            {
                Logger.Log($"[RlMap] Retrying with {node.Point.PointType} at "
                           + $"({node.Point.coord.row},{node.Point.coord.col})");
            }

            try
            {
                await WaitHelper.Until(() => node.IsEnabled, ct,
                    TimeSpan.FromSeconds(10), "Map point not enabled");
            }
            catch (Exception ex)
            {
                Logger.Log($"[RlMap] {node.Point.PointType} at "
                           + $"({node.Point.coord.row},{node.Point.coord.col}) never enabled: "
                           + ex.GetType().Name);
                continue;
            }

            _roomEnteredTcs = new TaskCompletionSource();
            RunManager.Instance.RoomEntered += OnRoomEntered;
            try
            {
                await UiHelper.Click(node);
                await WaitHelper.ForTask(_roomEnteredTcs.Task, ct,
                    AutoSlayConfig.mapScreenTimeout, "Room not entered after map click");
                Logger.Log("[RlMap] Map navigation complete");
                return;
            }
            catch (Exception ex)
            {
                Logger.Log($"[RlMap] Click did not enter a room ({ex.GetType().Name}).");
            }
            finally
            {
                RunManager.Instance.RoomEntered -= OnRoomEntered;
                _roomEnteredTcs = null;
            }
        }

        // Returning rather than throwing: the run is still alive and the map screen
        // is still up, so the caller gets another attempt. Throwing reported the
        // run as terminated.
        Logger.Log("[RlMap] No map node could be entered; leaving the screen for "
                   + "another attempt rather than ending the run.");
    }

    private void OnRoomEntered()
    {
        _roomEnteredTcs?.TrySetResult();
    }
}
