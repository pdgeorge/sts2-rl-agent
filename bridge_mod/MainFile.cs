// MainFile.cs -- Entry point for the STS2 RL Bridge Mod.
//
// Strategy: Patch NGame.IsReleaseGame() to return false, which unlocks AutoSlay.
// Then construct an AutoSlayer with our RL handlers replacing the random ones,
// and start it. The TCP server (BridgeServer) provides communication with Python.

using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes;

namespace STS2BridgeMod;

[ModInitializer(nameof(Initialize))]
public partial class MainFile : Node
{
    public const string ModId = "STS2BridgeMod";

    private static Harmony? _harmony;
    private static RlAutoSlayer? _autoSlayer;

    /// <summary>
    /// Lives for the whole session, unlike the 60s startup token. Cancelled on unload
    /// to stop the run loop.
    /// </summary>
    private static CancellationTokenSource? _loopCts;

    private const string MainMenuPath = "/root/Game/RootSceneContainer/MainMenu";
    private const int MainMenuTimeoutSeconds = 120;
    private const int RunTimeoutMinutes = 90;
    private const int MenuSettleMs = 2000;

    public static void Initialize()
    {
        Logger.Log("=== STS2 RL Bridge Mod Initializing ===");

        // Phase 1: Harmony patches
        try
        {
            _harmony = new Harmony(ModId);

            var patchTypes = new Type[]
            {
                typeof(IsReleaseGamePatch),
                typeof(WaitSpeedPatch),

                // CLEARED of the Punch Off crash on 2026-08-11, and restored.
                //
                // The test that cleared it: game seed 6D038P4FSM2F, forced, with
                // this patch removed from the list entirely. It crashed anyway,
                // on the same signature --
                //
                //     PunchOff.PunchEachOther() -> MegaSpineBinding.Call
                //     ERROR: Signal '_internal_spine_objects_invalidated' is
                //            already connected to given callable
                //
                // so the crash does not need us. It is BaseLib's
                // NCreature.SetAnimationTrigger_Patch1 against a game build
                // BaseLib predates: BaseLib.dll is Jul 31 09:59, the game .pck is
                // Jul 31 19:28, 3.4.0 is the newest published, and it already
                // throws MissingFieldException on NTreasureRoom._chestNode in
                // every treasure room.
                //
                // Note for anyone tempted to repeat the test with `--speed
                // normal`: that is NOT this test. It sets AnimMultiplier to 1.0
                // while the prefix stays installed and keeps wrapping every Spine
                // call. Remove the line below instead.
                typeof(AnimationSpeedPatch),
            };

            int patched = 0;
            foreach (var patchType in patchTypes)
            {
                try
                {
                    _harmony.CreateClassProcessor(patchType).Patch();
                    Logger.Log($"  Patched: {patchType.Name}");
                    patched++;
                }
                catch (Exception ex)
                {
                    Logger.Log($"  SKIP: {patchType.Name} - {ex.Message}");
                }
            }
            Logger.Log($"Harmony: {patched}/{patchTypes.Length} patches applied.");
        }
        catch (Exception ex)
        {
            Logger.Log($"Harmony init failed: {ex.Message}");
        }

        // Phase 2: TCP bridge server
        try
        {
            int port = 9002;
            BridgeServer.Instance.Start(port);
            Logger.Log($"TCP server started on port {port}.");
        }
        catch (Exception ex)
        {
            Logger.Log($"TCP server failed: {ex.Message}");
        }

        // Phase 3: Launch AutoSlay with RL handlers on Godot main thread.
        TaskHelper.RunSafely(LaunchRlAutoSlayAsync());
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try
            {
                _loopCts?.Cancel();
            }
            catch { }
            try
            {
                _autoSlayer?.Stop();
            }
            catch { }
            try
            {
                BridgeServer.Instance.Stop();
            }
            catch { }
        };

        Logger.Log("=== STS2 RL Bridge Mod Ready ===");
    }

    /// <summary>
    /// Wait for the game to initialize, then run runs back to back forever.
    ///
    /// This used to start exactly one run and return, which left a dead mod sitting
    /// in a live game after the first death -- the only recovery was restarting the
    /// whole game. A stream needs runs to keep coming, so this loops.
    ///
    /// RlGameOverScreenHandler already clicks Continue and then Return to Main Menu,
    /// so the game is back at the menu by the time a run's task completes; all that
    /// was missing was something to start the next one.
    /// </summary>
    private static async Task LaunchRlAutoSlayAsync()
    {
        // Startup gets a bounded token -- if the game never reaches a menu, failing
        // fast is correct. Note this token EXPIRES after 60s, so it must not be
        // reused for the run loop; doing so would cancel the loop one minute in.
        Logger.Log("[RlAutoSlay] Waiting for NGame.Instance...");
        var startupCts = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        await WaitHelper.Until(() => NGame.Instance != null, startupCts.Token,
            TimeSpan.FromSeconds(60), "NGame.Instance not available");
        Logger.Log("[RlAutoSlay] NGame.Instance available.");

        Node root = ((SceneTree)Engine.GetMainLoop()).Root;

        _loopCts = new CancellationTokenSource();
        CancellationToken ct = _loopCts.Token;

        int runNumber = 0;
        while (!ct.IsCancellationRequested)
        {
            runNumber++;
            try
            {
                // If a previous run died mid-game -- a transition timing out, say --
                // the game is left sitting inside that run and the menu never
                // appears. This used to abort, loop, and abort again: ten identical
                // "Main menu not visible" failures while a run that had just beaten
                // a boss sat abandoned on the act 2 map. Say so plainly, once,
                // rather than spinning silently.
                bool atMenu = root.GetNodeOrNull<Control>(MainMenuPath)?.IsVisibleInTree() ?? false;
                if (!atMenu && runNumber > 1)
                {
                    // A run that ended badly leaves the game over screen up, and
                    // nothing clicks it once the run loop has thrown -- so every
                    // later run waits for a menu that never comes. Observed live:
                    // one death on floor 21 blocked the session until Continue and
                    // Return to Main Menu were clicked by hand.
                    if (!await RlGameOverScreenHandler.TryDismissGameOverAsync(ct))
                    {
                        Logger.Log("[RlAutoSlay] Not at the main menu -- a previous run is "
                                   + "probably still open in-game. Waiting; if this repeats, "
                                   + "the game needs returning to the menu by hand.");
                    }
                }

                await WaitHelper.Until(
                    () => root.GetNodeOrNull<Control>(MainMenuPath)?.IsVisibleInTree() ?? false,
                    ct, TimeSpan.FromSeconds(MainMenuTimeoutSeconds), "Main menu not visible");

                // Wait for an agent before starting. Runs used to begin the moment
                // the game reached a menu, so you always joined one already in
                // progress, and a run started with nobody attached was played
                // entirely by the random fallback while looking like a real run.
                if (!BridgeServer.Instance.IsClientConnected)
                {
                    Logger.Log("[RlAutoSlay] Waiting for an agent to connect before starting a run...");
                    await WaitHelper.Until(() => BridgeServer.Instance.IsClientConnected, ct,
                        TimeSpan.FromHours(12), "No agent connected");
                    Logger.Log("[RlAutoSlay] Agent connected.");
                    // Let the client send its start options before the run begins.
                    await Task.Delay(250, ct);
                }

                // A fresh slayer per run: RunAsync's finally tears its state down, and
                // a new seed per run is the difference between a stream and a rerun.
                _autoSlayer = new RlAutoSlayer();
                string? forced = SeedOverride.Get();
                string seed = forced ?? SeedHelper.GetRandomSeed();
                Logger.Log($"[RlAutoSlay] Starting RL run #{runNumber} with seed: {seed}"
                           + (forced != null ? $"  (FORCED via {SeedOverride.EnvVar})" : ""));
                _autoSlayer.Start(seed);

                // Start() is fire-and-forget. RlAutoSlayer.IsActive is static, set true
                // synchronously inside Start() and cleared in RunAsync's finally, so it
                // is the completion signal.
                await WaitHelper.Until(() => !RlAutoSlayer.IsActive, ct,
                    TimeSpan.FromMinutes(RunTimeoutMinutes), "Run did not finish");

                Logger.Log($"[RlAutoSlay] Run #{runNumber} finished.");
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // One bad run must not end the stream. Log, settle, try again.
                Logger.Log($"[RlAutoSlay] Run #{runNumber} aborted: {ex.Message}");
                try { _autoSlayer?.Stop(); } catch { }
            }

            try
            {
                await Task.Delay(MenuSettleMs, ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        Logger.Log("[RlAutoSlay] Run loop stopped.");
    }
}

// ---------------------------------------------------------------------------
// Harmony Patches
// ---------------------------------------------------------------------------

/// <summary>
/// Patch NGame.IsReleaseGame() to return false. This unlocks the AutoSlay
/// system and other debug features needed for automation.
/// </summary>
[HarmonyPatch(typeof(NGame), nameof(NGame.IsReleaseGame))]
public static class IsReleaseGamePatch
{
    [HarmonyPrefix]
    static bool Prefix(ref bool __result)
    {
        __result = false;
        return false; // skip original
    }
}

/// <summary>Patch Cmd.CustomScaledWait to reduce all timed delays.</summary>
[HarmonyPatch(typeof(Cmd), nameof(Cmd.CustomScaledWait))]
public static class WaitSpeedPatch
{
    public static float WaitMultiplier = 0.1f;

    [HarmonyPrefix]
    static void Prefix(ref float fastSeconds, ref float standardSeconds)
    {
        fastSeconds *= WaitMultiplier;
        standardSeconds *= WaitMultiplier;
    }
}

/// <summary>
/// Force every run onto one seed, for reproducing a specific run.
///
/// `STS2_RL_SEED=VHHTGKTPEZWF` replays that run instead of rolling a new one.
/// The seed was always chosen here and always logged; nothing could ask for a
/// particular one, so a run that crashed the game could only be re-found by
/// playing until it happened again.
///
/// The immediate use is the Punch Off crash. VHHTGKTPEZWF is run #34 of the
/// 2026-08-11T17.41 session, whose log ends inside
/// PunchOff.PunchEachOther -> MegaSpineBinding.Call with a double-connected
/// `_internal_spine_objects_invalidated` signal. Punch Off appeared ZERO times
/// in a later 40-run session, so waiting for it to recur is not a test plan.
///
/// The larger use is A/B testing. Two live arms on the same seeds is a paired
/// comparison, and pairing is what made the offline sweeps able to see effects
/// that 40 unpaired runs cannot -- live act 1 clear rate carries about +/-5.6%
/// at n=40, which is wider than most changes worth making.
///
/// Read per run rather than cached, so it can be changed without a restart.
/// </summary>
internal static class SeedOverride
{
    public const string EnvVar = "STS2_RL_SEED";

    /// <summary>Set by the client's `configure` message. Wins over the env var.
    ///
    /// THE ENV VAR ALONE WAS USELESS FROM THE CLIENT. This code runs inside the
    /// GAME process, which Steam launches separately, so exporting the variable
    /// in front of the Python command set it on the wrong process entirely and
    /// every run still rolled a fresh seed. The env var is kept because it is
    /// the only way in when launching the game by hand, but the bridge is the
    /// path that actually works from live_eval.
    /// </summary>
    public static string? FromClient;

    /// <summary>The forced seed, or null to roll a fresh one.</summary>
    public static string? Get()
    {
        if (!string.IsNullOrWhiteSpace(FromClient)) return FromClient!.Trim();
        string? seed = System.Environment.GetEnvironmentVariable(EnvVar);
        return string.IsNullOrWhiteSpace(seed) ? null : seed.Trim();
    }
}

/// <summary>Patch MegaAnimationState.SetTimeScale to speed up Spine animations.</summary>
[HarmonyPatch(typeof(MegaCrit.Sts2.Core.Bindings.MegaSpine.MegaAnimationState),
    nameof(MegaCrit.Sts2.Core.Bindings.MegaSpine.MegaAnimationState.SetTimeScale))]
public static class AnimationSpeedPatch
{
    public static float AnimMultiplier = 5.0f;

    // The parameter must be named `scale`, not `timeScale`. Harmony binds prefix
    // parameters to the original method's by NAME, and the original is
    // `SetTimeScale(float scale)` -- so the old name threw at patch time and the
    // mod logged "SKIP: AnimationSpeedPatch". Two of three patches applied, the
    // game reported no error, and animations simply ran at 1x forever.
    [HarmonyPrefix]
    static void Prefix(ref float scale)
    {
        scale *= AnimMultiplier;
    }
}

/// <summary>
/// Logging wrapper using GD.Print for Godot console and log file output.
/// </summary>
internal static class Logger
{
    public static void Log(string message)
    {
        GD.Print($"[STS2Bridge] {message}");
    }
}
