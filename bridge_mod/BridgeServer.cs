// BridgeServer.cs -- TCP server for RL agent communication.
//
// Protocol: newline-delimited JSON over TCP (one JSON object per line).
//   Game -> Agent:  state messages (combat_action, map_select, card_reward, etc.)
//   Agent -> Game:  action messages (play, end_turn, choose, skip)
//
// Threading model:
//   - TCP accept/read loop runs on a background thread (Task.Run)
//   - Game handlers call SendState() and WaitForActionAsync() from the game thread
//   - WaitForActionAsync() blocks the calling async context until a response arrives
//
// The server accepts exactly one client at a time. If the client disconnects,
// it goes back to listening for a new connection.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace STS2BridgeMod;

/// <summary>
/// A pending request was superseded by a newer one before the agent answered.
///
/// This is NOT a failure, and the difference matters enormously. Only one
/// request can be outstanding at a time, so opening a nested prompt cancels
/// whatever was already waiting -- and until this type existed, that cancellation
/// arrived at the caller as a bare `null`, indistinguishable from "the agent
/// timed out" or "the client is gone".
///
/// RlCombatHandler treats a null as the agent being unreachable and ends the
/// run. Observed live on 2026-08-08: playing Battle Trance, Armaments, Burning
/// Pact or Acrobatics opened a card-select mid-resolution, that select's request
/// superseded the combat request still in flight, the combat handler read null,
/// declared the agent gone and went to wait for a rewards screen -- while the
/// fight was still going. Thirty seconds later the AutoSlay watchdog killed the
/// run and it was recorded as "terminated" with the player alive and healthy.
///
/// Three of five live sessions died this way. Pre-emption is routine; say so.
/// </summary>
public sealed class RequestPreemptedException : Exception
{
    public RequestPreemptedException(string? supersededId, string? bySender)
        : base($"Request {supersededId ?? "?"} was superseded by {bySender ?? "a newer request"}")
    {
    }
}

public class BridgeServer
{
    public static readonly BridgeServer Instance = new();

    private TcpListener? _listener;
    private TcpClient? _client;
    private NetworkStream? _stream;
    private readonly object _lock = new();
    private bool _running;
    private CancellationTokenSource? _cts;

    private readonly byte[] _readBuffer = new byte[8192];
    private string _readRemainder = "";

    // Pending action response mechanism: when a handler sends state and waits
    // for a response, it sets _pendingAction. The read loop sets the result
    // when a complete line arrives.
    private TaskCompletionSource<string>? _pendingAction;
    private string? _pendingRequestId;
    private readonly object _pendingLock = new();
    private long _requestCounter;

    /// <summary>
    /// Whether a Python client is currently connected.
    /// </summary>
    public bool IsClientConnected
    {
        get
        {
            lock (_lock)
            {
                return _client?.Connected == true;
            }
        }
    }

    private BridgeServer() { }

    /// <summary>
    /// Start listening for client connections on the given port.
    /// </summary>
    public void Start(int port)
    {
        if (_running) return;
        _running = true;
        _cts = new CancellationTokenSource();

        _listener = new TcpListener(IPAddress.Loopback, port);
        _listener.Start();
        Logger.Log($"[BridgeServer] Listening on 127.0.0.1:{port}");

        Task.Run(() => AcceptLoopAsync(_cts.Token));
    }

    /// <summary>
    /// Stop the server and disconnect any client.
    /// </summary>
    public void Stop()
    {
        _running = false;
        _cts?.Cancel();
        CancelPendingAction("Server stopped");
        DisconnectClient();
        _listener?.Stop();
        Logger.Log("[BridgeServer] Server stopped.");
    }

    /// <summary>
    /// Send a state JSON message to the connected client.
    /// Thread-safe; can be called from any thread.
    /// </summary>
    public void SendState(string stateJson)
    {
        SendStateInternal(stateJson);
    }

    private bool SendStateInternal(string stateJson)
    {
        lock (_lock)
        {
            if (_stream == null || _client?.Connected != true)
                return false;

            try
            {
                byte[] data = Encoding.UTF8.GetBytes(stateJson + "\n");
                _stream.Write(data, 0, data.Length);
                _stream.Flush();
                return true;
            }
            catch (Exception ex)
            {
                Logger.Log($"[BridgeServer] Error sending state: {ex.Message}");
                DisconnectClient();
                return false;
            }
        }
    }

    public async Task<string?> SendStateAndWaitForActionAsync(
        string stateJson, TimeSpan timeout, CancellationToken ct)
    {
        string requestId = Interlocked.Increment(ref _requestCounter).ToString();
        TaskCompletionSource<string> tcs;
        lock (_pendingLock)
        {
            // Fault rather than cancel, so the superseded caller can tell this
            // apart from a timeout. See RequestPreemptedException.
            _pendingAction?.TrySetException(
                new RequestPreemptedException(_pendingRequestId, requestId));
            tcs = new TaskCompletionSource<string>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            _pendingAction = tcs;
            _pendingRequestId = requestId;
        }

        try
        {
            string payload = AttachRequestId(stateJson, requestId);
            if (!SendStateInternal(payload))
            {
                return null;
            }
            return await WaitForPendingActionAsync(tcs, timeout, ct);
        }
        finally
        {
            lock (_pendingLock)
            {
                if (_pendingAction == tcs)
                {
                    _pendingAction = null;
                    _pendingRequestId = null;
                }
            }
        }
    }

    /// <summary>
    /// Wait for the next action message from the Python client.
    /// This is the primary mechanism for handlers to receive agent decisions.
    ///
    /// Returns the raw JSON string of the action message, or null on timeout.
    /// </summary>
    public async Task<string?> WaitForActionAsync(
        TimeSpan timeout, CancellationToken ct)
    {
        TaskCompletionSource<string> tcs = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            lock (_pendingLock)
            {
                _pendingAction?.TrySetException(
                    new RequestPreemptedException(_pendingRequestId, "an uncorrelated wait"));
                _pendingAction = tcs;
                _pendingRequestId = null;
            }
            return await WaitForPendingActionAsync(tcs, timeout, ct);
        }
        finally
        {
            lock (_pendingLock)
            {
                if (_pendingAction == tcs)
                {
                    _pendingAction = null;
                    _pendingRequestId = null;
                }
            }
        }
    }

    private static async Task<string?> WaitForPendingActionAsync(
        TaskCompletionSource<string> tcs, TimeSpan timeout, CancellationToken ct)
    {
        try
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
                ct, timeoutCts.Token);

            using var reg = linkedCts.Token.Register(() =>
            {
                tcs.TrySetCanceled();
            });

            return await tcs.Task;
        }
        catch (RequestPreemptedException)
        {
            // Deliberately propagated rather than folded into null: the caller
            // has to be able to retry instead of concluding the agent is gone.
            throw;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] WaitForAction error: {ex.Message}");
            return null;
        }
    }

    // ----------------------------------------------------------------
    // Background thread methods
    // ----------------------------------------------------------------

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        while (_running && !ct.IsCancellationRequested)
        {
            try
            {
                Logger.Log("[BridgeServer] Waiting for client connection...");
                var client = await _listener!.AcceptTcpClientAsync(ct);
                Logger.Log(
                    $"[BridgeServer] Client connected from {client.Client.RemoteEndPoint}");

                lock (_lock)
                {
                    client.SendTimeout = 5000;
                    client.ReceiveTimeout = 5000;
                    _client = client;
                    _stream = client.GetStream();
                    _stream.WriteTimeout = 5000;
                    _stream.ReadTimeout = 5000;
                    _readRemainder = "";
                }

                await HandleClientAsync(ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Logger.Log($"[BridgeServer] Accept error: {ex.Message}");
                await Task.Delay(1000, ct);
            }
        }
    }

    private async Task HandleClientAsync(CancellationToken ct)
    {
        try
        {
            while (_running && !ct.IsCancellationRequested)
            {
                NetworkStream? stream;
                lock (_lock)
                {
                    stream = _stream;
                }
                if (stream == null) break;

                int bytesRead = await stream.ReadAsync(
                    _readBuffer, 0, _readBuffer.Length, ct);
                if (bytesRead == 0)
                {
                    Logger.Log("[BridgeServer] Client disconnected (read 0 bytes).");
                    break;
                }

                _readRemainder += Encoding.UTF8.GetString(_readBuffer, 0, bytesRead);

                while (_readRemainder.Contains('\n'))
                {
                    int idx = _readRemainder.IndexOf('\n');
                    string line = _readRemainder[..idx].Trim();
                    _readRemainder = _readRemainder[(idx + 1)..];

                    if (string.IsNullOrEmpty(line))
                        continue;

                    ProcessIncomingMessage(line);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Logger.Log($"[BridgeServer] Client read error: {ex.Message}");
        }
        finally
        {
            CancelPendingAction("Client disconnected");
            DisconnectClient();
        }
    }

    /// <summary>
    /// Process an incoming message from the Python client.
    /// If there's a pending WaitForActionAsync, deliver the message to it.
    /// Otherwise handle special messages (PING).
    /// </summary>
    private void ProcessIncomingMessage(string json)
    {
        try
        {
            // Check for PING
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("action", out var actionProp))
            {
                string action = actionProp.GetString() ?? "";
                if (action.Equals("ping", StringComparison.OrdinalIgnoreCase))
                {
                    SendState("{\"type\":\"pong\"}");
                    return;
                }

                // Session options, sent once on connect. Handled here rather than as
                // a game action because they configure the session rather than a
                // turn, and must not be mistaken for a pending action response.
                if (action.Equals("configure", StringComparison.OrdinalIgnoreCase))
                {
                    if (root.TryGetProperty("speed", out var speedProp))
                    {
                        RlSpeed.Set(speedProp.GetString() ?? "");
                    }
                    if (root.TryGetProperty("allow_random_fallback", out var fbProp)
                        && (fbProp.ValueKind == JsonValueKind.True
                            || fbProp.ValueKind == JsonValueKind.False))
                    {
                        RlSpeed.AllowRandomFallback = fbProp.GetBoolean();
                        Logger.Log("[BridgeServer] Random fallback "
                                   + (RlSpeed.AllowRandomFallback ? "enabled." : "disabled."));
                    }
                    // Force every run onto one seed. Sent from the client so a
                    // reproduction or a paired A/B can be asked for without
                    // relaunching the game; an empty string clears it.
                    if (root.TryGetProperty("seed", out var seedProp)
                        && seedProp.ValueKind == JsonValueKind.String)
                    {
                        string s = seedProp.GetString() ?? "";
                        SeedOverride.FromClient = string.IsNullOrWhiteSpace(s) ? null : s.Trim();
                        Logger.Log(SeedOverride.FromClient == null
                            ? "[BridgeServer] Seed override cleared; runs roll their own."
                            : $"[BridgeServer] Seed FORCED to {SeedOverride.FromClient} for every run.");
                    }
                    SendState("{\"type\":\"configured\",\"speed\":\""
                              + RlSpeed.Current.Name + "\"}");
                    return;
                }
            }

            // Also support legacy "type" field for ping
            if (root.TryGetProperty("type", out var typeProp))
            {
                string type = typeProp.GetString() ?? "";
                if (type.Equals("PING", StringComparison.OrdinalIgnoreCase))
                {
                    SendState("{\"type\":\"pong\"}");
                    return;
                }
            }
        }
        catch
        {
            // If we can't parse, still deliver it to the pending action
        }

        // Deliver to pending WaitForActionAsync
        lock (_pendingLock)
        {
            if (_pendingAction != null)
            {
                try
                {
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    string? requestId = root.TryGetProperty("request_id", out var requestProp)
                        ? requestProp.GetString()
                        : null;
                    if (_pendingRequestId != null && requestId != _pendingRequestId)
                    {
                        Logger.Log(
                            $"[BridgeServer] Dropping stale action for request_id={requestId ?? "null"}, expected {_pendingRequestId}");
                        return;
                    }
                }
                catch
                {
                    if (_pendingRequestId != null)
                    {
                        Logger.Log("[BridgeServer] Dropping unparsable action while waiting for a correlated request.");
                        return;
                    }
                }
                _pendingAction.TrySetResult(json);
                _pendingAction = null;
                _pendingRequestId = null;
                return;
            }
        }

        // No pending action -- log and discard
        Logger.Log($"[BridgeServer] Received action with no handler waiting: {json}");
    }

    private void CancelPendingAction(string reason)
    {
        lock (_pendingLock)
        {
            _pendingAction?.TrySetCanceled();
            _pendingAction = null;
            _pendingRequestId = null;
        }
    }

    private void DisconnectClient()
    {
        lock (_lock)
        {
            _stream?.Close();
            _stream = null;
            _client?.Close();
            _client = null;
        }
    }

    private static string AttachRequestId(string stateJson, string requestId)
    {
        try
        {
            Dictionary<string, object?> payload =
                JsonSerializer.Deserialize<Dictionary<string, object?>>(stateJson)
                ?? new Dictionary<string, object?>();
            payload["request_id"] = requestId;
            return JsonSerializer.Serialize(payload);
        }
        catch
        {
            return stateJson;
        }
    }
}
