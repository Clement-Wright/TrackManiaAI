const string PLUGIN_VERSION = "0.2.0";
const string BRIDGE_HOST = "127.0.0.1";
const uint16 TELEMETRY_PORT = 9100;
const uint16 COMMAND_PORT = 9101;

const string RACE_STATE_OUTSIDE = "outside_race";
const string RACE_STATE_START_LINE = "start_line";
const string RACE_STATE_RUNNING = "running";
const string RACE_STATE_FINISHED = "finished";

const string TERMINAL_REASON_FINISHED = "finished";
const string TERMINAL_REASON_OUTSIDE_ACTIVE_RACE = "outside_active_race";
const string TERMINAL_REASON_MAP_CHANGED = "map_changed";

const int RESET_START_LINE_MAX_RACE_TIME_MS = 250;

BridgeState@ g_Bridge = BridgeState();

void LogBridge(const string &in message) {
    trace("[TM20AIBridge] " + message);
}

uint64 GetMonotonicTimestampNs() {
    auto app = GetApp();
    if (app is null) {
        return Time::Now * 1000000;
    }
    return uint64(app.TimeSinceInitMs) * 1000000;
}

string FormatRunId(const string &in sessionId, uint64 runCounter) {
    return sessionId + "-run-" + Text::Format("%06llu", runCounter);
}

void AdvanceRun() {
    g_Bridge.runCounter += 1;
    g_Bridge.runId = FormatRunId(g_Bridge.sessionId, g_Bridge.runCounter);
}

void InitializeBridgeState() {
    g_Bridge.pluginVersion = PLUGIN_VERSION;
    g_Bridge.sessionId = "session-" + Text::Format("%llu", Time::Now) + "-" + Text::Format("%06u", uint(Math::Rand(0, 1000000)));
    g_Bridge.runCounter = 1;
    g_Bridge.runId = FormatRunId(g_Bridge.sessionId, g_Bridge.runCounter);
    g_Bridge.frameId = 0;
    g_Bridge.heartbeatNs = GetMonotonicTimestampNs();
    g_Bridge.recordingMode = false;
    g_Bridge.telemetryClients = 0;
    g_Bridge.commandClients = 0;
    g_Bridge.mapCpTarget = 0;
    g_Bridge.lastObservedMapUid = "";
    g_Bridge.previousRaceState = RACE_STATE_OUTSIDE;
    g_Bridge.nextTerminalReason = "";
    g_Bridge.pendingReset = false;
    g_Bridge.pendingResetPreviousRunId = "";
    g_Bridge.pendingResetExpectedMapUid = "";
    g_Bridge.pendingResetAfterFrameId = 0;
    g_Bridge.lastMessage = "bridge initialized";
}

Json::Value@ HandleCommandLine(const string &in line) {
    string requestId;
    string command;
    Json::Value@ payload;
    string error;
    if (!TryParseCommandRequest(line, requestId, command, payload, error)) {
        return BuildCommandResponse("", false, error, Json::Object());
    }

    if (command == "health") {
        return BuildCommandResponse(requestId, true, "health snapshot", BuildHealthPayload());
    }
    if (command == "race_state") {
        return BuildCommandResponse(requestId, true, "race state snapshot", BuildRaceStatePayload());
    }
    if (command == "set_recording_mode") {
        if (payload is null || !payload.HasKey("enabled") || payload["enabled"].GetType() != Json::Type::Boolean) {
            return BuildCommandResponse(requestId, false, "set_recording_mode requires a boolean enabled payload field", Json::Object());
        }
        SetRecordingMode(bool(payload["enabled"]));
        return BuildCommandResponse(requestId, true, "recording mode updated", BuildRecordingModePayload());
    }
    if (command == "reset_to_start") {
        if (payload is null || !payload.HasKey("timeout_ms") || payload["timeout_ms"].GetType() != Json::Type::Number) {
            return BuildCommandResponse(requestId, false, "reset_to_start requires a numeric timeout_ms payload field", Json::Object());
        }

        Json::Value@ resetPayload;
        string resetMessage;
        if (!ExecuteResetToStart(int(payload["timeout_ms"]), resetPayload, resetMessage)) {
            return BuildCommandResponse(requestId, false, resetMessage, Json::Object());
        }
        return BuildCommandResponse(requestId, true, resetMessage, resetPayload);
    }

    return BuildCommandResponse(requestId, false, "unsupported command: " + command, Json::Object());
}

void CommandServerLoop() {
    auto server = Net::Socket();
    if (!server.Listen(BRIDGE_HOST, COMMAND_PORT)) {
        g_Bridge.lastMessage = "failed to listen on command port";
        LogBridge(g_Bridge.lastMessage);
        return;
    }

    LogBridge("command listener ready on " + BRIDGE_HOST + ":" + Text::Format("%u", COMMAND_PORT));

    while (true) {
        if (!server.IsReady()) {
            yield();
            continue;
        }

        auto client = server.Accept();
        if (client is null) {
            yield();
            continue;
        }
        if (!IsLoopbackIp(client.GetRemoteIP())) {
            LogBridge("rejected non-loopback command client: " + client.GetRemoteIP());
            client.Close();
            continue;
        }

        g_Bridge.commandClients = 1;
        while (!client.IsHungUp()) {
            string line;
            if (client.ReadLine(line)) {
                auto response = HandleCommandLine(line);
                if (!client.WriteLine(Json::Write(response))) {
                    break;
                }
                continue;
            }
            yield();
        }

        g_Bridge.commandClients = 0;
        client.Close();
    }
}

void Main() {
    InitializeBridgeState();
    LogBridge("starting " + PLUGIN_VERSION + " with session_id=" + g_Bridge.sessionId);
    startnew(TelemetryServerLoop);
    startnew(CommandServerLoop);

    while (true) {
        UpdateTelemetrySnapshot();
        yield();
    }
}
