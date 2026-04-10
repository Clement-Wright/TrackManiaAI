class TelemetrySnapshot {
    string sessionId;
    string runId;
    uint64 frameId;
    uint64 timestampNs;
    string mapUid;
    int raceTimeMs;
    uint cpCount;
    uint cpTarget;
    float speedKmh;
    int gear;
    float rpm;
    bool hasPos;
    vec3 pos;
    bool hasVel;
    vec3 vel;
    bool hasYpr;
    vec3 ypr;
    bool finished;
    string terminalReason;
    string raceState;
}

class BridgeState {
    string sessionId;
    uint64 runCounter;
    string runId;
    uint64 frameId;
    uint64 heartbeatNs;
    string pluginVersion;
    string lastMessage;
    bool recordingMode;
    uint telemetryClients;
    uint commandClients;
    uint mapCpTarget;
    string lastObservedMapUid;
    string previousRaceState;
    string nextTerminalReason;
    bool pendingReset;
    string pendingResetPreviousRunId;
    string pendingResetExpectedMapUid;
    uint64 pendingResetAfterFrameId;
    string latestFrameJson;
    TelemetrySnapshot@ latest;

    BridgeState() {
        @latest = TelemetrySnapshot();
        latestFrameJson = "";
    }
}

Json::Value@ JsonNull() {
    return Json::Value();
}

Json::Value@ JsonStringOrNull(const string &in value) {
    if (value.Length == 0) {
        return JsonNull();
    }
    return Json::Value(value);
}

Json::Value@ JsonVec3OrNull(bool hasValue, const vec3 &in value) {
    if (!hasValue) {
        return JsonNull();
    }
    auto arr = Json::Array();
    arr.Add(Json::Value(value.x));
    arr.Add(Json::Value(value.y));
    arr.Add(Json::Value(value.z));
    return arr;
}

bool IsLoopbackIp(const string &in ip) {
    return ip == "127.0.0.1" || ip == "::1" || ip == "::ffff:127.0.0.1";
}

bool IsActiveRaceState(const string &in raceState) {
    return raceState == RACE_STATE_START_LINE || raceState == RACE_STATE_RUNNING || raceState == RACE_STATE_FINISHED;
}

bool IsStartLineSnapshot(const TelemetrySnapshot@ snapshot) {
    return snapshot !is null
        && snapshot.raceState == RACE_STATE_START_LINE
        && snapshot.raceTimeMs >= 0
        && snapshot.raceTimeMs <= RESET_START_LINE_MAX_RACE_TIME_MS;
}

Json::Value@ BuildTelemetryJson(const TelemetrySnapshot@ snapshot) {
    auto payload = Json::Object();
    payload["session_id"] = Json::Value(snapshot.sessionId);
    payload["run_id"] = Json::Value(snapshot.runId);
    payload["frame_id"] = Json::Value(snapshot.frameId);
    payload["timestamp_ns"] = Json::Value(snapshot.timestampNs);
    payload["map_uid"] = Json::Value(snapshot.mapUid);
    payload["race_time_ms"] = Json::Value(snapshot.raceTimeMs);
    payload["cp_count"] = Json::Value(snapshot.cpCount);
    payload["cp_target"] = Json::Value(snapshot.cpTarget);
    payload["speed_kmh"] = Json::Value(snapshot.speedKmh);
    payload["gear"] = Json::Value(snapshot.gear);
    payload["rpm"] = Json::Value(snapshot.rpm);
    payload["pos_xyz"] = JsonVec3OrNull(snapshot.hasPos, snapshot.pos);
    payload["vel_xyz"] = JsonVec3OrNull(snapshot.hasVel, snapshot.vel);
    payload["yaw_pitch_roll"] = JsonVec3OrNull(snapshot.hasYpr, snapshot.ypr);
    payload["finished"] = Json::Value(snapshot.finished);
    payload["terminal_reason"] = JsonStringOrNull(snapshot.terminalReason);
    return payload;
}

Json::Value@ BuildHealthPayload() {
    auto payload = Json::Object();
    payload["ok"] = Json::Value(true);
    payload["heartbeat_ns"] = Json::Value(g_Bridge.heartbeatNs);
    payload["session_id"] = JsonStringOrNull(g_Bridge.sessionId);
    payload["run_id"] = JsonStringOrNull(g_Bridge.runId);
    payload["map_uid"] = JsonStringOrNull(g_Bridge.latest.mapUid);
    payload["race_state"] = JsonStringOrNull(g_Bridge.latest.raceState);
    payload["recording_mode"] = Json::Value(g_Bridge.recordingMode);
    payload["telemetry_clients"] = Json::Value(int(g_Bridge.telemetryClients));
    payload["command_clients"] = Json::Value(int(g_Bridge.commandClients));
    payload["plugin_version"] = Json::Value(g_Bridge.pluginVersion);
    payload["last_frame_id"] = Json::Value(g_Bridge.latest.frameId);
    payload["last_timestamp_ns"] = Json::Value(g_Bridge.latest.timestampNs);
    payload["message"] = JsonStringOrNull(g_Bridge.lastMessage);
    return payload;
}

Json::Value@ BuildRaceStatePayload() {
    auto payload = Json::Object();
    payload["race_state"] = Json::Value(g_Bridge.latest.raceState);
    payload["session_id"] = Json::Value(g_Bridge.sessionId);
    payload["run_id"] = Json::Value(g_Bridge.runId);
    payload["map_uid"] = Json::Value(g_Bridge.latest.mapUid);
    return payload;
}

Json::Value@ BuildRecordingModePayload() {
    auto payload = Json::Object();
    payload["recording_mode"] = Json::Value(g_Bridge.recordingMode);
    payload["session_id"] = Json::Value(g_Bridge.sessionId);
    payload["run_id"] = Json::Value(g_Bridge.runId);
    payload["map_uid"] = Json::Value(g_Bridge.latest.mapUid);
    return payload;
}

Json::Value@ BuildCommandResponse(const string &in requestId, bool success, const string &in message, Json::Value@ payload) {
    auto response = Json::Object();
    response["request_id"] = Json::Value(requestId);
    response["success"] = Json::Value(success);
    response["message"] = Json::Value(message);
    if (payload is null) {
        response["payload"] = Json::Object();
    } else {
        response["payload"] = payload;
    }
    return response;
}

bool TryParseCommandRequest(const string &in line, string &out requestId, string &out command, Json::Value@ &out payload, string &out error) {
    Json::Value@ root;
    try {
        @root = Json::Parse(line);
    } catch {
        error = "Invalid JSON: " + getExceptionInfo();
        return false;
    }

    if (root is null || root.GetType() != Json::Type::Object) {
        error = "Command request must be a JSON object.";
        return false;
    }
    if (!root.HasKey("request_id") || root["request_id"].GetType() != Json::Type::String) {
        error = "Command request is missing request_id.";
        return false;
    }
    if (!root.HasKey("command") || root["command"].GetType() != Json::Type::String) {
        error = "Command request is missing command.";
        return false;
    }

    requestId = string(root["request_id"]);
    command = string(root["command"]);

    if (!root.HasKey("payload")) {
        @payload = Json::Object();
        return true;
    }
    @payload = root["payload"];
    if (payload is null || payload.GetType() == Json::Type::Null) {
        @payload = Json::Object();
        return true;
    }
    if (payload.GetType() != Json::Type::Object) {
        error = "Command payload must be a JSON object.";
        return false;
    }
    return true;
}
