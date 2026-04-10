bool TryRequestRestartMap() {
    auto clientApi = TryGetClientScriptApi();
    if (clientApi is null) {
        g_Bridge.lastMessage = "reset_to_start unavailable: client playground API is not ready";
        return false;
    }

    clientApi.RequestRestartMap();
    return true;
}

bool ExecuteResetToStart(int timeoutMs, Json::Value@ &out payload, string &out message) {
    if (timeoutMs <= 0) {
        message = "timeout_ms must be positive";
        return false;
    }

    if (g_Bridge.latest.mapUid.Length == 0) {
        message = "reset_to_start requires an active map";
        return false;
    }

    const string previousRunId = g_Bridge.runId;
    const string expectedMapUid = g_Bridge.latest.mapUid;
    const uint64 previousFrameId = g_Bridge.latest.frameId;
    const uint64 deadlineNs = GetMonotonicTimestampNs() + (uint64(timeoutMs) * 1000000);

    g_Bridge.pendingReset = true;
    g_Bridge.pendingResetPreviousRunId = previousRunId;
    g_Bridge.pendingResetExpectedMapUid = expectedMapUid;
    g_Bridge.pendingResetAfterFrameId = previousFrameId;

    if (!TryRequestRestartMap()) {
        g_Bridge.pendingReset = false;
        message = g_Bridge.lastMessage;
        return false;
    }

    while (GetMonotonicTimestampNs() < deadlineNs) {
        auto latest = g_Bridge.latest;
        if (latest !is null) {
            if (latest.mapUid.Length > 0 && latest.mapUid != expectedMapUid) {
                g_Bridge.pendingReset = false;
                message = "map changed during reset_to_start";
                return false;
            }
            if (latest.frameId > previousFrameId && latest.runId != previousRunId && IsStartLineSnapshot(latest)) {
                @payload = Json::Object();
                payload["run_id"] = Json::Value(latest.runId);
                payload["frame_id"] = Json::Value(latest.frameId);
                payload["timestamp_ns"] = Json::Value(latest.timestampNs);
                payload["map_uid"] = Json::Value(latest.mapUid);
                payload["race_state"] = Json::Value(latest.raceState);
                message = "reset_to_start acknowledged";
                return true;
            }
        }
        sleep(10);
    }

    g_Bridge.pendingReset = false;
    message = "reset_to_start timed out waiting for a new start-line frame";
    return false;
}
