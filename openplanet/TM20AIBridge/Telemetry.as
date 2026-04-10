CTrackMania@ TryGetTmApp() {
    return cast<CTrackMania>(GetApp());
}

CTrackManiaNetwork@ TryGetTmNetwork() {
    auto app = TryGetTmApp();
    if (app is null) {
        return null;
    }
    return cast<CTrackManiaNetwork>(app.Network);
}

CGamePlaygroundClientScriptAPI@ TryGetClientScriptApi() {
    auto network = TryGetTmNetwork();
    if (network is null) {
        return null;
    }
    return network.PlaygroundClientScriptAPI;
}

CSmArenaClient@ TryGetArenaClient() {
    auto app = TryGetTmApp();
    if (app is null) {
        return null;
    }
    return cast<CSmArenaClient>(app.CurrentPlayground);
}

CSmArena@ TryGetArena() {
    auto playground = TryGetArenaClient();
    if (playground is null) {
        return null;
    }
    return cast<CSmArena>(playground.Arena);
}

CSmScriptPlayer@ TryGetScriptPlayer() {
    auto arena = TryGetArena();
    if (arena is null || arena.Players.Length == 0 || arena.Players[0] is null) {
        return null;
    }
    auto player = arena.Players[0];
    if (player is null || player.ScriptAPI is null) {
        return null;
    }
    return cast<CSmScriptPlayer>(player.ScriptAPI);
}

uint DetectCheckpointTarget(CSmArena@ arena) {
    if (arena is null) {
        return g_Bridge.mapCpTarget;
    }

    uint maxWaypointOrder = 0;
    for (uint i = 0; i < arena.MapLandmarks.Length; i++) {
        auto landmark = arena.MapLandmarks[i];
        if (landmark is null || landmark.Waypoint is null) {
            continue;
        }
        if (landmark.Order > maxWaypointOrder) {
            maxWaypointOrder = landmark.Order;
        }
    }

    if (maxWaypointOrder > 0) {
        return maxWaypointOrder;
    }
    return g_Bridge.mapCpTarget;
}

string TryGetMapUid() {
    auto app = TryGetTmApp();
    if (app is null || app.RootMap is null || app.RootMap.MapInfo is null) {
        return "";
    }
    return app.RootMap.MapInfo.MapUid;
}

string DetectRaceState(CSmArenaClient@ playground, CSmScriptPlayer@ player, uint cpCount, float speedKmh) {
    if (playground is null || player is null) {
        return RACE_STATE_OUTSIDE;
    }

    if (playground.GameTerminals.Length > 0 && playground.GameTerminals[0] !is null) {
        if (playground.GameTerminals[0].UISequence_Current == SGamePlaygroundUIConfig::EUISequence::Finish) {
            return RACE_STATE_FINISHED;
        }
    }
    if (player.EndTime > 0) {
        return RACE_STATE_FINISHED;
    }
    if (player.CurrentRaceTime >= 0
        && player.CurrentRaceTime <= RESET_START_LINE_MAX_RACE_TIME_MS
        && cpCount == 0
        && Math::Abs(speedKmh) <= 1.0f) {
        return RACE_STATE_START_LINE;
    }
    return RACE_STATE_RUNNING;
}

void UpdateTelemetrySnapshot() {
    g_Bridge.frameId += 1;
    g_Bridge.heartbeatNs = GetMonotonicTimestampNs();

    auto snapshot = TelemetrySnapshot();
    snapshot.sessionId = g_Bridge.sessionId;
    snapshot.runId = g_Bridge.runId;
    snapshot.frameId = g_Bridge.frameId;
    snapshot.timestampNs = g_Bridge.heartbeatNs;
    snapshot.mapUid = TryGetMapUid();
    snapshot.raceTimeMs = 0;
    snapshot.cpCount = 0;
    snapshot.cpTarget = g_Bridge.mapCpTarget;
    snapshot.speedKmh = 0.0f;
    snapshot.gear = 0;
    snapshot.rpm = 0.0f;
    snapshot.finished = false;
    snapshot.terminalReason = "";
    snapshot.raceState = RACE_STATE_OUTSIDE;

    auto playground = TryGetArenaClient();
    auto arena = TryGetArena();
    auto player = TryGetScriptPlayer();
    const uint checkpointTarget = DetectCheckpointTarget(arena);
    if (checkpointTarget > 0) {
        g_Bridge.mapCpTarget = checkpointTarget;
    }
    if (player !is null) {
        snapshot.raceTimeMs = player.CurrentRaceTime < 0 ? 0 : player.CurrentRaceTime;
        snapshot.cpCount = uint(player.RaceWaypointTimes.Length);
        snapshot.speedKmh = float(player.DisplaySpeed);
        snapshot.gear = player.EngineCurGear;
        snapshot.rpm = player.EngineRpm;
        snapshot.raceState = DetectRaceState(playground, player, snapshot.cpCount, snapshot.speedKmh);
        snapshot.finished = snapshot.raceState == RACE_STATE_FINISHED;

        if (player.IsEntityStateAvailable) {
            snapshot.hasPos = true;
            snapshot.pos = player.Position;
            snapshot.hasVel = true;
            snapshot.vel = player.Velocity;
            snapshot.hasYpr = true;
            snapshot.ypr = vec3(player.AimYaw, player.AimPitch, player.AimRoll);
        }
        if (snapshot.finished && snapshot.cpCount > g_Bridge.mapCpTarget) {
            g_Bridge.mapCpTarget = snapshot.cpCount;
        }
    }

    bool mapChanged = false;
    if (snapshot.mapUid.Length > 0) {
        if (g_Bridge.lastObservedMapUid.Length == 0) {
            g_Bridge.lastObservedMapUid = snapshot.mapUid;
        } else if (snapshot.mapUid != g_Bridge.lastObservedMapUid) {
            g_Bridge.lastObservedMapUid = snapshot.mapUid;
            g_Bridge.mapCpTarget = 0;
            g_Bridge.nextTerminalReason = TERMINAL_REASON_MAP_CHANGED;
            AdvanceRun();
            snapshot.runId = g_Bridge.runId;
            mapChanged = true;
        }
    }

    if (g_Bridge.pendingReset
        && snapshot.mapUid == g_Bridge.pendingResetExpectedMapUid
        && snapshot.frameId > g_Bridge.pendingResetAfterFrameId
        && IsStartLineSnapshot(snapshot)) {
        AdvanceRun();
        g_Bridge.pendingReset = false;
        snapshot.runId = g_Bridge.runId;
    }

    snapshot.cpTarget = g_Bridge.mapCpTarget;

    if (snapshot.finished) {
        snapshot.terminalReason = TERMINAL_REASON_FINISHED;
    } else if (g_Bridge.nextTerminalReason.Length > 0) {
        snapshot.terminalReason = g_Bridge.nextTerminalReason;
        g_Bridge.nextTerminalReason = "";
    } else if (snapshot.raceState == RACE_STATE_OUTSIDE && IsActiveRaceState(g_Bridge.previousRaceState) && !mapChanged) {
        snapshot.terminalReason = TERMINAL_REASON_OUTSIDE_ACTIVE_RACE;
    }

    g_Bridge.previousRaceState = snapshot.raceState;
    @g_Bridge.latest = snapshot;
    g_Bridge.latestFrameJson = Json::Write(BuildTelemetryJson(snapshot));
}

void TelemetryServerLoop() {
    auto server = Net::Socket();
    if (!server.Listen(BRIDGE_HOST, TELEMETRY_PORT)) {
        g_Bridge.lastMessage = "failed to listen on telemetry port";
        LogBridge(g_Bridge.lastMessage);
        return;
    }

    LogBridge("telemetry listener ready on " + BRIDGE_HOST + ":" + Text::Format("%u", TELEMETRY_PORT));

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
            LogBridge("rejected non-loopback telemetry client: " + client.GetRemoteIP());
            client.Close();
            continue;
        }

        g_Bridge.telemetryClients = 1;
        uint64 lastSentFrameId = 0;
        while (!client.IsHungUp()) {
            auto latest = g_Bridge.latest;
            if (latest !is null && latest.frameId != lastSentFrameId) {
                if (!client.WriteLine(g_Bridge.latestFrameJson)) {
                    break;
                }
                lastSentFrameId = latest.frameId;
            }
            yield();
        }

        g_Bridge.telemetryClients = 0;
        client.Close();
    }
}
