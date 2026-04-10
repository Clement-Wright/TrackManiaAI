void SetRecordingMode(bool enabled) {
    g_Bridge.recordingMode = enabled;
    g_Bridge.lastMessage = enabled ? "recording mode enabled" : "recording mode disabled";
}
