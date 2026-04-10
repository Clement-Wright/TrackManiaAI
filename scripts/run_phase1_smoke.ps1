[CmdletBinding()]
param(
    [ValidateSet("Lidar", "Full", "Both")]
    [string]$Mode = "Both",
    [double]$Seconds = 3.0,
    [string]$Action = "0.8,0.0,0.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-PhaseHeader {
    param([string]$Label)
    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
}

function Invoke-Phase1Pass {
    param(
        [string]$EnvName,
        [string]$CameraHint
    )

    Write-PhaseHeader "Applying $EnvName config"
    & $PythonExe $ApplyConfigScript --env $EnvName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to apply the $EnvName config."
    }

    Write-Host ""
    Write-Host "Manual checklist before continuing:"
    Write-Host "- Trackmania is on the tmrl-test map"
    Write-Host "- The game is in windowed mode"
    Write-Host "- The ghost is hidden with g"
    Write-Host "- $CameraHint"
    Read-Host "Press Enter when Trackmania is ready"

    Write-PhaseHeader "Running $EnvName smoke test"
    & $PythonExe $SmokeScript --env $EnvName --seconds $Seconds --action $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$EnvName smoke test failed."
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ApplyConfigScript = Join-Path $PSScriptRoot "apply_tmrl_config.py"
$SmokeScript = Join-Path $PSScriptRoot "smoke_test_env.py"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment missing at $PythonExe. Run scripts\bootstrap_phase1.ps1 first."
}

switch ($Mode) {
    "Lidar" {
        Invoke-Phase1Pass -EnvName "lidar" -CameraHint "Use camera 3 until the car is hidden."
    }
    "Full" {
        Invoke-Phase1Pass -EnvName "full" -CameraHint "Use camera 1 so the car is visible."
    }
    "Both" {
        Invoke-Phase1Pass -EnvName "lidar" -CameraHint "Use camera 3 until the car is hidden."
        Write-Host ""
        Write-Host "Keep the same Trackmania session open and switch the camera for the full pass."
        Invoke-Phase1Pass -EnvName "full" -CameraHint "Use camera 1 so the car is visible."
    }
}

Write-PhaseHeader "Phase 1 smoke run complete"
