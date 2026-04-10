[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11.9",
    [switch]$ForceVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-CheckedCommand {
    param(
        [string]$Display,
        [scriptblock]$Command
    )

    Write-Host $Display -ForegroundColor DarkGray
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Display"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PyenvRoot = Join-Path $env:USERPROFILE ".pyenv\pyenv-win"
$PyenvBat = Join-Path $PyenvRoot "bin\pyenv.bat"
$PythonHome = Join-Path $PyenvRoot "versions\$PythonVersion"
$BootstrapPython = Join-Path $PythonHome "python.exe"
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$TmrlData = Join-Path $env:USERPROFILE "TmrlData"
$OpenplanetRoot = Join-Path $env:USERPROFILE "OpenplanetNext"
$OpenplanetPlugin = Join-Path $OpenplanetRoot "Plugins\TMRL_GrabData.op"
$SourcePluginsDir = Join-Path $TmrlData "resources\Plugins"

if (-not (Test-Path $PyenvBat)) {
    throw "pyenv-win was not found at $PyenvBat"
}

if (-not (Test-Path $BootstrapPython)) {
    Write-Step "Installing Python $PythonVersion with pyenv-win"
    Invoke-CheckedCommand "pyenv install $PythonVersion" { & $PyenvBat install $PythonVersion }
}
else {
    Write-Step "Python $PythonVersion already exists in pyenv-win"
}

if (-not (Test-Path $BootstrapPython)) {
    throw "Python $PythonVersion was not found at $BootstrapPython after install."
}

if ((Test-Path $VenvDir) -and $ForceVenv) {
    $ResolvedVenv = (Resolve-Path $VenvDir).Path
    if (-not $ResolvedVenv.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove virtual environment outside repo root: $ResolvedVenv"
    }

    Write-Step "Removing existing .venv"
    Remove-Item -LiteralPath $ResolvedVenv -Recurse -Force
}

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating local .venv"
    Invoke-CheckedCommand "$BootstrapPython -m venv .venv" { & $BootstrapPython -m venv $VenvDir }
}
else {
    Write-Step "Using existing .venv"
}

Write-Step "Upgrading pip tooling"
Invoke-CheckedCommand "python -m pip install --upgrade pip setuptools wheel" {
    & $VenvPython -m pip install --upgrade pip setuptools wheel
}

Write-Step "Installing CUDA-enabled PyTorch"
Invoke-CheckedCommand "python -m pip install torch torchvision torchaudio (cu126)" {
    & $VenvPython -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
}

Write-Step "Installing Phase 1 Python dependencies"
Invoke-CheckedCommand "python -m pip install tmrl rtgym vgamepad pywin32 numpy opencv-python pyarrow omegaconf hydra-core tensorboard" {
    & $VenvPython -m pip install tmrl==0.7.1 rtgym==0.16 vgamepad==0.1.0 pywin32 numpy opencv-python pyarrow omegaconf hydra-core tensorboard
}

Write-Step "Running python -m tmrl --install"
Invoke-CheckedCommand "python -m tmrl --install" { & $VenvPython -m tmrl --install }

if (-not (Test-Path $TmrlData)) {
    throw "TMRL data directory was not created at $TmrlData"
}

Write-Step "Running import verification"
$ImportCheck = @'
import importlib
import torch

modules = ['tmrl', 'rtgym', 'vgamepad', 'win32gui']
for module_name in modules:
    importlib.import_module(module_name)

print('torch_version=', torch.__version__)
print('torch_cuda_available=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available in the bootstrap environment.')
'@
$ImportCheckPath = Join-Path $env:TEMP "tm20_phase1_import_check.py"
Set-Content -LiteralPath $ImportCheckPath -Value $ImportCheck -Encoding Ascii
try {
    Invoke-CheckedCommand "python import check" { & $VenvPython $ImportCheckPath }
}
finally {
    if (Test-Path $ImportCheckPath) {
        Remove-Item -LiteralPath $ImportCheckPath -Force
    }
}

Write-Step "Verifying Openplanet"
if (-not (Test-Path $OpenplanetRoot)) {
    throw "OpenplanetNext was not found at $OpenplanetRoot. Install Openplanet and launch Trackmania once, then rerun this script."
}

if (-not (Test-Path $OpenplanetPlugin)) {
    if (-not (Test-Path $SourcePluginsDir)) {
        throw "TMRL plugin source directory was not found at $SourcePluginsDir"
    }

    Write-Step "Copying TMRL Openplanet plugin into OpenplanetNext"
    Copy-Item -LiteralPath $SourcePluginsDir -Destination $OpenplanetRoot -Recurse -Force
}

if (-not (Test-Path $OpenplanetPlugin)) {
    throw "TMRL_GrabData.op is still missing at $OpenplanetPlugin"
}

Write-Step "Bootstrap complete"
Write-Host "Python: $VenvPython"
Write-Host "TmrlData: $TmrlData"
Write-Host "Openplanet plugin: $OpenplanetPlugin"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Copy %USERPROFILE%\TmrlData\resources\tmrl-test.Map.Gbx into %USERPROFILE%\Documents\Trackmania\Maps\My Maps"
Write-Host "2. Open Trackmania and load the tmrl-test map"
Write-Host "3. Run scripts\run_phase1_smoke.ps1 -Mode Both"
