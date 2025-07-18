# PowerShell script for launching SRS training on Windows
# Usage: .\launch_training.ps1 [options]

param(
    [int]$NumEpochs = 100,
    [int]$BatchSize = 64,
    [double]$LearningRate = 0.0001,
    [string]$Method = "auto",
    [int]$WorldSize = 0,
    [switch]$NoDDP,
    [switch]$UseTorchrun,
    [switch]$Debug,
    [switch]$Profile,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "SRS Channel Estimation Training Launcher (Windows)"
    Write-Host "Usage: .\launch_training.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -NumEpochs <int>       Number of training epochs (default: 100)"
    Write-Host "  -BatchSize <int>       Total batch size (default: 64)"
    Write-Host "  -LearningRate <double> Learning rate (default: 0.0001)"
    Write-Host "  -Method <string>       Launch method: auto, single, ddp-spawn, ddp-torchrun (default: auto)"
    Write-Host "  -WorldSize <int>       Number of processes for DDP (default: auto-detect)"
    Write-Host "  -NoDDP                 Disable DDP even if multiple GPUs available"
    Write-Host "  -UseTorchrun           Prefer torchrun over mp.spawn"
    Write-Host "  -Debug                 Enable debug mode"
    Write-Host "  -Profile               Enable profiling"
    Write-Host "  -Help                  Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\launch_training.ps1                              # Auto-detect and launch"
    Write-Host "  .\launch_training.ps1 -Method single               # Single-process training"
    Write-Host "  .\launch_training.ps1 -Method ddp-spawn            # DDP with mp.spawn"
    Write-Host "  .\launch_training.ps1 -NumEpochs 200 -BatchSize 128  # Custom parameters"
    exit 0
}

Write-Host "🚀 SRS Channel Estimation Training Launcher (Windows)" -ForegroundColor Green
Write-Host "=" * 50

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "🐍 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if the training script exists
if (-not (Test-Path "launch_training.py")) {
    Write-Host "❌ launch_training.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    exit 1
}

# Build command arguments
$args = @()
$args += "--num-epochs", $NumEpochs
$args += "--batch-size", $BatchSize
$args += "--learning-rate", $LearningRate
$args += "--method", $Method

if ($WorldSize -gt 0) {
    $args += "--world-size", $WorldSize
}

if ($NoDDP) {
    $args += "--no-ddp"
}

if ($UseTorchrun) {
    $args += "--use-torchrun"
}

if ($Debug) {
    $args += "--debug"
}

if ($Profile) {
    $args += "--profile"
}

# Launch the training
Write-Host "💻 Launching training..." -ForegroundColor Cyan
Write-Host "Command: python launch_training.py $($args -join ' ')" -ForegroundColor Gray

try {
    & python launch_training.py @args
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "❌ Failed to launch training: $_" -ForegroundColor Red
    exit 1
}
