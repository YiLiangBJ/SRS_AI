# PowerShell script for launching NUMA-aware SRS training on Windows
# Usage: .\launch_training.ps1 [options]

param(
    [int]$NumEpochs = 100,
    [int]$BatchSize = 64,
    [double]$LearningRate = 0.0001,
    [int]$WorldSize = 0,
    [switch]$EnableDDP,
    [switch]$TestNuma,
    [switch]$Debug,
    [switch]$Profile,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "NUMA-aware SRS Channel Estimation Training Launcher (Windows)"
    Write-Host "Usage: .\launch_training.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -NumEpochs <int>       Number of training epochs (default: 100)"
    Write-Host "  -BatchSize <int>       Total batch size (default: 64)"
    Write-Host "  -LearningRate <double> Learning rate (default: 0.0001)"
    Write-Host "  -WorldSize <int>       Number of processes (default: auto-detect based on NUMA)"
    Write-Host "  -EnableDDP             Enable distributed training (recommended for Linux servers)"
    Write-Host "  -TestNuma              Run NUMA optimization test instead of training"
    Write-Host "  -Debug                 Enable debug mode"
    Write-Host "  -Profile               Enable profiling"
    Write-Host "  -Help                  Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\launch_training.ps1                              # Single-process training (Windows)"
    Write-Host "  .\launch_training.ps1 -TestNuma                    # Test NUMA detection and binding"
    Write-Host "  .\launch_training.ps1 -EnableDDP                   # Force DDP (not recommended on Windows)"
    Write-Host "  .\launch_training.ps1 -NumEpochs 200 -BatchSize 128  # Custom parameters"
    Write-Host ""
    Write-Host "Note: On Windows, single-process training is recommended."
    Write-Host "      NUMA-aware DDP is optimized for Linux servers with multiple NUMA nodes."
    exit 0
}

Write-Host "🚀 NUMA-aware SRS Channel Estimation Training Launcher (Windows)" -ForegroundColor Green
Write-Host "=" * 60

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "🐍 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if the training script exists
if (-not (Test-Path "train_distributed.py")) {
    Write-Host "❌ train_distributed.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    exit 1
}

# Check if NUMA test is requested
if ($TestNuma) {
    Write-Host "🧪 Running NUMA optimization test..." -ForegroundColor Cyan
    try {
        & python test_numa_optimization.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ NUMA test completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ NUMA test failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } catch {
        Write-Host "❌ Failed to run NUMA test: $_" -ForegroundColor Red
        exit 1
    }
    exit 0
}

# Build command arguments
$args = @()
$args += "--num-epochs", $NumEpochs
$args += "--batch-size", $BatchSize
$args += "--learning-rate", $LearningRate

if ($WorldSize -gt 0) {
    $args += "--world-size", $WorldSize
}

if ($EnableDDP) {
    $args += "--enable-ddp"
    Write-Host "⚠️ DDP enabled - Note: Single-process training is recommended on Windows" -ForegroundColor Yellow
}

if ($Debug) {
    $args += "--debug"
}

if ($Profile) {
    $args += "--profile"
}

# Launch the training
Write-Host "💻 Launching NUMA-aware training..." -ForegroundColor Cyan
Write-Host "Command: python train_distributed.py $($args -join ' ')" -ForegroundColor Gray

try {
    & python train_distributed.py @args
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
