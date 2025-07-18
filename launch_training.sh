#!/bin/bash
# Shell script for launching SRS training on Linux
# Usage: ./launch_training.sh [options]

# Default values
NUM_EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.0001
METHOD="auto"
WORLD_SIZE=0
NO_DDP=false
USE_TORCHRUN=false
DEBUG=false
PROFILE=false

# Function to show help
show_help() {
    echo "SRS Channel Estimation Training Launcher (Linux)"
    echo "Usage: ./launch_training.sh [options]"
    echo ""
    echo "Options:"
    echo "  -e, --epochs <int>        Number of training epochs (default: 100)"
    echo "  -b, --batch-size <int>    Total batch size (default: 64)"
    echo "  -l, --learning-rate <float> Learning rate (default: 0.0001)"
    echo "  -m, --method <string>     Launch method: auto, single, ddp-spawn, ddp-torchrun (default: auto)"
    echo "  -w, --world-size <int>    Number of processes for DDP (default: auto-detect)"
    echo "  --no-ddp                  Disable DDP even if multiple GPUs available"
    echo "  --use-torchrun            Prefer torchrun over mp.spawn"
    echo "  --debug                   Enable debug mode"
    echo "  --profile                 Enable profiling"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch_training.sh                              # Auto-detect and launch"
    echo "  ./launch_training.sh -m single                    # Single-process training"
    echo "  ./launch_training.sh -m ddp-spawn                 # DDP with mp.spawn"
    echo "  ./launch_training.sh -e 200 -b 128                # Custom parameters"
    echo "  ./launch_training.sh --use-torchrun               # Use torchrun"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -w|--world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --no-ddp)
            NO_DDP=true
            shift
            ;;
        --use-torchrun)
            USE_TORCHRUN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

echo "🚀 SRS Channel Estimation Training Launcher (Linux)"
echo "=================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo "🐍 Python: $PYTHON_VERSION"

# Check if the training script exists
if [ ! -f "launch_training.py" ]; then
    echo "❌ launch_training.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Build command arguments
ARGS=()
ARGS+=("--num-epochs" "$NUM_EPOCHS")
ARGS+=("--batch-size" "$BATCH_SIZE")
ARGS+=("--learning-rate" "$LEARNING_RATE")
ARGS+=("--method" "$METHOD")

if [ "$WORLD_SIZE" -gt 0 ]; then
    ARGS+=("--world-size" "$WORLD_SIZE")
fi

if [ "$NO_DDP" = true ]; then
    ARGS+=("--no-ddp")
fi

if [ "$USE_TORCHRUN" = true ]; then
    ARGS+=("--use-torchrun")
fi

if [ "$DEBUG" = true ]; then
    ARGS+=("--debug")
fi

if [ "$PROFILE" = true ]; then
    ARGS+=("--profile")
fi

# Launch the training
echo "💻 Launching training..."
echo "Command: python launch_training.py ${ARGS[*]}"

python launch_training.py "${ARGS[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
