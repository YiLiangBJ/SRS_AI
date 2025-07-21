#!/bin/bash
# Linux script for launching NUMA-aware SRS training
# Usage: ./launch_training_numa.sh [options]

# Default parameters
NUM_EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.0001
WORLD_SIZE=0
ENABLE_DDP=false
TEST_NUMA=false
DEBUG=false
PROFILE=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --enable-ddp)
            ENABLE_DDP=true
            shift
            ;;
        --test-numa)
            TEST_NUMA=true
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
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Show help
if [[ "$HELP" == true ]]; then
    echo "NUMA-aware SRS Channel Estimation Training Launcher (Linux)"
    echo "Usage: ./launch_training_numa.sh [options]"
    echo ""
    echo "Options:"
    echo "  --num-epochs <int>       Number of training epochs (default: 100)"
    echo "  --batch-size <int>       Total batch size (default: 64)"
    echo "  --learning-rate <float>  Learning rate (default: 0.0001)"
    echo "  --world-size <int>       Number of processes (default: auto-detect based on NUMA)"
    echo "  --enable-ddp             Enable distributed training (recommended for multi-NUMA servers)"
    echo "  --test-numa              Run NUMA optimization test instead of training"
    echo "  --debug                  Enable debug mode"
    echo "  --profile                Enable profiling"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch_training_numa.sh                                    # Auto-detect NUMA and launch"
    echo "  ./launch_training_numa.sh --test-numa                        # Test NUMA detection and binding"
    echo "  ./launch_training_numa.sh --enable-ddp                       # NUMA-aware DDP training"
    echo "  ./launch_training_numa.sh --num-epochs 200 --batch-size 128  # Custom parameters"
    echo ""
    echo "Note: This script automatically detects NUMA topology and optimizes training accordingly."
    echo "      On servers with multiple NUMA nodes, DDP is recommended for optimal performance."
    exit 0
fi

echo "🚀 NUMA-aware SRS Channel Estimation Training Launcher (Linux)"
echo "=================================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "🐍 Python: $PYTHON_VERSION"

# Check if the training script exists
if [[ ! -f "train_distributed.py" ]]; then
    echo "❌ train_distributed.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check system capabilities
echo "🔍 Checking system capabilities..."

# Check for NUMA tools
if command -v lscpu &> /dev/null; then
    echo "✅ lscpu available - NUMA detection enabled"
else
    echo "⚠️ lscpu not available - limited NUMA detection"
fi

if command -v taskset &> /dev/null; then
    echo "✅ taskset available - CPU affinity control enabled"
else
    echo "⚠️ taskset not available - CPU affinity control disabled"
fi

if command -v numactl &> /dev/null; then
    echo "✅ numactl available - Advanced NUMA control available"
else
    echo "⚠️ numactl not available - Using basic NUMA detection"
fi

# Check if NUMA test is requested
if [[ "$TEST_NUMA" == true ]]; then
    echo "🧪 Running NUMA optimization test..."
    python3 test_numa_optimization.py
    exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ NUMA test completed successfully!"
    else
        echo "❌ NUMA test failed with exit code $exit_code"
        exit $exit_code
    fi
    exit 0
fi

# Build command arguments
ARGS=(
    "--num-epochs" "$NUM_EPOCHS"
    "--batch-size" "$BATCH_SIZE"
    "--learning-rate" "$LEARNING_RATE"
)

if [[ $WORLD_SIZE -gt 0 ]]; then
    ARGS+=("--world-size" "$WORLD_SIZE")
fi

if [[ "$ENABLE_DDP" == true ]]; then
    ARGS+=("--enable-ddp")
fi

if [[ "$DEBUG" == true ]]; then
    ARGS+=("--debug")
fi

if [[ "$PROFILE" == true ]]; then
    ARGS+=("--profile")
fi

# Launch the training
echo "💻 Launching NUMA-aware training..."
echo "Command: python3 train_distributed.py ${ARGS[*]}"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=1  # Prevent OpenMP from interfering with PyTorch threading
export MKL_NUM_THREADS=1  # Prevent MKL from interfering with PyTorch threading

python3 train_distributed.py "${ARGS[@]}"
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $exit_code"
    exit $exit_code
fi
