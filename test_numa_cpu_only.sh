#!/bin/bash
# Single NUMA Node CPU Performance Test Script for Linux
# This script disables CUDA and forces CPU-only training to test NUMA performance

echo "🧪 Single NUMA Node CPU Performance Test"
echo "========================================"

# Function to check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check if we're on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        echo "❌ This script is designed for Linux systems"
        exit 1
    fi
    
    # Check for NUMA tools
    if ! command -v numactl &> /dev/null; then
        echo "⚠️ numactl not found. Install with: sudo apt-get install numactl"
    fi
    
    if ! command -v lscpu &> /dev/null; then
        echo "⚠️ lscpu not found. Install with: sudo apt-get install util-linux"
    fi
    
    # Check Python and PyTorch
    if ! command -v python3 &> /dev/null; then
        echo "❌ python3 not found"
        exit 1
    fi
    
    echo "✅ Prerequisites check completed"
}

# Function to display system information
show_system_info() {
    echo ""
    echo "🖥️ System Information"
    echo "===================="
    
    echo "CPU Information:"
    lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core|Socket|NUMA)"
    echo ""
    
    echo "NUMA Topology:"
    numactl --hardware 2>/dev/null || echo "NUMA information not available"
    echo ""
    
    echo "Memory Information:"
    free -h
    echo ""
}

# Function to set environment for CPU-only training
setup_cpu_environment() {
    echo "⚙️ Setting up CPU-only environment"
    echo "=================================="
    
    # Disable CUDA completely
    export CUDA_VISIBLE_DEVICES=""
    export TF_CPP_MIN_LOG_LEVEL=2
    export SIONNA_DISABLE_GPU=1
    
    # Set optimal CPU settings
    export OMP_NUM_THREADS=$(nproc)
    export MKL_NUM_THREADS=$(nproc)
    export NUMEXPR_NUM_THREADS=$(nproc)
    
    # Disable TensorFlow GPU
    export TF_FORCE_CPU_DEVICE=1
    
    echo "✅ Environment variables set:"
    echo "   CUDA_VISIBLE_DEVICES=''"
    echo "   OMP_NUM_THREADS=$(nproc)"
    echo "   TF_FORCE_CPU_DEVICE=1"
    echo ""
}

# Function to test specific NUMA node
test_numa_node() {
    local numa_node=$1
    local num_epochs=${2:-2}
    local batch_size=${3:-16}
    
    echo "🎯 Testing NUMA Node $numa_node"
    echo "==============================="
    echo "Parameters:"
    echo "  - NUMA node: $numa_node"
    echo "  - Epochs: $num_epochs"
    echo "  - Batch size: $batch_size"
    echo "  - Device: CPU (forced)"
    echo ""
    
    # Start training with monitoring
    echo "🚀 Starting training..."
    echo "💡 Monitor in another terminal with:"
    echo "   htop"
    echo "   watch -n 1 'ps -eLo pid,tid,stat,pcpu,comm | grep python'"
    echo "   watch -n 1 numastat"
    echo ""
    
    # Run training
    python3 train_distributed.py \
        --force-single-numa \
        --numa-node-id $numa_node \
        --num-epochs $num_epochs \
        --batch-size $batch_size \
        --debug 2>&1 | tee "numa_node_${numa_node}_test.log"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Training completed successfully"
    else
        echo "⚠️ Training ended with exit code: $exit_code"
    fi
    
    echo "📄 Log saved to: numa_node_${numa_node}_test.log"
    return $exit_code
}

# Function to monitor resources during training
monitor_resources() {
    local numa_node=$1
    local duration=${2:-60}
    
    echo "📊 Monitoring Resources for NUMA Node $numa_node"
    echo "==============================================="
    
    # Start training in background
    python3 train_distributed.py \
        --force-single-numa \
        --numa-node-id $numa_node \
        --num-epochs 10 \
        --batch-size 32 \
        --debug > training_output.log 2>&1 &
    
    local train_pid=$!
    echo "✅ Training started with PID: $train_pid"
    
    # Monitor for specified duration
    echo "⏱️ Monitoring for $duration seconds..."
    for ((i=1; i<=duration; i++)); do
        if ! kill -0 $train_pid 2>/dev/null; then
            echo "⚠️ Training process ended early"
            break
        fi
        
        echo "📊 Sample $i/$duration:"
        echo "  CPU usage: $(ps -p $train_pid -o %cpu= 2>/dev/null || echo 'N/A')%"
        echo "  Memory usage: $(ps -p $train_pid -o %mem= 2>/dev/null || echo 'N/A')%"
        echo "  Thread count: $(ps -eLo pid | grep $train_pid 2>/dev/null | wc -l)"
        
        # Check thread states
        if ps -eLo pid,stat | grep $train_pid &>/dev/null; then
            thread_states=$(ps -eLo pid,stat | grep $train_pid | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')
            echo "  Thread states: $thread_states"
        fi
        
        echo ""
        sleep 1
    done
    
    # Stop training
    echo "🛑 Stopping training..."
    kill $train_pid 2>/dev/null
    wait $train_pid 2>/dev/null
    
    echo "📄 Training output saved to: training_output.log"
}

# Function to run comprehensive test
run_comprehensive_test() {
    echo "🔬 Comprehensive NUMA CPU Performance Test"
    echo "=========================================="
    
    # Test both NUMA nodes if available
    numa_nodes=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
    
    echo "📊 Testing $numa_nodes NUMA node(s)..."
    
    for ((node=0; node<numa_nodes; node++)); do
        echo ""
        echo "🎯 Testing NUMA Node $node"
        echo "=========================="
        
        test_numa_node $node 2 16
        
        if [ $? -eq 0 ]; then
            echo "✅ NUMA node $node: SUCCESS"
        else
            echo "❌ NUMA node $node: FAILED"
        fi
        
        echo ""
        echo "⏱️ Waiting 5 seconds before next test..."
        sleep 5
    done
    
    echo "🏁 Comprehensive test completed"
}

# Main menu
main_menu() {
    echo "🎯 Single NUMA Node CPU Test Options"
    echo "===================================="
    echo "1. Show system information"
    echo "2. Test NUMA node 0"
    echo "3. Test NUMA node 1" 
    echo "4. Monitor resources during training"
    echo "5. Run comprehensive test"
    echo "6. Quick test (recommended)"
    echo ""
    read -p "Choose an option (1-6): " choice
    
    case $choice in
        1)
            show_system_info
            ;;
        2)
            test_numa_node 0
            ;;
        3)
            test_numa_node 1
            ;;
        4)
            read -p "Which NUMA node to monitor? (0 or 1): " node
            read -p "Monitor duration in seconds? (default: 60): " duration
            duration=${duration:-60}
            monitor_resources $node $duration
            ;;
        5)
            run_comprehensive_test
            ;;
        6)
            echo "🚀 Running quick test..."
            show_system_info
            test_numa_node 0 2 16
            ;;
        *)
            echo "❌ Invalid option"
            exit 1
            ;;
    esac
}

# Script execution
echo "🧪 Single NUMA Node CPU Performance Test for Linux"
echo "=================================================="

check_prerequisites
setup_cpu_environment

# Run based on command line arguments or interactive menu
if [ $# -eq 0 ]; then
    main_menu
else
    case $1 in
        "info")
            show_system_info
            ;;
        "test")
            test_numa_node ${2:-0} ${3:-2} ${4:-16}
            ;;
        "monitor")
            monitor_resources ${2:-0} ${3:-60}
            ;;
        "comprehensive")
            run_comprehensive_test
            ;;
        *)
            echo "Usage: $0 [info|test [node] [epochs] [batch_size]|monitor [node] [duration]|comprehensive]"
            echo ""
            echo "Examples:"
            echo "  $0                           # Interactive menu"
            echo "  $0 info                      # Show system info"
            echo "  $0 test 0 5 32              # Test NUMA node 0, 5 epochs, batch size 32"
            echo "  $0 monitor 1 120            # Monitor NUMA node 1 for 120 seconds"
            echo "  $0 comprehensive            # Test all NUMA nodes"
            exit 1
            ;;
    esac
fi
