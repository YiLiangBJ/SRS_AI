#!/bin/bash
# Test script for single NUMA node CPU utilization
# This script forces training to use only one NUMA node to test CPU utilization
# without cross-NUMA communication overhead

echo "🔧 Testing Single NUMA Node CPU Utilization"
echo "=============================================="

# Function to monitor CPU and memory usage
monitor_resources() {
    echo "📊 Starting resource monitoring..."
    
    # Monitor CPU usage by NUMA node
    if command -v numastat &> /dev/null; then
        echo "📈 NUMA statistics (before training):"
        numastat
        echo ""
    fi
    
    # Monitor process and thread status
    echo "🧵 Process/Thread monitoring will start during training..."
    echo "   Use: htop, ps -eLf | grep python, or top -H -p <PID>"
    echo ""
}

# Function to show NUMA topology
show_numa_info() {
    echo "🗺️ NUMA Topology Information:"
    echo "=============================="
    
    if command -v lscpu &> /dev/null; then
        echo "💻 CPU Information:"
        lscpu | grep -E "(NUMA|Socket|Core|Thread)"
        echo ""
    fi
    
    if command -v numactl &> /dev/null; then
        echo "🎯 NUMA Control Information:"
        numactl --hardware
        echo ""
    fi
    
    if [ -d "/sys/devices/system/node" ]; then
        echo "📁 Available NUMA nodes:"
        ls /sys/devices/system/node/node* 2>/dev/null | wc -l
        echo ""
    fi
}

# Function to test different NUMA nodes
test_numa_node() {
    local numa_node=$1
    echo "🧪 Testing NUMA Node $numa_node"
    echo "==============================="
    
    echo "Starting training on NUMA node $numa_node..."
    echo "Monitor with: htop, iotop, or vmstat 1"
    echo ""
    
    # Run training with forced single NUMA node
    python train_distributed.py \
        --force-single-numa \
        --numa-node-id $numa_node \
        --num-epochs 5 \
        --batch-size 64 \
        --debug
}

# Function to compare CPU states
check_cpu_states() {
    echo "🔍 Checking CPU and Thread States"
    echo "=================================="
    
    echo "📋 Current running Python processes:"
    ps aux | grep python | grep -v grep
    echo ""
    
    echo "🧵 Thread states for Python processes:"
    # Find Python processes and show their thread states
    for pid in $(pgrep -f "python.*train_distributed"); do
        echo "Process $pid thread states:"
        ps -eLo pid,tid,stat,comm | grep $pid | head -20
        echo ""
    done
}

# Main execution
main() {
    echo "🚀 Single NUMA Node CPU Utilization Test"
    echo "========================================"
    echo ""
    
    # Check if we're on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        echo "❌ This script is designed for Linux systems with NUMA support"
        exit 1
    fi
    
    # Show system information
    show_numa_info
    
    # Start resource monitoring
    monitor_resources
    
    # Test NUMA node 0 first (most common)
    echo "🎯 Testing NUMA Node 0 (Primary)"
    echo "================================"
    test_numa_node 0 &
    TRAIN_PID=$!
    
    # Monitor for a few seconds, then show CPU states
    sleep 10
    check_cpu_states
    
    # Wait a bit more then check again
    echo "⏱️ Waiting 30 seconds for training to ramp up..."
    sleep 30
    
    echo "🔍 CPU States after 30 seconds:"
    check_cpu_states
    
    # Show NUMA statistics if available
    if command -v numastat &> /dev/null; then
        echo "📈 NUMA statistics (during training):"
        numastat
        echo ""
    fi
    
    # Kill the training process
    echo "🛑 Stopping training process..."
    kill $TRAIN_PID 2>/dev/null
    wait $TRAIN_PID 2>/dev/null
    
    echo ""
    echo "✅ Single NUMA node test completed!"
    echo ""
    echo "💡 Analysis Tips:"
    echo "  - If CPU usage is still low, check if data generation is the bottleneck"
    echo "  - Monitor with 'htop' to see per-core utilization"
    echo "  - Use 'perf top' to see where CPU time is spent"
    echo "  - Check if threads are in 'R' (running) state vs 'S' (sleeping)"
    echo ""
    echo "🔧 Next steps if CPU usage is good:"
    echo "  - Test NUMA node 1: ./test_single_numa.sh 1"
    echo "  - Test cross-NUMA distributed training: python train_distributed.py --enable-ddp"
}

# Allow testing specific NUMA node as argument
if [ $# -eq 1 ]; then
    echo "🎯 Testing specific NUMA node: $1"
    show_numa_info
    test_numa_node $1
else
    main
fi
