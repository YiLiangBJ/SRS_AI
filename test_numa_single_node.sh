#!/bin/bash
# NUMA Node CPU Utilization Test Script for Linux
# This script helps test CPU utilization on a single NUMA node to diagnose threading issues

echo "🔧 NUMA Node CPU Utilization Test for Linux"
echo "============================================="

# Function to check if we're on Linux
check_linux() {
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        echo "❌ This script is designed for Linux systems with NUMA support"
        echo "💡 For Windows, use: python train_distributed.py --force-single-numa"
        exit 1
    fi
}

# Function to show NUMA information
show_numa_info() {
    echo "🗺️ NUMA Topology Information"
    echo "============================"
    
    echo "💻 CPU Information:"
    lscpu | grep -E "(NUMA|Socket|Core|Thread|CPU\(s\))"
    echo ""
    
    echo "🎯 NUMA Hardware Information:"
    numactl --hardware
    echo ""
    
    echo "📊 Current NUMA Statistics:"
    numastat
    echo ""
}

# Function to monitor CPU usage in real-time
monitor_cpu_usage() {
    local numa_node=$1
    echo "📈 Monitoring CPU usage for NUMA node $numa_node"
    echo "Press Ctrl+C to stop monitoring"
    echo ""
    
    # Monitor overall CPU usage
    echo "🖥️ System-wide CPU usage:"
    vmstat 1 5 &
    VMSTAT_PID=$!
    
    # Monitor per-core CPU usage
    echo ""
    echo "📊 Per-core CPU usage (updating every 2 seconds):"
    mpstat -P ALL 2 &
    MPSTAT_PID=$!
    
    # Wait for user to stop
    sleep 10
    
    # Stop monitoring
    kill $VMSTAT_PID 2>/dev/null
    kill $MPSTAT_PID 2>/dev/null
    
    echo "✅ CPU monitoring stopped"
}

# Function to test training on specific NUMA node
test_single_numa_training() {
    local numa_node=$1
    echo "🧪 Testing Training on NUMA Node $numa_node"
    echo "==========================================="
    
    echo "🚀 Starting training with forced NUMA node $numa_node..."
    echo "💡 Monitor CPU usage with: htop, top -H, or another terminal with 'watch -n 1 numastat'"
    echo ""
    
    # Start training in background and get its PID
    python train_distributed.py \
        --force-single-numa \
        --numa-node-id $numa_node \
        --num-epochs 2 \
        --batch-size 32 \
        --debug &
    
    TRAIN_PID=$!
    echo "✅ Training started with PID: $TRAIN_PID"
    
    # Give training time to start up
    sleep 10
    
    # Check if process is still running
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "❌ Training process died. Check for errors."
        return 1
    fi
    
    echo "🔍 Process and Thread Information:"
    echo "Process status:"
    ps -eLo pid,tid,stat,comm | grep $TRAIN_PID | head -10
    echo ""
    
    echo "CPU affinity:"
    taskset -cp $TRAIN_PID 2>/dev/null || echo "Could not get CPU affinity"
    echo ""
    
    echo "NUMA memory binding:"
    numactl --show --pid $TRAIN_PID 2>/dev/null || echo "Could not get NUMA binding"
    echo ""
    
    # Monitor for 30 seconds
    echo "⏱️ Monitoring for 30 seconds..."
    for i in {1..6}; do
        echo "📊 Sample $i/6:"
        echo "  CPU usage: $(ps -p $TRAIN_PID -o %cpu= 2>/dev/null || echo 'N/A')%"
        echo "  Memory usage: $(ps -p $TRAIN_PID -o %mem= 2>/dev/null || echo 'N/A')%"
        echo "  Thread count: $(ps -eLo pid | grep $TRAIN_PID | wc -l)"
        
        # Check thread states
        thread_states=$(ps -eLo pid,stat | grep $TRAIN_PID | awk '{print $2}' | sort | uniq -c)
        echo "  Thread states: $thread_states"
        echo ""
        sleep 5
    done
    
    # Stop training
    echo "🛑 Stopping training..."
    kill $TRAIN_PID 2>/dev/null
    wait $TRAIN_PID 2>/dev/null
    
    echo "✅ Training test completed"
}

# Function to compare CPU states before and during training
compare_cpu_states() {
    local numa_node=$1
    
    echo "🔍 CPU State Comparison for NUMA Node $numa_node"
    echo "==============================================="
    
    echo "📋 BEFORE training:"
    echo "CPU usage:"
    mpstat 1 1 | tail -n +4
    echo ""
    
    echo "NUMA statistics:"
    numastat
    echo ""
    
    echo "🚀 Starting training..."
    python train_distributed.py \
        --force-single-numa \
        --numa-node-id $numa_node \
        --num-epochs 1 \
        --batch-size 16 \
        --debug &
    
    TRAIN_PID=$!
    sleep 15  # Let training ramp up
    
    echo "📋 DURING training:"
    echo "CPU usage:"
    mpstat 1 1 | tail -n +4
    echo ""
    
    echo "NUMA statistics:"
    numastat
    echo ""
    
    echo "Per-core CPU usage:"
    mpstat -P ALL 1 1 | grep -E "(Average|CPU)"
    echo ""
    
    # Stop training
    kill $TRAIN_PID 2>/dev/null
    wait $TRAIN_PID 2>/dev/null
    
    echo "✅ Comparison completed"
}

# Main menu
main_menu() {
    echo "🎯 NUMA CPU Utilization Test Options"
    echo "===================================="
    echo "1. Show NUMA topology information"
    echo "2. Test training on NUMA node 0"
    echo "3. Test training on NUMA node 1"
    echo "4. Compare CPU states (before/during training)"
    echo "5. Monitor CPU usage in real-time"
    echo "6. Quick test (recommended)"
    echo ""
    read -p "Choose an option (1-6): " choice
    
    case $choice in
        1)
            show_numa_info
            ;;
        2)
            test_single_numa_training 0
            ;;
        3)
            test_single_numa_training 1
            ;;
        4)
            read -p "Which NUMA node to test? (0 or 1): " node
            compare_cpu_states $node
            ;;
        5)
            read -p "Which NUMA node to monitor? (0 or 1): " node
            monitor_cpu_usage $node
            ;;
        6)
            echo "🚀 Running quick test on NUMA node 0..."
            show_numa_info
            test_single_numa_training 0
            ;;
        *)
            echo "❌ Invalid option"
            exit 1
            ;;
    esac
}

# Script execution
check_linux

echo "🔧 Prerequisites check:"
echo "======================="

# Check for required tools
missing_tools=""
for tool in numactl lscpu mpstat vmstat htop; do
    if ! command -v $tool &> /dev/null; then
        missing_tools="$missing_tools $tool"
    else
        echo "✅ $tool: available"
    fi
done

if [ ! -z "$missing_tools" ]; then
    echo "⚠️ Missing tools:$missing_tools"
    echo "💡 Install with: sudo apt-get install numactl sysstat htop"
    echo ""
fi

# Check if we have NUMA nodes
numa_nodes=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
if [ "$numa_nodes" -lt 2 ]; then
    echo "⚠️ Only $numa_nodes NUMA node(s) detected"
    echo "💡 This test is most useful on multi-NUMA systems"
    echo ""
fi

echo ""

# Run main menu or direct command
if [ $# -eq 0 ]; then
    main_menu
else
    case $1 in
        "info")
            show_numa_info
            ;;
        "test")
            test_single_numa_training ${2:-0}
            ;;
        "compare")
            compare_cpu_states ${2:-0}
            ;;
        "monitor")
            monitor_cpu_usage ${2:-0}
            ;;
        *)
            echo "Usage: $0 [info|test [node]|compare [node]|monitor [node]]"
            exit 1
            ;;
    esac
fi
