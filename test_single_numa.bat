@echo off
REM Test script for single NUMA node CPU utilization on Windows
REM This script forces training to use only one NUMA node to test CPU utilization

echo 🔧 Testing Single NUMA Node CPU Utilization (Windows)
echo ======================================================

echo 💻 System Information:
echo =====================
wmic computersystem get TotalPhysicalMemory,NumberOfProcessors,NumberOfLogicalProcessors
echo.

echo 🎯 Testing Single Process Training (Windows doesn't support NUMA binding like Linux)
echo ===================================================================================

echo Starting training with single process...
echo Monitor with: Task Manager (Performance tab) or Resource Monitor
echo.

REM Run training with forced single NUMA node (Windows equivalent)
python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 5 --batch-size 64 --debug

echo.
echo ✅ Single process test completed!
echo.
echo 💡 Analysis Tips for Windows:
echo   - Use Task Manager Performance tab to monitor CPU usage
echo   - Check if python.exe is using multiple cores
echo   - Look for high CPU usage vs low CPU usage
echo   - Monitor memory usage patterns
echo.
echo 🔧 If you're on Linux, use the bash script instead:
echo   bash test_single_numa.sh
