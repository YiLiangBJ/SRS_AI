#!/bin/bash

# 优化的训练脚本 - 针对至强多核处理器
# 适用于56核心/112线程的服务器

echo "🚀 启动优化训练 - 至强多核处理器版本"
echo "=========================================="

# 检查CPU信息
echo "📊 CPU信息:"
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket"
echo ""

# 方案1: 使用NUMA绑定 + CPU优化（推荐）
echo "🎯 方案1: NUMA优化 + CPU多线程 (推荐)"
numactl --physcpubind=0-55 --membind=0 python3 trainMLPmmse.py \
    --epochs 10 \
    --train_batches 100 \
    --batch_size 128 \
    --optimize_cpu \
    --num_threads 56

echo ""
echo "如果上面的命令效果不好，尝试以下方案："
echo ""

# 方案2: 不使用NUMA绑定，让系统自动调度
echo "🎯 方案2: 系统自动调度"
echo "python3 trainMLPmmse.py --epochs 10 --train_batches 100 --batch_size 128 --optimize_cpu"

echo ""

# 方案3: 保守的线程数设置
echo "🎯 方案3: 保守线程设置（避免过度订阅）"
echo "python3 trainMLPmmse.py --epochs 10 --train_batches 100 --batch_size 128 --optimize_cpu --num_threads 28"

echo ""

# 方案4: 调试模式 - 查看具体的线程使用情况
echo "🎯 方案4: 调试模式"
echo "OMP_DISPLAY_ENV=TRUE KMP_SETTINGS=1 python3 trainMLPmmse.py --epochs 1 --train_batches 10 --batch_size 32 --optimize_cpu"

echo ""
echo "💡 优化提示:"
echo "1. 使用 htop 或 top 监控CPU使用率"
echo "2. 如果CPU使用率低，尝试增加 --batch_size"
echo "3. 如果内存不足，减少 --batch_size 或 --num_threads"
echo "4. 使用 --optimize_cpu 启用CPU优化"
