#!/bin/bash
# 快速测试不同损失函数（小规模）

echo "=========================================="
echo "快速损失函数对比测试"
echo "=========================================="

# 配置
BATCHES=50
BATCH_SIZE=128
STAGES="2,3"
SHARE="True,False"
LOSS_TYPES="nmse,normalized,log,weighted"
PORTS="0,3,6,9"
SNR="0,30"
SAVE_DIR="./quick_loss_test"

echo ""
echo "配置:"
echo "  Stages: $STAGES"
echo "  Share weights: $SHARE"
echo "  Loss types: $LOSS_TYPES"
echo "  Batches: $BATCHES (快速测试)"
echo "  组合数: 2 × 1 × 2 = 4"
echo "=========================================="

# 训练
echo ""
echo "开始训练..."
python Model_AIIC/test_separator.py \
  --batches $BATCHES \
  --batch_size $BATCH_SIZE \
  --stages "$STAGES" \
  --share_weights "$SHARE" \
  --loss_type "$LOSS_TYPES" \
  --ports "$PORTS" \
  --snr "$SNR" \
  --save_dir "$SAVE_DIR"

echo ""
echo "=========================================="
echo "✅ 快速测试完成！"
echo "=========================================="
echo ""
echo "查看结果: $SAVE_DIR"
echo ""
