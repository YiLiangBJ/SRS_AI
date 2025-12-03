#!/bin/bash
# 对比不同损失函数的网格搜索实验

echo "=========================================="
echo "Loss Function Grid Search Comparison"
echo "=========================================="

# 配置
BATCHES=1000
BATCH_SIZE=2048
STAGES="2,3,4"
SHARE="True,False"
LOSS_TYPES="nmse,normalized,log,weighted"
PORTS="0,3,6,9"
SNR="0,30"  # 大范围 SNR
TDL="A-30,B-100,C-300"
SAVE_DIR="./loss_comparison_full"

echo ""
echo "实验配置:"
echo "  Stages: $STAGES"
echo "  Share weights: $SHARE"
echo "  Loss types: $LOSS_TYPES"
echo "  Ports: $PORTS"
echo "  SNR range: $SNR"
echo "  TDL configs: $TDL"
echo "  Batches: $BATCHES"
echo "  Batch size: $BATCH_SIZE"
echo "  Save dir: $SAVE_DIR"
echo ""
echo "总实验数: 3 stages × 2 share × 4 loss = 24 个组合"
echo "=========================================="

# 使用网格搜索训练所有组合
echo ""
echo "开始训练（网格搜索所有组合）..."
python Model_AIIC/test_separator.py \
  --batches $BATCHES \
  --batch_size $BATCH_SIZE \
  --stages "$STAGES" \
  --share_weights "$SHARE" \
  --loss_type "$LOSS_TYPES" \
  --ports "$PORTS" \
  --snr "$SNR" \
  --tdl "$TDL" \
  --early_stop 0.01 \
  --val_interval 100 \
  --patience 5 \
  --save_dir "$SAVE_DIR"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

# 评估所有模型
echo ""
echo "开始评估所有模型..."
python Model_AIIC/evaluate_models.py \
  --exp_dir "$SAVE_DIR" \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output "${SAVE_DIR}_eval"

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="

# 绘制对比图
echo ""
echo "生成可视化图表..."
python Model_AIIC/plot_results.py \
  --input "${SAVE_DIR}_eval" \
  --layout "subplots_tdl"

echo ""
echo "=========================================="
echo "✅ 全部完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - 训练日志: $SAVE_DIR"
echo "  - 评估结果: ${SAVE_DIR}_eval/evaluation_results.json"
echo "  - 可视化图: ${SAVE_DIR}_eval/*.png"
echo ""
echo "TensorBoard 监控:"
echo "  tensorboard --logdir $SAVE_DIR"
echo ""
echo "=========================================="
