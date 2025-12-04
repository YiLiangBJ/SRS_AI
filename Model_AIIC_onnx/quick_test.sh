#!/bin/bash
# Quick test for Model_AIIC_onnx training script

cd "$(dirname "$0")/.."

echo "============================================"
echo "Testing Model_AIIC_onnx Training Script"
echo "============================================"

python Model_AIIC_onnx/test_separator.py \
  --batches 50 \
  --batch_size 128 \
  --stages "2,3" \
  --share_weights "False" \
  --loss_type "nmse" \
  --activation_type "split_relu,cardioid" \
  --ports "0,3,6,9" \
  --snr "20.0" \
  --tdl "A-30" \
  --save_dir "./Model_AIIC_onnx/test_experiments"

echo ""
echo "✓ Test complete!"
echo "Check ./Model_AIIC_onnx/test_experiments for results"
