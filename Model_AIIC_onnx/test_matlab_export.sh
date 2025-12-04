#!/bin/bash
# Test MATLAB ONNX export

echo "=========================================="
echo "Testing MATLAB ONNX Export"
echo "=========================================="
echo ""

# Check if checkpoint exists
CHECKPOINT="./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Please train a model first:"
    echo "  python Model_AIIC_onnx/test_separator.py --batches 50 --save_dir ./Model_AIIC_onnx/test"
    exit 1
fi

echo "✓ Checkpoint found: $CHECKPOINT"
echo ""

# Export model
echo "Exporting model for MATLAB..."
echo ""

python Model_AIIC_onnx/export_onnx_matlab.py \
    --checkpoint "$CHECKPOINT" \
    --output model_matlab.onnx \
    --opset 9

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Export successful!"
    echo "=========================================="
    echo ""
    echo "Model saved to: model_matlab.onnx"
    echo ""
    echo "Next steps:"
    echo "  1. Open MATLAB"
    echo "  2. Navigate to: $(pwd)"
    echo "  3. Run: read_onnx_matlab"
    echo ""
else
    echo ""
    echo "❌ Export failed!"
    echo ""
    echo "Common issues:"
    echo "  1. ONNX not installed: pip install onnx onnxruntime"
    echo "  2. Checkpoint corrupted"
    echo "  3. Python environment not activated"
    exit 1
fi
