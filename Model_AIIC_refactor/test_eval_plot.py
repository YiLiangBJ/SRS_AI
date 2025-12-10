"""
Test evaluation and plotting scripts
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("Testing Evaluation and Plotting Scripts")
print("="*80)

# Test 1: Check if we have a trained model
test_model_dir = Path('./test_final/separator1_small_quick_test/separator1_small_hd32_stages2_depth3_share0_ports4')

if not test_model_dir.exists():
    print("\n❌ No test model found. Please run training first:")
    print("   python train.py --model_config separator1_small --training_config quick_test --num_batches 2 --save_dir ./test_final")
    sys.exit(1)

print(f"\n✓ Found test model: {test_model_dir}")

# Test 2: Run evaluation
print("\n" + "="*80)
print("Test 1: Running evaluation")
print("="*80)

eval_cmd = [
    sys.executable,
    'evaluate_models_refactored.py',
    '--model_dir', str(test_model_dir),
    '--snr_values', '10,15,20',
    '--tdl_configs', 'A-30',
    '--num_samples', '200',
    '--output', 'test_eval_results'
]

print(f"Command: {' '.join(eval_cmd)}")
print()

result = subprocess.run(eval_cmd, capture_output=False)

if result.returncode != 0:
    print(f"\n❌ Evaluation failed with code {result.returncode}")
    sys.exit(1)

print("\n✓ Evaluation completed")

# Test 3: Check if results exist
results_dir = Path('test_eval_results')
result_files = list(results_dir.glob('*_results.json'))

if not result_files:
    print(f"\n❌ No result files found in {results_dir}")
    sys.exit(1)

print(f"✓ Found {len(result_files)} result file(s)")
for f in result_files:
    print(f"  - {f.name}")

# Test 4: Run plotting
print("\n" + "="*80)
print("Test 2: Running plotting")
print("="*80)

plot_cmd = [
    sys.executable,
    'plot_results.py',
    '--input', 'test_eval_results',
    '--output', 'test_plots'
]

print(f"Command: {' '.join(plot_cmd)}")
print()

result = subprocess.run(plot_cmd, capture_output=False)

if result.returncode != 0:
    print(f"\n❌ Plotting failed with code {result.returncode}")
    sys.exit(1)

print("\n✓ Plotting completed")

# Test 5: Check if plots exist
plots_dir = Path('test_plots')
plot_files = list(plots_dir.glob('*.png'))

if not plot_files:
    print(f"\n❌ No plot files found in {plots_dir}")
    sys.exit(1)

print(f"✓ Found {len(plot_files)} plot file(s)")
for f in plot_files:
    print(f"  - {f.name}")

# Summary
print("\n" + "="*80)
print("Test Summary")
print("="*80)
print("✓ Evaluation script: working")
print("✓ Plotting script: working")
print(f"✓ Results: {results_dir}")
print(f"✓ Plots: {plots_dir}")
print("\n🎉 All tests passed!")
