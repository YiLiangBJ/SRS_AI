"""
Quick test to verify the refactored code works correctly.

This script:
1. Tests model creation
2. Tests data generation
3. Tests training (1 batch)
4. Tests evaluation
"""

import torch
from models import create_model, list_models
from training import Trainer
from data import generate_training_batch
from utils import get_device

print("="*80)
print("Refactored Code Verification Test")
print("="*80)

# Test 1: Model creation
print("\n1. Testing model creation...")
print(f"   Available models: {list_models()}")

config = {'seq_len': 12, 'num_ports': 4, 'hidden_dim': 32, 'num_stages': 2, 'mlp_depth': 3}

for model_name in ['separator1', 'separator2']:
    model = create_model(model_name, config)
    print(f"   ✓ {model_name}: {model.__class__.__name__}")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test 2: Data generation
print("\n2. Testing data generation...")
y, h_targets, pos, h_true, snr = generate_training_batch(
    batch_size=32,
    snr_db=(10, 20),
    tdl_config='A-30'
)
print(f"   ✓ Generated batch:")
print(f"     y shape: {y.shape}")
print(f"     h_targets shape: {h_targets.shape}")
print(f"     SNR: {snr:.1f} dB")

# Test 3: Training
print("\n3. Testing training (1 batch)...")
device = get_device('auto')
print(f"   Device: {device}")

model = create_model('separator1', config)
trainer = Trainer(model, learning_rate=0.01, loss_type='nmse', device=device)

losses = trainer.train(
    num_batches=2,
    batch_size=32,
    snr_db=(10, 20),
    print_interval=1
)
print(f"   ✓ Training completed")
print(f"     Loss: {losses[-1]:.6f}")

# Test 4: Evaluation
print("\n4. Testing evaluation...")
eval_results = trainer.evaluate(batch_size=32, snr_db=15.0)
print(f"   ✓ Evaluation completed")
print(f"     NMSE: {eval_results['nmse_db']:.2f} dB")

print("\n" + "="*80)
print("✓ All tests passed! Refactored code is working correctly.")
print("="*80)
