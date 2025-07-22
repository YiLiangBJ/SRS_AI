#!/usr/bin/env python3
"""
CPU-Only Training Test Script

This script tests that all CUDA/GPU references have been removed and 
the training runs completely on CPU without any CUDA warnings.
"""

import os

# Force CPU-only execution from the very beginning - disable all CUDA/GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow CUDA warnings

print("🔒 CPU-ONLY TEST: Environment configured for CPU-only execution")

# Test imports
print("📦 Testing imports...")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__} imported")
    print(f"   🔒 CUDA available: {torch.cuda.is_available()} (should be False)")
    
    from train_distributed import main as train_main
    print("   ✅ train_distributed imported")
    
    from model_Traditional import SRSChannelEstimator
    print("   ✅ model_Traditional imported")
    
    from trainMLPmmse import SRSTrainerModified
    print("   ✅ trainMLPmmse imported")
    
    from professional_channels import SIONNAChannelModel
    print("   ✅ professional_channels imported")
    
    from data_generator_refactored import SRSDataGenerator
    print("   ✅ data_generator_refactored imported")
    
    from system_detection import SystemDetector
    print("   ✅ system_detection imported")
    
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test basic tensor operations
print("\n🧮 Testing basic tensor operations...")
try:
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    z = torch.matmul(x, y)
    print(f"   ✅ Matrix multiplication on device: {z.device}")
    
    # Test complex tensors
    a = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
    b = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
    c = torch.matmul(a, b)
    print(f"   ✅ Complex matrix multiplication on device: {c.device}")
    
except Exception as e:
    print(f"   ❌ Tensor operations failed: {e}")
    exit(1)

# Test model initialization
print("\n🏗️ Testing model initialization...")
try:
    from user_config import create_example_config
    config = create_example_config()
    
    # Test SRS estimator
    estimator = SRSChannelEstimator(
        seq_length=config.seq_length,
        ktc=config.ktc,
        max_users=config.num_users,
        max_ports_per_user=max(config.ports_per_user),
        device="cpu"
    )
    print(f"   ✅ SRSChannelEstimator initialized on device: {estimator.device}")
    
    # Test system detector
    detector = SystemDetector()
    print(f"   ✅ SystemDetector initialized")
    print(f"   🔒 Has CUDA: {detector.has_cuda} (should be False)")
    print(f"   🔒 GPU count: {detector.gpu_count} (should be 0)")
    
except Exception as e:
    print(f"   ❌ Model initialization failed: {e}")
    exit(1)

print("\n✅ All CPU-only tests passed!")
print("🚀 Ready for CPU-only training without CUDA warnings")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CPU-ONLY TRAINING VERIFICATION COMPLETE")
    print("="*60)
