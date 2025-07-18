#!/usr/bin/env python3
"""
Test script for NUMA detection and DDP configuration

This script tests the NUMA detection functionality and verifies that
DDP is properly configured for multi-socket systems.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system_detection import SystemDetector, setup_distributed_training


def test_numa_detection():
    """Test NUMA detection functionality"""
    print("="*60)
    print("🧪 Testing NUMA Detection")
    print("="*60)
    
    detector = SystemDetector()
    
    # Test basic detection
    platform_type = detector.detect_platform_type()
    numa_nodes = detector.detect_numa_nodes()
    
    print(f"Platform type: {platform_type}")
    print(f"NUMA nodes detected: {numa_nodes}")
    print(f"CPU cores (physical): {detector.cpu_count}")
    print(f"CPU cores (logical): {detector.logical_cpu_count}")
    print(f"GPU count: {detector.gpu_count}")
    
    # Test optimal settings
    print("\n" + "="*60)
    print("⚙️  Testing Optimal Settings")
    print("="*60)
    
    settings = detector.get_optimal_settings()
    for key, value in settings.items():
        print(f"{key}: {value}")
    
    # Test DDP configuration
    print("\n" + "="*60)
    print("🔧 Testing DDP Configuration")
    print("="*60)
    
    if settings['use_ddp']:
        print("✅ DDP is recommended for this system")
        print(f"   Recommended world_size: {settings['world_size']}")
        print(f"   Backend: {settings['backend']}")
        
        if numa_nodes > 1 and detector.gpu_count == 0:
            print("🎯 CPU-only NUMA system detected - DDP enabled for performance")
        elif detector.gpu_count > 1:
            print("🎯 Multi-GPU system detected - DDP enabled for parallelism")
        else:
            print("🎯 DDP enabled for unknown reason")
    else:
        print("❌ DDP is not recommended for this system")
        print("   Reason: Single-socket system or insufficient resources")
    
    return settings


def test_ddp_setup():
    """Test DDP setup (without actually initializing)"""
    print("\n" + "="*60)
    print("🔄 Testing DDP Setup (Dry Run)")
    print("="*60)
    
    try:
        # Test setup_distributed_training function
        ddp_enabled, settings = setup_distributed_training(
            enable_ddp=None,  # Auto-detect
            rank=0,
            world_size=1
        )
        
        print(f"DDP enabled: {ddp_enabled}")
        print("Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ DDP setup test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 NUMA Detection and DDP Configuration Test")
    print("="*60)
    
    # Test NUMA detection
    settings = test_numa_detection()
    
    # Test DDP setup
    ddp_test_success = test_ddp_setup()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    if settings['numa_nodes'] > 1:
        print("✅ Multi-socket NUMA system detected")
    else:
        print("ℹ️  Single-socket system or NUMA not detected")
    
    if settings['use_ddp']:
        print("✅ DDP is recommended and should be used")
    else:
        print("ℹ️  DDP is not recommended for this system")
    
    if ddp_test_success:
        print("✅ DDP setup test passed")
    else:
        print("❌ DDP setup test failed")
    
    print("\n🎯 Recommendation:")
    if settings['use_ddp']:
        print(f"   Use DDP with world_size={settings['world_size']}")
        print(f"   Backend: {settings['backend']}")
        if settings['numa_nodes'] > 1 and settings.get('gpu_count', 0) == 0:
            print("   This is a CPU-only NUMA system - DDP will improve performance")
    else:
        print("   Use single-process training")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
