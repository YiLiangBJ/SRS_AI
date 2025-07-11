#!/usr/bin/env python3
"""
测试SNR配置是否正确统一使用config.py中的设置
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.getcwd())

import torch
from config import create_example_config
from trainMLPmmse import SRSTrainerModified


def test_snr_config():
    """测试SNR配置是否正确从config.py获取"""
    print("🧪 测试SNR配置统一性...")
    
    # 创建配置 - 使用自定义SNR范围
    config = create_example_config()
    config.snr_range = (-5, 25)  # 自定义SNR范围
    
    print(f"📊 配置的SNR范围: {config.snr_range}")
    
    # 创建trainer
    trainer = SRSTrainerModified(
        srs_config=config,
        device="cpu",  # 使用CPU避免CUDA问题
        save_dir="./test_checkpoints",
        use_trainable_mmse=True,  # 需要可训练参数才能创建optimizer
        enable_plotting=False,
        use_professional_channels=False  # 简化测试，不使用专业信道
    )
    
    print(f"✅ Trainer创建成功")
    
    # 测试数据生成器是否使用正确的SNR范围
    data_gen = trainer.get_data_generator()
    print(f"📊 数据生成器使用的SNR范围: {data_gen.base_generator.config.snr_range}")
    
    # 验证SNR范围是否一致
    if data_gen.base_generator.config.snr_range == config.snr_range:
        print("✅ SNR配置一致性测试通过!")
    else:
        print(f"❌ SNR配置不一致!")
        print(f"   期望: {config.snr_range}")
        print(f"   实际: {data_gen.base_generator.config.snr_range}")
        return False
    
    # 测试生成数据时的SNR设置
    try:
        print("📊 测试批次生成...")
        batch = trainer.generate_batch_with_dynamic_channel(
            batch_size=2, 
            enable_debug=True
        )
        print("✅ 批次生成成功!")
        print(f"   生成的数据键: {list(batch.keys())}")
        
        if 'snr_db' in batch:
            print(f"   生成的SNR值: {batch['snr_db']}")
        
    except Exception as e:
        print(f"❌ 批次生成失败: {e}")
        return False
    
    return True


def test_timing_offset_config():
    """测试timing offset配置是否正确从config.py获取"""
    print("\n🧪 测试timing offset配置统一性...")
    
    # 创建配置 - 使用自定义timing offset范围
    config = create_example_config()
    config.timing_offset_range = (-50e-9, 50e-9)  # 自定义timing offset范围
    
    print(f"⏱️  配置的timing offset范围: {config.timing_offset_range}")
    
    # 测试配置的方法
    timing_offset_seconds = config.get_timing_offset_seconds()
    print(f"⏱️  生成的timing offset: {timing_offset_seconds*1e9:.1f} ns")
    
    timing_offset_samples = config.get_timing_offset_samples(122.88e6)
    print(f"⏱️  转换为采样点: {timing_offset_samples} 采样点")
    
    # 验证是否在范围内
    min_offset, max_offset = config.timing_offset_range
    if min_offset <= timing_offset_seconds <= max_offset:
        print("✅ Timing offset配置一致性测试通过!")
        return True
    else:
        print(f"❌ Timing offset不在范围内!")
        print(f"   范围: {min_offset*1e9:.1f} ~ {max_offset*1e9:.1f} ns")
        print(f"   实际: {timing_offset_seconds*1e9:.1f} ns")
        return False


if __name__ == "__main__":
    print("🚀 开始SNR和timing offset配置统一性测试...\n")
    
    # 测试SNR配置
    snr_test_passed = test_snr_config()
    
    # 测试timing offset配置
    timing_test_passed = test_timing_offset_config()
    
    print("\n📊 测试总结:")
    print(f"   SNR配置测试: {'✅ 通过' if snr_test_passed else '❌ 失败'}")
    print(f"   Timing offset配置测试: {'✅ 通过' if timing_test_passed else '❌ 失败'}")
    
    if snr_test_passed and timing_test_passed:
        print("\n🎉 所有配置统一性测试通过!")
        sys.exit(0)
    else:
        print("\n💥 部分测试失败，需要修复!")
        sys.exit(1)
