"""
模型性能评估脚本

功能:
1. 扫描 SNR: 30:-3:0 dB
2. 加载训练好的模型
3. 评估不同 SNR、TDL 信道下的 NMSE 性能
4. 保存结果为 JSON 和 NPY 格式

用法:
    python Model_AIIC/evaluate_models.py \
        --exp_dir "./experiments" \
        --models "stages=2_share=False,stages=3_share=False" \
        --tdl "A-30,B-100,C-300" \
        --snr_range "30:-3:0" \
        --num_samples 1000 \
        --output "./evaluation_results"
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC.channel_separator import ResidualRefinementSeparator
from Model_AIIC.test_separator import generate_training_data


def load_model(model_dir):
    """
    加载训练好的模型
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        model: 加载的模型
        config: 模型配置
    """
    model_path = Path(model_dir) / 'model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # 创建模型
    model = ResidualRefinementSeparator(
        seq_len=config['seq_len'],
        num_ports=config['num_ports'],
        hidden_dim=config['hidden_dim'],
        num_stages=config['num_stages'],
        share_weights_across_stages=config['share_weights'],
        normalize_energy=config['normalize_energy']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def evaluate_model_at_snr(model, snr_db, tdl_config, num_batches=10, batch_size=100):
    """
    评估模型在特定 SNR 和 TDL 配置下的性能
    
    Args:
        model: 模型
        snr_db: 信噪比 (dB)
        tdl_config: TDL 配置 (e.g., 'A-30')
        num_batches: 评估批次数
        batch_size: 批大小
        
    Returns:
        nmse: NMSE (线性)
        nmse_db: NMSE (dB)
        port_nmse: 每个端口的 NMSE (线性)
        port_nmse_db: 每个端口的 NMSE (dB)
        
    Note:
        总样本数 = num_batches × batch_size
    """
    seq_len = 12
    num_ports = 4
    
    total_mse = 0.0
    total_power = 0.0
    port_mse = np.zeros(num_ports)
    port_power = np.zeros(num_ports)
    
    with torch.no_grad():
        for _ in range(num_batches):
            # 生成测试数据
            y, h_targets, _, _ = generate_training_data(
                batch_size=batch_size,
                snr_db=snr_db,
                seq_len=seq_len,
                num_ports=num_ports,
                tdl_config=tdl_config
            )
            
            # 预测
            h_pred = model(y)
            
            # 计算 MSE
            mse = (h_pred - h_targets).abs().pow(2).sum().item()
            power = h_targets.abs().pow(2).sum().item()
            
            total_mse += mse
            total_power += power
            
            # 每个端口的 MSE
            for p in range(num_ports):
                port_mse[p] += (h_pred[:, p] - h_targets[:, p]).abs().pow(2).sum().item()
                port_power[p] += h_targets[:, p].abs().pow(2).sum().item()
    
    # 计算 NMSE
    nmse = total_mse / (total_power + 1e-10)
    nmse_db = 10 * np.log10(nmse) if nmse > 0 else -np.inf
    
    # 每个端口的 NMSE
    port_nmse = port_mse / (port_power + 1e-10)
    port_nmse_db = 10 * np.log10(port_nmse)
    port_nmse_db[np.isinf(port_nmse_db)] = -100  # 处理 inf
    
    return nmse, nmse_db, port_nmse.tolist(), port_nmse_db.tolist()


def parse_snr_range(snr_str):
    """
    解析 SNR 范围字符串
    
    Args:
        snr_str: SNR 范围字符串，格式: "start:step:end" 或 "start,end"
        
    Returns:
        snr_list: SNR 列表
    """
    if ':' in snr_str:
        parts = snr_str.split(':')
        if len(parts) == 3:
            start, step, end = map(float, parts)
            # Python range 不包括终点，所以需要调整
            snr_list = np.arange(start, end - 0.1 * abs(step), step)
        else:
            raise ValueError(f"Invalid SNR range format: {snr_str}")
    elif ',' in snr_str:
        snr_list = np.array([float(x) for x in snr_str.split(',')])
    else:
        snr_list = np.array([float(snr_str)])
    
    return snr_list.tolist()


def main():
    parser = argparse.ArgumentParser(description='评估训练好的模型在不同 SNR 和 TDL 配置下的性能')
    
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='实验目录（包含训练好的模型）')
    parser.add_argument('--models', type=str, default=None,
                       help='要评估的模型列表（逗号分隔），如 "stages=2_share=False,stages=3_share=False"。'
                            '如果不指定，则评估所有模型')
    parser.add_argument('--tdl', type=str, default='A-30,B-100,C-300',
                       help='TDL 配置列表（逗号分隔），如 "A-30,B-100,C-300"')
    parser.add_argument('--snr_range', type=str, default='30:-3:0',
                       help='SNR 范围，格式: "start:step:end"，如 "30:-3:0" 表示 [30, 27, ..., 3, 0]')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='每个 SNR 点的评估批次数（总样本数 = num_batches × batch_size）')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='评估批大小（总样本数 = num_batches × batch_size）')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 解析参数
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析 SNR 范围
    snr_list = parse_snr_range(args.snr_range)
    print(f"SNR 范围: {snr_list}")
    
    # 解析 TDL 配置
    tdl_list = args.tdl.split(',')
    print(f"TDL 配置: {tdl_list}")
    
    # 解析模型列表
    if args.models:
        model_names = args.models.split(',')
    else:
        # 自动发现所有模型
        if not exp_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
        model_names = [d.name for d in exp_dir.iterdir() if d.is_dir() and (d / 'model.pth').exists()]
    
    if not model_names:
        raise ValueError(f"在 {exp_dir} 中没有找到任何训练好的模型（包含 model.pth 的目录）")
    
    # 计算总样本数
    total_samples_per_point = args.num_batches * args.batch_size
    
    print(f"要评估的模型: {model_names}")
    print(f"总共: {len(model_names)} 个模型 × {len(tdl_list)} 个 TDL × {len(snr_list)} 个 SNR")
    print(f"每个点: {args.num_batches} batches × {args.batch_size} samples = {total_samples_per_point} 总样本")
    print("="*80)
    
    # 评估结果存储
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'snr_list': snr_list,
            'tdl_list': tdl_list,
            'num_batches': args.num_batches,
            'batch_size': args.batch_size,
            'total_samples_per_point': args.num_batches * args.batch_size
        },
        'models': {}
    }
    
    # 遍历所有模型
    for model_name in model_names:
        model_dir = exp_dir / model_name
        
        if not (model_dir / 'model.pth').exists():
            print(f"⚠️  跳过 {model_name}: 找不到 model.pth")
            continue
        
        print(f"\n{'='*80}")
        print(f"评估模型: {model_name}")
        print(f"{'='*80}")
        
        try:
            # 加载模型
            model, config = load_model(model_dir)
            print(f"✓ 模型加载成功")
            print(f"  配置: stages={config['num_stages']}, share_weights={config['share_weights']}")
            
            # 为该模型创建结果存储
            results['models'][model_name] = {
                'config': config,
                'tdl_results': {}
            }
            
            # 遍历所有 TDL 配置
            for tdl_config in tdl_list:
                print(f"\n  TDL: {tdl_config}")
                
                # 为该 TDL 创建结果存储
                tdl_results = {
                    'snr': [],
                    'nmse': [],
                    'nmse_db': [],
                    'port_nmse': [],
                    'port_nmse_db': []
                }
                
                # 遍历所有 SNR
                for snr_db in tqdm(snr_list, desc=f"    {tdl_config}"):
                    # 评估
                    nmse, nmse_db, port_nmse, port_nmse_db = evaluate_model_at_snr(
                        model, snr_db, tdl_config, 
                        num_batches=args.num_batches,
                        batch_size=args.batch_size
                    )
                    
                    # 保存结果
                    tdl_results['snr'].append(snr_db)
                    tdl_results['nmse'].append(nmse)
                    tdl_results['nmse_db'].append(nmse_db)
                    tdl_results['port_nmse'].append(port_nmse)
                    tdl_results['port_nmse_db'].append(port_nmse_db)
                
                # 保存该 TDL 的结果
                results['models'][model_name]['tdl_results'][tdl_config] = tdl_results
                
                print(f"    ✓ 完成 {tdl_config}")
            
            print(f"\n✓ 模型 {model_name} 评估完成")
            
        except Exception as e:
            print(f"✗ 模型 {model_name} 评估失败: {e}")
            continue
    
    # 保存结果
    print(f"\n{'='*80}")
    print("保存结果...")
    print(f"{'='*80}")
    
    # 保存 JSON
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ 保存 JSON: {json_path}")
    
    # 保存 NumPy 格式（便于绘图）
    numpy_data = {}
    for model_name, model_data in results['models'].items():
        numpy_data[model_name] = {}
        for tdl_config, tdl_data in model_data['tdl_results'].items():
            numpy_data[model_name][tdl_config] = {
                'snr': np.array(tdl_data['snr']),
                'nmse': np.array(tdl_data['nmse']),
                'nmse_db': np.array(tdl_data['nmse_db']),
                'port_nmse': np.array(tdl_data['port_nmse']),
                'port_nmse_db': np.array(tdl_data['port_nmse_db'])
            }
    
    npy_path = output_dir / 'evaluation_results.npy'
    np.save(npy_path, numpy_data, allow_pickle=True)
    print(f"✓ 保存 NumPy: {npy_path}")
    
    # 打印摘要
    print(f"\n{'='*80}")
    print("评估摘要")
    print(f"{'='*80}")
    print(f"评估的模型数: {len(results['models'])}")
    print(f"TDL 配置: {tdl_list}")
    print(f"SNR 范围: {snr_list[0]} 到 {snr_list[-1]} dB ({len(snr_list)} 个点)")
    print(f"每个点: {args.num_batches} batches × {args.batch_size} samples = {args.num_batches * args.batch_size} 总样本")
    print(f"\n结果保存到: {output_dir}")
    print(f"  - {json_path.name}")
    print(f"  - {npy_path.name}")
    print(f"\n使用绘图脚本查看结果:")
    print(f"  python Model_AIIC/plot_results.py --input {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
