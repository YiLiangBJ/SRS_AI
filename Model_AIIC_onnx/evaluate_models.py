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

try:
    from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
    from Model_AIIC_onnx.test_separator import generate_training_data
except ImportError:
    from channel_separator import ResidualRefinementSeparatorReal
    from test_separator import generate_training_data


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
    
    # 从 hyperparameters 中获取 pos_values 和 num_params（如果存在）
    hyperparams = checkpoint.get('hyperparameters', {})
    pos_values = hyperparams.get('pos_values', None)
    num_params = hyperparams.get('num_params', None)
    
    # 如果没有 pos_values，根据 num_ports 生成默认值
    if pos_values is None:
        num_ports = config.get('num_ports', 4)
        if num_ports == 4:
            pos_values = [0, 3, 6, 9]
        elif num_ports == 6:
            pos_values = [0, 2, 4, 6, 8, 10]
        else:
            # 均匀分布
            pos_values = list(range(0, 12, 12 // num_ports))[:num_ports]
    
    # 创建模型（num_ports 从 pos_values 推导）
    # 获取激活函数类型（从保存的 config 中）
    activation_type = config.get('activation_type', 'relu')
    onnx_mode = config.get('onnx_mode', False)
    
    # 新版本模型：normalize_energy 已外置，不再作为模型参数
    model = ResidualRefinementSeparatorReal(
        seq_len=config['seq_len'],
        num_ports=len(pos_values),
        hidden_dim=config['hidden_dim'],
        num_stages=config['num_stages'],
        share_weights_across_stages=config['share_weights'],
        activation_type=activation_type,
        onnx_mode=onnx_mode
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 如果没有保存 num_params，现在计算它
    if num_params is None:
        num_params = sum(p.numel() for p in model.parameters())
    
    # 将 pos_values 和 num_params 添加到 config 中返回
    config['pos_values'] = pos_values
    config['num_params'] = num_params
    
    return model, config


def evaluate_model_at_snr(model, snr_db, tdl_config, pos_values, num_batches=10, batch_size=100):
    """
    评估模型在特定 SNR 和 TDL 配置下的性能
    
    Args:
        model: 模型
        snr_db: 信噪比 (dB)
        tdl_config: TDL 配置 (e.g., 'A-30')
        pos_values: 端口位置列表 (e.g., [0, 3, 6, 9])
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
    num_ports = len(pos_values)
    
    total_mse = 0.0
    total_power = 0.0
    port_mse = np.zeros(num_ports)
    port_power = np.zeros(num_ports)
    
    with torch.no_grad():
        for _ in range(num_batches):
            # 生成测试数据（返回 5 个值：y, h_targets, pos_values, h_true, batch_snr）
            y, h_targets, _, _, _ = generate_training_data(
                batch_size=batch_size,
                snr_db=snr_db,
                seq_len=seq_len,
                pos_values=pos_values,
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
    
    parser.add_argument('--exp_dir', type=str, 
                       default=r'C:\Users\YiLiang\Downloads\models_4ports_1206',
                       help='实验目录（包含训练好的模型）')
    parser.add_argument('--models', type=str, default=None,
                       help='要评估的模型列表（逗号分隔），如 "stages=2_share=False,stages=3_share=False"。'
                            '如果不指定，则评估所有模型')
    parser.add_argument('--tdl', type=str, default='A-30,B-100,C-300',
                       help='TDL 配置列表（逗号分隔），如 "A-30,B-100,C-300"')
    parser.add_argument('--snr_range', type=str, default='30:-3:0',
                       help='SNR 范围，格式: "start:step:end"，如 "30:-3:0" 表示 [30, 27, ..., 3, 0]')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='每个 SNR 点的评估批次数（总样本数 = num_batches × batch_size）')
    parser.add_argument('--batch_size', type=int, default=2048,
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
            print(f"  激活函数: {config.get('activation_type', 'N/A')}")
            print(f"  端口位置: {config.get('pos_values', 'N/A')}")
            
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
                        model, snr_db, tdl_config, config['pos_values'],
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
            import traceback
            print(f"详细错误信息:")
            traceback.print_exc()
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
