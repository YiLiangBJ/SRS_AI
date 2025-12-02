"""
性能曲线绘图脚本

功能:
1. 读取评估结果
2. 绘制 NMSE vs SNR 曲线
3. 支持选择特定模型、TDL 配置
4. 支持多种绘图风格

用法:
    # 绘制所有模型和 TDL
    python Model_AIIC/plot_results.py --input ./evaluation_results
    
    # 只绘制特定模型
    python Model_AIIC/plot_results.py \
        --input ./evaluation_results \
        --models "stages=2_share=False,stages=3_share=False"
    
    # 只绘制特定 TDL
    python Model_AIIC/plot_results.py \
        --input ./evaluation_results \
        --tdl "A-30,B-100"
    
    # 分图绘制（每个 TDL 一个子图）
    python Model_AIIC/plot_results.py \
        --input ./evaluation_results \
        --layout subplots
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams

# 设置字体（支持中文）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_results(result_dir):
    """
    加载评估结果
    
    Args:
        result_dir: 结果目录
        
    Returns:
        results: 评估结果字典
    """
    result_path = Path(result_dir) / 'evaluation_results.json'
    
    if not result_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_single_figure(results, model_filter=None, tdl_filter=None, output_dir=None):
    """
    在单个图中绘制所有曲线
    
    Args:
        results: 评估结果
        model_filter: 模型过滤列表
        tdl_filter: TDL 过滤列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 8))
    
    # 颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    color_idx = 0
    
    # 遍历所有模型
    for model_name, model_data in results['models'].items():
        # 过滤模型
        if model_filter and model_name not in model_filter:
            continue
        
        # 遍历所有 TDL
        for tdl_idx, (tdl_config, tdl_data) in enumerate(model_data['tdl_results'].items()):
            # 过滤 TDL
            if tdl_filter and tdl_config not in tdl_filter:
                continue
            
            snr = tdl_data['snr']
            nmse_db = tdl_data['nmse_db']
            
            # 绘制曲线
            label = f"{model_name} - {tdl_config}"
            linestyle = linestyles[tdl_idx % len(linestyles)]
            marker = markers[color_idx % len(markers)]
            
            plt.plot(snr, nmse_db, 
                    color=colors[color_idx % len(colors)],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=6,
                    linewidth=2,
                    label=label,
                    markevery=2)  # 每隔2个点标记一次
        
        color_idx += 1
    
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('NMSE (dB)', fontsize=14, fontweight='bold')
    plt.title('Channel Estimation Performance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        output_path = Path(output_dir) / 'nmse_vs_snr_single.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存图像: {output_path}")
        
        # 也保存 PDF
        output_path_pdf = Path(output_dir) / 'nmse_vs_snr_single.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ 保存图像: {output_path_pdf}")
    
    plt.show()


def plot_subplots_by_tdl(results, model_filter=None, tdl_filter=None, output_dir=None):
    """
    为每个 TDL 配置创建一个子图
    
    Args:
        results: 评估结果
        model_filter: 模型过滤列表
        tdl_filter: TDL 过滤列表
        output_dir: 输出目录
    """
    # 获取所有 TDL 配置
    all_tdl = set()
    for model_data in results['models'].values():
        all_tdl.update(model_data['tdl_results'].keys())
    
    # 过滤 TDL
    if tdl_filter:
        all_tdl = [t for t in all_tdl if t in tdl_filter]
    else:
        all_tdl = sorted(all_tdl)
    
    num_tdl = len(all_tdl)
    
    # 创建子图
    fig, axes = plt.subplots(1, num_tdl, figsize=(6*num_tdl, 5))
    if num_tdl == 1:
        axes = [axes]
    
    # 颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    
    # 为每个 TDL 绘制子图
    for tdl_idx, tdl_config in enumerate(all_tdl):
        ax = axes[tdl_idx]
        color_idx = 0
        
        # 遍历所有模型
        for model_name, model_data in results['models'].items():
            # 过滤模型
            if model_filter and model_name not in model_filter:
                continue
            
            # 检查该模型是否有这个 TDL 的结果
            if tdl_config not in model_data['tdl_results']:
                continue
            
            tdl_data = model_data['tdl_results'][tdl_config]
            snr = tdl_data['snr']
            nmse_db = tdl_data['nmse_db']
            
            # 绘制曲线
            ax.plot(snr, nmse_db,
                   color=colors[color_idx % len(colors)],
                   marker=markers[color_idx % len(markers)],
                   markersize=6,
                   linewidth=2,
                   label=model_name,
                   markevery=2)
            
            color_idx += 1
        
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
        ax.set_title(f'TDL-{tdl_config}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        output_path = Path(output_dir) / 'nmse_vs_snr_subplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存图像: {output_path}")
        
        output_path_pdf = Path(output_dir) / 'nmse_vs_snr_subplots.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ 保存图像: {output_path_pdf}")
    
    plt.show()


def plot_subplots_by_model(results, model_filter=None, tdl_filter=None, output_dir=None):
    """
    为每个模型创建一个子图
    
    Args:
        results: 评估结果
        model_filter: 模型过滤列表
        tdl_filter: TDL 过滤列表
        output_dir: 输出目录
    """
    # 获取所有模型
    all_models = list(results['models'].keys())
    
    # 过滤模型
    if model_filter:
        all_models = [m for m in all_models if m in model_filter]
    
    num_models = len(all_models)
    
    # 创建子图
    ncols = min(3, num_models)
    nrows = (num_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten() if num_models > 1 else [axes]
    
    # 颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    
    # 为每个模型绘制子图
    for model_idx, model_name in enumerate(all_models):
        ax = axes[model_idx]
        model_data = results['models'][model_name]
        
        color_idx = 0
        
        # 遍历所有 TDL
        for tdl_config, tdl_data in model_data['tdl_results'].items():
            # 过滤 TDL
            if tdl_filter and tdl_config not in tdl_filter:
                continue
            
            snr = tdl_data['snr']
            nmse_db = tdl_data['nmse_db']
            
            # 绘制曲线
            ax.plot(snr, nmse_db,
                   color=colors[color_idx % len(colors)],
                   marker=markers[color_idx % len(markers)],
                   markersize=6,
                   linewidth=2,
                   label=f'TDL-{tdl_config}',
                   markevery=2)
            
            color_idx += 1
        
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        output_path = Path(output_dir) / 'nmse_vs_snr_by_model.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 保存图像: {output_path}")
        
        output_path_pdf = Path(output_dir) / 'nmse_vs_snr_by_model.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ 保存图像: {output_path_pdf}")
    
    plt.show()


def print_summary(results, model_filter=None, tdl_filter=None):
    """
    打印结果摘要
    
    Args:
        results: 评估结果
        model_filter: 模型过滤列表
        tdl_filter: TDL 过滤列表
    """
    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)
    
    for model_name, model_data in results['models'].items():
        # 过滤模型
        if model_filter and model_name not in model_filter:
            continue
        
        print(f"\n模型: {model_name}")
        print("-" * 80)
        
        for tdl_config, tdl_data in model_data['tdl_results'].items():
            # 过滤 TDL
            if tdl_filter and tdl_config not in tdl_filter:
                continue
            
            snr = tdl_data['snr']
            nmse_db = tdl_data['nmse_db']
            
            # 找到最佳和最差性能
            best_idx = np.argmin(nmse_db)
            worst_idx = np.argmax(nmse_db)
            
            print(f"  TDL-{tdl_config}:")
            print(f"    SNR 范围: {snr[0]:.1f} ~ {snr[-1]:.1f} dB")
            print(f"    最佳性能: {nmse_db[best_idx]:.2f} dB @ SNR={snr[best_idx]:.1f} dB")
            print(f"    最差性能: {nmse_db[worst_idx]:.2f} dB @ SNR={snr[worst_idx]:.1f} dB")
            print(f"    性能提升: {nmse_db[worst_idx] - nmse_db[best_idx]:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description='绘制模型性能曲线')
    
    parser.add_argument('--input', type=str, required=True,
                       help='评估结果目录')
    parser.add_argument('--models', type=str, default=None,
                       help='要绘制的模型列表（逗号分隔）。不指定则绘制所有模型')
    parser.add_argument('--tdl', type=str, default=None,
                       help='要绘制的 TDL 配置（逗号分隔）。不指定则绘制所有 TDL')
    parser.add_argument('--layout', type=str, default='single',
                       choices=['single', 'subplots_tdl', 'subplots_model'],
                       help='绘图布局: single=单图, subplots_tdl=按TDL分子图, subplots_model=按模型分子图')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图像（只保存）')
    
    args = parser.parse_args()
    
    # 加载结果
    print("加载评估结果...")
    results = load_results(args.input)
    print(f"✓ 加载成功")
    
    # 解析过滤器
    model_filter = args.models.split(',') if args.models else None
    tdl_filter = args.tdl.split(',') if args.tdl else None
    
    # 打印摘要
    print_summary(results, model_filter, tdl_filter)
    
    # 绘图
    print("\n" + "="*80)
    print("绘制性能曲线...")
    print("="*80)
    
    output_dir = Path(args.input)
    
    if args.layout == 'single':
        plot_single_figure(results, model_filter, tdl_filter, output_dir)
    elif args.layout == 'subplots_tdl':
        plot_subplots_by_tdl(results, model_filter, tdl_filter, output_dir)
    elif args.layout == 'subplots_model':
        plot_subplots_by_model(results, model_filter, tdl_filter, output_dir)
    
    if args.no_show:
        plt.close('all')
    
    print("\n" + "="*80)
    print("✓ 完成")
    print("="*80)


if __name__ == "__main__":
    main()
