"""
Simplified plotting interface for programmatic use

Provides a simple function to generate all plots from evaluation results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def generate_plots_programmatic(eval_results_path, output_dir):
    """
    从评估结果生成所有图表（程序化调用）
    
    Args:
        eval_results_path: 评估结果JSON文件路径
        output_dir: 图表保存目录
        
    Returns:
        List of generated plot files
    """
    eval_results_path = Path(eval_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(eval_results_path, 'r') as f:
        results = json.load(f)
    
    # ✅ 检查是否有模型数据
    if not results.get('models') or len(results['models']) == 0:
        print("⚠️  No models found in evaluation results. Skipping plot generation.")
        return []
    
    generated_files = []
    
    # Get TDL configurations
    first_model = next(iter(results['models'].values()))
    tdl_list = list(first_model['tdl_results'].keys())
    
    # Plot for each TDL configuration
    for tdl_config in tdl_list:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each model
        for model_name, model_data in results['models'].items():
            tdl_data = model_data['tdl_results'][tdl_config]
            
            snr = np.array(tdl_data['snr'])
            nmse_db = np.array(tdl_data['nmse_db'])
            
            # Plot
            ax.plot(snr, nmse_db, marker='o', label=model_name, linewidth=2, markersize=6)
        
        # Formatting
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title(f'NMSE vs SNR - TDL-{tdl_config}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Save
        plot_file = output_dir / f'nmse_vs_snr_TDL_{tdl_config.replace("-", "_")}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        generated_files.append(plot_file)
        print(f"  ✓ Generated: {plot_file.name}")
    
    # Combined plot (all TDL configs)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['models'])))
    
    for idx, (model_name, model_data) in enumerate(results['models'].items()):
        for tdl_idx, tdl_config in enumerate(tdl_list):
            tdl_data = model_data['tdl_results'][tdl_config]
            
            snr = np.array(tdl_data['snr'])
            nmse_db = np.array(tdl_data['nmse_db'])
            
            linestyle = ['-', '--', ':'][tdl_idx % 3]
            label = f"{model_name} - TDL-{tdl_config}"
            
            ax.plot(snr, nmse_db, 
                   color=colors[idx], 
                   linestyle=linestyle, 
                   marker='o',
                   label=label, 
                   linewidth=2, 
                   markersize=5)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title('NMSE vs SNR - All Configurations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    
    combined_plot = output_dir / 'nmse_vs_snr_combined.png'
    plt.savefig(combined_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    generated_files.append(combined_plot)
    print(f"  ✓ Generated: {combined_plot.name}")
    
    return generated_files


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from evaluation results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to evaluation_results.json')
    parser.add_argument('--output', type=str, default='./plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print(f"📈 Generating plots...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print()
    
    generated_files = generate_plots_programmatic(
        eval_results_path=args.input,
        output_dir=args.output
    )
    
    print(f"\n✓ Generated {len(generated_files)} plots")
    print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()
