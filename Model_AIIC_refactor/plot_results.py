"""
Plot evaluation results (Refactored)

Usage:
    python plot_results.py --input evaluation_results --models model1,model2
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_dir: Path, model_names: list = None):
    """
    Load evaluation results
    
    Args:
        results_dir: Directory containing results
        model_names: List of model names (if None, load all)
    
    Returns:
        dict: {model_name: results_data}
    """
    results_dir = Path(results_dir)
    all_results = {}
    
    # Find all result files
    result_files = list(results_dir.glob('*_results.json'))
    
    if not result_files:
        print(f"⚠️  No result files found in {results_dir}")
        return all_results
    
    for result_file in result_files:
        model_name = result_file.stem.replace('_results', '')
        
        # Filter by model names if specified
        if model_names and model_name not in model_names:
            continue
        
        with open(result_file, 'r', encoding='utf-8') as f:
            all_results[model_name] = json.load(f)
    
    return all_results


def plot_nmse_vs_snr(results_dict: dict, tdl_config: str = None, save_path: Path = None):
    """
    Plot NMSE vs SNR for multiple models
    
    Args:
        results_dict: {model_name: results_data}
        tdl_config: Specific TDL to plot (if None, use first available)
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, results in results_dict.items():
        # Get TDL configs
        tdl_configs = results['tdl_configs']
        
        # Use specified or first TDL
        tdl = tdl_config if tdl_config else tdl_configs[0]
        
        # Filter results for this TDL
        tdl_results = [r for r in results['results'] if r['tdl_config'] == tdl]
        
        if not tdl_results:
            print(f"⚠️  No results for {model_name} with TDL {tdl}")
            continue
        
        # Extract SNR and NMSE
        snr_values = [r['snr_db'] for r in tdl_results]
        nmse_db = [r['nmse_db'] for r in tdl_results]
        
        # Sort by SNR
        sorted_data = sorted(zip(snr_values, nmse_db))
        snr_values, nmse_db = zip(*sorted_data)
        
        # Plot
        plt.plot(snr_values, nmse_db, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title(f'NMSE vs SNR - TDL-{tdl}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_port_nmse(results_dict: dict, snr_db: float, tdl_config: str = None, save_path: Path = None):
    """
    Plot per-port NMSE for multiple models at specific SNR
    
    Args:
        results_dict: {model_name: results_data}
        snr_db: SNR to plot
        tdl_config: Specific TDL to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_offset = 0
    bar_width = 0.8 / len(results_dict)
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        # Get TDL
        tdl_configs = results['tdl_configs']
        tdl = tdl_config if tdl_config else tdl_configs[0]
        
        # Find result for this SNR and TDL
        matching_results = [
            r for r in results['results']
            if r['tdl_config'] == tdl and abs(r['snr_db'] - snr_db) < 0.1
        ]
        
        if not matching_results:
            continue
        
        result = matching_results[0]
        per_port_nmse_db = result['per_port_nmse_db']
        num_ports = len(per_port_nmse_db)
        
        # Plot bars
        x = np.arange(num_ports) + i * bar_width
        ax.bar(x, per_port_nmse_db, bar_width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Port Index', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title(f'Per-Port NMSE at SNR={snr_db}dB - TDL-{tdl}', fontsize=14)
    ax.set_xticks(np.arange(num_ports) + bar_width * (len(results_dict) - 1) / 2)
    ax.set_xticklabels([f'Port {i}' for i in range(num_ports)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_tdl_comparison(results_dict: dict, save_dir: Path = None):
    """
    Plot NMSE vs SNR for all TDL configurations
    
    Args:
        results_dict: {model_name: results_data}
        save_dir: Directory to save figures
    """
    # Get all TDL configs
    all_tdls = set()
    for results in results_dict.values():
        all_tdls.update(results['tdl_configs'])
    
    all_tdls = sorted(all_tdls)
    
    # Plot for each TDL
    for tdl in all_tdls:
        save_path = save_dir / f'nmse_vs_snr_{tdl}.png' if save_dir else None
        plot_nmse_vs_snr(results_dict, tdl_config=tdl, save_path=save_path)


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--input', type=str, required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--models', type=str, default=None,
                       help='Comma-separated list of model names (default: all)')
    parser.add_argument('--tdl', type=str, default=None,
                       help='TDL configuration to plot (default: first available)')
    parser.add_argument('--snr', type=float, default=None,
                       help='SNR for per-port plot (default: skip per-port plot)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for figures (default: show plots)')
    parser.add_argument('--all_tdls', action='store_true',
                       help='Plot all TDL configurations separately')
    
    args = parser.parse_args()
    
    # Parse arguments
    results_dir = Path(args.input)
    model_names = args.models.split(',') if args.models else None
    output_dir = Path(args.output) if args.output else None
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Plot Evaluation Results")
    print("="*80)
    print(f"Input: {results_dir}")
    if model_names:
        print(f"Models: {model_names}")
    if output_dir:
        print(f"Output: {output_dir}")
    print()
    
    # Load results
    results_dict = load_results(results_dir, model_names)
    
    if not results_dict:
        print("❌ No results loaded!")
        return
    
    print(f"✓ Loaded {len(results_dict)} model(s):")
    for model_name in results_dict.keys():
        print(f"  - {model_name}")
    print()
    
    # Plot NMSE vs SNR
    if args.all_tdls:
        plot_all_tdl_comparison(results_dict, save_dir=output_dir)
    else:
        save_path = output_dir / 'nmse_vs_snr.png' if output_dir else None
        plot_nmse_vs_snr(results_dict, tdl_config=args.tdl, save_path=save_path)
    
    # Plot per-port NMSE if SNR specified
    if args.snr is not None:
        save_path = output_dir / f'per_port_nmse_snr{args.snr}.png' if output_dir else None
        plot_per_port_nmse(results_dict, snr_db=args.snr, tdl_config=args.tdl, save_path=save_path)
    
    print("\n✓ Plotting completed!")


if __name__ == '__main__':
    main()
