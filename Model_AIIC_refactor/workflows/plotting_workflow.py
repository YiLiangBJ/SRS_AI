"""Programmatic plotting workflow."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_plots_programmatic(eval_results_path, output_dir):
    """Generate plots from a refactored evaluation_results.json file."""
    eval_results_path = Path(eval_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_results_path, 'r', encoding='utf-8') as input_file:
        results = json.load(input_file)

    if not results.get('models'):
        print("⚠️  No models found in evaluation results. Skipping plot generation.")
        return []

    generated_files = []
    first_model = next(iter(results['models'].values()))
    tdl_list = list(first_model['tdl_results'].keys())

    for tdl_config in tdl_list:
        fig, axis = plt.subplots(figsize=(10, 6))

        for model_name, model_data in results['models'].items():
            tdl_data = model_data['tdl_results'][tdl_config]
            axis.plot(
                np.array(tdl_data['snr']),
                np.array(tdl_data['nmse_db']),
                marker='o',
                label=model_name,
                linewidth=2,
                markersize=6,
            )

        axis.set_xlabel('SNR (dB)', fontsize=12)
        axis.set_ylabel('NMSE (dB)', fontsize=12)
        axis.set_title(f'NMSE vs SNR - TDL-{tdl_config}', fontsize=14, fontweight='bold')
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)

        plot_file = output_dir / f'nmse_vs_snr_TDL_{tdl_config.replace("-", "_")}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        generated_files.append(plot_file)
        print(f"  ✓ Generated: {plot_file.name}")

    fig, axis = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['models'])))

    for index, (model_name, model_data) in enumerate(results['models'].items()):
        for tdl_index, tdl_config in enumerate(tdl_list):
            tdl_data = model_data['tdl_results'][tdl_config]
            axis.plot(
                np.array(tdl_data['snr']),
                np.array(tdl_data['nmse_db']),
                color=colors[index],
                linestyle=['-', '--', ':'][tdl_index % 3],
                marker='o',
                label=f"{model_name} - TDL-{tdl_config}",
                linewidth=2,
                markersize=5,
            )

    axis.set_xlabel('SNR (dB)', fontsize=12)
    axis.set_ylabel('NMSE (dB)', fontsize=12)
    axis.set_title('NMSE vs SNR - All Configurations', fontsize=14, fontweight='bold')
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9, ncol=2)

    combined_plot = output_dir / 'nmse_vs_snr_combined.png'
    plt.savefig(combined_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)

    generated_files.append(combined_plot)
    print(f"  ✓ Generated: {combined_plot.name}")
    return generated_files
