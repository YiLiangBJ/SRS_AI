"""Programmatic plotting workflow."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import resolve_existing_path


def _resolve_input_path(path_value) -> Path:
    """Resolve a plotting input path against the project roots."""
    resolved = resolve_existing_path(path_value)
    if isinstance(resolved, tuple):
        _, candidates = resolved
        candidate_text = '\n'.join(str(path) for path in candidates)
        raise FileNotFoundError('Plot input not found. Checked:\n' + candidate_text)
    return resolved


def _discover_latest_evaluation_dir(root: Path) -> Path | None:
    """Find the newest evaluation directory containing evaluation_results.json."""
    candidates = [
        child for child in root.iterdir()
        if child.is_dir() and (child / 'evaluation_results.json').exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_plot_inputs(eval_results_path, output_dir=None):
    """Resolve plotting input and output paths.

    The input may be an experiment directory, an evaluation directory, or an
    evaluation_results.json file.
    """
    resolved_input = _resolve_input_path(eval_results_path)

    if resolved_input.is_file():
        if resolved_input.name != 'evaluation_results.json':
            raise ValueError('Plot input file must be evaluation_results.json')
        resolved_eval_json = resolved_input
        resolved_evaluation_dir = resolved_input.parent
    elif (resolved_input / 'evaluation_results.json').exists():
        resolved_evaluation_dir = resolved_input
        resolved_eval_json = resolved_input / 'evaluation_results.json'
    else:
        evaluations_root = resolved_input / 'evaluations' if (resolved_input / 'evaluations').is_dir() else resolved_input
        latest_eval_dir = _discover_latest_evaluation_dir(evaluations_root)
        if latest_eval_dir is None:
            raise FileNotFoundError(
                'Could not find evaluation_results.json. Provide an experiment directory with evaluations/, '
                'an evaluation directory, or the JSON file itself.'
            )
        resolved_evaluation_dir = latest_eval_dir
        resolved_eval_json = latest_eval_dir / 'evaluation_results.json'

    resolved_output_dir = Path(output_dir) if output_dir else resolved_evaluation_dir / 'plots'
    return resolved_eval_json, resolved_output_dir


def generate_plots_programmatic(eval_results_path, output_dir):
    """Generate plots from a refactored evaluation_results.json file."""
    eval_results_path, output_dir = resolve_plot_inputs(eval_results_path, output_dir)
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
