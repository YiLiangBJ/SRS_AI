"""Reporting helpers for workflow summaries."""

from datetime import datetime
from pathlib import Path

import numpy as np


def generate_training_report(
    report_path: Path,
    results: list,
    training_recipe_name: str,
    start_time: datetime,
    end_time: datetime,
    total_duration: float,
    device: str,
):
    """Generate detailed training report in Markdown format."""
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write("# Training Report\n\n")

        report_file.write("## Time Information\n\n")
        report_file.write(f"- **Start Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write(f"- **End Time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write(f"- **Total Duration**: {total_duration/3600:.2f} hours ({total_duration:.1f} seconds)\n")
        report_file.write(f"- **Device**: {device}\n\n")

        report_file.write("## Training Recipe\n\n")
        report_file.write(f"- **Training Recipe**: {training_recipe_name}\n")
        report_file.write(f"- **Total Runs**: {len(results)}\n\n")

        report_file.write("## Results Summary\n\n")
        report_file.write("| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |\n")
        report_file.write("|------|-----|------|-------|----------|-----------|------------|-------------|\n")

        for index, result in enumerate(results, 1):
            report_file.write(
                f"| {index} | `{result['run_name']}` | `{result.get('task_label', '-')}` | "
                f"`{result.get('model_label', result.get('model_recipe_name', '-'))}` | `{result['training_label']}` | "
                f"{result['eval_nmse_db']:.2f} | {result['num_params']:,} | {result['training_duration']:.1f} |\n"
            )

        report_file.write("\n")

        if results:
            best = results[0]
            report_file.write("## 🏆 Best Run\n\n")
            report_file.write(f"**Run**: `{best['run_name']}`\n\n")
            report_file.write(f"- **Eval NMSE**: {best['eval_nmse_db']:.2f} dB\n")
            report_file.write(f"- **Final Loss**: {best['final_loss']:.6f}\n")
            report_file.write(f"- **Min Loss**: {best['min_loss']:.6f}\n")
            report_file.write(f"- **Parameters**: {best['num_params']:,}\n")
            report_file.write(f"- **Training Duration**: {best['training_duration']:.1f}s\n\n")

        report_file.write("## Detailed Results\n\n")

        for index, result in enumerate(results, 1):
            report_file.write(f"### {index}. {result['run_name']}\n\n")
            report_file.write(f"- **Task Label**: {result.get('task_label', '-') }\n")
            report_file.write(f"- **Model Recipe**: {result['model_recipe_name']}\n")
            report_file.write(f"- **Model Label**: {result.get('model_label', result['model_recipe_name'])}\n")
            report_file.write(f"- **Training Label**: {result['training_label']}\n")
            if result.get('init_checkpoint_path'):
                report_file.write(f"- **Init Checkpoint**: {result['init_checkpoint_path']}\n")
            report_file.write(f"- **Evaluation NMSE**: {result['eval_nmse_db']:.2f} dB\n")
            report_file.write(f"- **Final Training Loss**: {result['final_loss']:.6f}\n")
            report_file.write(f"- **Minimum Training Loss**: {result['min_loss']:.6f}\n")
            report_file.write(f"- **Total Parameters**: {result['num_params']:,}\n")
            report_file.write(f"- **Samples Processed**: {result.get('samples_processed', 0):,}\n")
            report_file.write(f"- **Average Throughput**: {result.get('avg_training_throughput', 0.0):,.0f} samples/s\n")
            report_file.write(
                f"- **Training Duration**: {result['training_duration']:.1f}s "
                f"({result['training_duration']/60:.1f} min)\n\n"
            )

            if result.get('stage_summaries'):
                report_file.write("- **Stage Summaries**:\n")
                for stage in result['stage_summaries']:
                    report_file.write(
                        f"  - `{stage['stage_name']}`: batches={stage['num_batches']}, batch_size={stage['batch_size']}, "
                        f"loss={stage['loss_type']}, lr={stage['learning_rate']}, final_loss={stage['final_loss']:.6f}\n"
                    )
                report_file.write("\n")

        report_file.write("---\n\n")
        report_file.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def generate_evaluation_summary(report_path: Path, results: dict):
    """Generate a human-readable evaluation summary beside evaluation results."""
    models = results.get('models', {})
    ranked_models = []
    for run_name, model_data in models.items():
        nmse_db_values = []
        for tdl_data in model_data.get('tdl_results', {}).values():
            nmse_db_values.extend(tdl_data.get('nmse_db', []))
        if not nmse_db_values:
            continue
        ranked_models.append({
            'run_name': run_name,
            'task_label': model_data.get('metadata', {}).get('task_label') or model_data.get('metadata', {}).get('task_recipe_name') or '-',
            'model_label': model_data.get('metadata', {}).get('model_label') or model_data.get('metadata', {}).get('model_recipe_name') or run_name,
            'training_label': model_data.get('metadata', {}).get('training_label') or '-',
            'mean_nmse_db': float(np.mean(nmse_db_values)),
            'best_nmse_db': float(np.min(nmse_db_values)),
            'worst_nmse_db': float(np.max(nmse_db_values)),
            'num_points': len(nmse_db_values),
        })

    ranked_models.sort(key=lambda item: item['mean_nmse_db'])

    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('# Evaluation Summary\n\n')
        report_file.write(f"- Evaluation name: `{results.get('evaluation_name', report_path.parent.name)}`\n")
        report_file.write(f"- Run count: {results.get('config', {}).get('run_count', len(ranked_models))}\n")
        report_file.write(f"- SNR values: `{results.get('config', {}).get('snr_list', [])}`\n")
        report_file.write(f"- TDL values: `{results.get('config', {}).get('tdl_list', [])}`\n\n")

        report_file.write('## Ranked Runs\n\n')
        report_file.write('| Rank | Run | Task | Model | Training | Mean NMSE (dB) | Best | Worst | Points |\n')
        report_file.write('|---|---|---|---|---|---|---|---|---|\n')
        for index, item in enumerate(ranked_models, 1):
            report_file.write(
                f"| {index} | `{item['run_name']}` | `{item['task_label']}` | `{item['model_label']}` | `{item['training_label']}` | "
                f"{item['mean_nmse_db']:.2f} | {item['best_nmse_db']:.2f} | {item['worst_nmse_db']:.2f} | {item['num_points']} |\n"
            )

        report_file.write('\n## Per-Run Notes\n\n')
        for item in ranked_models:
            report_file.write(f"### {item['run_name']}\n\n")
            report_file.write(f"- Task: `{item['task_label']}`\n")
            report_file.write(f"- Model: `{item['model_label']}`\n")
            report_file.write(f"- Training: `{item['training_label']}`\n")
            report_file.write(f"- Mean NMSE: {item['mean_nmse_db']:.2f} dB\n")
            report_file.write(f"- Best point: {item['best_nmse_db']:.2f} dB\n")
            report_file.write(f"- Worst point: {item['worst_nmse_db']:.2f} dB\n")
            report_file.write(f"- Evaluated points: {item['num_points']}\n\n")
