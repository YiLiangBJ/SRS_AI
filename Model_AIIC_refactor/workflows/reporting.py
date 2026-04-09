"""Reporting helpers for workflow summaries."""

from datetime import datetime
from pathlib import Path


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
        report_file.write("| Rank | Run | NMSE (dB) | Parameters | Duration (s) |\n")
        report_file.write("|------|--------------|-----------|------------|-------------|\n")

        for index, result in enumerate(results, 1):
            report_file.write(
                f"| {index} | `{result['run_name']}` | {result['eval_nmse_db']:.2f} | "
                f"{result['num_params']:,} | {result['training_duration']:.1f} |\n"
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
            report_file.write(f"- **Model Recipe**: {result['model_recipe_name']}\n")
            report_file.write(f"- **Training Label**: {result['training_label']}\n")
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

        report_file.write("---\n\n")
        report_file.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
