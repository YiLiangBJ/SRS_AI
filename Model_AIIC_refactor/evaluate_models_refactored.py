"""Thin CLI entrypoint for the refactored evaluation workflow."""

import argparse
from pathlib import Path

from utils import split_csv_arg, discover_run_dirs, resolve_run_selection
from workflows.evaluation_workflow import (
    parse_snr_range,
    resolve_device,
    resolve_evaluation_output_dir,
    evaluate_models_programmatic,
)
from workflows.plotting_workflow import generate_plots_for_target_programmatic, generate_plots_programmatic
from workflows.types import EvaluationRequest


def build_parser():
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(description='Evaluate trained runs across SNR and TDL settings')
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory. Evaluates all runs inside by default, or a subset with --runs')
    parser.add_argument('--run_dir', type=str, default=None, help='Single trained run directory')
    parser.add_argument('--run_dirs', type=str, default=None, help='Multiple trained run directories, comma-separated')
    parser.add_argument('--runs', type=str, default=None, help='Run names inside --exp_dir, comma-separated')
    parser.add_argument('--list_runs', action='store_true', help='List evaluable runs inside --exp_dir and exit')
    parser.add_argument('--snr_range', type=str, default='30:-3:0', help='SNR setting, supports 30:-3:0 or 30,25,20,15,10,5,0')
    parser.add_argument('--tdl', type=str, default='A-30,B-100,C-300', help='TDL configurations, comma-separated')
    parser.add_argument('--num_batches', type=int, default=100, help='Evaluation batches per SNR point')
    parser.add_argument('--batches_per_snr', type=int, default=None, help='Friendly alias for --num_batches')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--override', action='append', default=None, help='Universal override for evaluation CLI args in key=value form')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable AMP on GPU during evaluation')
    parser.add_argument('--no-compile', dest='compile', action='store_false', help='Disable torch.compile on GPU')
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--plot_after_eval', dest='plot_after_eval', action='store_true', help='Generate plots after evaluation (default)')
    plot_group.add_argument('--no-plot_after_eval', dest='plot_after_eval', action='store_false', help='Skip plot generation after evaluation')
    parser.set_defaults(use_amp=True, compile=True, plot_after_eval=True)
    parser.add_argument('--output', type=str, default=None, help='Output directory for one evaluation run (default: single run -> <run_dir>/evaluations/<timestamp>, multiple runs -> <exp_dir>/evaluations/<timestamp>_<scope>)')
    return parser


def main():
    """Parse CLI args and dispatch to the evaluation workflow."""
    request = EvaluationRequest.from_namespace(build_parser().parse_args())
    if request.runs and not request.exp_dir:
        raise ValueError('--runs requires --exp_dir')

    if request.list_runs:
        if not request.exp_dir:
            raise ValueError('--list_runs requires --exp_dir')
        run_dirs = discover_run_dirs(request.exp_dir)
        print(f'Evaluable runs: {len(run_dirs)}')
        for run_dir in run_dirs:
            print(f'  - {run_dir.name}')
        return

    device = resolve_device(request.device)
    if device.type == 'cpu':
        request.compile = False
        request.use_amp = False

    target_dirs = resolve_run_selection(
        exp_dir=request.exp_dir,
        run_dir=request.run_dir,
        run_dirs=request.run_dirs,
        runs=request.runs,
    )
    snr_values = parse_snr_range(request.snr_range)
    tdl_configs = split_csv_arg(request.tdl)
    output_dir = resolve_evaluation_output_dir(
        explicit_output=request.output,
        exp_dir=Path(request.exp_dir) if request.exp_dir else None,
        model_dirs=target_dirs,
        force_exp_dir=bool(request.exp_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('Model Evaluation (Refactored)')
    print('=' * 80)
    print(f'Device: {device}')
    print(f'Runs: {[run_dir.name for run_dir in target_dirs]}')
    print(f'SNR values: {snr_values}')
    print(f'TDL configs: {tdl_configs}')
    print(f'Batches per SNR: {request.num_batches}')
    print(f'Batch size: {request.batch_size}')
    print(f'Output: {output_dir}')
    print()

    results = evaluate_models_programmatic(
        exp_dir=Path(request.exp_dir) if request.exp_dir else None,
        output_dir=output_dir,
        snr_range=request.snr_range,
        tdl_list=tdl_configs,
        num_batches=request.num_batches,
        batch_size=request.batch_size,
        device=device,
        use_amp=request.use_amp,
        compile=request.compile,
        model_dirs=target_dirs,
    )

    if request.plot_after_eval:
        print('\n' + '=' * 80)
        print('Generating Evaluation Plots')
        print('=' * 80)
        if request.exp_dir:
            generated_files = generate_plots_for_target_programmatic(request.exp_dir, request.output and Path(request.output) / 'plots')
        elif len(target_dirs) == 1:
            run_eval_dir = Path(next(iter(results['artifacts']['per_run_output_dirs'].values())))
            generated_files = generate_plots_programmatic(
                eval_results_path=run_eval_dir,
                output_dir=run_eval_dir / 'plots',
            )
        else:
            generated_files = []
            for run_eval_dir in results['artifacts']['per_run_output_dirs'].values():
                generated_files.extend(generate_plots_programmatic(run_eval_dir, Path(run_eval_dir) / 'plots'))
            aggregate_dir = results['artifacts']['aggregate_output_dir']
            if aggregate_dir:
                generated_files.extend(generate_plots_programmatic(aggregate_dir, Path(aggregate_dir) / 'plots'))
        print(f'Generated {len(generated_files)} plot(s)')
        if request.exp_dir and results['artifacts']['aggregate_output_dir']:
            print(f"Experiment comparison plots: {Path(results['artifacts']['aggregate_output_dir']) / 'plots'}")


if __name__ == '__main__':
    main()
