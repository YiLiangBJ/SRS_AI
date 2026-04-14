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
from workflows.plotting_workflow import generate_plots_programmatic


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
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable AMP on GPU during evaluation')
    parser.add_argument('--no-compile', dest='compile', action='store_false', help='Disable torch.compile on GPU')
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--plot_after_eval', dest='plot_after_eval', action='store_true', help='Generate plots after evaluation (default)')
    plot_group.add_argument('--no-plot_after_eval', dest='plot_after_eval', action='store_false', help='Skip plot generation after evaluation')
    parser.set_defaults(use_amp=True, compile=True, plot_after_eval=True)
    parser.add_argument('--output', type=str, default=None, help='Output directory for one evaluation run (default: <exp_dir>/evaluations/<timestamp>_<scope>)')
    return parser


def main():
    """Parse CLI args and dispatch to the evaluation workflow."""
    args = build_parser().parse_args()
    if args.batches_per_snr is not None:
        args.num_batches = args.batches_per_snr
    if args.runs and not args.exp_dir:
        raise ValueError('--runs requires --exp_dir')

    if args.list_runs:
        if not args.exp_dir:
            raise ValueError('--list_runs requires --exp_dir')
        run_dirs = discover_run_dirs(args.exp_dir)
        print(f'Evaluable runs: {len(run_dirs)}')
        for run_dir in run_dirs:
            print(f'  - {run_dir.name}')
        return

    device = resolve_device(args.device)
    if device.type == 'cpu':
        args.compile = False
        args.use_amp = False

    target_dirs = resolve_run_selection(
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
    )
    snr_values = parse_snr_range(args.snr_range)
    tdl_configs = split_csv_arg(args.tdl)
    output_dir = resolve_evaluation_output_dir(
        explicit_output=args.output,
        exp_dir=Path(args.exp_dir) if args.exp_dir else None,
        model_dirs=target_dirs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('Model Evaluation (Refactored)')
    print('=' * 80)
    print(f'Device: {device}')
    print(f'Runs: {[run_dir.name for run_dir in target_dirs]}')
    print(f'SNR values: {snr_values}')
    print(f'TDL configs: {tdl_configs}')
    print(f'Batches per SNR: {args.num_batches}')
    print(f'Batch size: {args.batch_size}')
    print(f'Output: {output_dir}')
    print()

    evaluate_models_programmatic(
        exp_dir=Path(args.exp_dir) if args.exp_dir else target_dirs[0].parent,
        output_dir=output_dir,
        snr_range=args.snr_range,
        tdl_list=tdl_configs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        device=device,
        use_amp=args.use_amp,
        compile=args.compile,
        model_dirs=target_dirs,
    )

    if args.plot_after_eval:
        print('\n' + '=' * 80)
        print('Generating Evaluation Plots')
        print('=' * 80)
        plot_output_dir = output_dir / 'plots'
        generated_files = generate_plots_programmatic(
            eval_results_path=output_dir,
            output_dir=plot_output_dir,
        )
        print(f'Generated {len(generated_files)} plot(s)')
        print(f'Plot output: {plot_output_dir}')


if __name__ == '__main__':
    main()
