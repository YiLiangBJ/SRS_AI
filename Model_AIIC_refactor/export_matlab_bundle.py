"""Thin CLI entrypoint for Matlab explicit-weight bundle export."""

import argparse

from utils import discover_run_dirs
from workflows.matlab_export_workflow import export_runs_to_matlab_bundle


def build_parser():
    parser = argparse.ArgumentParser(description='Export trained refactored runs to Matlab explicit-weight bundles')
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory. Exports all runs inside by default, or a subset with --runs')
    parser.add_argument('--run_dir', type=str, default=None, help='Single trained run directory')
    parser.add_argument('--run_dirs', type=str, default=None, help='Multiple trained run directories, comma-separated')
    parser.add_argument('--runs', type=str, default=None, help='Run names inside --exp_dir, comma-separated')
    parser.add_argument('--list_runs', action='store_true', help='List exportable runs inside --exp_dir and exit')
    parser.add_argument('--output', type=str, default=None, help='Directory where Matlab bundle artifacts are written for a single run (default: each run directory under matlab_exports/)')
    parser.add_argument('--batch_size', type=int, default=2, help='Reference batch size stored in sample_input/reference_output')
    return parser


def main():
    args = build_parser().parse_args()
    if args.runs and not args.exp_dir:
        raise ValueError('--runs requires --exp_dir')

    if args.list_runs:
        if not args.exp_dir:
            raise ValueError('--list_runs requires --exp_dir')
        run_dirs = discover_run_dirs(args.exp_dir)
        print(f'Exportable runs: {len(run_dirs)}')
        for resolved_run_dir in run_dirs:
            print(f'  - {resolved_run_dir.name}')
        return

    manifests = export_runs_to_matlab_bundle(
        output_root=args.output,
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
        batch_size=args.batch_size,
    )

    if args.output:
        print(f'✓ Exported {len(manifests)} run(s) to {args.output}')
    else:
        print(f'✓ Exported {len(manifests)} run(s) to per-run matlab_exports directories')
    for manifest in manifests:
        print(f"  - {manifest['run_name']}: {manifest['mat_path']}")


if __name__ == '__main__':
    main()