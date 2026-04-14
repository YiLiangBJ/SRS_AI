"""Thin CLI entrypoint for the ONNX export workflow."""

import argparse

from utils import discover_run_dirs
from workflows.export_workflow import validate_exported_model, export_run_to_onnx, export_runs_to_onnx


def build_parser():
    """Build the export CLI parser."""
    parser = argparse.ArgumentParser(description='Export trained refactored runs to ONNX')
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory. Exports all runs inside by default, or a subset with --runs')
    parser.add_argument('--run_dir', type=str, default=None, help='Single trained run directory')
    parser.add_argument('--run_dirs', type=str, default=None, help='Multiple trained run directories, comma-separated')
    parser.add_argument('--runs', type=str, default=None, help='Run names inside --exp_dir, comma-separated')
    parser.add_argument('--list_runs', action='store_true', help='List exportable runs inside --exp_dir and exit')
    parser.add_argument('--output', type=str, default=None, help='Directory where exported ONNX artifacts are written (default: each run directory under onnx_exports/)')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version (13 is a Matlab-friendly default)')
    parser.add_argument('--batch_size', type=int, default=1, help='Dummy batch size used for export tracing')
    dynamic_group = parser.add_mutually_exclusive_group()
    dynamic_group.add_argument('--dynamic_batch', dest='dynamic_batch', action='store_true', help='Export with a dynamic batch dimension (default)')
    dynamic_group.add_argument('--fixed_batch', dest='dynamic_batch', action='store_false', help='Export with a fixed batch dimension')
    parser.set_defaults(dynamic_batch=True)
    parser.add_argument('--validate', action='store_true', help='Run ONNX checker and ONNX Runtime smoke validation after export')
    return parser


def main():
    """Parse CLI args and dispatch to the export workflow."""
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

    manifests = export_runs_to_onnx(
        output_root=args.output,
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
        opset_version=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        validate=args.validate,
    )

    if args.output:
        print(f'✓ Exported {len(manifests)} run(s) to {args.output}')
    else:
        print(f'✓ Exported {len(manifests)} run(s) to per-run onnx_exports directories')
    for manifest in manifests:
        print(f"  - {manifest['run_name']}: {manifest['onnx_path']}")


if __name__ == '__main__':
    main()
