"""Thin CLI entrypoint for Matlab explicit-weight bundle export."""

import argparse

from workflows.matlab_export_workflow import export_checkpoint_to_matlab_bundle


def build_parser():
    parser = argparse.ArgumentParser(description='Export one trained refactored checkpoint to a Matlab explicit-weight bundle')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a specific checkpoint file, e.g. model.pth or checkpoint_batch_87000.pth')
    parser.add_argument('--output', type=str, default=None, help='Directory where Matlab bundle artifacts are written for a single run (default: each run directory under matlab_exports/)')
    return parser


def main():
    args = build_parser().parse_args()
    manifest = export_checkpoint_to_matlab_bundle(
        checkpoint_path=args.checkpoint,
        output_root=args.output,
    )

    if args.output:
        print(f'✓ Exported Matlab bundle to {args.output}')
    else:
        print('✓ Exported Matlab bundle to the run-specific matlab_exports directory')
    print(f"  - {manifest['run_name']}: {manifest['mat_path']}")


if __name__ == '__main__':
    main()