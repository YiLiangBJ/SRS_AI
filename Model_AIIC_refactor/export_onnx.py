"""Thin CLI entrypoint for the ONNX export workflow."""

import argparse

from workflows.export_workflow import export_checkpoint_to_onnx


def build_parser():
    """Build the export CLI parser."""
    parser = argparse.ArgumentParser(description='Export one trained refactored checkpoint to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a specific checkpoint file, e.g. model.pth or checkpoint_batch_87000.pth')
    parser.add_argument('--output', type=str, default=None, help='Directory where ONNX artifacts are written for a single run (default: each run directory under onnx_exports/)')
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
    manifest = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_root=args.output,
        opset_version=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        validate=args.validate,
    )

    if args.output:
        print(f'✓ Exported ONNX to {args.output}')
    else:
        print('✓ Exported ONNX to the run-specific onnx_exports directory')
    print(f"  - {manifest['run_name']}: {manifest['onnx_path']}")


if __name__ == '__main__':
    main()
