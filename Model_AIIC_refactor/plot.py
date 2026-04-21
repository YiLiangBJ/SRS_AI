"""Thin CLI entrypoint for the plotting workflow."""

import argparse
from pathlib import Path

from workflows.plotting_workflow import generate_plots_for_target_programmatic, generate_plots_programmatic, resolve_plot_inputs


def build_parser():
    """Build the plot CLI parser."""
    parser = argparse.ArgumentParser(description='Generate plots from evaluation results')
    parser.add_argument('--input', type=str, required=True, help='Path to an experiment directory, evaluation directory, or evaluation_results.json')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots (default: <evaluation_dir>/plots)')
    return parser


def main():
    """Parse CLI args and dispatch to the plotting workflow."""
    args = build_parser().parse_args()
    print(f"📈 Generating plots...")
    print(f"  Input: {args.input}")
    if args.output:
        print(f"  Aggregate output override: {args.output}")
    print()
    generated_files = generate_plots_for_target_programmatic(args.input, Path(args.output) if args.output else None)
    print(f"\n✓ Generated {len(generated_files)} plots")
    if args.output:
        print(f"  Aggregate plots saved to: {args.output}")


if __name__ == '__main__':
    main()
