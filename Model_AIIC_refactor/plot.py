"""Thin CLI entrypoint for the plotting workflow."""

import argparse

from workflows.plotting_workflow import generate_plots_programmatic


def build_parser():
    """Build the plot CLI parser."""
    parser = argparse.ArgumentParser(description='Generate plots from evaluation results')
    parser.add_argument('--input', type=str, required=True, help='Path to evaluation_results.json')
    parser.add_argument('--output', type=str, default='./plots', help='Output directory for plots')
    return parser


def main():
    """Parse CLI args and dispatch to the plotting workflow."""
    args = build_parser().parse_args()
    print(f"📈 Generating plots...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print()
    generated_files = generate_plots_programmatic(eval_results_path=args.input, output_dir=args.output)
    print(f"\n✓ Generated {len(generated_files)} plots")
    print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()
