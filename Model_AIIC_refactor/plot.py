"""Thin CLI entrypoint for the plotting workflow."""

import argparse

from workflows.plotting_workflow import generate_plots_programmatic, resolve_plot_inputs


def build_parser():
    """Build the plot CLI parser."""
    parser = argparse.ArgumentParser(description='Generate plots from evaluation results')
    parser.add_argument('--input', type=str, required=True, help='Path to an experiment directory, evaluation directory, or evaluation_results.json')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots (default: <evaluation_dir>/plots)')
    return parser


def main():
    """Parse CLI args and dispatch to the plotting workflow."""
    args = build_parser().parse_args()
    eval_results_path, output_dir = resolve_plot_inputs(args.input, args.output)
    print(f"📈 Generating plots...")
    print(f"  Evaluation data: {eval_results_path}")
    print(f"  Output: {output_dir}")
    print()
    generated_files = generate_plots_programmatic(eval_results_path=eval_results_path, output_dir=output_dir)
    print(f"\n✓ Generated {len(generated_files)} plots")
    print(f"  Saved to: {output_dir}")


if __name__ == '__main__':
    main()
