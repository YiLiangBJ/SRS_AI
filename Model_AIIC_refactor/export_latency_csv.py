"""Export or backfill latency benchmark CSV tables from saved latency_results.json files."""

import argparse

from benchmarks.workflow import backfill_latency_csv_tree, export_latency_csv_from_results


def build_parser():
    parser = argparse.ArgumentParser(description='Export latency_results.csv from saved latency benchmark outputs')
    parser.add_argument('--input', type=str, required=True, help='Latency directory, latency_results.json, or a parent directory to scan with --recursive')
    parser.add_argument('--output', type=str, default=None, help='Optional explicit CSV output path when exporting a single result')
    parser.add_argument('--recursive', action='store_true', help='Recursively backfill latency_results.csv for every latency_results.json under --input')
    return parser


def main():
    args = build_parser().parse_args()
    if args.recursive:
        generated = backfill_latency_csv_tree(args.input)
        print(f'Generated {len(generated)} CSV file(s)')
        for path in generated:
            print(f'  - {path}')
        return

    output_path = export_latency_csv_from_results(args.input, output_path=args.output)
    print(output_path)


if __name__ == '__main__':
    main()