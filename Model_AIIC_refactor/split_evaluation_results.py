"""Create single-run evaluation JSON views from a multi-run evaluation result."""

import argparse
import json
from pathlib import Path


def resolve_eval_json(input_path: str) -> Path:
    candidate = Path(input_path)
    if candidate.is_dir():
        candidate = candidate / 'evaluation_results.json'
    if not candidate.exists() or candidate.name != 'evaluation_results.json':
        raise FileNotFoundError(f'Could not find evaluation_results.json from: {input_path}')
    return candidate


def filter_evaluation_to_run(eval_json_path: Path, run_name: str, output_dir: Path | None = None) -> Path:
    with open(eval_json_path, 'r', encoding='utf-8') as input_file:
        payload = json.load(input_file)

    models = payload.get('models', {})
    if run_name not in models:
        available = ', '.join(sorted(models.keys()))
        raise KeyError(f'Run {run_name!r} not found. Available runs: {available}')

    target_output_dir = output_dir or eval_json_path.parent
    target_output_dir.mkdir(parents=True, exist_ok=True)

    filtered_payload = dict(payload)
    filtered_payload['evaluation_name'] = target_output_dir.name
    filtered_payload['output_dir'] = str(target_output_dir)
    filtered_payload['models'] = {run_name: models[run_name]}

    config = dict(payload.get('config', {}))
    config['run_names'] = [run_name]
    config['run_count'] = 1
    filtered_payload['config'] = config

    output_json_path = target_output_dir / 'evaluation_results.json'
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(filtered_payload, output_file, indent=2, ensure_ascii=False)

    return output_json_path


def build_parser():
    parser = argparse.ArgumentParser(description='Split a multi-run evaluation JSON into a single-run JSON view')
    parser.add_argument('--input', required=True, help='Evaluation directory or evaluation_results.json path')
    parser.add_argument('--run_name', required=True, help='Run name to keep in the output JSON')
    parser.add_argument('--output_dir', default=None, help='Where to write the filtered evaluation_results.json (default: overwrite in input dir)')
    return parser


def main():
    args = build_parser().parse_args()
    eval_json_path = resolve_eval_json(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_json_path = filter_evaluation_to_run(eval_json_path, args.run_name, output_dir=output_dir)
    print(f'✓ Wrote filtered evaluation JSON: {output_json_path}')


if __name__ == '__main__':
    main()