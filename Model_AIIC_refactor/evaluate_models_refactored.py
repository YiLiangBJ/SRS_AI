"""Refactored evaluation entry point with run selection and friendly scan controls."""

import argparse
from datetime import datetime
from pathlib import Path

import json
import numpy as np
import torch
from tqdm import tqdm

from data import generate_training_batch
from utils import (
    split_csv_arg,
    discover_run_dirs,
    resolve_run_selection,
    load_trained_model_from_run,
)


def parse_snr_range(snr_str):
    """Parse SNR strings such as 30:-3:0 or 30,20,10."""
    if ':' in snr_str:
        parts = snr_str.split(':')
        if len(parts) != 3:
            raise ValueError(f'Invalid SNR range format: {snr_str}')
        start, step, end = map(float, parts)
        return np.arange(start, end - 0.1 * abs(step), step).tolist()
    if ',' in snr_str:
        return [float(value) for value in snr_str.split(',')]
    return [float(snr_str)]


def resolve_device(device):
    """Normalize device selection and set CUDA device when needed."""
    if device == 'auto':
        resolved = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        resolved = torch.device(device)

    if resolved.type == 'cuda':
        if resolved.index is not None:
            torch.cuda.set_device(resolved)
        else:
            torch.cuda.set_device(0)

    return resolved


def evaluate_at_snr(
    model,
    model_spec: dict,
    snr_db: float,
    tdl_config: str,
    num_batches: int = 100,
    batch_size: int = 2048,
    device='cpu',
    use_amp: bool = False,
):
    """
    Evaluate model at specific SNR
    
    Args:
        model: Model to evaluate
        model_spec: Model spec
        snr_db: SNR in dB
        tdl_config: TDL configuration
        num_batches: Number of simulation batches per SNR point
        batch_size: Batch size
    
    Returns:
        dict: Evaluation results
    """
    seq_len = model_spec['seq_len']
    pos_values = model_spec['pos_values']
    num_ports = len(pos_values)

    total_mse = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    port_mse = torch.zeros(num_ports, device=device)
    port_power = torch.zeros(num_ports, device=device)
    autocast_context = torch.cuda.amp.autocast if use_amp and device.type == 'cuda' else None

    with torch.no_grad():
        for _ in range(num_batches):
            y, h_targets, _, _, _ = generate_training_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                pos_values=pos_values,
                snr_db=snr_db,
                tdl_config=tdl_config,
                return_complex=False,
                device=device,
            )

            if autocast_context is not None:
                with autocast_context():
                    h_pred = model(y)
            else:
                h_pred = model(y)

            diff = h_pred - h_targets
            total_mse += diff.pow(2).sum()
            total_power += h_targets.pow(2).sum()
            port_mse += diff.pow(2).sum(dim=(0, 2))
            port_power += h_targets.pow(2).sum(dim=(0, 2))

    nmse = (total_mse / (total_power + 1e-10)).cpu().item()
    nmse_db = 10 * np.log10(nmse) if nmse > 0 else -100
    port_nmse = (port_mse / (port_power + 1e-10)).cpu().numpy()
    port_nmse_db = 10 * np.log10(port_nmse)
    port_nmse_db[np.isinf(port_nmse_db)] = -100

    return {
        'snr_db': float(snr_db),
        'tdl_config': tdl_config,
        'nmse': float(nmse),
        'nmse_db': float(nmse_db),
        'per_port_nmse': port_nmse.tolist(),
        'per_port_nmse_db': port_nmse_db.tolist(),
        'num_samples': num_batches * batch_size,
    }


def save_evaluation_results(results, output_dir: Path):
    """Persist evaluation results in JSON and NumPy formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)

    numpy_data = {}
    for run_name, run_data in results['models'].items():
        numpy_data[run_name] = {}
        for tdl_config, tdl_data in run_data['tdl_results'].items():
            numpy_data[run_name][tdl_config] = {
                'snr': np.array(tdl_data['snr']),
                'nmse': np.array(tdl_data['nmse']),
                'nmse_db': np.array(tdl_data['nmse_db']),
                'port_nmse': np.array(tdl_data['port_nmse']),
                'port_nmse_db': np.array(tdl_data['port_nmse_db']),
            }

    npy_path = output_dir / 'evaluation_results.npy'
    np.save(npy_path, numpy_data, allow_pickle=True)
    return json_path, npy_path


def evaluate_models_programmatic(
    exp_dir,
    output_dir,
    snr_range='30:-3:0',
    snr_values=None,
    tdl_list=None,
    num_batches=100,
    batch_size=2048,
    device='auto',
    use_amp=False,
    compile=False,
    models=None,
    model_dirs=None,
):
    """Programmatic multi-run evaluation entry point."""
    device = resolve_device(device)
    exp_dir = Path(exp_dir) if exp_dir is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_list = [float(value) for value in split_csv_arg(snr_values)] if snr_values else parse_snr_range(snr_range)
    tdl_list = split_csv_arg(tdl_list) if tdl_list is not None else ['A-30', 'B-100', 'C-300']

    if model_dirs:
        target_dirs = [Path(path) for path in model_dirs]
    elif models:
        if exp_dir is None:
            raise ValueError('exp_dir is required when selecting runs by name')
        target_dirs = [exp_dir / run_name for run_name in split_csv_arg(models)]
    else:
        if exp_dir is None:
            raise ValueError('exp_dir is required when model_dirs is not provided')
        target_dirs = discover_run_dirs(exp_dir)

    if not target_dirs:
        raise ValueError('No trained runs found to evaluate')

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'snr_list': snr_list,
            'tdl_list': tdl_list,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'total_samples_per_point': num_batches * batch_size,
        },
        'models': {},
    }

    for run_dir in target_dirs:
        run_name = run_dir.name
        try:
            model, artifacts = load_trained_model_from_run(run_dir, device=device)
            model_spec = artifacts.model_spec
            if compile and device.type == 'cuda' and hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
            model.eval()

            results['models'][run_name] = {
                'model_spec': model_spec,
                'training_spec': artifacts.training_spec,
                'metadata': artifacts.metadata,
                'checkpoint_path': str(artifacts.checkpoint_path),
                'tdl_results': {},
            }

            for tdl_config in tdl_list:
                tdl_results = {
                    'snr': [],
                    'nmse': [],
                    'nmse_db': [],
                    'port_nmse': [],
                    'port_nmse_db': [],
                }

                for snr_db in tqdm(snr_list, desc=f'  {run_name} - {tdl_config}', leave=False):
                    point_result = evaluate_at_snr(
                        model=model,
                        model_spec=model_spec,
                        snr_db=snr_db,
                        tdl_config=tdl_config,
                        num_batches=num_batches,
                        batch_size=batch_size,
                        device=device,
                        use_amp=use_amp,
                    )

                    tdl_results['snr'].append(point_result['snr_db'])
                    tdl_results['nmse'].append(point_result['nmse'])
                    tdl_results['nmse_db'].append(point_result['nmse_db'])
                    tdl_results['port_nmse'].append(point_result['per_port_nmse'])
                    tdl_results['port_nmse_db'].append(point_result['per_port_nmse_db'])

                results['models'][run_name]['tdl_results'][tdl_config] = tdl_results

        except Exception as error:
            print(f'✗ Run {run_name} evaluation failed: {error}')
            continue

    if not results['models']:
        raise RuntimeError(
            'No runs were evaluated successfully. If these are older checkpoints, retrain them with the current '
            'training pipeline so model_spec is saved.'
        )

    save_evaluation_results(results, output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained runs across SNR and TDL settings')
    parser.add_argument('--exp_dir', type=str, default=None,
                       help='Experiment directory. Evaluates all runs inside by default, or a subset with --runs')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Single trained run directory')
    parser.add_argument('--run_dirs', type=str, default=None,
                       help='Multiple trained run directories, comma-separated')
    parser.add_argument('--runs', type=str, default=None,
                       help='Run names inside --exp_dir, comma-separated')
    parser.add_argument('--list_runs', action='store_true',
                       help='List evaluable runs inside --exp_dir and exit')
    parser.add_argument('--snr_range', type=str, default='30:-3:0',
                       help='SNR range such as 30:-3:0')
    parser.add_argument('--snr_values', type=str, default=None,
                       help='Explicit SNR list such as 30,25,20,15,10,5,0. Overrides --snr_range')
    parser.add_argument('--tdl', type=str, default='A-30,B-100,C-300',
                       help='TDL configurations, comma-separated')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Evaluation batches per SNR point')
    parser.add_argument('--batches_per_snr', type=int, default=None,
                       help='Friendly alias for --num_batches')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--use_amp', action='store_true',
                       help='Enable AMP on GPU during evaluation')
    parser.add_argument('--no-compile', dest='compile', action='store_false',
                       help='Disable torch.compile on GPU')
    parser.set_defaults(compile=True)
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
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

    target_dirs = resolve_run_selection(
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
    )
    snr_values = [float(value) for value in split_csv_arg(args.snr_values)] if args.snr_values else parse_snr_range(args.snr_range)
    tdl_configs = split_csv_arg(args.tdl)
    output_dir = Path(args.output)
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

    results = evaluate_models_programmatic(
        exp_dir=Path(args.exp_dir) if args.exp_dir else target_dirs[0].parent,
        output_dir=output_dir,
        snr_range=args.snr_range,
        snr_values=args.snr_values,
        tdl_list=tdl_configs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        device=device,
        use_amp=args.use_amp,
        compile=args.compile,
        model_dirs=target_dirs,
    )

    json_path = output_dir / 'evaluation_results.json'
    npy_path = output_dir / 'evaluation_results.npy'
    print('\n' + '=' * 80)
    print('Evaluation Summary')
    print('=' * 80)
    print(f'Evaluated runs: {len(results["models"])}')
    print(f'TDL configs: {tdl_configs}')
    print(f'SNR values: {snr_values}')
    print(f'Total samples per point: {args.num_batches * args.batch_size}')
    print(f'JSON: {json_path}')
    print(f'NPY:  {npy_path}')


if __name__ == '__main__':
    main()
