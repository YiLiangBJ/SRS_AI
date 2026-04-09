"""Programmatic evaluation workflow."""

import json
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import torch
from tqdm import tqdm

from data import generate_training_batch
from utils import split_csv_arg, discover_run_dirs, load_trained_model_from_run


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
        torch.cuda.set_device(resolved if resolved.index is not None else 0)

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
    """Evaluate one run at one SNR point."""
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


def _slugify_label(value: str) -> str:
    """Normalize a path label for directory names."""
    normalized = re.sub(r'[^A-Za-z0-9._-]+', '-', value).strip('-')
    return normalized or 'evaluation'


def _build_evaluation_scope_label(model_dirs) -> str:
    """Build a short label describing the evaluated run selection."""
    if not model_dirs:
        return 'all-runs'

    run_names = [Path(model_dir).name for model_dir in model_dirs]
    if len(run_names) == 1:
        return _slugify_label(run_names[0])
    if len(run_names) <= 3:
        combined = '_'.join(_slugify_label(run_name) for run_name in run_names)
        return combined[:80].rstrip('_-')
    return f'{len(run_names)}-runs'


def resolve_evaluation_output_dir(explicit_output=None, exp_dir: Path | None = None, model_dirs=None) -> Path:
    """Resolve the default output directory for evaluation artifacts."""
    if explicit_output:
        return Path(explicit_output)

    evaluation_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_build_evaluation_scope_label(model_dirs)}"

    if exp_dir is not None:
        return Path(exp_dir) / 'evaluations' / evaluation_name

    if model_dirs:
        common_parent = Path(model_dirs[0]).parent
        if all(Path(model_dir).parent == common_parent for model_dir in model_dirs):
            return common_parent / 'evaluations' / evaluation_name

    return Path('evaluations') / evaluation_name


def evaluate_models_programmatic(
    exp_dir,
    output_dir=None,
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

    output_dir = Path(output_dir) if output_dir is not None else resolve_evaluation_output_dir(
        exp_dir=exp_dir,
        model_dirs=target_dirs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_name': output_dir.name,
        'output_dir': str(output_dir),
        'config': {
            'snr_list': snr_list,
            'tdl_list': tdl_list,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'total_samples_per_point': num_batches * batch_size,
            'run_names': [run_dir.name for run_dir in target_dirs],
            'run_count': len(target_dirs),
        },
        'models': {},
    }

    for run_dir in target_dirs:
        run_name = run_dir.name
        try:
            model, artifacts = load_trained_model_from_run(run_dir, device=device)
            model_spec = artifacts.model_spec
            if compile and device.type == 'cuda' and hasattr(torch, 'compile'):
                torch.set_float32_matmul_precision('high')
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
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
