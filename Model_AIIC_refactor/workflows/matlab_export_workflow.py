"""Programmatic Matlab bundle export workflow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat

from utils import build_dummy_input, load_trained_model_from_checkpoint, load_trained_model_from_run, resolve_run_selection


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _resolve_port_stage_module(model: torch.nn.Module, port_idx: int, stage_idx: int):
    if model.share_weights_across_stages:
        return model.port_mlps[port_idx]
    return model.port_mlps[port_idx][stage_idx]


def _export_separator2_weights(model: torch.nn.Module, num_ports: int, num_stages: int) -> Dict[str, np.ndarray]:
    mat_data: Dict[str, np.ndarray] = {}
    for port_idx in range(num_ports):
        for stage_idx in range(num_stages):
            mlp = _resolve_port_stage_module(model, port_idx, stage_idx)
            for layer_idx, layer in enumerate(mlp.layers, start=1):
                prefix = f'p{port_idx + 1:02d}_s{stage_idx + 1:02d}_l{layer_idx:02d}'
                mat_data[f'{prefix}_weight_real'] = _to_numpy(layer.weight_real)
                mat_data[f'{prefix}_weight_imag'] = _to_numpy(layer.weight_imag)
                mat_data[f'{prefix}_bias_real'] = _to_numpy(layer.bias_real)
                mat_data[f'{prefix}_bias_imag'] = _to_numpy(layer.bias_imag)
    return mat_data


def _linear_layers(sequence: nn.Sequential) -> List[nn.Linear]:
    return [layer for layer in sequence if isinstance(layer, nn.Linear)]


def _export_separator1_weights(model: torch.nn.Module, num_ports: int, num_stages: int) -> Dict[str, np.ndarray]:
    mat_data: Dict[str, np.ndarray] = {}
    for port_idx in range(num_ports):
        for stage_idx in range(num_stages):
            mlp = _resolve_port_stage_module(model, port_idx, stage_idx)
            real_layers = _linear_layers(mlp.mlp_real)
            imag_layers = _linear_layers(mlp.mlp_imag)

            for layer_idx, layer in enumerate(real_layers, start=1):
                prefix = f'p{port_idx + 1:02d}_s{stage_idx + 1:02d}_real_l{layer_idx:02d}'
                mat_data[f'{prefix}_weight'] = _to_numpy(layer.weight)
                mat_data[f'{prefix}_bias'] = _to_numpy(layer.bias)

            for layer_idx, layer in enumerate(imag_layers, start=1):
                prefix = f'p{port_idx + 1:02d}_s{stage_idx + 1:02d}_imag_l{layer_idx:02d}'
                mat_data[f'{prefix}_weight'] = _to_numpy(layer.weight)
                mat_data[f'{prefix}_bias'] = _to_numpy(layer.bias)

    return mat_data


def export_run_to_matlab_bundle(
    run_dir,
    output_root=None,
) -> Dict[str, object]:
    """Export a single trained run into a Matlab-friendly explicit-weight bundle."""
    model, artifacts = load_trained_model_from_run(run_dir, device='cpu')
    model.eval()
    model.cpu()

    model_spec = dict(artifacts.model_spec)
    model_type = model_spec['model_type']
    num_ports = int(model_spec['num_ports'])
    num_stages = int(model_spec['num_stages'])
    mlp_depth = int(model_spec['mlp_depth'])

    if output_root is None:
        output_root = artifacts.run_dir / 'matlab_exports'
    else:
        output_root = Path(output_root)
    run_output_dir = output_root
    run_output_dir.mkdir(parents=True, exist_ok=True)

    sample_input = build_dummy_input(model_spec, batch_size=1)
    with torch.no_grad():
        reference_output = model(sample_input)

    mat_data: Dict[str, np.ndarray] = {
        'sample_input': _to_numpy(sample_input),
        'reference_output': _to_numpy(reference_output),
        'pos_values': np.asarray(model_spec.get('pos_values', []), dtype=np.int32),
    }

    if model_type == 'separator2':
        mat_data.update(_export_separator2_weights(model, num_ports=num_ports, num_stages=num_stages))
    elif model_type == 'separator1':
        mat_data.update(_export_separator1_weights(model, num_ports=num_ports, num_stages=num_stages))
    else:
        raise ValueError(f'Unsupported model_type for Matlab bundle export: {model_type}')

    mat_path = run_output_dir / 'matlab_model_bundle.mat'
    savemat(mat_path, mat_data, do_compression=True)

    manifest = {
        'timestamp': datetime.now().isoformat(),
        'format': 'srs_ai_refactor_matlab_bundle_v1',
        'run_name': artifacts.run_dir.name,
        'run_dir': str(artifacts.run_dir),
        'checkpoint_path': str(artifacts.checkpoint_path),
        'mat_file': mat_path.name,
        'mat_path': str(mat_path),
        'model_spec': model_spec,
        'training_spec': artifacts.training_spec,
        'metadata': artifacts.metadata,
        'input_layout': 'N x (2*seq_len) real-stacked float32 = [real_part, imag_part]',
        'output_layout': 'N x num_ports x (2*seq_len) real-stacked float32',
        'sample_input_shape': list(sample_input.shape),
        'reference_output_shape': list(reference_output.shape),
        'reference_sample_rule': 'sample_input/reference_output are always exported with batch size 1; Matlab inference accepts arbitrary batch size N.',
        'materialization_rule': 'Every effective port-stage MLP is materialized explicitly, even when training used shared stage weights.',
        'matlab_entrypoints': [
            'import_refactor_matlab_bundle',
            'predict_refactor_matlab_bundle',
            'run_refactor_matlab_bundle_demo',
        ],
        'bundle_contents': {
            'sample_input_field': 'sample_input',
            'reference_output_field': 'reference_output',
            'pos_values_field': 'pos_values',
            'linear_layers_per_mlp': mlp_depth,
            'separator2_field_pattern': 'p##_s##_l##_weight_real/weight_imag/bias_real/bias_imag',
            'separator1_field_pattern': 'p##_s##_real_l##_weight/bias and p##_s##_imag_l##_weight/bias',
        },
        'input_normalization': {
            'enabled': bool(model_spec.get('normalize_energy', False)),
            'rule': 'Per-sample RMS over the complex sequence; output is rescaled by the same factor after separation.',
        },
    }

    manifest_path = run_output_dir / 'matlab_model_bundle_manifest.json'
    manifest['manifest_path'] = str(manifest_path)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    return manifest


def export_checkpoint_to_matlab_bundle(
    checkpoint_path,
    output_root=None,
) -> Dict[str, object]:
    """Export a single explicit checkpoint into a Matlab-friendly explicit-weight bundle."""
    model, artifacts = load_trained_model_from_checkpoint(checkpoint_path, device='cpu')
    model.eval()
    model.cpu()

    model_spec = dict(artifacts.model_spec)
    model_type = model_spec['model_type']
    num_ports = int(model_spec['num_ports'])
    num_stages = int(model_spec['num_stages'])
    mlp_depth = int(model_spec['mlp_depth'])

    if output_root is None:
        output_root = artifacts.run_dir / 'matlab_exports'
    else:
        output_root = Path(output_root)
    run_output_dir = output_root
    run_output_dir.mkdir(parents=True, exist_ok=True)

    sample_input = build_dummy_input(model_spec, batch_size=1)
    with torch.no_grad():
        reference_output = model(sample_input)

    mat_data: Dict[str, np.ndarray] = {
        'sample_input': _to_numpy(sample_input),
        'reference_output': _to_numpy(reference_output),
        'pos_values': np.asarray(model_spec.get('pos_values', []), dtype=np.int32),
    }

    if model_type == 'separator2':
        mat_data.update(_export_separator2_weights(model, num_ports=num_ports, num_stages=num_stages))
    elif model_type == 'separator1':
        mat_data.update(_export_separator1_weights(model, num_ports=num_ports, num_stages=num_stages))
    else:
        raise ValueError(f'Unsupported model_type for Matlab bundle export: {model_type}')

    mat_path = run_output_dir / 'matlab_model_bundle.mat'
    savemat(mat_path, mat_data, do_compression=True)

    manifest = {
        'timestamp': datetime.now().isoformat(),
        'format': 'srs_ai_refactor_matlab_bundle_v1',
        'run_name': artifacts.run_dir.name,
        'run_dir': str(artifacts.run_dir),
        'checkpoint_path': str(artifacts.checkpoint_path),
        'mat_file': mat_path.name,
        'mat_path': str(mat_path),
        'model_spec': model_spec,
        'training_spec': artifacts.training_spec,
        'metadata': artifacts.metadata,
        'input_layout': 'N x (2*seq_len) real-stacked float32 = [real_part, imag_part]',
        'output_layout': 'N x num_ports x (2*seq_len) real-stacked float32',
        'sample_input_shape': list(sample_input.shape),
        'reference_output_shape': list(reference_output.shape),
        'reference_sample_rule': 'sample_input/reference_output are always exported with batch size 1; Matlab inference accepts arbitrary batch size N.',
        'materialization_rule': 'Every effective port-stage MLP is materialized explicitly, even when training used shared stage weights.',
        'matlab_entrypoints': [
            'import_refactor_matlab_bundle',
            'predict_refactor_matlab_bundle',
            'run_refactor_matlab_bundle_demo',
        ],
        'bundle_contents': {
            'sample_input_field': 'sample_input',
            'reference_output_field': 'reference_output',
            'pos_values_field': 'pos_values',
            'linear_layers_per_mlp': mlp_depth,
            'separator2_field_pattern': 'p##_s##_l##_weight_real/weight_imag/bias_real/bias_imag',
            'separator1_field_pattern': 'p##_s##_real_l##_weight/bias and p##_s##_imag_l##_weight/bias',
        },
        'input_normalization': {
            'enabled': bool(model_spec.get('normalize_energy', False)),
            'rule': 'Per-sample RMS over the complex sequence; output is rescaled by the same factor after separation.',
        },
    }

    manifest_path = run_output_dir / 'matlab_model_bundle_manifest.json'
    manifest['manifest_path'] = str(manifest_path)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    return manifest


def export_runs_to_matlab_bundle(
    output_root,
    exp_dir=None,
    run_dir=None,
    run_dirs=None,
    runs=None,
) -> List[Dict[str, object]]:
    """Programmatic multi-run Matlab bundle export entry point."""
    target_dirs = resolve_run_selection(
        exp_dir=exp_dir,
        run_dir=run_dir,
        run_dirs=run_dirs,
        runs=runs,
    )
    if output_root is not None and len(target_dirs) > 1:
        raise ValueError('Shared output_root is only supported for a single run. For multiple runs, omit --output so each run writes to its own matlab_exports directory.')
    return [
        export_run_to_matlab_bundle(
            run_dir=target_dir,
            output_root=output_root,
        )
        for target_dir in target_dirs
    ]