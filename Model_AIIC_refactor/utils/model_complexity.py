"""Human-readable model complexity summaries for trained artifacts and exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from models.separator2 import ComplexLinearReal


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def _shape(*dims: int) -> List[int]:
    return [int(dim) for dim in dims]


def _shape_string(shape: List[int]) -> str:
    return '[' + ', '.join(str(dim) for dim in shape) + ']'


def _format_int(value: int) -> str:
    return f'{int(value):,}'


def _format_bytes(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    return f'{mib:.3f} MiB ({num_bytes:,} bytes)'


def _linear_ops(in_features: int, out_features: int) -> Dict[str, int]:
    macs = in_features * out_features
    return {
        'macs': macs,
        'multiplications': macs,
        'additions': macs,
        'other_ops': 0,
        'parameters': in_features * out_features + out_features,
    }


def _complex_linear_ops(in_features: int, out_features: int) -> Dict[str, int]:
    macs = 4 * in_features * out_features
    return {
        'macs': macs,
        'multiplications': macs,
        'additions': macs,
        'other_ops': 0,
        'parameters': 2 * in_features * out_features + 2 * out_features,
    }


def _normalization_ops(seq_len: int, num_ports: int, enabled: bool) -> Dict[str, int]:
    if not enabled:
        return {'macs': 0, 'multiplications': 0, 'additions': 0, 'other_ops': 0, 'parameters': 0}
    input_width = seq_len * 2
    output_width = input_width * num_ports
    return {
        'macs': 0,
        'multiplications': input_width + output_width,
        'additions': input_width - 1,
        'other_ops': input_width + 1,
        'parameters': 0,
    }


def _relu_ops(width: int) -> Dict[str, int]:
    return {'macs': 0, 'multiplications': 0, 'additions': 0, 'other_ops': width, 'parameters': 0}


def _layer_norm_ops(width: int) -> Dict[str, int]:
    return {
        'macs': 0,
        'multiplications': 3 * width,
        'additions': 2 * width,
        'other_ops': width + 2,
        'parameters': 2 * width,
    }


def _residual_ops(seq_len: int, num_ports: int) -> Dict[str, int]:
    width = seq_len * 2
    return {
        'macs': 0,
        'multiplications': 0,
        'additions': width * (2 * num_ports),
        'other_ops': 0,
        'parameters': 0,
    }


def _complex_activation_ops(width: int, activation_type: str) -> Dict[str, int]:
    if activation_type in {'relu', 'split_relu', 'z_relu'}:
        other_ops = width * 2
    elif activation_type == 'mod_relu':
        other_ops = width * 8
    elif activation_type == 'cardioid':
        other_ops = width * 10
    else:
        other_ops = width * 2
    return {'macs': 0, 'multiplications': 0, 'additions': 0, 'other_ops': other_ops, 'parameters': 0}


def _scale_ops(ops: Dict[str, int], multiplier: int) -> Dict[str, int]:
    return {key: int(value) * int(multiplier) for key, value in ops.items()}


def _finalize_entry(name: str, repeat: str, why: str, ops: Dict[str, int]) -> Dict[str, Any]:
    total_flops = ops['multiplications'] + ops['additions'] + ops['other_ops']
    return {
        'name': name,
        'repeat': repeat,
        'why': why,
        'parameters': int(ops['parameters']),
        'parameters_string': _format_int(ops['parameters']),
        'macs': int(ops['macs']),
        'macs_string': _format_int(ops['macs']),
        'multiplications': int(ops['multiplications']),
        'multiplications_string': _format_int(ops['multiplications']),
        'additions': int(ops['additions']),
        'additions_string': _format_int(ops['additions']),
        'other_ops': int(ops['other_ops']),
        'other_ops_string': _format_int(ops['other_ops']),
        'flops_estimate': int(total_flops),
        'flops_estimate_string': _format_int(total_flops),
    }


def _summarize(entries: List[Dict[str, Any]], trainable_parameters: int) -> Dict[str, Any]:
    macs = sum(item['macs'] for item in entries)
    multiplications = sum(item['multiplications'] for item in entries)
    additions = sum(item['additions'] for item in entries)
    other_ops = sum(item['other_ops'] for item in entries)
    flops = multiplications + additions + other_ops
    parameter_bytes = trainable_parameters * 4
    return {
        'trainable_parameters': int(trainable_parameters),
        'trainable_parameters_string': _format_int(trainable_parameters),
        'parameter_memory_bytes_fp32': int(parameter_bytes),
        'parameter_memory_bytes_fp32_string': _format_bytes(parameter_bytes),
        'macs_per_sample': int(macs),
        'macs_per_sample_string': _format_int(macs),
        'multiplications_per_sample': int(multiplications),
        'multiplications_per_sample_string': _format_int(multiplications),
        'additions_per_sample': int(additions),
        'additions_per_sample_string': _format_int(additions),
        'other_scalar_ops_per_sample_estimate': int(other_ops),
        'other_scalar_ops_per_sample_estimate_string': _format_int(other_ops),
        'flops_per_sample_estimate': int(flops),
        'flops_per_sample_estimate_string': _format_int(flops),
        'batch_scaling_rule': 'Multiply the per-sample counts by runtime batch size N for a first-order batch estimate.',
    }


def _full_mlp_entries(model: torch.nn.Module, model_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = _unwrap_model(model)
    linears = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    entries: List[Dict[str, Any]] = []
    for index, layer in enumerate(linears, 1):
        ops = _linear_ops(layer.in_features, layer.out_features)
        entries.append(_finalize_entry(
            name=f'joint_linear_{index:02d}',
            repeat='once per sample',
            why=f'Joint affine layer maps width {layer.in_features} to width {layer.out_features}.',
            ops=ops,
        ))
        if index < len(linears):
            relu = _relu_ops(layer.out_features)
            entries.append(_finalize_entry(
                name=f'joint_relu_{index:02d}',
                repeat='once per sample',
                why='ReLU is applied after every hidden affine layer in the joint MLP.',
                ops=relu,
            ))
    entries.insert(0, _finalize_entry(
        name='input_normalize_restore',
        repeat='once per sample',
        why='Per-sample RMS normalization and output rescaling are applied around the network when normalize_energy=true.',
        ops=_normalization_ops(model.seq_len, model.num_ports, model.normalize_energy),
    ))
    return entries


def _separator1_entries(model: torch.nn.Module, model_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = _unwrap_model(model)
    example_mlp = model.port_mlps[0] if model.share_weights_across_stages else model.port_mlps[0][0]
    repeat = f"per port, per stage ({model.num_ports} ports x {model.num_stages} stage executions)"
    stage_multiplier = model.num_ports * model.num_stages
    entries: List[Dict[str, Any]] = [
        _finalize_entry(
            name='input_normalize_restore',
            repeat='once per sample',
            why='Per-sample RMS normalization and output rescaling are applied around the network when normalize_energy=true.',
            ops=_normalization_ops(model.seq_len, model.num_ports, model.normalize_energy),
        )
    ]

    for branch_name, branch in (('real', example_mlp.mlp_real), ('imag', example_mlp.mlp_imag)):
        linear_index = 0
        hidden_width = None
        for layer in branch:
            if isinstance(layer, nn.Linear):
                linear_index += 1
                hidden_width = layer.out_features
                ops = _scale_ops(_linear_ops(layer.in_features, layer.out_features), stage_multiplier)
                entries.append(_finalize_entry(
                    name=f'{branch_name}_linear_{linear_index:02d}',
                    repeat=repeat,
                    why=f'{branch_name.title()} branch affine layer maps width {layer.in_features} to width {layer.out_features}.',
                    ops=ops,
                ))
            elif isinstance(layer, nn.LayerNorm):
                ops = _scale_ops(_layer_norm_ops(layer.normalized_shape[0]), stage_multiplier)
                entries.append(_finalize_entry(
                    name=f'{branch_name}_layer_norm_{linear_index:02d}',
                    repeat=repeat,
                    why=f'{branch_name.title()} branch LayerNorm keeps hidden width {layer.normalized_shape[0]} unchanged.',
                    ops=ops,
                ))
            elif isinstance(layer, nn.ReLU) and hidden_width is not None:
                ops = _scale_ops(_relu_ops(hidden_width), stage_multiplier)
                entries.append(_finalize_entry(
                    name=f'{branch_name}_relu_{linear_index:02d}',
                    repeat=repeat,
                    why=f'{branch_name.title()} branch ReLU keeps hidden width {hidden_width} unchanged.',
                    ops=ops,
                ))

    entries.append(_finalize_entry(
        name='residual_correction',
        repeat=f'per stage ({model.num_stages} stage executions)',
        why='Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate.',
        ops=_scale_ops(_residual_ops(model.seq_len, model.num_ports), model.num_stages),
    ))
    return entries


def _separator2_entries(model: torch.nn.Module, model_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = _unwrap_model(model)
    example_mlp = model.port_mlps[0] if model.share_weights_across_stages else model.port_mlps[0][0]
    repeat = f"per port, per stage ({model.num_ports} ports x {model.num_stages} stage executions)"
    stage_multiplier = model.num_ports * model.num_stages
    entries: List[Dict[str, Any]] = [
        _finalize_entry(
            name='input_normalize_restore',
            repeat='once per sample',
            why='Per-sample RMS normalization and output rescaling are applied around the network when normalize_energy=true.',
            ops=_normalization_ops(model.seq_len, model.num_ports, model.normalize_energy),
        )
    ]

    for index, layer in enumerate(example_mlp.layers, 1):
        if not isinstance(layer, ComplexLinearReal):
            continue
        ops = _scale_ops(_complex_linear_ops(layer.in_features, layer.out_features), stage_multiplier)
        entries.append(_finalize_entry(
            name=f'complex_linear_{index:02d}',
            repeat=repeat,
            why=f'Complex affine block maps complex width {layer.in_features} to complex width {layer.out_features}, stored as real-stacked width {2 * layer.out_features}.',
            ops=ops,
        ))
        if index < len(example_mlp.layers):
            activation_ops = _scale_ops(_complex_activation_ops(layer.out_features, example_mlp.activation_type), stage_multiplier)
            entries.append(_finalize_entry(
                name=f'complex_activation_{index:02d}',
                repeat=repeat,
                why=f'Configured complex activation `{example_mlp.activation_type}` is applied after hidden complex block {index}.',
                ops=activation_ops,
            ))

    entries.append(_finalize_entry(
        name='residual_correction',
        repeat=f'per stage ({model.num_stages} stage executions)',
        why='Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate.',
        ops=_scale_ops(_residual_ops(model.seq_len, model.num_ports), model.num_stages),
    ))
    return entries


def _separator3_entries(model: torch.nn.Module, model_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    model = _unwrap_model(model)
    expanded_dim = model.expanded_dim
    entries: List[Dict[str, Any]] = [
        _finalize_entry(
            name='input_normalize_restore',
            repeat='once per sample',
            why='Per-sample RMS normalization and output rescaling are applied around the network when normalize_energy=true.',
            ops=_normalization_ops(model.seq_len, model.num_ports, model.normalize_energy),
        ),
    ]
    for stage_idx, stage in enumerate(model.stages, start=1):
        linear_layers = [layer for layer in stage.network if isinstance(layer, nn.Linear)]
        stage_hidden_dim = model.stage_hidden_dims[stage_idx - 1]
        stage_input_dim = model.input_dim if stage_idx == 1 else expanded_dim
        entries.append(_finalize_entry(
            name=f'stage_{stage_idx:02d}_hidden_linear_01',
            repeat='once per sample',
            why=f'Stage {stage_idx} first affine layer maps width {stage_input_dim} to hidden width {stage_hidden_dim}.',
            ops=_linear_ops(stage_input_dim, stage_hidden_dim),
        ))
        entries.append(_finalize_entry(
            name=f'stage_{stage_idx:02d}_hidden_relu_01',
            repeat='once per sample',
            why=f'Stage {stage_idx} applies ReLU after the first hidden affine layer.',
            ops=_relu_ops(stage_hidden_dim),
        ))
        for layer_idx in range(2, len(linear_layers)):
            entries.append(_finalize_entry(
                name=f'stage_{stage_idx:02d}_hidden_linear_{layer_idx:02d}',
                repeat='once per sample',
                why=f'Stage {stage_idx} additional hidden affine layer keeps width {stage_hidden_dim}.',
                ops=_linear_ops(stage_hidden_dim, stage_hidden_dim),
            ))
            entries.append(_finalize_entry(
                name=f'stage_{stage_idx:02d}_hidden_relu_{layer_idx:02d}',
                repeat='once per sample',
                why=f'Stage {stage_idx} applies ReLU after hidden affine layer {layer_idx}.',
                ops=_relu_ops(stage_hidden_dim),
            ))
        entries.append(_finalize_entry(
            name=f'stage_{stage_idx:02d}_joint_output',
            repeat='once per sample',
            why=f'Stage {stage_idx} output affine layer maps hidden width {stage_hidden_dim} to expanded width {expanded_dim}.',
            ops=_linear_ops(stage_hidden_dim, expanded_dim),
        ))
        entries.append(_finalize_entry(
            name=f'stage_{stage_idx:02d}_residual_correction',
            repeat='once per sample',
            why=f'Stage {stage_idx} output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask.',
            ops=_residual_ops(model.seq_len, model.num_ports),
        ))
    return entries


def generate_model_complexity_spec(
    model: torch.nn.Module,
    model_spec: Mapping[str, Any],
    component_specs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model = _unwrap_model(model)
    model_spec = dict(model_spec or {})
    model_type = model_spec['model_type']
    input_width = int(model_spec['seq_len']) * 2
    output_shape = _shape(-1, int(model_spec['num_ports']), input_width)

    if model_type == 'full_mlp':
        entries = _full_mlp_entries(model, model_spec)
    elif model_type == 'separator1':
        entries = _separator1_entries(model, model_spec)
    elif model_type == 'separator2':
        entries = _separator2_entries(model, model_spec)
    elif model_type == 'separator3':
        entries = _separator3_entries(model, model_spec)
    else:
        raise ValueError(f'Unsupported model_type for complexity description: {model_type}')

    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    summary = _summarize(entries, trainable_parameters=trainable_parameters)
    return {
        'format': 'srs_ai_model_complexity_v1',
        'model_type': model_type,
        'input_shape': _shape(-1, input_width),
        'input_shape_string': _shape_string(_shape(-1, input_width)),
        'output_shape': output_shape,
        'output_shape_string': _shape_string(output_shape),
        'counting_assumptions': [
            'All counts are estimated for one forward pass of one sample; multiply by runtime batch size N for a first-order batch estimate.',
            'MACs and multiply/add counts are counted from affine layers using scalar CPU-style arithmetic.',
            'Bias accumulation is counted as additions inside the addition totals.',
            'Normalization, activations, and residual corrections are included as estimated scalar ops where relevant.',
            'Tensor reshapes, concatenations, indexing, and memory traffic are not counted as arithmetic FLOPs.',
            'Latency depends on implementation and hardware; these counts are for fast complexity screening only.',
        ],
        'summary': summary,
        'entries': entries,
        'model_spec': model_spec,
        'component_specs': dict(component_specs or {}),
    }


def render_model_complexity_markdown(complexity_spec: Mapping[str, Any]) -> str:
    summary = complexity_spec['summary']
    lines = [
        '# Model Complexity',
        '',
        f"- Model type: `{complexity_spec['model_type']}`",
        f"- Input shape: `{complexity_spec['input_shape_string']}`",
        f"- Output shape: `{complexity_spec['output_shape_string']}`",
        f"- Trainable parameters: `{summary['trainable_parameters_string']}`",
        f"- Parameter memory (FP32): `{summary['parameter_memory_bytes_fp32_string']}`",
        f"- MACs / sample: `{summary['macs_per_sample_string']}`",
        f"- Multiplications / sample: `{summary['multiplications_per_sample_string']}`",
        f"- Additions / sample: `{summary['additions_per_sample_string']}`",
        f"- Other scalar ops / sample: `{summary['other_scalar_ops_per_sample_estimate_string']}`",
        f"- FLOPs / sample estimate: `{summary['flops_per_sample_estimate_string']}`",
        f"- Batch scaling: {summary['batch_scaling_rule']}",
        '',
        '## Counting Assumptions',
        '',
    ]

    for item in complexity_spec['counting_assumptions']:
        lines.append(f'- {item}')

    lines.extend([
        '',
        '## Operator Breakdown',
        '',
        '| Block | Repeat | Parameters | MACs | Multiplies | Adds | Other ops | FLOPs est. | Why |',
        '|---|---|---|---|---|---|---|---|---|',
    ])

    for entry in complexity_spec['entries']:
        lines.append(
            f"| `{entry['name']}` | `{entry['repeat']}` | `{entry['parameters_string']}` | `{entry['macs_string']}` | `{entry['multiplications_string']}` | `{entry['additions_string']}` | `{entry['other_ops_string']}` | `{entry['flops_estimate_string']}` | {entry['why']} |"
        )

    lines.extend([
        '',
        '## Raw Summary',
        '',
        '```json',
        json.dumps(summary, indent=2, ensure_ascii=False),
        '```',
    ])
    return '\n'.join(lines) + '\n'


def save_model_complexity_artifacts(
    output_dir: Path | str,
    model: torch.nn.Module,
    model_spec: Mapping[str, Any],
    component_specs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complexity_spec = generate_model_complexity_spec(model=model, model_spec=model_spec, component_specs=component_specs)
    json_path = output_dir / 'model_complexity.json'
    markdown_path = output_dir / 'MODEL_COMPLEXITY.md'

    with open(json_path, 'w', encoding='utf-8') as output_file:
        json.dump(complexity_spec, output_file, indent=2, ensure_ascii=False)

    with open(markdown_path, 'w', encoding='utf-8') as output_file:
        output_file.write(render_model_complexity_markdown(complexity_spec))

    return {
        'complexity_spec': complexity_spec,
        'json_path': str(json_path),
        'markdown_path': str(markdown_path),
    }


__all__ = [
    'generate_model_complexity_spec',
    'render_model_complexity_markdown',
    'save_model_complexity_artifacts',
]