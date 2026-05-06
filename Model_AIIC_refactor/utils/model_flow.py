"""Human-readable model flow descriptions for trained artifacts and exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _shape(*dims: int) -> List[int]:
    return [int(dim) for dim in dims]


def _shape_string(shape: List[int]) -> str:
    return '[' + ', '.join(str(dim) for dim in shape) + ']'


def _seq_len(model_spec: Mapping[str, Any]) -> int:
    return int(model_spec['seq_len'])


def _num_ports(model_spec: Mapping[str, Any]) -> int:
    return int(model_spec['num_ports'])


def _linear_param_count(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def _complex_linear_param_count(in_features: int, out_features: int) -> int:
    return 2 * in_features * out_features + 2 * out_features


def _format_param_count(value: int) -> str:
    return f'{value:,}'


def _add_node(
    nodes: List[Dict[str, Any]],
    name: str,
    shape: List[int],
    description: str,
    repeat: Optional[str] = None,
    why: Optional[str] = None,
    param_count_per_occurrence: int = 0,
    effective_total_param_count: int = 0,
) -> None:
    node = {
        'name': name,
        'shape': list(shape),
        'shape_string': _shape_string(shape),
        'description': description,
        'why': why or description,
        'param_count_per_occurrence': int(param_count_per_occurrence),
        'param_count_per_occurrence_string': _format_param_count(int(param_count_per_occurrence)),
        'effective_total_param_count': int(effective_total_param_count),
        'effective_total_param_count_string': _format_param_count(int(effective_total_param_count)),
    }
    if repeat:
        node['repeat'] = repeat
    nodes.append(node)


def _full_mlp_flow(model_spec: Mapping[str, Any]) -> Dict[str, Any]:
    seq_len = _seq_len(model_spec)
    num_ports = _num_ports(model_spec)
    hidden_dim = int(model_spec.get('hidden_dim', 128))
    mlp_depth = int(model_spec.get('mlp_depth', 3))
    input_dim = seq_len * 2
    output_dim = input_dim * num_ports
    total_trainable_params = int(model_spec.get('num_params', 0))

    nodes: List[Dict[str, Any]] = []
    _add_node(nodes, 'mixed_signal', _shape(-1, input_dim), 'Real-stacked mixed input [real, imag].', why='Task output is one complex sequence flattened into real and imaginary blocks.')
    _add_node(nodes, 'normalized_input', _shape(-1, input_dim), 'Optional per-sample RMS normalization inside the model.', why='Normalization rescales values but does not change tensor rank or width.')

    first_hidden_params = _linear_param_count(input_dim, hidden_dim)
    _add_node(nodes, 'joint_hidden_1', _shape(-1, hidden_dim), f'First hidden linear layer with hidden_dim={hidden_dim}.', why=f'The first affine layer maps the input width {input_dim} into the configured hidden width {hidden_dim}.', param_count_per_occurrence=first_hidden_params, effective_total_param_count=first_hidden_params)
    _add_node(nodes, 'joint_hidden_1_relu', _shape(-1, hidden_dim), 'ReLU activation after the first hidden layer.', why='ReLU is elementwise, so it keeps the hidden shape unchanged.')
    for layer_idx in range(2, mlp_depth):
        hidden_params = _linear_param_count(hidden_dim, hidden_dim)
        _add_node(nodes, f'joint_hidden_{layer_idx}', _shape(-1, hidden_dim), f'Additional hidden linear layer {layer_idx}.', why=f'This affine layer keeps the same hidden width {hidden_dim} while adding capacity.', param_count_per_occurrence=hidden_params, effective_total_param_count=hidden_params)
        _add_node(nodes, f'joint_hidden_{layer_idx}_relu', _shape(-1, hidden_dim), f'ReLU activation after hidden layer {layer_idx}.', why='ReLU is elementwise, so the hidden width stays the same.')
    final_params = _linear_param_count(hidden_dim, output_dim)
    _add_node(nodes, 'joint_linear_output', _shape(-1, output_dim), 'Final linear layer predicts all ports jointly.', why=f'The output affine layer expands hidden width {hidden_dim} to flat joint output width {output_dim}.', param_count_per_occurrence=final_params, effective_total_param_count=final_params)

    _add_node(nodes, 'reshaped_channels', _shape(-1, num_ports, input_dim), 'Flat output reshaped into one real-stacked channel tensor per port.', why=f'The flat width {output_dim} is partitioned into {num_ports} port blocks of width {input_dim}.')
    _add_node(nodes, 'separated_channels', _shape(-1, num_ports, input_dim), 'Optional output RMS restoration to the original input scale.', why='Rescaling restores amplitude but keeps the separated tensor shape unchanged.')

    return {
        'family': 'full_mlp',
        'loop_summary': 'One joint MLP processes the mixed signal once and reshapes the output into all ports.',
        'total_trainable_params': total_trainable_params,
        'total_trainable_params_string': _format_param_count(total_trainable_params),
        'nodes': nodes,
    }


def _separator1_flow(model_spec: Mapping[str, Any]) -> Dict[str, Any]:
    seq_len = _seq_len(model_spec)
    num_ports = _num_ports(model_spec)
    hidden_dim = int(model_spec.get('hidden_dim', 64))
    num_stages = int(model_spec.get('num_stages', 3))
    mlp_depth = int(model_spec.get('mlp_depth', 3))
    input_dim = seq_len * 2
    share_weights = bool(model_spec.get('share_weights_across_stages', False))
    use_hidden_layer_norm = bool(model_spec.get('use_hidden_layer_norm', False))
    stage_instantiations = 1 if share_weights else num_stages
    branch_instance_multiplier = num_ports * stage_instantiations
    total_trainable_params = int(model_spec.get('num_params', 0))

    nodes: List[Dict[str, Any]] = []
    repeat_text = f'per port, per stage ({num_stages} stages total{", stage weights shared" if share_weights else ""})'
    _add_node(nodes, 'mixed_signal', _shape(-1, input_dim), 'Real-stacked mixed input [real, imag].', why='The task provides one mixed complex sequence flattened into real and imaginary blocks.')
    _add_node(nodes, 'normalized_input', _shape(-1, input_dim), 'Optional per-sample RMS normalization inside the model.', why='Normalization rescales values but does not change tensor width.')
    _add_node(nodes, 'replicated_port_features', _shape(-1, num_ports, input_dim), f'Input copied to all {num_ports} ports before refinement.', why=f'The separator starts each port estimate from the same mixed input, so a port axis of size {num_ports} is introduced.')
    _add_node(nodes, 'port_stage_input', _shape(-1, input_dim), 'One port slice entering one refinement stage.', repeat=repeat_text, why=f'Each port-stage block consumes one real-stacked sequence of width {input_dim}.')

    input_hidden_params = _linear_param_count(input_dim, hidden_dim)
    hidden_hidden_params = _linear_param_count(hidden_dim, hidden_dim)
    output_params = _linear_param_count(hidden_dim, seq_len)
    layer_norm_params = 2 * hidden_dim if use_hidden_layer_norm else 0
    for hidden_idx in range(1, mlp_depth):
        hidden_params = input_hidden_params if hidden_idx == 1 else hidden_hidden_params
        hidden_reason = (
            f'The first branch affine layer maps width {input_dim} to hidden width {hidden_dim}.'
            if hidden_idx == 1 else
            f'This branch affine layer keeps the hidden width at {hidden_dim}.'
        )
        _add_node(nodes, f'real_branch_hidden_{hidden_idx}', _shape(-1, hidden_dim), f'Real branch hidden layer {hidden_idx}.', repeat=repeat_text, why=hidden_reason, param_count_per_occurrence=hidden_params, effective_total_param_count=hidden_params * branch_instance_multiplier)
        if use_hidden_layer_norm:
            _add_node(nodes, f'real_branch_hidden_{hidden_idx}_layer_norm', _shape(-1, hidden_dim), f'Real branch LayerNorm after hidden layer {hidden_idx}.', repeat=repeat_text, why='LayerNorm keeps the hidden width unchanged and adds one learned scale and bias per hidden feature.', param_count_per_occurrence=layer_norm_params, effective_total_param_count=layer_norm_params * branch_instance_multiplier)
        _add_node(nodes, f'imag_branch_hidden_{hidden_idx}', _shape(-1, hidden_dim), f'Imag branch hidden layer {hidden_idx}.', repeat=repeat_text, why=hidden_reason, param_count_per_occurrence=hidden_params, effective_total_param_count=hidden_params * branch_instance_multiplier)
        if use_hidden_layer_norm:
            _add_node(nodes, f'imag_branch_hidden_{hidden_idx}_layer_norm', _shape(-1, hidden_dim), f'Imag branch LayerNorm after hidden layer {hidden_idx}.', repeat=repeat_text, why='LayerNorm keeps the hidden width unchanged and adds one learned scale and bias per hidden feature.', param_count_per_occurrence=layer_norm_params, effective_total_param_count=layer_norm_params * branch_instance_multiplier)
    _add_node(nodes, 'real_branch_output', _shape(-1, seq_len), 'Real branch final linear output.', repeat=repeat_text, why=f'The real branch final affine layer reduces hidden width {hidden_dim} to one real channel width {seq_len}.', param_count_per_occurrence=output_params, effective_total_param_count=output_params * branch_instance_multiplier)
    _add_node(nodes, 'imag_branch_output', _shape(-1, seq_len), 'Imag branch final linear output.', repeat=repeat_text, why=f'The imaginary branch final affine layer reduces hidden width {hidden_dim} to one imaginary channel width {seq_len}.', param_count_per_occurrence=output_params, effective_total_param_count=output_params * branch_instance_multiplier)

    _add_node(nodes, 'port_output', _shape(-1, input_dim), 'Real and imaginary branch outputs concatenated back to one port tensor.', repeat=repeat_text, why=f'Concatenating one real width-{seq_len} output and one imag width-{seq_len} output reconstructs one width-{input_dim} port tensor.')
    _add_node(nodes, 'stacked_stage_output', _shape(-1, num_ports, input_dim), 'All port outputs stacked for the current stage.', repeat=f'per stage ({num_stages} stages total)', why=f'Stacking all {num_ports} ports reintroduces the port axis while keeping each port width at {input_dim}.')
    _add_node(nodes, 'residual_corrected_output', _shape(-1, num_ports, input_dim), 'Residual correction enforces that the separated outputs sum back to the mixed input.', repeat=f'per stage ({num_stages} stages total)', why='Residual correction adds the same mixed-signal residual back to every port estimate, so the shape is unchanged.')
    _add_node(nodes, 'separated_channels', _shape(-1, num_ports, input_dim), 'Optional output RMS restoration to the original input scale.', why='Rescaling restores amplitude but keeps the separated tensor shape unchanged.')

    return {
        'family': 'separator1',
        'loop_summary': f'{num_stages} refinement stages; each stage runs one real branch and one imag branch per port, then applies residual correction.',
        'weight_sharing_across_stages': share_weights,
        'total_trainable_params': total_trainable_params,
        'total_trainable_params_string': _format_param_count(total_trainable_params),
        'nodes': nodes,
    }


def _separator2_flow(model_spec: Mapping[str, Any]) -> Dict[str, Any]:
    seq_len = _seq_len(model_spec)
    num_ports = _num_ports(model_spec)
    hidden_dim = int(model_spec.get('hidden_dim', 64))
    num_stages = int(model_spec.get('num_stages', 3))
    mlp_depth = int(model_spec.get('mlp_depth', 3))
    input_dim = seq_len * 2
    complex_hidden_dim = hidden_dim * 2
    share_weights = bool(model_spec.get('share_weights_across_stages', False))
    stage_instantiations = 1 if share_weights else num_stages
    block_instance_multiplier = num_ports * stage_instantiations
    total_trainable_params = int(model_spec.get('num_params', 0))

    nodes: List[Dict[str, Any]] = []
    repeat_text = f'per port, per stage ({num_stages} stages total{", stage weights shared" if share_weights else ""})'
    _add_node(nodes, 'mixed_signal', _shape(-1, input_dim), 'Real-stacked mixed input [real, imag].', why='The task provides one mixed complex sequence flattened into real and imaginary blocks.')
    _add_node(nodes, 'normalized_input', _shape(-1, input_dim), 'Optional per-sample RMS normalization inside the model.', why='Normalization rescales values but does not change tensor width.')
    _add_node(nodes, 'replicated_port_features', _shape(-1, num_ports, input_dim), f'Input copied to all {num_ports} ports before refinement.', why=f'The separator starts each port estimate from the same mixed input, so a port axis of size {num_ports} is introduced.')
    _add_node(nodes, 'port_stage_input', _shape(-1, input_dim), 'One port slice entering one refinement stage.', repeat=repeat_text, why=f'Each complex block consumes one real-stacked width-{input_dim} sequence.')

    if mlp_depth == 2:
        output_params = _complex_linear_param_count(seq_len, seq_len)
        _add_node(nodes, 'complex_output_block', _shape(-1, input_dim), 'Direct complex-linear output in real-stacked form.', repeat=repeat_text, why=f'One complex affine block maps complex width {seq_len} back to complex width {seq_len}, stored as real-stacked width {input_dim}.', param_count_per_occurrence=output_params, effective_total_param_count=output_params * block_instance_multiplier)
    else:
        input_hidden_params = _complex_linear_param_count(seq_len, hidden_dim)
        hidden_hidden_params = _complex_linear_param_count(hidden_dim, hidden_dim)
        output_params = _complex_linear_param_count(hidden_dim, seq_len)
        for hidden_idx in range(1, mlp_depth - 1):
            hidden_params = input_hidden_params if hidden_idx == 1 else hidden_hidden_params
            hidden_reason = (
                f'The first complex affine block maps complex width {seq_len} to hidden complex width {hidden_dim}, stored as real-stacked width {complex_hidden_dim}.'
                if hidden_idx == 1 else
                f'This complex affine block keeps the hidden complex width {hidden_dim}, so the real-stacked width stays {complex_hidden_dim}.'
            )
            _add_node(nodes, f'complex_hidden_{hidden_idx}', _shape(-1, complex_hidden_dim), f'Complex hidden affine block {hidden_idx} stored as one real-stacked tensor [real_hidden, imag_hidden].', repeat=repeat_text, why=hidden_reason, param_count_per_occurrence=hidden_params, effective_total_param_count=hidden_params * block_instance_multiplier)
            _add_node(nodes, f'complex_hidden_{hidden_idx}_activation', _shape(-1, complex_hidden_dim), f'Configured complex activation after hidden block {hidden_idx}.', repeat=repeat_text, why='The configured complex activation is elementwise or blockwise, so it keeps the same real-stacked hidden width.')
        _add_node(nodes, 'complex_output_block', _shape(-1, input_dim), 'Final complex output affine block in real-stacked form.', repeat=repeat_text, why=f'The output complex affine block reduces hidden complex width {hidden_dim} back to complex sequence width {seq_len}, stored as real-stacked width {input_dim}.', param_count_per_occurrence=output_params, effective_total_param_count=output_params * block_instance_multiplier)

    _add_node(nodes, 'stacked_stage_output', _shape(-1, num_ports, input_dim), 'All port outputs stacked for the current stage.', repeat=f'per stage ({num_stages} stages total)', why=f'Stacking all {num_ports} ports reintroduces the port axis while keeping each port width at {input_dim}.')
    _add_node(nodes, 'residual_corrected_output', _shape(-1, num_ports, input_dim), 'Residual correction enforces that the separated outputs sum back to the mixed input.', repeat=f'per stage ({num_stages} stages total)', why='Residual correction adds the same mixed-signal residual back to every port estimate, so the shape is unchanged.')
    _add_node(nodes, 'separated_channels', _shape(-1, num_ports, input_dim), 'Optional output RMS restoration to the original input scale.', why='Rescaling restores amplitude but keeps the separated tensor shape unchanged.')

    return {
        'family': 'separator2',
        'loop_summary': f'{num_stages} refinement stages; each stage runs one complex-valued MLP per port in real-stacked form, then applies residual correction.',
        'weight_sharing_across_stages': share_weights,
        'total_trainable_params': total_trainable_params,
        'total_trainable_params_string': _format_param_count(total_trainable_params),
        'nodes': nodes,
    }


def _separator3_flow(model_spec: Mapping[str, Any]) -> Dict[str, Any]:
    seq_len = _seq_len(model_spec)
    num_ports = _num_ports(model_spec)
    input_dim = seq_len * 2
    expanded_dim = num_ports * input_dim
    num_stages = int(model_spec.get('num_stages', 2))
    mlp_depth = int(model_spec.get('mlp_depth', 2))
    stage_hidden_dims = model_spec.get('stage_hidden_dims')
    if stage_hidden_dims is None:
        stage_hidden_dims = [int(model_spec.get('hidden_dim', 128))] * num_stages
    else:
        stage_hidden_dims = [int(value) for value in stage_hidden_dims]
    total_trainable_params = int(model_spec.get('num_params', 0))

    nodes: List[Dict[str, Any]] = []
    _add_node(nodes, 'mixed_signal', _shape(-1, input_dim), 'Real-stacked mixed input [real, imag].', why='The task provides one mixed complex sequence flattened into real and imaginary blocks.')
    _add_node(nodes, 'normalized_input', _shape(-1, input_dim), 'Optional per-sample RMS normalization inside the model.', why='Normalization rescales values but does not change tensor width.')
    for stage_idx in range(num_stages):
        stage_num = stage_idx + 1
        hidden_dim = stage_hidden_dims[stage_idx]
        stage_input_dim = input_dim if stage_idx == 0 else expanded_dim
        repeat = 'once' if stage_idx == 0 else f'stage {stage_num}'
        _add_node(nodes, f'stage_{stage_num}_input', _shape(-1, stage_input_dim), f'Input to stage {stage_num}.', repeat=repeat, why=(
            f'The first stage consumes the mixed signal width {input_dim} directly.' if stage_idx == 0 else
            f'Stage {stage_num} consumes the flattened joint output of the previous stage, so the width is {expanded_dim}.'
        ))
        first_hidden_params = _linear_param_count(stage_input_dim, hidden_dim)
        _add_node(nodes, f'stage_{stage_num}_hidden_1', _shape(-1, hidden_dim), f'First hidden linear layer of stage {stage_num}.', repeat=repeat, why=f'The first affine layer of stage {stage_num} maps width {stage_input_dim} to hidden width {hidden_dim}.', param_count_per_occurrence=first_hidden_params, effective_total_param_count=first_hidden_params)
        _add_node(nodes, f'stage_{stage_num}_hidden_1_relu', _shape(-1, hidden_dim), f'ReLU after the first hidden layer of stage {stage_num}.', repeat=repeat, why='ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged.')
        for layer_idx in range(2, mlp_depth):
            hidden_params = _linear_param_count(hidden_dim, hidden_dim)
            _add_node(nodes, f'stage_{stage_num}_hidden_{layer_idx}', _shape(-1, hidden_dim), f'Additional hidden linear layer {layer_idx} of stage {stage_num}.', repeat=repeat, why=f'This affine layer keeps hidden width {hidden_dim} inside stage {stage_num}.', param_count_per_occurrence=hidden_params, effective_total_param_count=hidden_params)
            _add_node(nodes, f'stage_{stage_num}_hidden_{layer_idx}_relu', _shape(-1, hidden_dim), f'ReLU after hidden layer {layer_idx} of stage {stage_num}.', repeat=repeat, why='ReLU is elementwise, so the hidden width stays the same.')
        output_params = _linear_param_count(hidden_dim, expanded_dim)
        _add_node(nodes, f'stage_{stage_num}_joint_output', _shape(-1, expanded_dim), f'Joint stage output before residual correction for stage {stage_num}.', repeat=repeat, why=f'The final affine layer of stage {stage_num} maps hidden width {hidden_dim} to expanded width {expanded_dim} = num_ports * (2 * seq_len).', param_count_per_occurrence=output_params, effective_total_param_count=output_params)
        _add_node(nodes, f'stage_{stage_num}_port_features', _shape(-1, num_ports, input_dim), f'Stage {stage_num} output reshaped into one real-stacked port tensor per port.', repeat=repeat, why=f'The expanded width {expanded_dim} is partitioned into {num_ports} port blocks of width {input_dim}.')
        _add_node(nodes, f'stage_{stage_num}_residual_corrected', _shape(-1, num_ports, input_dim), f'Learned-dense residual correction after stage {stage_num}.', repeat=repeat, why='Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage.')
        if stage_idx < num_stages - 1:
            _add_node(nodes, f'stage_{stage_num}_flattened_output', _shape(-1, expanded_dim), f'Flattened stage {stage_num} output passed to the next stage.', repeat=repeat, why='The per-port output is flattened back to one joint vector so the next stage can process all ports jointly again.')
    _add_node(nodes, 'separated_channels', _shape(-1, num_ports, input_dim), 'Optional output RMS restoration to the original input scale.', why='Rescaling restores amplitude but keeps the separated tensor shape unchanged.')

    return {
        'family': 'separator3',
        'loop_summary': f'{num_stages} joint refinement stages; stage 1 maps {input_dim} to {expanded_dim}, later stages map {expanded_dim} to {expanded_dim}, and every stage ends with learned-dense residual correction.',
        'total_trainable_params': total_trainable_params,
        'total_trainable_params_string': _format_param_count(total_trainable_params),
        'nodes': nodes,
    }


def generate_model_flow_spec(model_spec: Mapping[str, Any], component_specs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Generate a model-family-specific shape flow description with dynamic batch dimension -1."""
    model_spec = dict(model_spec or {})
    seq_len = _seq_len(model_spec)
    num_ports = _num_ports(model_spec)
    input_dim = seq_len * 2
    model_type = model_spec['model_type']

    if model_type == 'full_mlp':
        details = _full_mlp_flow(model_spec)
    elif model_type == 'separator1':
        details = _separator1_flow(model_spec)
    elif model_type == 'separator2':
        details = _separator2_flow(model_spec)
    elif model_type == 'separator3':
        details = _separator3_flow(model_spec)
    else:
        raise ValueError(f'Unsupported model_type for flow description: {model_type}')

    return {
        'format': 'srs_ai_model_flow_v1',
        'model_type': model_type,
        'input_layout': 'N x (2*seq_len) real-stacked float32 = [real_part, imag_part]',
        'output_layout': 'N x num_ports x (2*seq_len) real-stacked float32',
        'dynamic_dimension_convention': 'Dynamic dimensions are written as -1.',
        'input_shape': _shape(-1, input_dim),
        'output_shape': _shape(-1, num_ports, input_dim),
        'input_shape_string': _shape_string(_shape(-1, input_dim)),
        'output_shape_string': _shape_string(_shape(-1, num_ports, input_dim)),
        'model_spec': model_spec,
        'component_specs': dict(component_specs or {}),
        **details,
    }


def render_model_flow_markdown(flow_spec: Mapping[str, Any]) -> str:
    """Render the flow spec into a human-readable Markdown document."""
    lines = [
        '# Model Flow',
        '',
        f"- Model type: `{flow_spec['model_type']}`",
        f"- Input layout: `{flow_spec['input_layout']}`",
        f"- Input shape: `{flow_spec['input_shape_string']}`",
        f"- Output layout: `{flow_spec['output_layout']}`",
        f"- Output shape: `{flow_spec['output_shape_string']}`",
        f"- Total trainable parameters: `{flow_spec.get('total_trainable_params_string', '0')}`",
        f"- Dynamic dimensions: {flow_spec['dynamic_dimension_convention']}",
        '',
        '## Flow Summary',
        '',
        flow_spec['loop_summary'],
        '',
        '## Node Shapes',
        '',
        '| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |',
        '|---|---|---|---|---|---|',
    ]

    for node in flow_spec['nodes']:
        lines.append(
            f"| `{node['name']}` | `{node['shape_string']}` | `{node.get('repeat', '-')}` | `{node['param_count_per_occurrence_string']}` | `{node['effective_total_param_count_string']}` | {node['why']} |"
        )

    lines.extend([
        '',
        '## Model Spec',
        '',
        '```json',
        json.dumps(flow_spec['model_spec'], indent=2, ensure_ascii=False),
        '```',
    ])
    return '\n'.join(lines) + '\n'


def save_model_flow_artifacts(
    output_dir: Path | str,
    model_spec: Mapping[str, Any],
    component_specs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write model_flow.json and MODEL_FLOW.md to one artifact directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_spec = generate_model_flow_spec(model_spec=model_spec, component_specs=component_specs)
    json_path = output_dir / 'model_flow.json'
    markdown_path = output_dir / 'MODEL_FLOW.md'

    with open(json_path, 'w', encoding='utf-8') as output_file:
        json.dump(flow_spec, output_file, indent=2, ensure_ascii=False)

    with open(markdown_path, 'w', encoding='utf-8') as output_file:
        output_file.write(render_model_flow_markdown(flow_spec))

    return {
        'flow_spec': flow_spec,
        'json_path': str(json_path),
        'markdown_path': str(markdown_path),
    }


__all__ = [
    'generate_model_flow_spec',
    'render_model_flow_markdown',
    'save_model_flow_artifacts',
]