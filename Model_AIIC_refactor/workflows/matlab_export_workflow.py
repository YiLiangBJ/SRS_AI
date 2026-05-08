"""Programmatic Matlab bundle export workflow."""

import json
from datetime import datetime
from pathlib import Path
import shutil
import textwrap
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat

from utils import build_dummy_input, load_trained_model_from_checkpoint, load_trained_model_from_run, resolve_run_selection, save_model_complexity_artifacts, save_model_flow_artifacts


MATLAB_COMPONENT_RUNTIME_FILES = [
    'import_refactor_matlab_bundle.m',
    'predict_refactor_matlab_bundle.m',
    'describe_refactor_model_io.m',
    'resolve_refactor_export_dir.m',
    'prepare_refactor_input.m',
    'import_refactor_model.m',
    'predict_refactor_model.m',
]


def _matlab_runtime_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'matlab'


def _sanitize_matlab_identifier(raw_name: str) -> str:
    cleaned = ''.join(character if character.isalnum() else '_' for character in raw_name)
    cleaned = cleaned.strip('_') or 'model_component'
    if not cleaned[0].isalpha():
        cleaned = f'model_{cleaned}'
    return cleaned


def _short_model_tag(manifest: Dict[str, object]) -> str:
    model_spec = manifest.get('model_spec', {}) if isinstance(manifest, dict) else {}
    model_type = str(model_spec.get('model_type', 'model'))
    prefix_map = {
        'separator1': 'sep1',
        'separator2': 'sep2',
        'separator3': 'sep3',
        'full_mlp': 'fmlp',
    }
    prefix = prefix_map.get(model_type, _sanitize_matlab_identifier(model_type).lower())
    parts = [prefix]
    if 'hidden_dim' in model_spec:
        parts.append(f"hd{int(model_spec['hidden_dim'])}")
    if 'mlp_depth' in model_spec:
        parts.append(f"d{int(model_spec['mlp_depth'])}")
    if 'num_stages' in model_spec:
        parts.append(f"s{int(model_spec['num_stages'])}")
    return '_'.join(parts)


def _write_component_package_files(component_dir: Path, manifest: Dict[str, object]) -> Dict[str, str]:
    run_name = str(manifest['run_name'])
    short_tag = _short_model_tag(manifest)
    demo_dir = component_dir / 'demo'
    demo_dir.mkdir(parents=True, exist_ok=True)

    load_script = textwrap.dedent(
        """
        function component = load_srs_ai_matlab_component(componentDir)
        %LOAD_SRS_AI_MATLAB_COMPONENT Load the colocated SRS AI Matlab bundle component.
        if nargin < 1 || isempty(componentDir)
            componentDir = fileparts(mfilename('fullpath'));
        end
        component = import_refactor_matlab_bundle(componentDir);
        end
        """
    ).lstrip()

    init_script = textwrap.dedent(
        f"""
        function state = init_model(componentDir)
        %INIT_MODEL One-time initialization for deployed Matlab inference.
        if nargin < 1 || isempty(componentDir)
            componentDir = fileparts(mfilename('fullpath'));
        end
        state = load_srs_ai_matlab_component(componentDir);
        state.component_dir = string(componentDir);
        state.model_name = "{short_tag}";
        state.seq_len = double(state.manifest.model_spec.seq_len);
        state.input_width = state.seq_len * 2;
        state.num_ports = double(state.manifest.model_spec.num_ports);
        state.output_width = state.input_width;
        end
        """
    ).lstrip()

    predict_script = textwrap.dedent(
        f"""
        function [outputData, debug, state] = predict_srs_ai_matlab_component(inputData, stateOrDir)
        %PREDICT_SRS_AI_MATLAB_COMPONENT Lower-level inference entrypoint for {run_name}.
        %
        % Usage:
        %   outputData = predict_srs_ai_matlab_component(inputData, state)
        %   outputData = predict_srs_ai_matlab_component(inputData, componentDir)
        %
        % Input shape:
        %   N x 24 real-stacked float32 = [real_part, imag_part]
        % Output shape:
        %   N x 6 x 24 real-stacked float32
        if nargin < 2 || isempty(stateOrDir)
            state = init_model(fileparts(mfilename('fullpath')));
        elseif isstruct(stateOrDir)
            state = stateOrDir;
        else
            state = init_model(stateOrDir);
        end
        [outputData, debug] = predict_refactor_matlab_bundle(state, inputData);
        end
        """
    ).lstrip()

    predict_model_script = textwrap.dedent(
        """
        function [outputData, ports, debug] = predict_model(state, inputData)
        %PREDICT_MODEL Fast deployed inference using preinitialized state.
        [outputData, debug] = predict_refactor_matlab_bundle(state, single(inputData));
        if nargout >= 2
            ports = split_ports(outputData);
        end
        end
        """
    ).lstrip()

    split_script = textwrap.dedent(
        """
        function ports = split_ports(outputData)
        %SPLIT_PORTS Convert N x 6 x 24 output into a 1x6 cell array of N x 24 slices.
        validateattributes(outputData, {'numeric'}, {'3d'});
        numPorts = size(outputData, 2);
        ports = cell(1, numPorts);
        for portIdx = 1:numPorts
            ports{portIdx} = squeeze(outputData(:, portIdx, :));
            if size(outputData, 1) == 1
                ports{portIdx} = reshape(ports{portIdx}, 1, []);
            end
        end
        end
        """
    ).lstrip()

    demo_script = textwrap.dedent(
        """
        function [inputData, outputData, ports, debug, component] = demo_srs_ai_matlab_component(batchSize)
        %DEMO_SRS_AI_MATLAB_COMPONENT Quick self-test for the colocated component package.
        if nargin < 1 || isempty(batchSize)
            batchSize = 4;
        end
        component = init_model(fileparts(mfilename('fullpath')));
        inputData = prepare_refactor_input(component, batchSize, "bundle");
        [outputData, debug] = predict_srs_ai_matlab_component(inputData, component);
        ports = split_ports(outputData);
        disp("Component demo finished.");
        disp("  Input size: " + mat2str(size(inputData)));
        disp("  Output size: " + mat2str(size(outputData)));
        end
        """
    ).lstrip()

    model_predict_script = textwrap.dedent(
        f"""
        function [outputData, ports, debug] = predict_{short_tag}(state, inputData)
        %PREDICT_{short_tag.upper()} Short model-specific deployed inference entrypoint.
        [outputData, ports, debug] = predict_model(state, inputData);
        end
        """
    ).lstrip()

    model_init_script = textwrap.dedent(
        f"""
        function state = init_{short_tag}(componentDir)
        %INIT_{short_tag.upper()} Short model-specific initialization entrypoint.
        state = init_model(componentDir);
        end
        """
    ).lstrip()

    demo_quick_start = textwrap.dedent(
        """
        %% Quick start: one-time init, then inference
        componentDir = fileparts(fileparts(mfilename('fullpath')));
        state = init_model(componentDir);
        inputData = randn(8, 24, 'single');
        [outputData, ports, debug] = predict_model(state, inputData);
        disp(size(inputData));
        disp(size(outputData));
        disp(size(ports{1}));
        %#ok<NASGU>
        """
    ).lstrip()

    demo_step_by_step = textwrap.dedent(
        f"""
        %% Step 1: locate the component package root
        componentDir = fileparts(fileparts(mfilename('fullpath')));

        %% Step 2: one-time initialization
        state = init_model(componentDir);
        manifest = state.manifest;
        ioSpec = state.io_spec;
        disp(manifest.run_name);
        disp(ioSpec.input);
        disp(ioSpec.output);

        %% Step 3: inspect exported reference tensors
        sampleInput = single(state.weights.sample_input);
        referenceOutput = single(state.weights.reference_output);
        disp(size(sampleInput));
        disp(size(referenceOutput));

        %% Step 4: create your own dynamic-batch input
        batchSize = 4;
        inputData = prepare_refactor_input(state, batchSize, "bundle");
        disp(size(inputData));

        %% Step 5: run deployed inference with preloaded state
        [outputData, ports, debug] = predict_model(state, inputData);
        disp(size(outputData));
        disp(size(ports{{1}}));

        %% Step 6: verify the reference sample path once
        [referencePrediction, referencePorts, referenceDebug] = predict_model(state, sampleInput);
        maxAbsDiff = max(abs(referencePrediction(:) - referenceOutput(:)));
        disp("Max abs diff vs reference_output: " + string(maxAbsDiff));

        %% Step 7: short model-specific aliases
        state2 = init_{short_tag}(componentDir);
        [outputData2, ports2, debug2] = predict_{short_tag}(state2, inputData);
        disp(size(outputData2));
        %#ok<NASGU>
        """
    ).lstrip()

    demo_sim_platform = textwrap.dedent(
        """
        function [outputData, ports, state] = demo_sim_platform_loop(inputData, resetState)
        %DEMO_SIM_PLATFORM_LOOP Template for first-slot init and later-slot reuse.
        persistent cachedState
        if nargin < 2
            resetState = false;
        end
        if resetState
            cachedState = [];
        end
        if isempty(cachedState)
            componentDir = fileparts(fileparts(mfilename('fullpath')));
            cachedState = init_model(componentDir);
        end
        [outputData, ports] = predict_model(cachedState, inputData);
        state = cachedState;
        end
        """
    ).lstrip()

    demo_readme = textwrap.dedent(
        f"""
        # Demo Guide

        Recommended order:

        1. Run `demo_quick_start.m` to confirm the basic API.
        2. Run `demo_step_by_step.m` section by section to inspect initialization, sample tensors, and reference-output matching.
        3. Use `demo_sim_platform_loop.m` as the template for slot-based platform integration.

        Deployment-first API:

        ```matlab
        state = init_model(componentDir);      % once
        outputData = predict_model(state, x);  % every slot
        ```

        Short model-specific aliases:

        ```matlab
        state = init_{short_tag}(componentDir);
        outputData = predict_{short_tag}(state, x);
        ```
        """
    ).lstrip()

    readme = textwrap.dedent(
        f"""
        # SRS AI Matlab Component

        This folder is a copyable off-the-shelf Matlab component package for run:

        - `{run_name}`

    Short deployment tag:

    - `{short_tag}`

    Copy this entire folder to your Matlab project.

    Recommended deployment pattern:

        ```matlab
    state = init_model();
    outputData = predict_model(state, randn(8, 24, 'single'));
        ```

    Short model-specific aliases:

        ```matlab
    state = init_{short_tag}();
    [outputData, ports] = predict_{short_tag}(state, randn(8, 24, 'single'));
        ```

    If you want a quick smoke test:

        ```matlab
        [inputData, outputData, ports] = demo_srs_ai_matlab_component(8);
        ```

    Demo scripts are under `demo/`.

        Interface contract:

        - input: `N x 24` real-stacked float32
        - output: `N x 6 x 24` real-stacked float32
        - `ports{{k}}`: `N x 24` output for port `k`

        Deployment-first workflow:

        1. `init_model(...)` is the one-time load/parse step.
        2. `predict_model(state, inputData)` is the per-slot fast path.
        3. Keep `state` in a persistent variable in your simulation platform.
        4. Use `demo/demo_sim_platform_loop.m` as the integration template.

        Required colocated files in this folder:

        - `matlab_model_bundle.mat`
        - `matlab_model_bundle_manifest.json`
        - runtime `*.m` helpers

        This package is versioned so future migrated models can coexist without overwriting each other.
        """
    ).lstrip()

    files = {
        'load_srs_ai_matlab_component.m': load_script,
        'init_model.m': init_script,
        'predict_srs_ai_matlab_component.m': predict_script,
        'predict_model.m': predict_model_script,
        'split_ports.m': split_script,
        'demo_srs_ai_matlab_component.m': demo_script,
        f'init_{short_tag}.m': model_init_script,
        f'predict_{short_tag}.m': model_predict_script,
        'README_COMPONENT.md': readme,
    }
    for file_name, content in files.items():
        (component_dir / file_name).write_text(content, encoding='utf-8')

    demo_files = {
        'demo_quick_start.m': demo_quick_start,
        'demo_step_by_step.m': demo_step_by_step,
        'demo_sim_platform_loop.m': demo_sim_platform,
        'README_DEMO.md': demo_readme,
    }
    for file_name, content in demo_files.items():
        (demo_dir / file_name).write_text(content, encoding='utf-8')

    file_paths = {name: str(component_dir / name) for name in files}
    file_paths.update({f'demo/{name}': str(demo_dir / name) for name in demo_files})
    return file_paths


def _build_matlab_component_package(run_output_dir: Path, manifest: Dict[str, object]) -> Dict[str, object]:
    component_root = run_output_dir / 'matlab_component'
    component_root.mkdir(parents=True, exist_ok=True)
    version_tag = datetime.now().strftime('v1_%Y%m%d_%H%M%S')
    component_dir = component_root / version_tag
    component_dir.mkdir(parents=True, exist_ok=False)

    for file_name in MATLAB_COMPONENT_RUNTIME_FILES:
        shutil.copy2(_matlab_runtime_dir() / file_name, component_dir / file_name)

    mat_path = Path(str(manifest['mat_path']))
    manifest_path = Path(str(manifest['manifest_path']))
    copied_mat_path = component_dir / mat_path.name
    copied_manifest_path = component_dir / manifest_path.name
    shutil.copy2(mat_path, copied_mat_path)
    shutil.copy2(manifest_path, copied_manifest_path)

    wrapper_paths = _write_component_package_files(component_dir, manifest)
    return {
        'component_root': str(component_root),
        'component_dir': str(component_dir),
        'version_tag': version_tag,
        'short_tag': _short_model_tag(manifest),
        'mat_file': str(copied_mat_path),
        'manifest_file': str(copied_manifest_path),
        'entrypoints': {
            'init': str(component_dir / 'init_model.m'),
            'predict': str(component_dir / 'predict_model.m'),
            'split_ports': str(component_dir / 'split_ports.m'),
            'quick_demo': str(component_dir / 'demo' / 'demo_quick_start.m'),
            'step_by_step_demo': str(component_dir / 'demo' / 'demo_step_by_step.m'),
            'sim_platform_demo': str(component_dir / 'demo' / 'demo_sim_platform_loop.m'),
        },
        'wrapper_files': wrapper_paths,
    }


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


def _layer_norm_layers(sequence: nn.Sequential) -> List[nn.LayerNorm]:
    return [layer for layer in sequence if isinstance(layer, nn.LayerNorm)]


def _export_separator1_weights(model: torch.nn.Module, num_ports: int, num_stages: int) -> Dict[str, np.ndarray]:
    mat_data: Dict[str, np.ndarray] = {}
    for port_idx in range(num_ports):
        for stage_idx in range(num_stages):
            mlp = _resolve_port_stage_module(model, port_idx, stage_idx)
            real_layers = _linear_layers(mlp.mlp_real)
            imag_layers = _linear_layers(mlp.mlp_imag)
            real_norm_layers = _layer_norm_layers(mlp.mlp_real)
            imag_norm_layers = _layer_norm_layers(mlp.mlp_imag)

            for layer_idx, layer in enumerate(real_layers, start=1):
                prefix = f'p{port_idx + 1:02d}_s{stage_idx + 1:02d}_real_l{layer_idx:02d}'
                mat_data[f'{prefix}_weight'] = _to_numpy(layer.weight)
                mat_data[f'{prefix}_bias'] = _to_numpy(layer.bias)
                if layer_idx < len(real_layers) and layer_idx <= len(real_norm_layers):
                    layer_norm = real_norm_layers[layer_idx - 1]
                    mat_data[f'{prefix}_ln_weight'] = _to_numpy(layer_norm.weight)
                    mat_data[f'{prefix}_ln_bias'] = _to_numpy(layer_norm.bias)
                    mat_data[f'{prefix}_ln_eps'] = np.asarray(layer_norm.eps, dtype=np.float32)

            for layer_idx, layer in enumerate(imag_layers, start=1):
                prefix = f'p{port_idx + 1:02d}_s{stage_idx + 1:02d}_imag_l{layer_idx:02d}'
                mat_data[f'{prefix}_weight'] = _to_numpy(layer.weight)
                mat_data[f'{prefix}_bias'] = _to_numpy(layer.bias)
                if layer_idx < len(imag_layers) and layer_idx <= len(imag_norm_layers):
                    layer_norm = imag_norm_layers[layer_idx - 1]
                    mat_data[f'{prefix}_ln_weight'] = _to_numpy(layer_norm.weight)
                    mat_data[f'{prefix}_ln_bias'] = _to_numpy(layer_norm.bias)
                    mat_data[f'{prefix}_ln_eps'] = np.asarray(layer_norm.eps, dtype=np.float32)

    return mat_data


def _export_full_mlp_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    mat_data: Dict[str, np.ndarray] = {}
    for layer_idx, layer in enumerate(_linear_layers(model.network), start=1):
        prefix = f'joint_l{layer_idx:02d}'
        mat_data[f'{prefix}_weight'] = _to_numpy(layer.weight)
        mat_data[f'{prefix}_bias'] = _to_numpy(layer.bias)
    return mat_data


def _export_separator3_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    mat_data: Dict[str, np.ndarray] = {}
    for stage_idx, stage in enumerate(model.stages, start=1):
        for layer_idx, layer in enumerate(_linear_layers(stage.network), start=1):
            prefix = f'stage{stage_idx:02d}_joint_l{layer_idx:02d}'
            mat_data[f'{prefix}_weight'] = _to_numpy(layer.weight)
            mat_data[f'{prefix}_bias'] = _to_numpy(layer.bias)
        if model.learned_residual_masks is not None:
            mat_data[f'stage{stage_idx:02d}_residual_mask'] = _to_numpy(model.learned_residual_masks[stage_idx - 1])
    return mat_data


def _build_bundle_contents(model_type: str, mlp_depth: int | None, linear_layer_count: int) -> Dict[str, object]:
    bundle_contents: Dict[str, object] = {
        'sample_input_field': 'sample_input',
        'reference_output_field': 'reference_output',
        'pos_values_field': 'pos_values',
    }

    if model_type == 'separator2':
        bundle_contents['linear_layers_per_mlp'] = mlp_depth
        bundle_contents['separator2_field_pattern'] = 'p##_s##_l##_weight_real/weight_imag/bias_real/bias_imag'
    elif model_type == 'separator1':
        bundle_contents['linear_layers_per_mlp'] = mlp_depth
        bundle_contents['separator1_field_pattern'] = (
            'p##_s##_real_l##_weight/bias[/ln_weight/ln_bias/ln_eps] '
            'and p##_s##_imag_l##_weight/bias[/ln_weight/ln_bias/ln_eps]'
        )
    elif model_type == 'full_mlp':
        bundle_contents['linear_layers_in_joint_network'] = linear_layer_count
        bundle_contents['full_mlp_field_pattern'] = 'joint_l##_weight/bias'
    elif model_type == 'separator3':
        bundle_contents['linear_layers_in_separator3'] = linear_layer_count
        bundle_contents['linear_layers_per_stage'] = mlp_depth
        bundle_contents['separator3_field_pattern'] = 'stage##_joint_l##_weight/bias, stage##_residual_mask'
    else:
        raise ValueError(f'Unsupported model_type for Matlab bundle export: {model_type}')

    return bundle_contents


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
    mlp_depth = int(model_spec['mlp_depth']) if 'mlp_depth' in model_spec else None
    linear_layer_count = len(_linear_layers(model.network)) if model_type == 'full_mlp' else (sum(len(_linear_layers(stage.network)) for stage in model.stages) if model_type == 'separator3' else int(model_spec['mlp_depth']))

    if output_root is None:
        output_root = artifacts.run_dir / 'matlab_exports'
    else:
        output_root = Path(output_root)
    run_output_dir = output_root
    run_output_dir.mkdir(parents=True, exist_ok=True)

    sample_input = build_dummy_input(
        model_spec,
        batch_size=1,
        component_specs=artifacts.component_specs,
    )
    with torch.no_grad():
        reference_output = model(sample_input)

    mat_data: Dict[str, np.ndarray] = {
        'sample_input': _to_numpy(sample_input),
        'reference_output': _to_numpy(reference_output),
        'pos_values': np.asarray(model_spec.get('pos_values', []), dtype=np.int32),
    }

    if model_type == 'separator2':
        num_stages = int(model_spec['num_stages'])
        mat_data.update(_export_separator2_weights(model, num_ports=num_ports, num_stages=num_stages))
    elif model_type == 'separator1':
        num_stages = int(model_spec['num_stages'])
        mat_data.update(_export_separator1_weights(model, num_ports=num_ports, num_stages=num_stages))
    elif model_type == 'full_mlp':
        mat_data.update(_export_full_mlp_weights(model))
    elif model_type == 'separator3':
        mat_data.update(_export_separator3_weights(model))
    else:
        raise ValueError(f'Unsupported model_type for Matlab bundle export: {model_type}')

    mat_path = run_output_dir / 'matlab_model_bundle.mat'
    savemat(mat_path, mat_data, do_compression=True)

    flow_artifacts = save_model_flow_artifacts(
        output_dir=run_output_dir,
        model_spec=model_spec,
        component_specs=artifacts.component_specs,
    )
    complexity_artifacts = save_model_complexity_artifacts(
        output_dir=run_output_dir,
        model=model,
        model_spec=model_spec,
        component_specs=artifacts.component_specs,
    )

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
        'materialization_rule': 'Every learned affine layer used during inference is materialized explicitly. For staged models this includes each effective port-stage block, even when training used shared stage weights.',
        'matlab_entrypoints': [
            'import_refactor_matlab_bundle',
            'predict_refactor_matlab_bundle',
            'run_refactor_matlab_bundle_demo',
        ],
        'bundle_contents': _build_bundle_contents(
            model_type=model_type,
            mlp_depth=mlp_depth,
            linear_layer_count=linear_layer_count,
        ),
        'model_flow': flow_artifacts['flow_spec'],
        'model_flow_json_path': flow_artifacts['json_path'],
        'model_flow_markdown_path': flow_artifacts['markdown_path'],
        'model_complexity': complexity_artifacts['complexity_spec'],
        'model_complexity_json_path': complexity_artifacts['json_path'],
        'model_complexity_markdown_path': complexity_artifacts['markdown_path'],
        'input_normalization': {
            'enabled': bool(model_spec.get('normalize_energy', False)),
            'rule': 'Per-sample RMS over the complex sequence; output is rescaled by the same factor after separation.',
        },
    }

    manifest_path = run_output_dir / 'matlab_model_bundle_manifest.json'
    manifest['manifest_path'] = str(manifest_path)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    manifest['matlab_component'] = _build_matlab_component_package(run_output_dir, manifest)
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
    mlp_depth = int(model_spec['mlp_depth']) if 'mlp_depth' in model_spec else None
    linear_layer_count = len(_linear_layers(model.network)) if model_type == 'full_mlp' else (sum(len(_linear_layers(stage.network)) for stage in model.stages) if model_type == 'separator3' else int(model_spec['mlp_depth']))

    if output_root is None:
        output_root = artifacts.run_dir / 'matlab_exports'
    else:
        output_root = Path(output_root)
    run_output_dir = output_root
    run_output_dir.mkdir(parents=True, exist_ok=True)

    sample_input = build_dummy_input(
        model_spec,
        batch_size=1,
        component_specs=artifacts.component_specs,
    )
    with torch.no_grad():
        reference_output = model(sample_input)

    mat_data: Dict[str, np.ndarray] = {
        'sample_input': _to_numpy(sample_input),
        'reference_output': _to_numpy(reference_output),
        'pos_values': np.asarray(model_spec.get('pos_values', []), dtype=np.int32),
    }

    if model_type == 'separator2':
        num_stages = int(model_spec['num_stages'])
        mat_data.update(_export_separator2_weights(model, num_ports=num_ports, num_stages=num_stages))
    elif model_type == 'separator1':
        num_stages = int(model_spec['num_stages'])
        mat_data.update(_export_separator1_weights(model, num_ports=num_ports, num_stages=num_stages))
    elif model_type == 'full_mlp':
        mat_data.update(_export_full_mlp_weights(model))
    elif model_type == 'separator3':
        mat_data.update(_export_separator3_weights(model))
    else:
        raise ValueError(f'Unsupported model_type for Matlab bundle export: {model_type}')

    mat_path = run_output_dir / 'matlab_model_bundle.mat'
    savemat(mat_path, mat_data, do_compression=True)

    flow_artifacts = save_model_flow_artifacts(
        output_dir=run_output_dir,
        model_spec=model_spec,
        component_specs=artifacts.component_specs,
    )
    complexity_artifacts = save_model_complexity_artifacts(
        output_dir=run_output_dir,
        model=model,
        model_spec=model_spec,
        component_specs=artifacts.component_specs,
    )

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
        'materialization_rule': 'Every learned affine layer used during inference is materialized explicitly. For staged models this includes each effective port-stage block, even when training used shared stage weights.',
        'matlab_entrypoints': [
            'import_refactor_matlab_bundle',
            'predict_refactor_matlab_bundle',
            'run_refactor_matlab_bundle_demo',
        ],
        'bundle_contents': _build_bundle_contents(
            model_type=model_type,
            mlp_depth=mlp_depth,
            linear_layer_count=linear_layer_count,
        ),
        'model_flow': flow_artifacts['flow_spec'],
        'model_flow_json_path': flow_artifacts['json_path'],
        'model_flow_markdown_path': flow_artifacts['markdown_path'],
        'model_complexity': complexity_artifacts['complexity_spec'],
        'model_complexity_json_path': complexity_artifacts['json_path'],
        'model_complexity_markdown_path': complexity_artifacts['markdown_path'],
        'input_normalization': {
            'enabled': bool(model_spec.get('normalize_energy', False)),
            'rule': 'Per-sample RMS over the complex sequence; output is rescaled by the same factor after separation.',
        },
    }

    manifest_path = run_output_dir / 'matlab_model_bundle_manifest.json'
    manifest['manifest_path'] = str(manifest_path)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    manifest['matlab_component'] = _build_matlab_component_package(run_output_dir, manifest)
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