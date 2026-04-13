"""Programmatic ONNX export workflow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from utils import discover_run_dirs, resolve_run_selection, load_trained_model_from_run, build_dummy_input


def _prepare_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """Return an exportable module, unwrapping torch.compile when needed."""
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    model.eval()
    model.cpu()
    return model


def validate_exported_model(
    onnx_path: Path,
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
) -> Dict[str, object]:
    """Run lightweight ONNX checker and ONNX Runtime smoke validation."""
    validation: Dict[str, object] = {'checker': False, 'onnxruntime': False}

    import onnx

    exported_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported_model)
    validation['checker'] = True

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        ort_output = session.run(None, {input_name: dummy_input.numpy()})[0]
        with torch.no_grad():
            torch_output = model(dummy_input).detach().cpu().numpy()

        validation['onnxruntime'] = True
        validation['max_abs_diff'] = float(abs(torch_output - ort_output).max())
    except Exception as error:
        validation['onnxruntime_error'] = str(error)

    return validation


def export_run_to_onnx(
    run_dir,
    output_root=None,
    opset_version: int = 13,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    validate: bool = False,
    export_params: bool = True,
) -> Dict[str, object]:
    """Export a single trained run directory to ONNX."""
    model, artifacts = load_trained_model_from_run(run_dir, device='cpu')
    model = _prepare_model_for_export(model)

    if output_root is None:
        output_root = artifacts.run_dir / 'onnx_exports'
    else:
        output_root = Path(output_root)
    run_output_dir = output_root / artifacts.run_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    dummy_input = build_dummy_input(artifacts.model_spec, batch_size=batch_size)
    input_names = ['mixed_signal']
    output_names = ['separated_channels']
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'mixed_signal': {0: 'batch_size'},
            'separated_channels': {0: 'batch_size'},
        }

    onnx_path = run_output_dir / f'{artifacts.run_dir.name}.onnx'
    with torch.no_grad():
        dummy_output = model(dummy_input)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=export_params,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    validation = validate_exported_model(onnx_path, model, dummy_input) if validate else {}
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'run_name': artifacts.run_dir.name,
        'run_dir': str(artifacts.run_dir),
        'checkpoint_path': str(artifacts.checkpoint_path),
        'onnx_path': str(onnx_path),
        'model_spec': artifacts.model_spec,
        'training_spec': artifacts.training_spec,
        'metadata': artifacts.metadata,
        'input_names': input_names,
        'output_names': output_names,
        'dummy_input_shape': list(dummy_input.shape),
        'dummy_output_shape': list(dummy_output.shape),
        'opset_version': opset_version,
        'dynamic_batch': dynamic_batch,
        'validation': validation,
        'matlab_notes': {
            'recommended_import': 'importNetworkFromONNX',
            'input_name': input_names[0],
            'output_name': output_names[0],
            'input_layout': 'N x (2*seq_len) real-stacked float32',
            'output_layout': 'N x num_ports x (2*seq_len) real-stacked float32',
        },
    }

    manifest_path = run_output_dir / 'export_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    return manifest


def export_runs_to_onnx(
    output_root,
    exp_dir=None,
    run_dir=None,
    run_dirs=None,
    runs=None,
    opset_version: int = 13,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    validate: bool = False,
) -> List[Dict[str, object]]:
    """Programmatic multi-run ONNX export entry point."""
    target_dirs = resolve_run_selection(
        exp_dir=exp_dir,
        run_dir=run_dir,
        run_dirs=run_dirs,
        runs=runs,
    )
    return [
        export_run_to_onnx(
            run_dir=target_dir,
            output_root=output_root,
            opset_version=opset_version,
            batch_size=batch_size,
            dynamic_batch=dynamic_batch,
            validate=validate,
        )
        for target_dir in target_dirs
    ]
