"""Export trained refactored runs to ONNX with Matlab-friendly metadata."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from utils import (
    split_csv_arg,
    discover_run_dirs,
    resolve_run_selection,
    load_trained_model_from_run,
    build_dummy_input,
)


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

        max_abs_diff = float(abs(torch_output - ort_output).max())
        validation['onnxruntime'] = True
        validation['max_abs_diff'] = max_abs_diff
    except Exception as error:
        validation['onnxruntime_error'] = str(error)

    return validation


def export_run_to_onnx(
    run_dir,
    output_root,
    opset_version: int = 13,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    validate: bool = False,
    export_params: bool = True,
) -> Dict[str, object]:
    """Export a single trained run directory to ONNX."""
    model, artifacts = load_trained_model_from_run(run_dir, device='cpu')
    model = _prepare_model_for_export(model)

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

    validation = {}
    if validate:
        validation = validate_exported_model(onnx_path, model, dummy_input)

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
        'dummy_output_shape': list(model(dummy_input).shape),
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
    manifests = []
    for target_dir in target_dirs:
        manifests.append(
            export_run_to_onnx(
                run_dir=target_dir,
                output_root=output_root,
                opset_version=opset_version,
                batch_size=batch_size,
                dynamic_batch=dynamic_batch,
                validate=validate,
            )
        )
    return manifests


def main():
    parser = argparse.ArgumentParser(description='Export trained refactored runs to ONNX')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory. Exports all runs inside by default, or a subset with --runs')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Single trained run directory')
    parser.add_argument('--run_dirs', type=str, default=None,
                        help='Multiple trained run directories, comma-separated')
    parser.add_argument('--runs', type=str, default=None,
                        help='Run names inside --exp_dir, comma-separated')
    parser.add_argument('--list_runs', action='store_true',
                        help='List exportable runs inside --exp_dir and exit')
    parser.add_argument('--output', type=str, default='onnx_exports',
                        help='Directory where exported ONNX artifacts are written')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version (13 is a Matlab-friendly default)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Dummy batch size used for export tracing')
    parser.add_argument('--dynamic_batch', action='store_true',
                        help='Export with a dynamic batch dimension')
    parser.add_argument('--validate', action='store_true',
                        help='Run ONNX checker and ONNX Runtime smoke validation after export')

    args = parser.parse_args()

    if args.runs and not args.exp_dir:
        raise ValueError('--runs requires --exp_dir')

    if args.list_runs:
        if not args.exp_dir:
            raise ValueError('--list_runs requires --exp_dir')
        run_dirs = discover_run_dirs(args.exp_dir)
        print(f'Exportable runs: {len(run_dirs)}')
        for resolved_run_dir in run_dirs:
            print(f'  - {resolved_run_dir.name}')
        return

    manifests = export_runs_to_onnx(
        output_root=args.output,
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
        opset_version=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        validate=args.validate,
    )

    print(f'✓ Exported {len(manifests)} run(s) to {args.output}')
    for manifest in manifests:
        print(f"  - {manifest['run_name']}: {manifest['onnx_path']}")


if __name__ == '__main__':
    main()
