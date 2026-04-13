"""Shared helpers for loading trained run artifacts and export metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import yaml

try:
    from ..models import create_model
except ImportError:
    from models import create_model


REQUIRED_MODEL_SPEC_FIELDS = ('model_type', 'pos_values', 'seq_len')


def _first_non_empty(mapping: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Return the first non-empty dictionary-like payload from the given keys."""
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return {}


def _normalize_legacy_metadata(metadata: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Map older metadata field names into the canonical refactor schema."""
    resolved = dict(metadata or {})
    if not resolved:
        return {}

    resolved.setdefault('run_name', resolved.get('config_instance_name') or run_dir.name)
    resolved.setdefault('model_recipe_name', resolved.get('model_config_name'))
    resolved.setdefault('training_recipe_name', resolved.get('training_config_name'))
    resolved.setdefault('model_label', resolved.get('model_label') or resolved.get('model_config_name'))
    resolved.setdefault('training_label', resolved.get('training_label') or resolved.get('training_config_name'))
    resolved.setdefault('experiment_name', resolved.get('experiment_name'))
    return resolved


@dataclass(frozen=True)
class RunArtifacts:
    """Resolved metadata and checkpoint assets for a trained run."""

    run_dir: Path
    checkpoint_path: Path
    config_path: Optional[Path]
    checkpoint: Dict[str, Any]
    model_spec: Dict[str, Any]
    training_spec: Dict[str, Any]
    metadata: Dict[str, Any]
    eval_results: Dict[str, Any]


def find_checkpoint_path(run_dir: Union[str, Path]) -> Optional[Path]:
    """Return the preferred checkpoint path for a trained run directory."""
    run_dir = Path(run_dir)
    primary = run_dir / 'model.pth'
    if primary.exists():
        return primary

    checkpoints = sorted(run_dir.glob('checkpoint_batch_*.pth'))
    if checkpoints:
        return checkpoints[-1]

    return None


def normalize_model_spec(model_spec: Dict[str, Any], num_params: Optional[int] = None) -> Dict[str, Any]:
    """Fill derived/default fields so all downstream workflows see one schema."""
    resolved = dict(model_spec or {})
    if 'pos_values' in resolved and 'num_ports' not in resolved:
        resolved['num_ports'] = len(resolved['pos_values'])

    resolved.setdefault('hidden_dim', 64)
    resolved.setdefault('num_stages', 2)
    resolved.setdefault('mlp_depth', 3)
    resolved.setdefault('share_weights_across_stages', False)
    resolved.setdefault('activation_type', 'relu')
    resolved.setdefault('onnx_mode', False)

    if num_params is not None:
        resolved['num_params'] = int(num_params)

    return resolved


def build_model_artifact_spec(model_spec: Dict[str, Any], num_params: Optional[int] = None) -> Dict[str, Any]:
    """Create the serialized model spec stored beside checkpoints."""
    return normalize_model_spec(model_spec, num_params=num_params)


def build_training_artifact_spec(training_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create a stable serialized training spec for run artifacts."""
    return dict(training_spec or {})


def build_run_metadata(
    experiment_name: Optional[str],
    model_recipe_name: str,
    model_label: str,
    run_name: str,
    training_recipe_name: str,
    training_label: str,
    training_duration: float,
) -> Dict[str, Any]:
    """Build the canonical run metadata payload."""
    from datetime import datetime

    return {
        'experiment_name': experiment_name,
        'model_recipe_name': model_recipe_name,
        'model_label': model_label,
        'run_name': run_name,
        'training_recipe_name': training_recipe_name,
        'training_label': training_label,
        'training_duration': training_duration,
        'timestamp': datetime.now().isoformat(),
    }


def save_run_config(
    run_dir: Union[str, Path],
    model_spec: Dict[str, Any],
    training_spec: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Path:
    """Persist the canonical config.yaml stored in each run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / 'config.yaml'
    with open(config_path, 'w', encoding='utf-8') as config_file:
        yaml.safe_dump(
            {
                'model_spec': dict(model_spec or {}),
                'training_spec': dict(training_spec or {}),
                'metadata': dict(metadata or {}),
            },
            config_file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    return config_path


def load_run_artifacts(
    run_dir: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
) -> RunArtifacts:
    """Load config/checkpoint metadata for a trained run."""
    run_dir = Path(run_dir)
    checkpoint_path = find_checkpoint_path(run_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f'No checkpoint found in {run_dir}')

    config_path = run_dir / 'config.yaml'
    config_data: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config_data = yaml.safe_load(config_file) or {}

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_spec = _first_non_empty(config_data, 'model_spec', 'model_config') or checkpoint.get('model_spec') or {}
    if not model_spec:
        raise KeyError(
            f"Checkpoint missing 'model_spec' in {checkpoint_path}. Retrain with the current training pipeline."
        )

    missing_fields = [field for field in REQUIRED_MODEL_SPEC_FIELDS if field not in model_spec]
    if missing_fields:
        raise KeyError(f'Model spec missing required fields: {missing_fields}')

    model_spec = normalize_model_spec(
        model_spec,
        num_params=checkpoint.get('model_info', {}).get('num_params') or model_spec.get('num_params'),
    )
    training_spec = _first_non_empty(config_data, 'training_spec', 'training_config') or checkpoint.get('training_spec') or {}
    metadata = _normalize_legacy_metadata(
        _first_non_empty(config_data, 'metadata') or checkpoint.get('metadata') or {},
        run_dir=run_dir,
    )
    eval_results = checkpoint.get('eval_results', {})

    return RunArtifacts(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path if config_path.exists() else None,
        checkpoint=checkpoint,
        model_spec=model_spec,
        training_spec=training_spec,
        metadata=metadata,
        eval_results=eval_results,
    )


def load_trained_model_from_run(
    run_dir: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[torch.nn.Module, RunArtifacts]:
    """Load a trained model and its resolved run artifacts."""
    artifacts = load_run_artifacts(run_dir, device=device)
    model_type = artifacts.model_spec.get('model_type', 'separator1')
    model = create_model(model_name=model_type, config=artifacts.model_spec)

    state_dict = artifacts.checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, artifacts


def build_dummy_input(model_spec: Dict[str, Any], batch_size: int = 1) -> torch.Tensor:
    """Create a representative real-stacked input tensor for export or smoke tests."""
    seq_len = int(model_spec['seq_len'])
    return torch.randn(batch_size, seq_len * 2, dtype=torch.float32)
