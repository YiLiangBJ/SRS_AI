"""Helpers for validating and loading initialization checkpoints for continued training."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .run_artifacts import load_run_artifacts_from_checkpoint, normalize_model_spec


IGNORED_MODEL_SPEC_KEYS = {'num_params'}


def compare_model_specs(expected_model_spec: Mapping[str, Any], loaded_model_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Compare two normalized model specs and return user-facing mismatches."""
    expected = normalize_model_spec(dict(expected_model_spec or {}))
    loaded = normalize_model_spec(dict(loaded_model_spec or {}))

    mismatches: List[Dict[str, Any]] = []
    keys = sorted((set(expected.keys()) | set(loaded.keys())) - IGNORED_MODEL_SPEC_KEYS)
    for key in keys:
        expected_value = expected.get(key)
        loaded_value = loaded.get(key)
        if expected_value != loaded_value:
            mismatches.append({
                'field': key,
                'expected': expected_value,
                'loaded': loaded_value,
            })
    return mismatches


def format_model_spec_mismatches(mismatches: List[Dict[str, Any]]) -> str:
    if not mismatches:
        return ''
    lines = ['Model spec does not match the initialization checkpoint:']
    for mismatch in mismatches:
        lines.append(
            f"- {mismatch['field']}: expected={mismatch['expected']!r}, checkpoint={mismatch['loaded']!r}"
        )
    return '\n'.join(lines)


def load_initial_checkpoint_state(
    checkpoint_path,
    expected_model_spec: Mapping[str, Any],
    device: str = 'cpu',
) -> Tuple[Dict[str, Any], Any]:
    """Load one checkpoint for continued training after validating model-spec compatibility."""
    artifacts = load_run_artifacts_from_checkpoint(checkpoint_path, device=device)
    mismatches = compare_model_specs(expected_model_spec, artifacts.model_spec)
    if mismatches:
        raise ValueError(format_model_spec_mismatches(mismatches))

    state_dict = artifacts.checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

    return state_dict, artifacts


__all__ = [
    'compare_model_specs',
    'format_model_spec_mismatches',
    'load_initial_checkpoint_state',
]