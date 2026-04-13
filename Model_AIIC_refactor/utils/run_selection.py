"""Helpers for resolving trained run directories from CLI-style selectors."""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from .run_artifacts import find_checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def default_refactor_experiments_root() -> Path:
    """Return the default directory where refactor artifacts should live."""
    return PACKAGE_ROOT / 'experiments_refactored'


def split_csv_arg(value: Optional[Union[str, Iterable[str]]]) -> Optional[List[str]]:
    """Split a comma-separated CLI argument into a cleaned list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(',') if item.strip()]


def resolve_existing_path(path_value: Union[str, Path]):
    """Resolve a user path against common project roots."""
    raw_path = Path(path_value)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            raw_path,
            Path.cwd() / raw_path,
            PROJECT_ROOT / raw_path,
            PACKAGE_ROOT / raw_path,
        ])

    seen = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.append(resolved)

    for candidate in seen:
        if candidate.exists():
            return candidate

    return seen[0], seen


def discover_run_dirs(exp_dir: Union[str, Path]) -> List[Path]:
    """Discover evaluable run directories inside an experiment directory."""
    resolved = resolve_existing_path(exp_dir)
    if isinstance(resolved, tuple):
        _, candidates = resolved
        candidate_text = '\n'.join(str(path) for path in candidates)
        raise FileNotFoundError('Experiment directory not found. Checked:\n' + candidate_text)

    exp_dir = resolved
    return sorted(
        [
            child for child in exp_dir.iterdir()
            if child.is_dir() and find_checkpoint_path(child) is not None
        ],
        key=lambda path: path.name,
    )


def resolve_run_selection(
    exp_dir=None,
    run_dir=None,
    run_dirs=None,
    runs=None,
) -> List[Path]:
    """Resolve trained runs from one of exp_dir, run_dir, run_dirs, or runs."""
    modes = [exp_dir is not None, run_dir is not None, run_dirs is not None]
    if sum(bool(mode) for mode in modes) == 0:
        raise ValueError('Must provide one of --exp_dir, --run_dir, or --run_dirs')
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError('--exp_dir, --run_dir, and --run_dirs are mutually exclusive')

    if run_dir is not None:
        target_dirs = [resolve_existing_path(run_dir)]
    elif run_dirs is not None:
        target_dirs = [resolve_existing_path(path) for path in split_csv_arg(run_dirs)]
    else:
        resolved_exp_dir = resolve_existing_path(exp_dir)
        if isinstance(resolved_exp_dir, tuple):
            _, candidates = resolved_exp_dir
            candidate_text = '\n'.join(str(path) for path in candidates)
            raise FileNotFoundError('Experiment directory not found. Checked:\n' + candidate_text)
        exp_dir = resolved_exp_dir
        if runs:
            target_dirs = [exp_dir / run_name for run_name in split_csv_arg(runs)]
        else:
            target_dirs = discover_run_dirs(exp_dir)

    normalized_dirs = []
    missing_path_candidates = []
    for target in target_dirs:
        if isinstance(target, tuple):
            primary_candidate, candidates = target
            missing_path_candidates.extend(str(path) for path in candidates)
            normalized_dirs.append(primary_candidate)
        else:
            normalized_dirs.append(target)

    missing_targets = [str(path) for path in normalized_dirs if find_checkpoint_path(path) is None]
    if missing_path_candidates:
        missing_targets.extend(missing_path_candidates)
    if missing_targets:
        unique_missing_targets = list(dict.fromkeys(missing_targets))
        raise FileNotFoundError(
            'Missing evaluable checkpoint in the following directories:\n'
            + '\n'.join(unique_missing_targets)
        )

    return normalized_dirs
