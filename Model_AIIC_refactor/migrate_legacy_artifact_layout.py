"""Migrate legacy single-run artifact directories to the simplified layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _merge_directory_contents(source_dir: Path, target_dir: Path) -> bool:
    """Move all items from source_dir into target_dir when no conflicts exist."""
    if not source_dir.is_dir():
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    source_items = list(source_dir.iterdir())
    for source_item in source_items:
        destination = target_dir / source_item.name
        if destination.exists():
            raise FileExistsError(f'Cannot migrate {source_item} -> {destination}: destination already exists')
        shutil.move(str(source_item), str(destination))

    source_dir.rmdir()
    return True


def _rename_single_run_eval_dir(eval_dir: Path) -> bool:
    """Rename YYYYMMDD_HHMMSS_<run_name> to YYYYMMDD_HHMMSS for single-run evaluations."""
    parent = eval_dir.parent
    if parent.name != 'evaluations':
        return False

    run_dir = parent.parent
    run_name = run_dir.name
    prefix = eval_dir.name[:15]
    expected_prefix = f'{prefix}_'
    if len(prefix) != 15 or eval_dir.name[8] != '_' or not eval_dir.name.startswith(expected_prefix):
        return False
    suffix = eval_dir.name[16:]
    if suffix != run_name:
        return False

    destination = parent / prefix
    if destination.exists():
        raise FileExistsError(f'Cannot rename {eval_dir} -> {destination}: destination already exists')
    eval_dir.rename(destination)
    return True


def migrate_workspace(root: Path) -> dict[str, int]:
    """Apply layout migration under the given workspace root."""
    root = Path(root)
    migrated = {
        'onnx_dirs': 0,
        'matlab_dirs': 0,
        'eval_dirs': 0,
    }

    for legacy_dir in sorted(root.glob('**/onnx_exports/*')):
        if legacy_dir.is_dir() and legacy_dir.parent.parent.name == legacy_dir.name:
            if _merge_directory_contents(legacy_dir, legacy_dir.parent):
                migrated['onnx_dirs'] += 1

    for legacy_dir in sorted(root.glob('**/matlab_exports/*')):
        if legacy_dir.is_dir() and legacy_dir.parent.parent.name == legacy_dir.name:
            if _merge_directory_contents(legacy_dir, legacy_dir.parent):
                migrated['matlab_dirs'] += 1

    for eval_dir in sorted(root.glob('**/evaluations/*')):
        if eval_dir.is_dir() and _rename_single_run_eval_dir(eval_dir):
            migrated['eval_dirs'] += 1

    return migrated


def main():
    parser = argparse.ArgumentParser(description='Migrate legacy single-run artifact directories to the simplified layout')
    parser.add_argument('--root', type=str, default=str(Path(__file__).resolve().parent / 'experiments_refactored'), help='Root directory to scan for legacy artifacts')
    args = parser.parse_args()

    migrated = migrate_workspace(Path(args.root))
    print('Migration complete')
    print(f"  ONNX directories flattened: {migrated['onnx_dirs']}")
    print(f"  Matlab directories flattened: {migrated['matlab_dirs']}")
    print(f"  Evaluation directories renamed: {migrated['eval_dirs']}")


if __name__ == '__main__':
    main()