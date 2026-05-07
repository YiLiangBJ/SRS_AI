"""V2 component-based experiment planning helpers."""

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from .config_parser import parse_search_space_value

try:
    from tasks import get_task_class
    from training_strategies import get_training_strategy_class
except ImportError:
    from ..tasks import get_task_class
    from ..training_strategies import get_training_strategy_class


def _load_yaml_mapping(config_path: Path, root_key: str) -> Dict[str, Any]:
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as config_file:
        payload = yaml.safe_load(config_file) or {}

    if root_key in payload:
        return payload[root_key] or {}
    return payload


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _format_name_value(value: Any) -> str:
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float):
        return f"{value:.4f}".rstrip('0').rstrip('.')
    if isinstance(value, list):
        return '-'.join(_format_name_value(item) for item in value)
    if isinstance(value, dict):
        if value.get('type') == 'range':
            parts = [
                str(value.get('type', 'range')),
                _format_name_value(value.get('min')),
                _format_name_value(value.get('max')),
            ]
            return '-'.join(part for part in parts if part and part != 'None')
        if 'values' in value:
            return '-'.join(_format_name_value(item) for item in value['values'])
        return str(len(value))
    return str(value).replace(' ', '')


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _set_nested_value(mapping: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split('.') if part]
    if not parts:
        raise ValueError('Sweep target path cannot be empty')

    current = mapping
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _apply_recipe_override(base_payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split('.') if part]
    if not parts:
        raise ValueError('Recipe override path cannot be empty')

    if parts[0] != 'sweeps':
        _set_nested_value(base_payload, path, value)
        return

    if len(parts) < 3:
        raise ValueError(
            f"Sweep override '{path}' must use sweeps.<alias|target|index>.<field> form"
        )

    sweep_identifier = parts[1]
    remaining_path = '.'.join(parts[2:])
    sweeps = base_payload.get('sweeps', [])
    for index, sweep in enumerate(sweeps):
        alias = str(sweep.get('alias') or '')
        target = str(sweep.get('target') or '')
        if sweep_identifier in {alias, target, str(index)}:
            _set_nested_value(sweep, remaining_path, value)
            return

    raise ValueError(
        f"Sweep override '{path}' did not match any sweep alias/target/index in recipe"
    )


def _append_tokens(label: str, tokens: Sequence[str]) -> str:
    if not tokens:
        return label
    return f"{label}_{'_'.join(tokens)}"


def _normalize_local_variant(
    recipe_name: str,
    base_payload: Dict[str, Any],
    raw_spec: Dict[str, Any],
    tokens: Sequence[str],
) -> tuple[Dict[str, Any], List[str]]:
    """Normalize locally expanded variants to remove semantically duplicate combinations."""
    normalized_spec = deepcopy(raw_spec)
    normalized_tokens = list(tokens)

    if normalized_spec.get('type') == 'separator1':
        params = normalized_spec.get('params', {})
        if int(params.get('mlp_depth', 3)) == 2:
            base_hidden_dim = _nested_get(base_payload, 'params', 'hidden_dim', default=params.get('hidden_dim'))
            params['hidden_dim'] = base_hidden_dim
            normalized_tokens = [token for token in normalized_tokens if not token.startswith('hd')]

    return normalized_spec, normalized_tokens


def _expand_local_sweeps(
    recipe_name: str,
    base_payload: Dict[str, Any],
    sweeps: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not sweeps:
        return [{'label': recipe_name, 'raw_spec': deepcopy(base_payload)}]

    parsed_sweeps = []
    for sweep in sweeps:
        target = sweep.get('target')
        if not target:
            raise ValueError(f"Recipe '{recipe_name}' has a sweep without target")
        values = parse_search_space_value(sweep.get('values'), target)
        alias = sweep.get('alias') or target.split('.')[-1]
        parsed_sweeps.append((target, alias, values))

    variants = []
    seen_variant_keys = set()
    for combination in product(*[values for _, _, values in parsed_sweeps]):
        raw_spec = deepcopy(base_payload)
        tokens: List[str] = []
        for (target, alias, _), selected_value in zip(parsed_sweeps, combination):
            _set_nested_value(raw_spec, target, selected_value)
            tokens.append(f"{alias}{_format_name_value(selected_value)}")

        normalized_spec, normalized_tokens = _normalize_local_variant(
            recipe_name=recipe_name,
            base_payload=base_payload,
            raw_spec=raw_spec,
            tokens=tokens,
        )
        variant_key = yaml.safe_dump(normalized_spec, sort_keys=True)
        if variant_key in seen_variant_keys:
            continue
        seen_variant_keys.add(variant_key)
        variants.append({'label': _append_tokens(recipe_name, normalized_tokens), 'raw_spec': normalized_spec})

    return variants


def _expand_experiment_sweeps(
    task_raw_spec: Dict[str, Any],
    model_raw_spec: Dict[str, Any],
    training_raw_spec: Dict[str, Any],
    sweeps: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not sweeps:
        return [{
            'task_raw_spec': deepcopy(task_raw_spec),
            'model_raw_spec': deepcopy(model_raw_spec),
            'training_raw_spec': deepcopy(training_raw_spec),
            'tokens_by_component': {'task': [], 'model': [], 'training_strategy': []},
        }]

    parsed_sweeps = []
    for sweep in sweeps:
        target = sweep.get('target')
        if not target:
            raise ValueError('Experiment sweep is missing target')
        root, _, _ = target.partition('.')
        if root not in {'task', 'model', 'training_strategy'}:
            raise ValueError(
                f"Unsupported experiment sweep target '{target}'. "
                f"Expected root in task/model/training_strategy"
            )
        values = parse_search_space_value(sweep.get('values'), target)
        alias = sweep.get('alias') or target.split('.')[-1]
        parsed_sweeps.append((root, target, alias, values))

    expanded = []
    for combination in product(*[values for _, _, _, values in parsed_sweeps]):
        task_variant = deepcopy(task_raw_spec)
        model_variant = deepcopy(model_raw_spec)
        training_variant = deepcopy(training_raw_spec)
        tokens_by_component = {'task': [], 'model': [], 'training_strategy': []}

        for (root, target, alias, _), selected_value in zip(parsed_sweeps, combination):
            token = f"{alias}{_format_name_value(selected_value)}"
            tokens_by_component[root].append(token)

            if root == 'task':
                nested_target = target.split('.', 1)[1]
                _set_nested_value(task_variant, nested_target, selected_value)
            elif root == 'model':
                nested_target = target.split('.', 1)[1]
                _set_nested_value(model_variant, nested_target, selected_value)
            else:
                nested_target = target.split('.', 1)[1]
                _set_nested_value(training_variant, nested_target, selected_value)

        expanded.append({
            'task_raw_spec': task_variant,
            'model_raw_spec': model_variant,
            'training_raw_spec': training_variant,
            'tokens_by_component': tokens_by_component,
        })

    return expanded


def load_component_catalog(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    config_dir = Path(config_dir)
    v2_dir = config_dir / 'v2'
    return {
        'tasks': _load_yaml_mapping(v2_dir / 'tasks.yaml', 'tasks'),
        'models': _load_yaml_mapping(v2_dir / 'models.yaml', 'models'),
        'training_strategies': _load_yaml_mapping(v2_dir / 'training_strategies.yaml', 'training_strategies'),
        'experiments': _load_yaml_mapping(v2_dir / 'experiments.yaml', 'experiments'),
    }
def build_component_experiment_data(
    config_dir: Path,
    experiment_name: str,
    batch_size_override: Optional[int] = None,
    num_batches_override: Optional[int] = None,
    task_recipe_overrides: Optional[Mapping[str, Any]] = None,
    model_recipe_overrides: Optional[Mapping[str, Any]] = None,
    training_recipe_overrides: Optional[Mapping[str, Any]] = None,
    default_training_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    catalog = load_component_catalog(config_dir)
    experiment_definition = catalog['experiments'].get(experiment_name)
    if not experiment_definition:
        raise ValueError(f"V2 experiment '{experiment_name}' not found")

    task_recipe_name = experiment_definition.get('task')
    training_recipe_name = experiment_definition.get('training_strategy') or experiment_definition.get('training')
    model_recipe_names = [name.strip() for name in _ensure_list(experiment_definition.get('model') or experiment_definition.get('models')) if str(name).strip()]
    experiment_sweeps = experiment_definition.get('sweeps', [])

    if not task_recipe_name:
        raise ValueError(f"V2 experiment '{experiment_name}' is missing task")
    if not training_recipe_name:
        raise ValueError(f"V2 experiment '{experiment_name}' is missing training_strategy")
    if not model_recipe_names:
        raise ValueError(f"V2 experiment '{experiment_name}' is missing model")

    task_recipe = catalog['tasks'].get(task_recipe_name)
    if not task_recipe:
        raise ValueError(f"Task recipe '{task_recipe_name}' not found")
    task_recipe = deepcopy(task_recipe)
    for path, value in (task_recipe_overrides or {}).items():
        _apply_recipe_override(task_recipe, path, value)

    training_recipe = catalog['training_strategies'].get(training_recipe_name)
    if not training_recipe:
        raise ValueError(f"Training strategy '{training_recipe_name}' not found")
    training_recipe = deepcopy(training_recipe)
    for path, value in (training_recipe_overrides or {}).items():
        _apply_recipe_override(training_recipe, path, value)

    task_variants_raw = _expand_local_sweeps(
        recipe_name=task_recipe_name,
        base_payload={**deepcopy(task_recipe), 'params': deepcopy(task_recipe.get('params', {}))},
        sweeps=task_recipe.get('sweeps', []),
    )

    training_overrides: Dict[str, Any] = {}
    if batch_size_override is not None:
        training_overrides['params.batch_size'] = batch_size_override
    if num_batches_override is not None:
        training_overrides['params.num_batches'] = num_batches_override

    training_variants_raw = _expand_local_sweeps(
        recipe_name=training_recipe_name,
        base_payload={**deepcopy(training_recipe), 'params': deepcopy(training_recipe.get('params', {}))},
        sweeps=training_recipe.get('sweeps', []),
    )
    if training_overrides:
        for variant in training_variants_raw:
            for path, selected_value in training_overrides.items():
                _set_nested_value(variant['raw_spec'], path, selected_value)

    model_variants_raw_by_recipe: Dict[str, List[Dict[str, Any]]] = {}
    missing_model_recipes: List[str] = []
    for model_recipe_name in model_recipe_names:
        model_recipe = catalog['models'].get(model_recipe_name)
        if not model_recipe:
            missing_model_recipes.append(model_recipe_name)
            continue
        model_recipe = deepcopy(model_recipe)
        for path, value in (model_recipe_overrides or {}).items():
            _apply_recipe_override(model_recipe, path, value)
        model_variants_raw_by_recipe[model_recipe_name] = _expand_local_sweeps(
            recipe_name=model_recipe_name,
            base_payload={**deepcopy(model_recipe), 'params': deepcopy(model_recipe.get('params', {}))},
            sweeps=model_recipe.get('sweeps', []),
        )

    if not model_variants_raw_by_recipe:
        raise ValueError(f"No valid model recipes found in v2 experiment: {experiment_name}")

    primary_task_raw_spec = task_variants_raw[0]['raw_spec']
    task_class = get_task_class(primary_task_raw_spec.get('type'))
    training_strategy_class = get_training_strategy_class(training_recipe.get('type'))
    model_variants_by_recipe = {
        recipe_name: [
            {
                'recipe_name': recipe_name,
                'label': variant['label'],
                'spec': task_class.compile_model_spec(primary_task_raw_spec, variant['raw_spec']),
                'index': index,
                'total': len(variants),
            }
            for index, variant in enumerate(variants, 1)
        ]
        for recipe_name, variants in model_variants_raw_by_recipe.items()
    }
    training_variants = [
        {
            'recipe_name': training_recipe_name,
            'label': variant['label'],
            'spec': training_strategy_class.compile_runtime_spec(
                primary_task_raw_spec,
                variant['raw_spec'],
                default_training_config,
            ),
            'index': index,
            'total': len(training_variants_raw),
        }
        for index, variant in enumerate(training_variants_raw, 1)
    ]

    plan: List[Dict[str, Any]] = []
    task_labels: List[str] = []
    task_total = len(task_variants_raw)
    include_task_label = task_total > 1
    include_training_label = len(training_variants_raw) > 1
    task_label_to_index = {variant['label']: index for index, variant in enumerate(task_variants_raw, 1)}

    task_labels = [variant['label'] for variant in task_variants_raw]
    task_index = 0
    for task_variant_raw in task_variants_raw:
        task_index += 1
        for training_index, training_variant_raw in enumerate(training_variants_raw, 1):
            for model_recipe_name in model_recipe_names:
                model_variants_raw = model_variants_raw_by_recipe.get(model_recipe_name, [])
                for model_index, model_variant_raw in enumerate(model_variants_raw, 1):
                    expanded_variants = _expand_experiment_sweeps(
                        task_raw_spec=task_variant_raw['raw_spec'],
                        model_raw_spec=model_variant_raw['raw_spec'],
                        training_raw_spec=training_variant_raw['raw_spec'],
                        sweeps=experiment_sweeps,
                    )
                    for expanded_variant in expanded_variants:
                        tokens_by_component = expanded_variant['tokens_by_component']
                        task_label = _append_tokens(task_variant_raw['label'], tokens_by_component['task'])
                        model_label = _append_tokens(model_variant_raw['label'], tokens_by_component['model'])
                        training_label = _append_tokens(training_variant_raw['label'], tokens_by_component['training_strategy'])

                        run_name_parts = [model_label]
                        if include_training_label or tokens_by_component['training_strategy']:
                            run_name_parts.append(training_label)
                        if include_task_label or tokens_by_component['task']:
                            run_name_parts.append(task_label)

                        plan.append({
                            'task_index': len(plan) + 1,
                            'run_name': '_'.join(run_name_parts),
                            'task_recipe_name': task_recipe_name,
                            'task_label': task_label,
                            'task_variant_index': task_label_to_index[task_variant_raw['label']],
                            'task_variant_total': task_total,
                            'model_variant': {
                                'recipe_name': model_recipe_name,
                                'label': model_label,
                                'spec': task_class.compile_model_spec(
                                    expanded_variant['task_raw_spec'],
                                    expanded_variant['model_raw_spec'],
                                ),
                                'index': model_index,
                                'total': len(model_variants_raw),
                            },
                            'training_variant': {
                                'recipe_name': training_recipe_name,
                                'label': training_label,
                                'spec': training_strategy_class.compile_runtime_spec(
                                    expanded_variant['task_raw_spec'],
                                    expanded_variant['training_raw_spec'],
                                    default_training_config,
                                ),
                                'index': training_index,
                                'total': len(training_variants_raw),
                            },
                            'component_specs': {
                                'task': expanded_variant['task_raw_spec'],
                                'model': expanded_variant['model_raw_spec'],
                                'training_strategy': expanded_variant['training_raw_spec'],
                            },
                        })

    return {
        'task_recipe_name': task_recipe_name,
        'task_labels': task_labels,
        'model_recipe_names': model_recipe_names,
        'training_recipe_name': training_recipe_name,
        'model_variants_by_recipe': model_variants_by_recipe,
        'training_variants': training_variants,
        'missing_model_recipes': missing_model_recipes,
        'plan': plan,
    }


__all__ = [
    'build_component_experiment_data',
    'load_component_catalog',
]