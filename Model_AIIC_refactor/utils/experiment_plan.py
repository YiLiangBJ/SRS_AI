"""Helpers for building v2 component-based experiment plans."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .component_experiment_plan import _append_tokens, _format_name_value, _set_nested_value, build_component_experiment_data, load_component_catalog

try:
    from tasks import get_task_class
    from training_strategies import get_training_strategy_class
except ImportError:
    from ..tasks import get_task_class
    from ..training_strategies import get_training_strategy_class


DEFAULT_TRAINING_CONFIG = {
    'batch_size': 2048,
    'num_batches': 10000,
    'learning_rate': 0.01,
    'loss_type': 'nmse',
    'print_interval': 100,
    'validation_batches': 4,
    'patience': 3,
    'keep_last_n_checkpoints': 2,
    'lr_scheduler': {
        'enabled': True,
        'factor': 0.8,
        'patience': 30,
        'threshold': 5e-3,
        'threshold_mode': 'abs',
        'cooldown': 10,
        'min_lr': 1e-6,
    },
}


@dataclass(frozen=True)
class ModelVariant:
    """A fully resolved model configuration variant."""

    recipe_name: str
    label: str
    spec: Dict[str, Any]
    index: int
    total: int


@dataclass(frozen=True)
class TrainingVariant:
    """A fully resolved training-strategy runtime variant."""

    recipe_name: str
    label: str
    spec: Dict[str, Any]
    index: int
    total: int


@dataclass(frozen=True)
class ExperimentPlanItem:
    """A single executable task-model-training combination."""

    task_index: int
    run_name: str
    model_variant: ModelVariant
    training_variant: TrainingVariant
    task_recipe_name: Optional[str] = None
    task_label: Optional[str] = None
    task_variant_index: Optional[int] = None
    task_variant_total: Optional[int] = None
    component_specs: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = 'v2'

    @property
    def model_spec(self) -> Dict[str, Any]:
        return self.model_variant.spec

    @property
    def training_spec(self) -> Dict[str, Any]:
        return self.training_variant.spec

    @property
    def model_recipe_name(self) -> str:
        return self.model_variant.recipe_name

    @property
    def model_label(self) -> str:
        return self.model_variant.label

    @property
    def model_index(self) -> int:
        return self.model_variant.index

    @property
    def model_total(self) -> int:
        return self.model_variant.total

    @property
    def training_recipe_name(self) -> str:
        return self.training_variant.recipe_name

    @property
    def training_label(self) -> str:
        return self.training_variant.label

    @property
    def training_index(self) -> int:
        return self.training_variant.index

    @property
    def training_total(self) -> int:
        return self.training_variant.total

    @property
    def task_spec(self) -> Dict[str, Any]:
        return self.component_specs.get('task', {})

    @property
    def training_strategy_spec(self) -> Dict[str, Any]:
        return self.component_specs.get('training_strategy', {})

    @property
    def model_component_spec(self) -> Dict[str, Any]:
        return self.component_specs.get('model', {})


@dataclass(frozen=True)
class ConfigCatalog:
    """Loaded v2 component catalog for the project."""

    config_dir: Path
    component_catalog: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ExperimentSuite:
    """All prepared state needed to execute one v2 experiment."""

    catalog: ConfigCatalog
    experiment_name: Optional[str]
    model_recipe_names: List[str]
    training_recipe_name: str
    model_variants_by_recipe: Dict[str, List[ModelVariant]]
    training_variants: List[TrainingVariant]
    missing_model_recipes: List[str]
    plan: List[ExperimentPlanItem]
    schema_version: str = 'v2'
    task_recipe_name: Optional[str] = None
    task_labels: List[str] = field(default_factory=list)


def _build_override_tokens(prefix: str, overrides: Optional[Dict[str, Any]]) -> List[str]:
    if not overrides:
        return []
    return [
        f"{prefix}{path.replace('.', '_')}{_format_name_value(value)}"
        for path, value in sorted(overrides.items())
    ]


def _apply_component_override(raw_spec: Dict[str, Any], path: str, value: Any) -> None:
    if path.startswith('params.'):
        _set_nested_value(raw_spec, path, value)
        return

    root_key = path.split('.', 1)[0]
    if root_key in raw_spec:
        _set_nested_value(raw_spec, path, value)
        return

    if isinstance(raw_spec.get('params'), dict):
        _set_nested_value(raw_spec, f'params.{path}', value)
        return

    _set_nested_value(raw_spec, path, value)


def _deduplicate_plan_items(plan: Sequence[ExperimentPlanItem]) -> List[ExperimentPlanItem]:
    deduplicated: List[ExperimentPlanItem] = []
    seen_signatures = set()
    for item in plan:
        signature = json.dumps(
            {
                'task_spec': item.task_spec,
                'model_spec': item.model_spec,
                'training_spec': item.training_spec,
            },
            sort_keys=True,
            separators=(',', ':'),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduplicated.append(item)
    return [
        ExperimentPlanItem(
            task_index=index,
            run_name=item.run_name,
            model_variant=item.model_variant,
            training_variant=item.training_variant,
            task_recipe_name=item.task_recipe_name,
            task_label=item.task_label,
            task_variant_index=item.task_variant_index,
            task_variant_total=item.task_variant_total,
            component_specs=item.component_specs,
            schema_version=item.schema_version,
        )
        for index, item in enumerate(deduplicated, 1)
    ]


def _collect_suite_views_from_plan(
    plan: Sequence[ExperimentPlanItem],
) -> tuple[List[str], List[str], Dict[str, List[ModelVariant]], List[TrainingVariant]]:
    model_recipe_names: List[str] = []
    task_labels: List[str] = []
    model_variants_by_recipe: Dict[str, List[ModelVariant]] = {}
    seen_model_labels_by_recipe: Dict[str, set[str]] = {}
    training_variants: List[TrainingVariant] = []
    seen_training_labels: set[str] = set()

    for item in plan:
        if item.model_recipe_name not in model_recipe_names:
            model_recipe_names.append(item.model_recipe_name)

        task_label = item.task_label or item.task_recipe_name
        if task_label and task_label not in task_labels:
            task_labels.append(task_label)

        recipe_variants = model_variants_by_recipe.setdefault(item.model_recipe_name, [])
        seen_model_labels = seen_model_labels_by_recipe.setdefault(item.model_recipe_name, set())
        if item.model_label not in seen_model_labels:
            seen_model_labels.add(item.model_label)
            recipe_variants.append(item.model_variant)

        if item.training_label not in seen_training_labels:
            seen_training_labels.add(item.training_label)
            training_variants.append(item.training_variant)

    return model_recipe_names, task_labels, model_variants_by_recipe, training_variants


def build_experiment_suite(
    config_dir: Path,
    batch_size_override: Optional[int] = None,
    num_batches_override: Optional[int] = None,
    experiment_name: Optional[str] = None,
    run_names: Optional[Sequence[str]] = None,
    task_overrides: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    training_overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentSuite:
    """Load configs/v2 and prepare the full experiment suite in one place."""
    config_dir = Path(config_dir)
    if not experiment_name:
        raise ValueError('Component-based experiment suite requires an experiment_name from configs/v2/experiments.yaml')

    component_catalog = load_component_catalog(config_dir)
    component_data = build_component_experiment_data(
        config_dir=config_dir,
        experiment_name=experiment_name,
        batch_size_override=batch_size_override,
        num_batches_override=num_batches_override,
        default_training_config=DEFAULT_TRAINING_CONFIG,
    )

    model_variants_by_recipe = {
        recipe_name: [
            ModelVariant(
                recipe_name=variant['recipe_name'],
                label=variant['label'],
                spec=variant['spec'],
                index=variant['index'],
                total=variant['total'],
            )
            for variant in variants
        ]
        for recipe_name, variants in component_data['model_variants_by_recipe'].items()
    }
    training_variants = [
        TrainingVariant(
            recipe_name=variant['recipe_name'],
            label=variant['label'],
            spec=variant['spec'],
            index=variant['index'],
            total=variant['total'],
        )
        for variant in component_data['training_variants']
    ]
    plan = [
        ExperimentPlanItem(
            task_index=item['task_index'],
            run_name=item['run_name'],
            model_variant=ModelVariant(
                recipe_name=item['model_variant']['recipe_name'],
                label=item['model_variant']['label'],
                spec=item['model_variant']['spec'],
                index=item['model_variant']['index'],
                total=item['model_variant']['total'],
            ),
            training_variant=TrainingVariant(
                recipe_name=item['training_variant']['recipe_name'],
                label=item['training_variant']['label'],
                spec=item['training_variant']['spec'],
                index=item['training_variant']['index'],
                total=item['training_variant']['total'],
            ),
            task_recipe_name=item['task_recipe_name'],
            task_label=item['task_label'],
            task_variant_index=item.get('task_variant_index'),
            task_variant_total=item.get('task_variant_total'),
            component_specs=item.get('component_specs', {}),
        )
        for item in component_data['plan']
    ]

    requested_run_names = [item for item in (run_names or []) if item]
    if requested_run_names:
        requested = set(requested_run_names)
        plan = [item for item in plan if item.run_name in requested]
        if not plan:
            available = ', '.join(sorted(item['run_name'] for item in component_data['plan']))
            raise ValueError(
                f"Requested runs not found in experiment plan: {requested_run_names}. "
                f"Available runs: {available}"
            )

    def _set_nested_value(mapping: Dict[str, Any], path: str, value: Any) -> None:
        parts = [part for part in path.split('.') if part]
        if not parts:
            raise ValueError('Override path cannot be empty')
        current = mapping
        for part in parts[:-1]:
            next_value = current.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        current[parts[-1]] = value

    task_override_tokens = _build_override_tokens('tover_', task_overrides)
    model_override_tokens = _build_override_tokens('mo_', model_overrides)
    training_override_tokens = _build_override_tokens('to_', training_overrides)

    if task_overrides or model_overrides or training_overrides:
        overridden_plan: List[ExperimentPlanItem] = []
        for item in plan:
            task_raw_spec = deepcopy(item.component_specs.get('task', {}))
            model_raw_spec = deepcopy(item.component_specs.get('model', {}))
            training_raw_spec = deepcopy(item.component_specs.get('training_strategy', {}))
            task_class = get_task_class(task_raw_spec.get('type'))
            training_strategy_class = get_training_strategy_class(training_raw_spec.get('type'))

            for key, value in (task_overrides or {}).items():
                _apply_component_override(task_raw_spec, key, value)
            for key, value in (model_overrides or {}).items():
                _apply_component_override(model_raw_spec, key, value)
            for key, value in (training_overrides or {}).items():
                _apply_component_override(training_raw_spec, key, value)

            model_spec = task_class.compile_model_spec(task_raw_spec, model_raw_spec)
            training_spec = training_strategy_class.compile_runtime_spec(
                task_raw_spec,
                training_raw_spec,
                DEFAULT_TRAINING_CONFIG,
            )
            task_label = _append_tokens(item.task_label or item.task_recipe_name or '', task_override_tokens)
            model_label = _append_tokens(item.model_variant.label, model_override_tokens)
            training_label = _append_tokens(item.training_variant.label, training_override_tokens)
            run_name = _append_tokens(item.run_name, [*task_override_tokens, *model_override_tokens, *training_override_tokens])
            overridden_plan.append(
                ExperimentPlanItem(
                    task_index=item.task_index,
                    run_name=run_name,
                    model_variant=ModelVariant(
                        recipe_name=item.model_variant.recipe_name,
                        label=model_label,
                        spec=model_spec,
                        index=item.model_variant.index,
                        total=item.model_variant.total,
                    ),
                    training_variant=TrainingVariant(
                        recipe_name=item.training_variant.recipe_name,
                        label=training_label,
                        spec=training_spec,
                        index=item.training_variant.index,
                        total=item.training_variant.total,
                    ),
                    task_recipe_name=item.task_recipe_name,
                    task_label=task_label,
                    task_variant_index=item.task_variant_index,
                    task_variant_total=item.task_variant_total,
                    component_specs={
                        'task': task_raw_spec,
                        'model': model_raw_spec,
                        'training_strategy': training_raw_spec,
                    },
                )
            )
        plan = _deduplicate_plan_items(overridden_plan)

    model_recipe_names = component_data['model_recipe_names']
    task_labels = component_data['task_labels']

    if plan:
        model_recipe_names, task_labels, model_variants_by_recipe, training_variants = _collect_suite_views_from_plan(plan)

    return ExperimentSuite(
        catalog=ConfigCatalog(config_dir=config_dir, component_catalog=component_catalog),
        experiment_name=experiment_name,
        model_recipe_names=model_recipe_names,
        training_recipe_name=component_data['training_recipe_name'],
        model_variants_by_recipe=model_variants_by_recipe,
        training_variants=training_variants,
        missing_model_recipes=component_data['missing_model_recipes'],
        plan=plan,
        task_recipe_name=component_data['task_recipe_name'],
        task_labels=task_labels,
    )


def print_experiment_plan_summary(plan: Sequence[ExperimentPlanItem]) -> None:
    """Print a concise execution plan preview."""
    print(f"Experiment plan: {len(plan)} runs")
    for item in plan:
        print(
            f"  {item.task_index:>3}. {item.run_name} "
            f"[task={item.task_label or item.task_recipe_name}, model={item.model_recipe_name}, training={item.training_label}]"
        )


__all__ = [
    'DEFAULT_TRAINING_CONFIG',
    'ModelVariant',
    'TrainingVariant',
    'ExperimentPlanItem',
    'ConfigCatalog',
    'ExperimentSuite',
    'build_experiment_suite',
    'print_experiment_plan_summary',
]
