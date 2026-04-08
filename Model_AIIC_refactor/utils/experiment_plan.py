"""Helpers for loading configs and building experiment plans."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .config_parser import generate_config_name, parse_config_variants


DEFAULT_TRAINING_CONFIG = {
    'batch_size': 2048,
    'num_batches': 10000,
    'learning_rate': 0.01,
    'loss_type': 'nmse',
    'snr_config': {'type': 'range', 'min': 0, 'max': 30},
    'tdl_config': 'A-30',
    'print_interval': 100,
    'patience': 3,
    'keep_last_n_checkpoints': 2,
}

TRAINING_NAME_ALIASES = {
    'loss_type': 'loss',
    'learning_rate': 'lr',
    'batch_size': 'bs',
    'num_batches': 'nb',
    'tdl_config': 'tdl',
    'save_interval': 'save',
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
    """A fully resolved training configuration variant."""

    recipe_name: str
    label: str
    spec: Dict[str, Any]
    index: int
    total: int


@dataclass(frozen=True)
class ExperimentPlanItem:
    """A single executable model-training combination."""

    task_index: int
    run_name: str
    model_variant: ModelVariant
    training_variant: TrainingVariant

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


@dataclass(frozen=True)
class ConfigCatalog:
    """Loaded raw configuration files for the project."""

    config_dir: Path
    model_configs: Dict[str, Any]
    training_configs: Dict[str, Any]
    experiments: Dict[str, Any]


@dataclass(frozen=True)
class ExperimentSuite:
    """All prepared state needed to execute a training sweep."""

    catalog: ConfigCatalog
    experiment_name: Optional[str]
    model_recipe_names: List[str]
    training_recipe_name: str
    model_variants_by_recipe: Dict[str, List[ModelVariant]]
    training_variants: List[TrainingVariant]
    missing_model_recipes: List[str]
    plan: List[ExperimentPlanItem]


def _format_name_value(value: Any) -> str:
    """Format a config value into a compact, file-system-friendly token."""
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


def _infer_varying_keys(
    configs: Sequence[Mapping[str, Any]],
    ignored_keys: Optional[Sequence[str]] = None
) -> List[str]:
    """Infer which keys differ across expanded variants."""
    if not configs:
        return []

    ignored = set(ignored_keys or [])
    varying_keys = []

    for key in configs[0].keys():
        if key in ignored:
            continue

        values = [config.get(key) for config in configs]
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)

        if len(unique_values) > 1:
            varying_keys.append(key)

    return varying_keys


def _build_variant_name(
    base_name: str,
    config: Mapping[str, Any],
    varying_keys: Sequence[str],
    variant_index: int,
    total_variants: int,
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """Build a readable variant name from the keys that intentionally vary."""
    if total_variants == 1:
        return base_name

    aliases = aliases or {}
    name_parts = [base_name]
    for key in varying_keys:
        if key not in config:
            continue
        label = aliases.get(key, key)
        name_parts.append(f"{label}{_format_name_value(config[key])}")

    if len(name_parts) == 1:
        name_parts.append(f"v{variant_index}")

    return '_'.join(name_parts)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def load_config_catalog(config_dir: Path) -> ConfigCatalog:
    """Load model and training config files from a single directory."""
    config_dir = Path(config_dir)
    experiments_path = config_dir / 'experiments.yaml'
    experiments = load_yaml_config(experiments_path) if experiments_path.exists() else {}
    return ConfigCatalog(
        config_dir=config_dir,
        model_configs=load_yaml_config(config_dir / 'model_configs.yaml'),
        training_configs=load_yaml_config(config_dir / 'training_configs.yaml'),
        experiments=experiments.get('experiments', experiments),
    )


def resolve_experiment_definition(
    catalog: ConfigCatalog,
    experiment_name: str,
) -> Dict[str, Any]:
    """Resolve an experiment entry from experiments.yaml."""
    experiment_definition = catalog.experiments.get(experiment_name)
    if not experiment_definition:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return experiment_definition


def prepare_model_config_variants(
    all_model_configs: Mapping[str, Any],
    model_recipe_names: Sequence[str]
) -> Tuple[Dict[str, List[ModelVariant]], List[str]]:
    """Merge common model config and expand all requested model variants once."""
    common_config = all_model_configs.get('common', {})
    available_model_configs = all_model_configs.get('models', {})

    model_variants_by_recipe: Dict[str, List[ModelVariant]] = {}
    missing_model_recipes = []

    for model_recipe_name in model_recipe_names:
        model_config = available_model_configs.get(model_recipe_name)
        if not model_config:
            missing_model_recipes.append(model_recipe_name)
            continue

        full_model_config = {**common_config, **model_config}
        parsed_configs = parse_config_variants(full_model_config)
        model_variants_by_recipe[model_recipe_name] = [
            ModelVariant(
                recipe_name=model_recipe_name,
                label=generate_config_name(config, model_recipe_name),
                spec=config,
                index=index,
                total=len(parsed_configs),
            )
            for index, config in enumerate(parsed_configs, 1)
        ]

    return model_variants_by_recipe, missing_model_recipes


def prepare_training_config_variants(
    all_training_configs: Mapping[str, Any],
    training_recipe_name: str,
    batch_size_override: Optional[int] = None,
    num_batches_override: Optional[int] = None,
) -> List[TrainingVariant]:
    """Expand and normalize training config variants with optional CLI overrides."""
    training_config_raw = all_training_configs.get(training_recipe_name, {})
    if not training_config_raw:
        raise ValueError(f"Training config '{training_recipe_name}' not found")

    parsed_training_configs = parse_config_variants(training_config_raw)
    normalized_configs = []
    for config in parsed_training_configs:
        normalized_config = {**DEFAULT_TRAINING_CONFIG, **config}
        if batch_size_override is not None:
            normalized_config['batch_size'] = batch_size_override
        if num_batches_override is not None:
            normalized_config['num_batches'] = num_batches_override
        normalized_configs.append(normalized_config)

    raw_search_space = training_config_raw.get('search_space', {})
    varying_keys = list(raw_search_space.keys()) or _infer_varying_keys(
        normalized_configs,
        ignored_keys=['snr_config', 'tdl_config', 'validation_interval', 'early_stop_loss', 'patience']
    )

    return [
        TrainingVariant(
            recipe_name=training_recipe_name,
            label=_build_variant_name(
                training_recipe_name,
                config,
                varying_keys,
                index,
                len(normalized_configs),
                aliases=TRAINING_NAME_ALIASES,
            ),
            spec=config,
            index=index,
            total=len(normalized_configs),
        )
        for index, config in enumerate(normalized_configs, 1)
    ]


def build_experiment_plan(
    model_recipe_names: Sequence[str],
    model_variants_by_recipe: Mapping[str, Sequence[ModelVariant]],
    training_variants: Sequence[TrainingVariant],
) -> List[ExperimentPlanItem]:
    """Build the full training matrix as an ordered execution plan."""
    experiment_plan: List[ExperimentPlanItem] = []
    task_index = 0
    include_training_suffix = len(training_variants) > 1

    for training_variant in training_variants:
        for model_recipe_name in model_recipe_names:
            model_variants = model_variants_by_recipe.get(model_recipe_name, [])
            for model_variant in model_variants:
                task_index += 1
                run_name = model_variant.label
                if include_training_suffix:
                    run_name = f"{run_name}_{training_variant.label}"

                experiment_plan.append(
                    ExperimentPlanItem(
                        task_index=task_index,
                        run_name=run_name,
                        model_variant=model_variant,
                        training_variant=training_variant,
                    )
                )

    return experiment_plan


def build_experiment_suite(
    config_dir: Path,
    model_recipe_names: Optional[Sequence[str]] = None,
    training_recipe_name: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    num_batches_override: Optional[int] = None,
    experiment_name: Optional[str] = None,
) -> ExperimentSuite:
    """Load configs and prepare the full experiment suite in one place."""
    catalog = load_config_catalog(config_dir)
    if experiment_name:
        experiment_definition = resolve_experiment_definition(catalog, experiment_name)
        model_recipe_names = experiment_definition.get('model_configs', [])
        training_recipe_name = experiment_definition.get('training_config')
        if batch_size_override is None:
            batch_size_override = experiment_definition.get('batch_size')
        if num_batches_override is None:
            num_batches_override = experiment_definition.get('num_batches')

    if not model_recipe_names:
        raise ValueError('No model configs specified for experiment suite')
    if not training_recipe_name:
        raise ValueError('No training config specified for experiment suite')

    model_recipe_names = [name.strip() for name in model_recipe_names if name.strip()]
    model_variants_by_recipe, missing_model_recipes = prepare_model_config_variants(
        catalog.model_configs,
        model_recipe_names,
    )
    training_variants = prepare_training_config_variants(
        catalog.training_configs,
        training_recipe_name,
        batch_size_override=batch_size_override,
        num_batches_override=num_batches_override,
    )
    plan = build_experiment_plan(
        model_recipe_names,
        model_variants_by_recipe,
        training_variants,
    )

    return ExperimentSuite(
        catalog=catalog,
        experiment_name=experiment_name,
        model_recipe_names=model_recipe_names,
        training_recipe_name=training_recipe_name,
        model_variants_by_recipe=model_variants_by_recipe,
        training_variants=training_variants,
        missing_model_recipes=missing_model_recipes,
        plan=plan,
    )


def print_experiment_plan_summary(plan: Sequence[ExperimentPlanItem]) -> None:
    """Print a concise execution plan preview."""
    print(f"Experiment plan: {len(plan)} runs")
    for item in plan:
        print(
            f"  {item.task_index:>3}. {item.run_name} "
            f"[model={item.model_recipe_name}, training={item.training_label}]"
        )


__all__ = [
    'DEFAULT_TRAINING_CONFIG',
    'ModelVariant',
    'TrainingVariant',
    'ExperimentPlanItem',
    'ConfigCatalog',
    'ExperimentSuite',
    'load_yaml_config',
    'load_config_catalog',
    'resolve_experiment_definition',
    'prepare_model_config_variants',
    'prepare_training_config_variants',
    'build_experiment_plan',
    'build_experiment_suite',
    'print_experiment_plan_summary',
]