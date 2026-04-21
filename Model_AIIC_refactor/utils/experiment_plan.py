"""Helpers for building v2 component-based experiment plans."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .component_experiment_plan import build_component_experiment_data, load_component_catalog


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


def build_experiment_suite(
    config_dir: Path,
    batch_size_override: Optional[int] = None,
    num_batches_override: Optional[int] = None,
    experiment_name: Optional[str] = None,
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

    return ExperimentSuite(
        catalog=ConfigCatalog(config_dir=config_dir, component_catalog=component_catalog),
        experiment_name=experiment_name,
        model_recipe_names=component_data['model_recipe_names'],
        training_recipe_name=component_data['training_recipe_name'],
        model_variants_by_recipe=model_variants_by_recipe,
        training_variants=training_variants,
        missing_model_recipes=component_data['missing_model_recipes'],
        plan=plan,
        task_recipe_name=component_data['task_recipe_name'],
        task_labels=component_data['task_labels'],
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
