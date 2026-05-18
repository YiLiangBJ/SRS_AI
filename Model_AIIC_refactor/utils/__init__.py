"""
Utility functions and classes.
"""

from .device_utils import get_device, print_device_info
from .logging_utils import setup_logger
from .snr_sampler import SNRSampler
from .snr_config import SNRConfig, parse_snr_config
from .config_parser import (
    parse_config_variants,
    parse_search_space_value,
    expand_search_space,
    generate_config_name,
    print_search_space_summary,
    load_and_parse_config
)
from .experiment_plan import (
    DEFAULT_TRAINING_CONFIG,
    ModelVariant,
    TrainingVariant,
    ExperimentPlanItem,
    ConfigCatalog,
    ExperimentSuite,
    build_experiment_suite,
    print_experiment_plan_summary,
)
from .progress_tracker import TrainingProgressTracker
from .run_artifacts import (
    RunArtifacts,
    find_checkpoint_path,
    normalize_model_spec,
    build_model_artifact_spec,
    build_training_artifact_spec,
    build_run_metadata,
    save_run_config,
    load_run_artifacts,
    load_run_artifacts_from_checkpoint,
    load_trained_model_from_run,
    load_trained_model_from_checkpoint,
    build_dummy_input,
)
from .model_flow import (
    generate_model_flow_spec,
    render_model_flow_markdown,
    save_model_flow_artifacts,
)
from .model_complexity import (
    generate_model_complexity_spec,
    render_model_complexity_markdown,
    save_model_complexity_artifacts,
)
from .checkpoint_resume import (
    compare_model_specs,
    format_model_spec_mismatches,
    load_initial_checkpoint_state,
)
from .run_selection import (
    split_csv_arg,
    resolve_existing_path,
    default_refactor_experiments_root,
    discover_run_dirs,
    resolve_run_selection,
)
from .plot_style import (
    bottom_legend_figure_size,
    color_for_index,
    linestyle_for_index,
    marker_for_index,
    style_for_series,
    outside_legend_figure_size,
    panel_figure_size,
    place_legend_below,
    place_legend_outside_right,
    max_legend_items,
)

__all__ = [
    'get_device',
    'print_device_info',
    'setup_logger',
    'SNRSampler',
    'SNRConfig',
    'parse_snr_config',
    'parse_config_variants',
    'parse_search_space_value',
    'expand_search_space',
    'generate_config_name',
    'print_search_space_summary',
    'load_and_parse_config',
    'DEFAULT_TRAINING_CONFIG',
    'ModelVariant',
    'TrainingVariant',
    'ExperimentPlanItem',
    'ConfigCatalog',
    'ExperimentSuite',
    'build_experiment_suite',
    'print_experiment_plan_summary',
    'TrainingProgressTracker',
    'RunArtifacts',
    'find_checkpoint_path',
    'normalize_model_spec',
    'build_model_artifact_spec',
    'build_training_artifact_spec',
    'build_run_metadata',
    'save_run_config',
    'load_run_artifacts',
    'load_run_artifacts_from_checkpoint',
    'load_trained_model_from_run',
    'load_trained_model_from_checkpoint',
    'build_dummy_input',
    'generate_model_flow_spec',
    'render_model_flow_markdown',
    'save_model_flow_artifacts',
    'generate_model_complexity_spec',
    'render_model_complexity_markdown',
    'save_model_complexity_artifacts',
    'compare_model_specs',
    'format_model_spec_mismatches',
    'load_initial_checkpoint_state',
    'split_csv_arg',
    'resolve_existing_path',
    'default_refactor_experiments_root',
    'discover_run_dirs',
    'resolve_run_selection',
    'color_for_index',
    'bottom_legend_figure_size',
    'linestyle_for_index',
    'marker_for_index',
    'style_for_series',
    'outside_legend_figure_size',
    'panel_figure_size',
    'place_legend_below',
    'place_legend_outside_right',
    'max_legend_items',
]
