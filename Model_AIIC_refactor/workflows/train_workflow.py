"""Training workflow that keeps the CLI thin."""

import time
from datetime import datetime
from pathlib import Path

from models import create_model, list_models
from tasks import create_task
from training_strategies import create_training_strategy
from utils import (
    build_experiment_suite,
    default_refactor_experiments_root,
    get_device,
    print_device_info,
    print_experiment_plan_summary,
    print_search_space_summary,
    TrainingProgressTracker,
    build_model_artifact_spec,
    build_training_artifact_spec,
    build_run_metadata,
    load_initial_checkpoint_state,
    save_model_flow_artifacts,
    save_run_config,
)

from .postprocess_workflow import run_post_training_pipeline
from .reporting import generate_training_report
from .types import TrainingSummary


def _resolve_experiment_output_dir(base_save_dir: str, suite) -> tuple[str, Path]:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = Path(base_save_dir)
    experiment_key = suite.experiment_name or f"{suite.model_recipe_names[0]}_{suite.training_recipe_name}"
    experiment_name = f"{timestamp}_{experiment_key}"
    return experiment_name, base_save_path / experiment_name


def _print_suite_overview(suite, experiment_name: str, experiment_output_dir: Path, device):
    print(f"\n{'='*80}")
    print(f"🚀 Experiment: {experiment_name}")
    print(f"{'='*80}")
    print(f"   Save directory: {experiment_output_dir}")
    print(f"{'='*80}\n")

    print("=" * 80)
    print("Channel Separator Training (Refactored)")
    print("=" * 80)
    print_device_info(device)
    print()

    if suite.missing_model_recipes:
        missing_configs_str = ', '.join(suite.missing_model_recipes)
        print(f"✗ Model recipe(s) not found: {missing_configs_str}")

    if not suite.model_variants_by_recipe:
        raise ValueError(f"No valid model recipes found in experiment: {suite.experiment_name}")

    print("Training plan:")
    print(f"  Experiment: {suite.experiment_name}")
    if suite.task_recipe_name:
        print(f"  Task recipe: {suite.task_recipe_name} ({max(len(suite.task_labels), 1)} variants)")
    if suite.schema_version == 'v2':
        print(f"  Training strategy: {suite.training_recipe_name} ({len(suite.training_variants)} variants)")
    else:
        print(f"  Training recipe: {suite.training_recipe_name} ({len(suite.training_variants)} variants)")
    print(f"  Model recipes: {suite.model_recipe_names}")
    print(f"  Available models: {list_models()}")
    print(f"  Planned runs: {len(suite.plan)}")
    print()

    if suite.schema_version == 'v2':
        if len(suite.task_labels) > 1:
            print(f"Task variants: {len(suite.task_labels)}")
            for label in suite.task_labels:
                print(f"  - {label}")
            print()

        if len(suite.training_variants) > 1:
            print(f"Training strategy variants: {len(suite.training_variants)}")
            for variant in suite.training_variants:
                print(f"  - {variant.label}")
            print()

        for model_recipe_name in suite.model_recipe_names:
            model_variants = suite.model_variants_by_recipe.get(model_recipe_name)
            if model_variants is None:
                continue
            print(f"Model recipe: {model_recipe_name} ({len(model_variants)} variants)")
            for variant in model_variants:
                print(f"  - {variant.label}")
            print()

        print_experiment_plan_summary(suite.plan)
        print()
        return

    if len(suite.training_variants) > 1:
        print(f"Training search space: {len(suite.training_variants)} variants")
        print_search_space_summary([variant.spec for variant in suite.training_variants], suite.training_recipe_name)
        print()

    for model_recipe_name in suite.model_recipe_names:
        model_variants = suite.model_variants_by_recipe.get(model_recipe_name)
        if model_variants is None:
            continue
        print_search_space_summary([variant.spec for variant in model_variants], model_recipe_name)
        print()

    print_experiment_plan_summary(suite.plan)
    print()


def _print_training_summary(training_summary: TrainingSummary):
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}\n")

    print(f"Total runs trained: {len(training_summary.results)}")
    print(f"Start time: {training_summary.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {training_summary.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {training_summary.total_duration/3600:.2f} hours ({training_summary.total_duration:.1f}s)")
    print()

    for index, result in enumerate(training_summary.results_sorted, 1):
        print(f"{index}. {result['run_name']}:")
        print(f"   Final loss: {result['final_loss']:.6f}")
        print(f"   Min loss: {result['min_loss']:.6f}")
        print(f"   Eval NMSE: {result['eval_nmse_db']:.2f} dB")
        print(f"   Parameters: {result['num_params']:,}")
        print(f"   Duration: {result['training_duration']:.1f}s")
        print()

    if training_summary.results_sorted:
        best = training_summary.results_sorted[0]
        print(f"🏆 Best run: {best['run_name']}")
        print(f"   NMSE: {best['eval_nmse_db']:.2f} dB")
        print()


def _run_single_plan_item(experiment, suite, request, device, progress_tracker, previous_labels):
    task = create_task(experiment.task_spec)
    training_strategy = create_training_strategy(experiment.training_strategy_spec)
    training_spec = experiment.training_spec
    model_spec = experiment.model_spec
    training_label = experiment.training_label
    model_recipe_name = experiment.model_recipe_name
    task_label = experiment.task_label or experiment.task_recipe_name

    batch_size = training_spec['batch_size']
    num_batches = training_spec['num_batches']

    previous_training_label, previous_model_recipe_name = previous_labels

    if len(suite.training_variants) > 1 and training_label != previous_training_label:
        print(f"\n{'='*80}")
        print(f"Training Variant {experiment.training_index}/{experiment.training_total}: {training_label}")
        print(f"{'='*80}")
        print(f"  Strategy type: {training_strategy.type}")
        print(f"  Runtime params: {training_spec}")
        print()

    if model_recipe_name != previous_model_recipe_name or training_label != previous_training_label:
        print(f"\n{'='*80}")
        if experiment.task_recipe_name:
            print(f"Task: {experiment.task_recipe_name}")
        print(f"Model: {model_recipe_name}")
        if len(suite.training_variants) > 1:
            print(f"Training: {training_label}")
        print(f"{'='*80}\n")

    print(f"\n{'─'*80}")
    if experiment.model_total > 1:
        print(f"Model Variant {experiment.model_index}/{experiment.model_total} of {model_recipe_name}")
    if len(suite.training_variants) > 1:
        print(f"Training Variant: {training_label}")
    if experiment.task_recipe_name and (experiment.task_variant_total or 0) > 1:
        print(f"Task Variant: {task_label}")
    print(f"{'─'*80}\n")

    run_name = experiment.run_name
    progress_tracker.start_task(run_name, experiment.task_index)
    print(f"Run: {run_name}")

    model_type = model_spec['model_type']
    model_params = {key: value for key, value in model_spec.items() if key != 'model_type'}
    print(f"  Model type: {model_type}")
    print(f"  Parameters: {model_params}")

    model = create_model(model_type, model_spec)
    init_checkpoint_artifacts = None
    if request.init_checkpoint:
        state_dict, init_checkpoint_artifacts = load_initial_checkpoint_state(
            checkpoint_path=request.init_checkpoint,
            expected_model_spec=model_spec,
            device='cpu',
        )
        model.load_state_dict(state_dict)
        print(f"  Initialized from checkpoint: {init_checkpoint_artifacts.checkpoint_path}")
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"  Total parameters: {num_params:,}")
    print()

    experiment_dir = Path(request.save_dir) / run_name
    tensorboard_dir = experiment_dir / 'tensorboard'

    trainer = training_strategy.create_trainer(
        model=model,
        training_spec=training_spec,
        request=request,
        device=device,
        tensorboard_dir=tensorboard_dir,
    )

    start_time = time.time()
    losses = training_strategy.run(
        trainer=trainer,
        task=task,
        model_spec=model_spec,
        training_spec=training_spec,
        experiment_dir=experiment_dir,
        progress_tracker=progress_tracker,
    )
    training_duration = time.time() - start_time

    print("\n" + "─" * 80)
    print("Final Evaluation")
    print("─" * 80)

    eval_results = training_strategy.final_evaluate(
        trainer=trainer,
        task=task,
        model_spec=model_spec,
        training_spec=training_spec,
    )
    effective_trainer = getattr(training_strategy, '_final_trainer', None) or trainer

    print(f"  NMSE: {eval_results['nmse']:.6f} ({eval_results['nmse_db']:.2f} dB)")
    print(f"  Per-port NMSE (dB): {eval_results['per_port_nmse_db']}")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_spec_dict = build_model_artifact_spec(model_spec, num_params=num_params)
    training_spec_dict = build_training_artifact_spec(training_spec)
    metadata_dict = build_run_metadata(
        experiment_name=suite.experiment_name,
        model_recipe_name=model_recipe_name,
        model_label=experiment.model_label,
        run_name=run_name,
        training_recipe_name=experiment.training_recipe_name,
        training_label=training_label,
        training_duration=training_duration,
        task_recipe_name=experiment.task_recipe_name,
        task_label=task_label,
        schema_version=suite.schema_version,
        init_checkpoint_path=str(init_checkpoint_artifacts.checkpoint_path) if init_checkpoint_artifacts else None,
    )

    additional_info = {
        'model_spec': model_spec_dict,
        'training_spec': training_spec_dict,
        'metadata': metadata_dict,
        'eval_results': eval_results,
    }
    if experiment.component_specs:
        additional_info['component_specs'] = experiment.component_specs

    effective_trainer.save_checkpoint(
        experiment_dir / 'model.pth',
        additional_info=additional_info,
    )
    save_run_config(
        run_dir=experiment_dir,
        model_spec=model_spec_dict,
        training_spec=training_spec_dict,
        metadata=metadata_dict,
        component_specs=experiment.component_specs,
    )
    flow_artifacts = save_model_flow_artifacts(
        output_dir=experiment_dir,
        model_spec=model_spec_dict,
        component_specs=experiment.component_specs,
    )
    print(f"✓ Model saved to: {experiment_dir}")

    result = {
        'task_label': task_label,
        'model_recipe_name': model_recipe_name,
        'model_label': experiment.model_label,
        'run_name': run_name,
        'training_label': training_label,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'samples_processed': batch_size * num_batches,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'eval_nmse_db': eval_results['nmse_db'],
        'training_duration': training_duration,
        'num_params': num_params,
        'init_checkpoint_path': str(init_checkpoint_artifacts.checkpoint_path) if init_checkpoint_artifacts else None,
        'stage_summaries': getattr(training_strategy, '_stage_summaries', []),
        'model_flow_markdown_path': flow_artifacts['markdown_path'],
        'avg_training_throughput': (batch_size * num_batches / training_duration) if training_duration > 0 else 0.0,
        'timing_breakdown': {
            'data_gen_time': effective_trainer.data_gen_time,
            'forward_time': effective_trainer.forward_time,
            'backward_time': effective_trainer.backward_time,
        },
    }
    progress_tracker.complete_task(result)
    return result, (training_label, model_recipe_name)


def run_training_experiment(request):
    """Run a named experiment through the full training workflow."""
    config_dir = Path(__file__).resolve().parent.parent / 'configs'
    if not request.save_dir:
        request.save_dir = str(default_refactor_experiments_root())
    suite = build_experiment_suite(
        config_dir=config_dir,
        batch_size_override=request.batch_size,
        num_batches_override=request.num_batches,
        experiment_name=request.experiment,
    )

    experiment_name, experiment_output_dir = _resolve_experiment_output_dir(request.save_dir, suite)
    request.save_dir = str(experiment_output_dir)
    device = get_device(request.device)
    if request.compile_model is None:
        request.compile_model = device.type == 'cuda'

    _print_suite_overview(suite, experiment_name, experiment_output_dir, device)

    training_summary = TrainingSummary(
        experiment_output_dir=experiment_output_dir,
        experiment_name=experiment_name,
        suite=suite,
        device=device,
        request=request,
        plan_only=request.plan_only,
    )
    if request.plan_only:
        print("✓ Plan generated. Exiting without training.")
        return training_summary

    training_summary.started_at = datetime.now()
    script_start_time = time.time()
    progress_tracker = TrainingProgressTracker(len(suite.plan), report_interval=300.0)

    previous_labels = (None, None)
    for experiment in suite.plan:
        result, previous_labels = _run_single_plan_item(
            experiment=experiment,
            suite=suite,
            request=request,
            device=device,
            progress_tracker=progress_tracker,
            previous_labels=previous_labels,
        )
        training_summary.results.append(result)

    training_summary.ended_at = datetime.now()
    training_summary.total_duration = time.time() - script_start_time
    training_summary.results_sorted = sorted(training_summary.results, key=lambda item: item['eval_nmse_db'])

    _print_training_summary(training_summary)
    training_summary.report_path = training_summary.experiment_output_dir / 'TRAINING_REPORT.md'
    generate_training_report(
        report_path=training_summary.report_path,
        results=training_summary.results_sorted,
        training_recipe_name=suite.training_recipe_name,
        start_time=training_summary.started_at,
        end_time=training_summary.ended_at,
        total_duration=training_summary.total_duration,
        device=device,
    )
    print(f"✓ Training report saved: {training_summary.report_path}")

    training_summary.postprocess = run_post_training_pipeline(training_summary)

    print(f"\n{'='*80}")
    print("🎉 Complete Pipeline Finished!")
    print(f"{'='*80}")
    print(f"  Training:   {training_summary.experiment_output_dir}")
    if training_summary.postprocess and training_summary.postprocess.evaluation_output_dir:
        print(f"  Evaluation: {training_summary.postprocess.evaluation_output_dir}")
        if training_summary.postprocess.plot_output_dir:
            print(f"  Plots:      {training_summary.postprocess.plot_output_dir}")
    if training_summary.postprocess and training_summary.postprocess.onnx_manifests:
        print(f"  ONNX:       per-run onnx_exports/ under selected run directories")
    if training_summary.postprocess and training_summary.postprocess.matlab_manifests:
        print(f"  Matlab:     per-run matlab_exports/ under selected run directories")
    print(f"{'='*80}\n")

    return training_summary
