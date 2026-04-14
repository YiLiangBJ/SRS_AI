"""Training workflow that keeps the CLI thin."""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from models import create_model, list_models
from training import Trainer
from utils import (
    build_experiment_suite,
    default_refactor_experiments_root,
    get_device,
    print_device_info,
    parse_snr_config,
    print_experiment_plan_summary,
    print_search_space_summary,
    TrainingProgressTracker,
    build_model_artifact_spec,
    build_training_artifact_spec,
    build_run_metadata,
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
    print(f"  Training recipe: {suite.training_recipe_name} ({len(suite.training_variants)} variants)")
    print(f"  Model recipes: {suite.model_recipe_names}")
    print(f"  Available models: {list_models()}")
    print(f"  Planned runs: {len(suite.plan)}")
    print()

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
    training_spec = experiment.training_spec
    model_spec = experiment.model_spec
    training_label = experiment.training_label
    model_recipe_name = experiment.model_recipe_name

    batch_size = training_spec['batch_size']
    num_batches = training_spec['num_batches']
    learning_rate = training_spec['learning_rate']
    loss_type = training_spec['loss_type']
    snr_config_dict = training_spec['snr_config']
    tdl_config = training_spec['tdl_config']
    print_interval = training_spec['print_interval']
    val_interval = training_spec.get('validation_interval')
    validation_batches = training_spec.get('validation_batches', 4)
    early_stop_loss = training_spec.get('early_stop_loss')
    patience = training_spec['patience']
    keep_last_n = training_spec['keep_last_n_checkpoints']
    scheduler_config = training_spec.get('lr_scheduler')
    snr_config = parse_snr_config(snr_config_dict)

    previous_training_label, previous_model_recipe_name = previous_labels

    if len(suite.training_variants) > 1 and training_label != previous_training_label:
        print(f"\n{'='*80}")
        print(f"Training Variant {experiment.training_index}/{experiment.training_total}: {training_label}")
        print(f"{'='*80}")
        print(f"  Loss type: {loss_type}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  SNR: {snr_config}")
        print()

    if model_recipe_name != previous_model_recipe_name or training_label != previous_training_label:
        print(f"\n{'='*80}")
        print(f"Model: {model_recipe_name}")
        if len(suite.training_variants) > 1:
            print(f"Training: {training_label}")
        print(f"{'='*80}\n")

    print(f"\n{'─'*80}")
    if experiment.model_total > 1:
        print(f"Model Variant {experiment.model_index}/{experiment.model_total} of {model_recipe_name}")
    if len(suite.training_variants) > 1:
        print(f"Training Variant: {training_label}")
    print(f"{'─'*80}\n")

    run_name = experiment.run_name
    progress_tracker.start_task(run_name, experiment.task_index)
    print(f"Run: {run_name}")

    model_type = model_spec['model_type']
    model_params = {key: value for key, value in model_spec.items() if key != 'model_type'}
    print(f"  Model type: {model_type}")
    print(f"  Parameters: {model_params}")

    model = create_model(model_type, model_spec)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"  Total parameters: {num_params:,}")
    print()

    experiment_dir = Path(request.save_dir) / run_name
    tensorboard_dir = experiment_dir / 'tensorboard'

    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
        loss_type=loss_type,
        device=device,
        use_amp=request.use_amp,
        compile_model=request.compile_model,
        tensorboard_dir=tensorboard_dir,
        scheduler_config=scheduler_config,
    )

    start_time = time.time()
    save_interval = training_spec.get('save_interval')
    if save_interval is None and num_batches >= 1000:
        save_interval = max(1000, num_batches // 20)
        print(f"  💾 Auto checkpoint: every {save_interval} batches (~{num_batches // save_interval} saves)")
    elif save_interval:
        print(f"  💾 Manual checkpoint: every {save_interval} batches (~{num_batches // save_interval} saves)")

    losses = trainer.train(
        num_batches=num_batches,
        batch_size=batch_size,
        snr_config=snr_config,
        pos_values=model_spec['pos_values'],
        tdl_config=tdl_config,
        seq_len=model_spec['seq_len'],
        print_interval=print_interval,
        val_interval=val_interval,
        validation_batches=validation_batches,
        early_stop_loss=early_stop_loss,
        patience=patience,
        progress_tracker=progress_tracker,
        save_interval=save_interval,
        save_dir=experiment_dir if save_interval is not None else None,
        keep_last_n=keep_last_n,
    )
    training_duration = time.time() - start_time

    print("\n" + "─" * 80)
    print("Final Evaluation")
    print("─" * 80)

    eval_snr = (
        (snr_config.min_snr + snr_config.max_snr) / 2
        if snr_config.config_type == 'range'
        else float(np.mean(snr_config.snr_values))
    )
    eval_results = trainer.evaluate(
        batch_size=200,
        snr_db=eval_snr,
        pos_values=model_spec['pos_values'],
        tdl_config=tdl_config,
    )

    print(f"  NMSE: {eval_results['nmse']:.6f} ({eval_results['nmse_db']:.2f} dB)")
    print(f"  Per-port NMSE (dB): {eval_results['per_port_nmse_db']}")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_spec_dict = build_model_artifact_spec(model_spec, num_params=num_params)
    training_spec_dict = build_training_artifact_spec({
        'loss_type': loss_type,
        'learning_rate': learning_rate,
        'num_batches': num_batches,
        'batch_size': batch_size,
        'snr_config': snr_config_dict,
        'tdl_config': tdl_config,
        'print_interval': print_interval,
        'validation_interval': val_interval,
        'validation_batches': validation_batches,
        'early_stop_loss': early_stop_loss,
        'patience': patience,
        'keep_last_n_checkpoints': keep_last_n,
        'save_interval': save_interval,
        'lr_scheduler': scheduler_config,
    })
    metadata_dict = build_run_metadata(
        experiment_name=suite.experiment_name,
        model_recipe_name=model_recipe_name,
        model_label=experiment.model_label,
        run_name=run_name,
        training_recipe_name=experiment.training_recipe_name,
        training_label=training_label,
        training_duration=training_duration,
    )

    trainer.save_checkpoint(
        experiment_dir / 'model.pth',
        additional_info={
            'model_spec': model_spec_dict,
            'training_spec': training_spec_dict,
            'metadata': metadata_dict,
            'eval_results': eval_results,
        },
    )
    save_run_config(
        run_dir=experiment_dir,
        model_spec=model_spec_dict,
        training_spec=training_spec_dict,
        metadata=metadata_dict,
    )
    print(f"✓ Model saved to: {experiment_dir}")

    result = {
        'model_recipe_name': model_recipe_name,
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
        'avg_training_throughput': (batch_size * num_batches / training_duration) if training_duration > 0 else 0.0,
        'timing_breakdown': {
            'data_gen_time': trainer.data_gen_time,
            'forward_time': trainer.forward_time,
            'backward_time': trainer.backward_time,
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
