"""Post-training evaluation/export/plot orchestration."""

import json
from pathlib import Path

from .evaluation_workflow import evaluate_models_programmatic, resolve_evaluation_output_dir
from .export_workflow import export_runs_to_onnx
from .plotting_workflow import generate_plots_programmatic
from .types import PostprocessSummary


def run_post_training_pipeline(training_summary):
    """Run optional ONNX export, evaluation, and plotting after training."""
    request = training_summary.request
    summary = PostprocessSummary()

    if request.export_onnx_after_train and training_summary.results_sorted:
        print(f"\n{'='*80}")
        print("📦 ONNX Export")
        print(f"{'='*80}")

        export_run_names = (
            [training_summary.results_sorted[0]['run_name']]
            if request.onnx_export_selection == 'best'
            else [result['run_name'] for result in training_summary.results_sorted]
        )
        onnx_output_dir = (
            Path(request.onnx_output_dir)
            if request.onnx_output_dir
            else training_summary.experiment_output_dir / 'onnx_exports'
        )
        summary.onnx_manifests = export_runs_to_onnx(
            output_root=onnx_output_dir,
            exp_dir=training_summary.experiment_output_dir,
            runs=','.join(export_run_names),
            opset_version=request.onnx_opset,
            batch_size=request.onnx_batch_size,
            dynamic_batch=request.onnx_dynamic_batch,
            validate=request.onnx_validate,
        )
        print(f"✓ ONNX export completed: {onnx_output_dir}")
        for manifest in summary.onnx_manifests:
            print(f"  - {manifest['run_name']}: {manifest['onnx_path']}")

    if request.eval_after_train:
        print(f"\n{'='*80}")
        print("📊 Post-Training Evaluation")
        print(f"{'='*80}")

        summary.evaluation_output_dir = resolve_evaluation_output_dir(
            exp_dir=training_summary.experiment_output_dir,
            model_dirs=[training_summary.experiment_output_dir / result['run_name'] for result in training_summary.results_sorted],
        )
        summary.evaluation_results = evaluate_models_programmatic(
            exp_dir=training_summary.experiment_output_dir,
            output_dir=summary.evaluation_output_dir,
            snr_range=request.eval_snr_range,
            tdl_list=request.eval_tdl.split(','),
            num_batches=request.eval_num_batches,
            batch_size=request.eval_batch_size,
            device=training_summary.device,
            use_amp=False,
            compile=training_summary.device.type == 'cuda',
        )
        print(f"\n✓ Evaluation completed!")
        print(f"  Results saved to: {summary.evaluation_output_dir}")

        if request.plot_after_eval:
            eval_json_path = summary.evaluation_output_dir / 'evaluation_results.json'
            if not eval_json_path.exists():
                print(f"\n⚠️  Skipping plot generation: evaluation results not found")
            else:
                with open(eval_json_path, 'r', encoding='utf-8') as input_file:
                    eval_data = json.load(input_file)

                if not eval_data.get('models'):
                    print(f"\n⚠️  Skipping plot generation: no models evaluated successfully")
                else:
                    print(f"\n{'='*80}")
                    print("📈 Generating Plots")
                    print(f"{'='*80}")
                    summary.plot_output_dir = summary.evaluation_output_dir / 'plots'
                    summary.generated_plots = generate_plots_programmatic(
                        eval_results_path=summary.evaluation_output_dir,
                        output_dir=summary.plot_output_dir,
                    )
                    print(f"\n✓ Plots generated!")
                    print(f"  Saved to: {summary.plot_output_dir}")

    return summary
