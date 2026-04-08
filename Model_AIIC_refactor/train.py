"""
Simplified training script using the refactored modules.

Usage:
    # Preview an experiment plan
    python train.py --experiment quick_separator1 --plan_only

    # Run a named experiment
    python train.py --experiment compare_default_models
"""

import argparse
import yaml
from pathlib import Path
import time
import numpy as np
from datetime import datetime

# Import refactored modules
from models import create_model, list_models
from training import Trainer
from utils import (
    build_experiment_suite,
    get_device, print_device_info,
    parse_snr_config,
    print_experiment_plan_summary,
    print_search_space_summary,
    TrainingProgressTracker
)


def generate_training_report(
    report_path: Path,
    results: list,
    training_recipe_name: str,
    start_time: datetime,
    end_time: datetime,
    total_duration: float,
    device: str
):
    """Generate detailed training report in Markdown format"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Training Report\n\n")
        
        # Time Information
        f.write("## Time Information\n\n")
        f.write(f"- **Start Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **End Time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Duration**: {total_duration/3600:.2f} hours ({total_duration:.1f} seconds)\n")
        f.write(f"- **Device**: {device}\n\n")
        
        # Training Configuration
        f.write("## Training Configuration\n\n")
        f.write(f"- **Training Recipe**: {training_recipe_name}\n")
        f.write(f"- **Total Configurations**: {len(results)}\n\n")
        
        # Results Summary
        f.write("## Results Summary\n\n")
        f.write("| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |\n")
        f.write("|------|--------------|-----------|------------|-------------|\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"| {i} | `{result['run_name']}` | "
                   f"{result['eval_nmse_db']:.2f} | {result['num_params']:,} | "
                   f"{result['training_duration']:.1f} |\n")
        
        f.write("\n")
        
        # Best Configuration
        if results:
            best = results[0]
            f.write("## 🏆 Best Configuration\n\n")
            f.write(f"**Run**: `{best['run_name']}`\n\n")
            f.write(f"- **Eval NMSE**: {best['eval_nmse_db']:.2f} dB\n")
            f.write(f"- **Final Loss**: {best['final_loss']:.6f}\n")
            f.write(f"- **Min Loss**: {best['min_loss']:.6f}\n")
            f.write(f"- **Parameters**: {best['num_params']:,}\n")
            f.write(f"- **Training Duration**: {best['training_duration']:.1f}s\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"### {i}. {result['run_name']}\n\n")
            f.write(f"- **Model Recipe**: {result['model_recipe_name']}\n")
            f.write(f"- **Training Label**: {result['training_label']}\n")
            f.write(f"- **Evaluation NMSE**: {result['eval_nmse_db']:.2f} dB\n")
            f.write(f"- **Final Training Loss**: {result['final_loss']:.6f}\n")
            f.write(f"- **Minimum Training Loss**: {result['min_loss']:.6f}\n")
            f.write(f"- **Total Parameters**: {result['num_params']:,}\n")
            f.write(f"- **Training Duration**: {result['training_duration']:.1f}s "
                   f"({result['training_duration']/60:.1f} min)\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    parser = argparse.ArgumentParser(description='Train channel separator models')
    
    # Configuration files
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name from experiments.yaml')
    
    # Overrides
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Override number of batches')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)')
    parser.add_argument('--save_dir', type=str, default='./experiments_refactored',
                       help='Directory to save models')
    
    # ✅ NEW: Performance optimization options
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                       help='Disable mixed precision training (FP16)')
    parser.add_argument('--no-compile', dest='compile_model', action='store_false',
                       help='Disable model compilation (torch.compile)')
    # ✅ GPU默认启用compile，CPU默认禁用（将在后面根据device调整）
    parser.set_defaults(use_amp=False, compile_model=None)
    
    # ✅ NEW: Post-training workflow options
    parser.add_argument('--eval_after_train', action='store_true',
                       help='自动评估训练后的模型')
    parser.add_argument('--eval_snr_range', type=str, default='30:-3:0',
                       help='评估SNR范围 (格式: "start:step:end")')
    parser.add_argument('--eval_tdl', type=str, default='A-30,B-100,C-300',
                       help='评估TDL配置 (逗号分隔)')
    parser.add_argument('--eval_num_batches', type=int, default=100,
                       help='评估批次数')
    parser.add_argument('--eval_batch_size', type=int, default=2048,
                       help='评估批大小')
    parser.add_argument('--plot_after_eval', action='store_true',
                       help='评估后自动绘图')
    parser.add_argument('--plan_only', action='store_true',
                       help='仅解析并打印实验计划，不执行训练')
    
    args = parser.parse_args()
    
    config_dir = Path(__file__).parent / 'configs'
    suite = build_experiment_suite(
        config_dir=config_dir,
        batch_size_override=args.batch_size,
        num_batches_override=args.num_batches,
        experiment_name=args.experiment,
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ✅ Create timestamped experiment directory (avoid conflicts)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_dir = Path(args.save_dir)
    experiment_key = suite.experiment_name or f"{suite.model_recipe_names[0]}_{suite.training_recipe_name}"
    experiment_name = f"{timestamp}_{experiment_key}"
    args.save_dir = str(base_save_dir / experiment_name)

    print(f"\n{'='*80}")
    print(f"🚀 Experiment: {experiment_name}")
    print(f"{'='*80}")
    print(f"   Save directory: {args.save_dir}")
    print(f"   Timestamp: {timestamp}")
    print(f"{'='*80}\n")
    
    # Print device info
    device = get_device(args.device)
    
    # ✅ Auto-adjust compile based on device (unless explicitly disabled)
    if args.compile_model is None:
        args.compile_model = device.type == 'cuda'
    
    print("="*80)
    print("Channel Separator Training (Refactored)")
    print("="*80)
    print_device_info(device)
    print()
    
    if suite.missing_model_recipes:
        missing_configs_str = ', '.join(suite.missing_model_recipes)
        print(f"✗ Model config(s) not found: {missing_configs_str}")

    if not suite.model_variants_by_recipe:
        raise ValueError(f"No valid model recipes found in experiment: {args.experiment}")
    
    print(f"Training configurations:")
    print(f"  Experiment: {suite.experiment_name}")
    print(f"  Training recipe: {suite.training_recipe_name} ({len(suite.training_variants)} variants)")
    print(f"  Model recipes: {suite.model_recipe_names}")
    print(f"  Available models: {list_models()}")
    print(f"  Planned runs: {len(suite.plan)}")
    print()
    
    # Show training config search space (if any)
    if len(suite.training_variants) > 1:
        print(f"Training search space: {len(suite.training_variants)} configurations")
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

    if args.plan_only:
        print("✓ Plan generated. Exiting without training.")
        return
    
    # Record start time
    script_start_time = time.time()
    script_start_datetime = datetime.now()
    
    # Count total configurations first (model configs × training configs)
    total_configs = len(suite.plan)
    
    # Initialize progress tracker (report every 5 minutes)
    progress_tracker = TrainingProgressTracker(total_configs, report_interval=300.0)
    
    # Train each combination of (training × model)
    results = []
    previous_training_label = None
    previous_model_recipe_name = None
    
    for experiment in suite.plan:
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
        early_stop_loss = training_spec.get('early_stop_loss')
        patience = training_spec['patience']
        keep_last_n = training_spec['keep_last_n_checkpoints']
        snr_config = parse_snr_config(snr_config_dict)

        if len(suite.training_variants) > 1 and training_label != previous_training_label:
            print(f"\n{'='*80}")
            print(
                f"Training Config Variant {experiment.training_variant_index}/{experiment.training_variant_total}: "
                f"{training_label}"
            )
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
            print(
                f"Model Variant {experiment.model_index}/{experiment.model_total} "
                f"of {model_recipe_name}"
            )
        if len(suite.training_variants) > 1:
            print(f"Training Variant: {training_label}")
        print(f"{'─'*80}\n")

        config_instance_name = experiment.run_name

        progress_tracker.start_task(config_instance_name, experiment.task_index)
        print(f"Configuration: {config_instance_name}")

        model_type = model_spec['model_type']
        model_params = {key: value for key, value in model_spec.items() if key != 'model_type'}

        print(f"  Model type: {model_type}")
        print(f"  Parameters: {model_params}")

        model = create_model(model_type, model_spec)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {num_params:,}")
        print()

        experiment_dir = Path(args.save_dir) / config_instance_name
        tensorboard_dir = experiment_dir / 'tensorboard'

        trainer = Trainer(
            model=model,
            learning_rate=learning_rate,
            loss_type=loss_type,
            device=device,
            use_amp=args.use_amp,
            compile_model=args.compile_model,
            tensorboard_dir=tensorboard_dir
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
            early_stop_loss=early_stop_loss,
            patience=patience,
            progress_tracker=progress_tracker,
            save_interval=save_interval,
            save_dir=experiment_dir if save_interval is not None else None,
            keep_last_n=keep_last_n,
        )

        training_duration = time.time() - start_time

        print("\n" + "─"*80)
        print("Final Evaluation")
        print("─"*80)

        if snr_config.config_type == 'range':
            eval_snr = (snr_config.min_snr + snr_config.max_snr) / 2
        else:
            eval_snr = float(np.mean(snr_config.snr_values))

        eval_results = trainer.evaluate(
            batch_size=200,
            snr_db=eval_snr,
            pos_values=model_spec['pos_values'],
            tdl_config=tdl_config
        )

        print(f"  NMSE: {eval_results['nmse']:.6f} ({eval_results['nmse_db']:.2f} dB)")
        print(f"  Per-port NMSE (dB): {eval_results['per_port_nmse_db']}")

        experiment_dir.mkdir(parents=True, exist_ok=True)

        save_path = experiment_dir / 'model.pth'
        model_spec_dict = {
            'model_type': model_spec['model_type'],
            'hidden_dim': model_spec.get('hidden_dim', 64),
            'num_stages': model_spec.get('num_stages', 2),
            'mlp_depth': model_spec.get('mlp_depth', 3),
            'share_weights_across_stages': model_spec.get('share_weights_across_stages', False),
            'activation_type': model_spec.get('activation_type', 'relu'),
            'seq_len': model_spec['seq_len'],
            'num_ports': len(model_spec['pos_values']),
            'pos_values': model_spec['pos_values'],
            'num_params': num_params,
        }

        training_spec_dict = {
            'loss_type': loss_type,
            'learning_rate': learning_rate,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'snr_config': snr_config_dict,
            'tdl_config': tdl_config,
        }

        metadata_dict = {
            'experiment_name': suite.experiment_name,
            'model_recipe_name': model_recipe_name,
            'model_label': experiment.model_label,
            'run_name': config_instance_name,
            'training_recipe_name': experiment.training_recipe_name,
            'training_label': training_label,
            'training_duration': training_duration,
            'timestamp': datetime.now().isoformat(),
        }

        trainer.save_checkpoint(
            save_path,
            additional_info={
                'model_spec': model_spec_dict,
                'training_spec': training_spec_dict,
                'metadata': metadata_dict,
                'eval_results': eval_results,
            }
        )

        config_save_path = experiment_dir / 'config.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'model_spec': model_spec_dict,
                'training_spec': training_spec_dict,
                'metadata': metadata_dict,
            }, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ Model saved to: {experiment_dir}")

        result = {
            'model_recipe_name': model_recipe_name,
            'run_name': config_instance_name,
            'training_label': training_label,
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'eval_nmse_db': eval_results['nmse_db'],
            'training_duration': training_duration,
            'num_params': num_params
        }
        results.append(result)
        progress_tracker.complete_task(result)
        previous_training_label = training_label
        previous_model_recipe_name = model_recipe_name
    
    # Calculate total time
    script_end_time = time.time()
    script_end_datetime = datetime.now()
    total_duration = script_end_time - script_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}\n")
    
    print(f"Total configurations trained: {len(results)}")
    print(f"Start time: {script_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration/3600:.2f} hours ({total_duration:.1f}s)")
    print()
    
    # Sort by evaluation NMSE (best first)
    results_sorted = sorted(results, key=lambda x: x['eval_nmse_db'])
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i}. {result['run_name']}:")
        print(f"   Final loss: {result['final_loss']:.6f}")
        print(f"   Min loss: {result['min_loss']:.6f}")
        print(f"   Eval NMSE: {result['eval_nmse_db']:.2f} dB")
        print(f"   Parameters: {result['num_params']:,}")
        print(f"   Duration: {result['training_duration']:.1f}s")
        print()
    
    # Highlight best configuration
    if results_sorted:
        best = results_sorted[0]
        print(f"🏆 Best run: {best['run_name']}")
        print(f"   NMSE: {best['eval_nmse_db']:.2f} dB")
        print()
    
    # Generate report
    report_path = Path(args.save_dir) / 'TRAINING_REPORT.md'
    generate_training_report(
        report_path=report_path,
        results=results_sorted,
        training_recipe_name=suite.training_recipe_name,
        start_time=script_start_datetime,
        end_time=script_end_datetime,
        total_duration=total_duration,
        device=device
    )
    print(f"✓ Training report saved: {report_path}")
    
    print("\n✓ All training completed!")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ✅ Post-training workflow: Evaluation + Plotting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if args.eval_after_train:
        print(f"\n{'='*80}")
        print("📊 Post-Training Evaluation")
        print(f"{'='*80}")
        
        # Import evaluation function
        from evaluate_models import evaluate_models_programmatic
        
        # Run evaluation
        eval_output_dir = Path(args.save_dir) / 'evaluation_results'
        eval_results = evaluate_models_programmatic(
            exp_dir=args.save_dir,
            output_dir=eval_output_dir,
            snr_range=args.eval_snr_range,
            tdl_list=args.eval_tdl.split(','),
            num_batches=args.eval_num_batches,
            batch_size=args.eval_batch_size,
            device=device,
            use_amp=False,  # Evaluation: precision > speed
            compile=True if device.type == 'cuda' else False  # ✅ GPU默认启用
        )
        
        print(f"\n✓ Evaluation completed!")
        print(f"  Results saved to: {eval_output_dir}")
        
        # ✅ Optional: Plotting (only if evaluation succeeded)
        if args.plot_after_eval:
            # Check if evaluation produced any results
            eval_json_path = eval_output_dir / 'evaluation_results.json'
            if not eval_json_path.exists():
                print(f"\n⚠️  Skipping plot generation: evaluation results not found")
            else:
                # Check if there are any models in results
                import json
                with open(eval_json_path, 'r') as f:
                    eval_data = json.load(f)
                
                if not eval_data.get('models') or len(eval_data['models']) == 0:
                    print(f"\n⚠️  Skipping plot generation: no models evaluated successfully")
                else:
                    print(f"\n{'='*80}")
                    print("📈 Generating Plots")
                    print(f"{'='*80}")
                    
                    # Import plotting function
                    from plot import generate_plots_programmatic
                    
                    # Generate plots
                    plot_output_dir = Path(args.save_dir) / 'plots'
                    try:
                        generate_plots_programmatic(
                            eval_results_path=eval_json_path,
                            output_dir=plot_output_dir
                        )
                        print(f"\n✓ Plots generated!")
                        print(f"  Saved to: {plot_output_dir}")
                    except Exception as e:
                        print(f"\n✗ Plot generation failed: {e}")
    
    print(f"\n{'='*80}")
    print("🎉 Complete Pipeline Finished!")
    print(f"{'='*80}")
    print(f"  Training:   {args.save_dir}")
    if args.eval_after_train:
        print(f"  Evaluation: {args.save_dir}/evaluation_results")
        if args.plot_after_eval:
            print(f"  Plots:      {args.save_dir}/plots")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
