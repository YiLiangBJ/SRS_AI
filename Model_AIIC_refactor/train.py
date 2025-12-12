"""
Simplified training script using the refactored modules.

Usage:
    # Train with config file
    python test_separator_refactored.py --model_config separator1_default --training_config default
    
    # Quick test
    python test_separator_refactored.py --model_config separator1_small --training_config quick_test
    
    # Train both models for comparison
    python test_separator_refactored.py --model_config separator1_default,separator2_default
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
    get_device, print_device_info,
    parse_model_config, generate_config_name, print_search_space_summary,
    SNRConfig, parse_snr_config,
    TrainingProgressTracker
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_training_report(
    report_path: Path,
    results: list,
    training_config_name: str,
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
        f.write(f"- **Training Config**: {training_config_name}\n")
        f.write(f"- **Total Configurations**: {len(results)}\n\n")
        
        # Results Summary
        f.write("## Results Summary\n\n")
        f.write("| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |\n")
        f.write("|------|--------------|-----------|------------|-------------|\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"| {i} | `{result['config_instance_name']}` | "
                   f"{result['eval_nmse_db']:.2f} | {result['num_params']:,} | "
                   f"{result['training_duration']:.1f} |\n")
        
        f.write("\n")
        
        # Best Configuration
        if results:
            best = results[0]
            f.write("## 🏆 Best Configuration\n\n")
            f.write(f"**Configuration**: `{best['config_instance_name']}`\n\n")
            f.write(f"- **Eval NMSE**: {best['eval_nmse_db']:.2f} dB\n")
            f.write(f"- **Final Loss**: {best['final_loss']:.6f}\n")
            f.write(f"- **Min Loss**: {best['min_loss']:.6f}\n")
            f.write(f"- **Parameters**: {best['num_params']:,}\n")
            f.write(f"- **Training Duration**: {best['training_duration']:.1f}s\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"### {i}. {result['config_instance_name']}\n\n")
            f.write(f"- **Model Config**: {result['model_config_name']}\n")
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
    parser.add_argument('--model_config', type=str, default='separator1_default',
                       help='Model configuration name(s) from model_configs.yaml. '
                            'Multiple: "separator1_default,separator2_default"')
    parser.add_argument('--training_config', type=str, default='default',
                       help='Training configuration name from training_configs.yaml')
    
    # Overrides
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Override number of batches')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./experiments_refactored',
                       help='Directory to save models')
    
    # ✅ NEW: Performance optimization options
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                       help='Disable mixed precision training (FP16)')
    parser.add_argument('--no-compile', dest='compile_model', action='store_false',
                       help='Disable model compilation (torch.compile)')
    # ✅ GPU默认启用compile，CPU默认禁用（将在后面根据device调整）
    parser.set_defaults(use_amp=True, compile_model=True)
    
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
    
    args = parser.parse_args()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ✅ Create timestamped experiment directory (avoid conflicts)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate experiment name with timestamp
    base_save_dir = Path(args.save_dir)
    
    # Extract meaningful name from model_config
    model_config_short = args.model_config.split(',')[0]  # Use first if multiple
    experiment_name = f"{timestamp}_{model_config_short}_{args.training_config}"
    
    # Update save_dir to include timestamp
    args.save_dir = str(base_save_dir / experiment_name)
    
    print(f"\n{'='*80}")
    print(f"🚀 Experiment: {experiment_name}")
    print(f"{'='*80}")
    print(f"   Save directory: {args.save_dir}")
    print(f"   Timestamp: {timestamp}")
    print(f"{'='*80}\n")
    
    # Load configurations
    model_configs_file = Path(__file__).parent / 'configs' / 'model_configs.yaml'
    training_configs_file = Path(__file__).parent / 'configs' / 'training_configs.yaml'
    
    all_model_configs = load_config(model_configs_file)
    all_training_configs = load_config(training_configs_file)
    
    # Get common settings
    common_config = all_model_configs.get('common', {})
    
    # Get training config
    training_config_raw = all_training_configs.get(args.training_config, {})
    if not training_config_raw:
        raise ValueError(f"Training config '{args.training_config}' not found")
    
    # Parse training config (supports search_space like model configs)
    training_configs = parse_model_config(training_config_raw)  # Reuse same parser
    
    # Override with command line arguments
    for tc in training_configs:
        if args.batch_size:
            tc['batch_size'] = args.batch_size
        if args.num_batches:
            tc['num_batches'] = args.num_batches
    
    # Print device info
    device = get_device(args.device)
    
    # ✅ Auto-adjust compile based on device (if not explicitly disabled)
    if 'compile_model' in vars(args):
        # User explicitly set --no-compile
        pass
    else:
        # Auto: GPU enables compile, CPU disables
        if device.type == 'cuda':
            args.compile_model = True
        else:
            args.compile_model = False
    
    print("="*80)
    print("Channel Separator Training (Refactored)")
    print("="*80)
    print_device_info(device)
    print()
    
    # Parse model configs (support multiple models)
    model_config_names = [name.strip() for name in args.model_config.split(',')]
    
    print(f"Training configurations:")
    print(f"  Training config: {args.training_config} ({len(training_configs)} variants)")
    print(f"  Model config(s): {model_config_names}")
    print(f"  Available models: {list_models()}")
    print()
    
    # Show training config search space (if any)
    if len(training_configs) > 1:
        print(f"Training search space: {len(training_configs)} configurations")
        print_search_space_summary(training_configs, args.training_config)
        print()
    
    # Record start time
    script_start_time = time.time()
    script_start_datetime = datetime.now()
    
    # Count total configurations first (model configs × training configs)
    total_configs = 0
    for model_config_name in model_config_names:
        model_config = all_model_configs['models'].get(model_config_name)
        if model_config:
            full_model_config = {**common_config, **model_config}
            parsed_model_configs = parse_model_config(full_model_config)
            total_configs += len(parsed_model_configs) * len(training_configs)
    
    # Initialize progress tracker (report every 5 minutes)
    progress_tracker = TrainingProgressTracker(total_configs, report_interval=300.0)
    
    # Train each combination of (training_config × model_config)
    results = []
    task_index = 0
    
    for training_config_idx, training_config in enumerate(training_configs, 1):
        # Parse SNR configuration for this training config
        snr_config_dict = training_config.get('snr_config', {'type': 'range', 'min': 0, 'max': 30})
        snr_config = parse_snr_config(snr_config_dict)
        
        # Generate training config name
        if len(training_configs) > 1:
            training_variant_name = f"{args.training_config}_v{training_config_idx}"
            # Add loss_type or learning_rate to name if they vary
            if 'loss_type' in training_config:
                training_variant_name = f"{args.training_config}_{training_config['loss_type']}"
            elif 'learning_rate' in training_config:
                training_variant_name = f"{args.training_config}_lr{training_config['learning_rate']}"
        else:
            training_variant_name = args.training_config
        
        if len(training_configs) > 1:
            print(f"\n{'='*80}")
            print(f"Training Config Variant {training_config_idx}/{len(training_configs)}: {training_variant_name}")
            print(f"{'='*80}")
            print(f"  Loss type: {training_config.get('loss_type', 'nmse')}")
            print(f"  Learning rate: {training_config.get('learning_rate', 0.01)}")
            print(f"  SNR: {snr_config}")
            print()
        
        for model_config_name in model_config_names:
            print(f"\n{'='*80}")
            print(f"Model: {model_config_name}")
            if len(training_configs) > 1:
                print(f"Training: {training_variant_name}")
            print(f"{'='*80}\n")
            
            # Get model config
            model_config = all_model_configs['models'].get(model_config_name)
            if not model_config:
                print(f"✗ Model config '{model_config_name}' not found. Skipping.")
                continue
            
            # Merge common config
            full_model_config = {**common_config, **model_config}
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Parse configuration (supports search space)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            parsed_model_configs = parse_model_config(full_model_config)
            
            print_search_space_summary(parsed_model_configs, model_config_name)
            print()
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Train each model configuration
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            for config_idx, config in enumerate(parsed_model_configs, 1):
                task_index += 1
                
                print(f"\n{'─'*80}")
                if len(parsed_model_configs) > 1:
                    print(f"Model Config {config_idx}/{len(parsed_model_configs)} of {model_config_name}")
                if len(training_configs) > 1:
                    print(f"Training Config: {training_variant_name}")
                print(f"{'─'*80}\n")
                
                # Generate descriptive name for this configuration
                config_instance_name = generate_config_name(config, model_config_name)
                if len(training_configs) > 1:
                    config_instance_name = f"{config_instance_name}_{training_variant_name}"
                
                # Start tracking this task
                progress_tracker.start_task(config_instance_name, task_index)
                print(f"Configuration: {config_instance_name}")
                
                # Create model
                model_type = config['model_type']
                model_params = {k: v for k, v in config.items() if k != 'model_type'}
                
                print(f"  Model type: {model_type}")
                print(f"  Parameters: {model_params}")
                
                model = create_model(model_type, config)
                num_params = sum(p.numel() for p in model.parameters())
                print(f"  Total parameters: {num_params:,}")
                print()
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    learning_rate=training_config.get('learning_rate', 0.01),
                    loss_type=training_config.get('loss_type', 'nmse'),
                    device=device,
                    use_amp=args.use_amp,  # ✅ Mixed precision
                    compile_model=args.compile_model  # ✅ Model compilation
                )
                
                # Train
                start_time = time.time()
                
                losses = trainer.train(
                    num_batches=training_config.get('num_batches', 10000),
                    batch_size=training_config.get('batch_size', 2048),
                    snr_config=snr_config,
                    pos_values=config.get('pos_values', [0, 3, 6, 9]),  # From model_config
                    tdl_config=training_config.get('tdl_config', 'A-30'),
                    seq_len=config.get('seq_len', 12),
                    print_interval=training_config.get('print_interval', 100),
                    val_interval=training_config.get('validation_interval'),
                    early_stop_loss=training_config.get('early_stop_loss'),
                    patience=training_config.get('patience', 3),
                    progress_tracker=progress_tracker  # Pass progress tracker
                )
                
                training_duration = time.time() - start_time
                
                # Evaluate
                print("\n" + "─"*80)
                print("Final Evaluation")
                print("─"*80)
                
                # Evaluate at mid-point SNR
                if snr_config.config_type == 'range':
                    eval_snr = (snr_config.min_snr + snr_config.max_snr) / 2
                else:
                    eval_snr = float(np.mean(snr_config.snr_values))
                
                eval_results = trainer.evaluate(
                    batch_size=200,
                    snr_db=eval_snr,
                    pos_values=config.get('pos_values', [0, 3, 6, 9]),
                    tdl_config=training_config.get('tdl_config', 'A-30')
                )
                
                print(f"  NMSE: {eval_results['nmse']:.6f} ({eval_results['nmse_db']:.2f} dB)")
                print(f"  Per-port NMSE (dB): {eval_results['per_port_nmse_db']}")
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # ✅ Save checkpoint with unified format
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                save_dir = Path(args.save_dir) / config_instance_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / 'model.pth'
                
                # ✅ Prepare config (all model architecture parameters)
                model_config_dict = {
                    'model_type': config.get('model_type', 'separator1'),
                    'hidden_dim': config.get('hidden_dim', 64),
                    'num_stages': config.get('num_stages', 2),
                    'mlp_depth': config.get('mlp_depth', 3),
                    'share_weights_across_stages': config.get('share_weights_across_stages', False),
                    'activation_type': config.get('activation_type', 'relu'),
                    'seq_len': config.get('seq_len', 12),
                    'num_ports': len(config.get('pos_values', [0, 3, 6, 9])),
                    'pos_values': config.get('pos_values', [0, 3, 6, 9]),
                    'num_params': sum(p.numel() for p in trainer.model.parameters()),
                }
                
                # ✅ Prepare training_config (all training parameters)
                training_config_dict = {
                    'loss_type': training_config.get('loss_type', 'nmse'),
                    'learning_rate': training_config.get('learning_rate', 0.01),
                    'num_batches': training_config.get('num_batches', 10000),
                    'batch_size': training_config.get('batch_size', 4096),
                    'snr_config': training_config.get('snr_config', {'type': 'range', 'min': 0, 'max': 30}),
                    'tdl_config': training_config.get('tdl_config', 'A-30'),
                }
                
                # ✅ Prepare metadata
                metadata_dict = {
                    'model_config_name': model_config_name,
                    'config_instance_name': config_instance_name,
                    'training_config_name': training_variant_name,
                    'training_duration': training_duration,
                    'timestamp': datetime.now().isoformat(),
                }
                
                trainer.save_checkpoint(
                    save_path,
                    additional_info={
                        'config': model_config_dict,           # ✅ Standard key for model config
                        'training_config': training_config_dict,  # ✅ Standard key for training config
                        'metadata': metadata_dict,                # ✅ Metadata
                        'eval_results': eval_results,             # ✅ Quick evaluation results
                    }
                )
                
                # Save human-readable config
                config_save_path = save_dir / 'config.yaml'
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    yaml.dump({
                        'model_config': model_config_dict,
                        'training_config': training_config_dict,
                        'metadata': metadata_dict,
                    }, f, default_flow_style=False, allow_unicode=True)
                
                print(f"✓ Model saved to: {save_dir}")
                
                result = {
                    'model_config_name': model_config_name,
                    'config_instance_name': config_instance_name,
                    'training_config_name': training_variant_name,
                    'final_loss': losses[-1],
                    'min_loss': min(losses),
                    'eval_nmse_db': eval_results['nmse_db'],
                    'training_duration': training_duration,
                    'num_params': num_params
                }
                results.append(result)
                
                # Complete this task in progress tracker
                progress_tracker.complete_task(result)
    
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
        print(f"{i}. {result['config_instance_name']}:")
        print(f"   Final loss: {result['final_loss']:.6f}")
        print(f"   Min loss: {result['min_loss']:.6f}")
        print(f"   Eval NMSE: {result['eval_nmse_db']:.2f} dB")
        print(f"   Parameters: {result['num_params']:,}")
        print(f"   Duration: {result['training_duration']:.1f}s")
        print()
    
    # Highlight best configuration
    if results_sorted:
        best = results_sorted[0]
        print(f"🏆 Best configuration: {best['config_instance_name']}")
        print(f"   NMSE: {best['eval_nmse_db']:.2f} dB")
        print()
    
    # Generate report
    report_path = Path(args.save_dir) / 'TRAINING_REPORT.md'
    generate_training_report(
        report_path=report_path,
        results=results_sorted,
        training_config_name=args.training_config,
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
