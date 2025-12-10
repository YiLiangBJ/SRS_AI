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

# Import refactored modules
from models import create_model, list_models
from training import Trainer
from utils import get_device, print_device_info, SNRSampler


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    
    args = parser.parse_args()
    
    # Load configurations
    model_configs_file = Path(__file__).parent / 'configs' / 'model_configs.yaml'
    training_configs_file = Path(__file__).parent / 'configs' / 'training_configs.yaml'
    
    all_model_configs = load_config(model_configs_file)
    all_training_configs = load_config(training_configs_file)
    
    # Get common settings
    common_config = all_model_configs.get('common', {})
    
    # Get training config
    training_config = all_training_configs.get(args.training_config, {})
    if not training_config:
        raise ValueError(f"Training config '{args.training_config}' not found")
    
    # Override with command line arguments
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.num_batches:
        training_config['num_batches'] = args.num_batches
    
    # Print device info
    device = get_device(args.device)
    print("="*80)
    print("Channel Separator Training (Refactored)")
    print("="*80)
    print_device_info(device)
    print()
    
    # Parse model configs (support multiple models)
    model_config_names = [name.strip() for name in args.model_config.split(',')]
    
    print(f"Training configurations:")
    print(f"  Training config: {args.training_config}")
    print(f"  Model config(s): {model_config_names}")
    print(f"  Available models: {list_models()}")
    print()
    
    # Setup SNR sampler if needed
    snr_sampler = None
    if training_config.get('snr_sampling') == 'stratified':
        snr_range = tuple(training_config['snr_range'])
        num_bins = training_config.get('snr_num_bins', 10)
        snr_sampler = SNRSampler(
            snr_min=snr_range[0],
            snr_max=snr_range[1],
            strategy='stratified',
            num_bins=num_bins
        )
        print(f"Using stratified SNR sampling: {num_bins} bins over {snr_range}")
    
    # Train each model
    results = []
    
    for model_config_name in model_config_names:
        print(f"\n{'='*80}")
        print(f"Training: {model_config_name}")
        print(f"{'='*80}\n")
        
        # Get model config
        model_config = all_model_configs['models'].get(model_config_name)
        if not model_config:
            print(f"✗ Model config '{model_config_name}' not found. Skipping.")
            continue
        
        # Merge common config
        full_model_config = {**common_config, **model_config}
        
        # Create model
        model_type = full_model_config.pop('model_type')
        print(f"Creating model: {model_type}")
        print(f"  Config: {full_model_config}")
        
        model = create_model(model_type, full_model_config)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            learning_rate=training_config.get('learning_rate', 0.01),
            loss_type=training_config.get('loss_type', 'nmse'),
            device=device
        )
        
        # Train
        start_time = time.time()
        
        losses = trainer.train(
            num_batches=training_config.get('num_batches', 10000),
            batch_size=training_config.get('batch_size', 2048),
            snr_db=tuple(training_config['snr_range']),
            pos_values=training_config.get('pos_values', [0, 3, 6, 9]),
            tdl_config=training_config.get('tdl_config', 'A-30'),
            snr_sampler=snr_sampler,
            snr_per_sample=training_config.get('snr_per_sample', False),
            seq_len=full_model_config.get('seq_len', 12),
            print_interval=training_config.get('print_interval', 100),
            val_interval=training_config.get('validation_interval'),
            early_stop_loss=training_config.get('early_stop_loss'),
            patience=training_config.get('patience', 3)
        )
        
        training_duration = time.time() - start_time
        
        # Evaluate
        print("\n" + "─"*80)
        print("Final Evaluation")
        print("─"*80)
        
        eval_results = trainer.evaluate(
            batch_size=200,
            snr_db=sum(training_config['snr_range']) / 2,  # Mid-point SNR
            pos_values=training_config.get('pos_values', [0, 3, 6, 9]),
            tdl_config=training_config.get('tdl_config', 'A-30')
        )
        
        print(f"  NMSE: {eval_results['nmse']:.6f} ({eval_results['nmse_db']:.2f} dB)")
        print(f"  Per-port NMSE (dB): {eval_results['per_port_nmse_db']}")
        
        # Save model
        save_dir = Path(args.save_dir) / model_config_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / 'model.pth'
        trainer.save_checkpoint(
            save_path,
            additional_info={
                'model_config_name': model_config_name,
                'training_config_name': args.training_config,
                'model_config': full_model_config,
                'training_config': training_config,
                'training_duration': training_duration,
                'eval_results': eval_results
            }
        )
        
        # Save config for reproducibility
        config_save_path = save_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump({
                'model_config': full_model_config,
                'training_config': training_config
            }, f, default_flow_style=False)
        
        print(f"✓ Model saved to: {save_dir}")
        
        results.append({
            'model_config_name': model_config_name,
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'eval_nmse_db': eval_results['nmse_db'],
            'training_duration': training_duration
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}\n")
    
    for result in results:
        print(f"  {result['model_config_name']}:")
        print(f"    Final loss: {result['final_loss']:.6f}")
        print(f"    Min loss: {result['min_loss']:.6f}")
        print(f"    Eval NMSE: {result['eval_nmse_db']:.2f} dB")
        print(f"    Duration: {result['training_duration']:.1f}s")
        print()
    
    print("✓ All training completed!")


if __name__ == '__main__':
    main()
