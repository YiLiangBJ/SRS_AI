"""
Grid Search Example - Demonstrate hyperparameter search space functionality

This script shows how to use the grid search feature to automatically
train and compare multiple model configurations.

Usage:
    # Quick test (2 configurations)
    python grid_search_example.py
    
    # Full grid search (9 configurations)
    python grid_search_example.py --full
"""

import argparse
from models import create_model
from utils import parse_model_config, print_search_space_summary, generate_config_name


def example_1_basic():
    """Example 1: Basic grid search"""
    print("="*80)
    print("Example 1: Basic Grid Search")
    print("="*80)
    
    config = {
        'model_type': 'separator1',
        'fixed_params': {
            'mlp_depth': 3,
            'share_weights_across_stages': False
        },
        'search_space': {
            'hidden_dim': [32, 64],
            'num_stages': [2, 3]
        }
    }
    
    # Parse configuration
    configs = parse_model_config(config)
    
    print_search_space_summary(configs, 'example_basic')
    
    # Show generated configurations
    print("\nGenerated configurations:")
    for i, cfg in enumerate(configs, 1):
        name = generate_config_name(cfg, 'example_basic')
        print(f"  {i}. {name}")
        print(f"     {cfg}")
    
    return configs


def example_2_full_search():
    """Example 2: Full grid search with more parameters"""
    print("\n" + "="*80)
    print("Example 2: Full Grid Search")
    print("="*80)
    
    config = {
        'model_type': 'separator1',
        'fixed_params': {
            'mlp_depth': 3
        },
        'search_space': {
            'hidden_dim': [32, 64, 128],
            'num_stages': [2, 3, 4],
            'share_weights_across_stages': [True, False]
        }
    }
    
    configs = parse_model_config(config)
    
    print_search_space_summary(configs, 'example_full')
    
    print(f"\nThis would train {len(configs)} different configurations!")
    print("Note: This is just for demonstration. Use train.py for actual training.")
    
    return configs


def example_3_range_search():
    """Example 3: Range-based search"""
    print("\n" + "="*80)
    print("Example 3: Range-Based Search")
    print("="*80)
    
    config = {
        'model_type': 'separator2',
        'fixed_params': {
            'mlp_depth': 3,
            'share_weights_across_stages': False,
            'onnx_mode': False
        },
        'search_space': {
            'hidden_dim': {
                'type': 'choice',
                'values': [32, 64, 128]
            },
            'num_stages': {
                'type': 'range',
                'min': 2,
                'max': 4,
                'step': 1
            },
            'activation_type': ['relu', 'split_relu']
        }
    }
    
    configs = parse_model_config(config)
    
    print_search_space_summary(configs, 'example_range')
    
    print(f"\nGenerated {len(configs)} configurations with range-based search")
    
    return configs


def example_4_model_creation():
    """Example 4: Create models from search space"""
    print("\n" + "="*80)
    print("Example 4: Creating Models from Search Space")
    print("="*80)
    
    config = {
        'model_type': 'separator1',
        'fixed_params': {
            'seq_len': 12,
            'num_ports': 4,
            'mlp_depth': 3
        },
        'search_space': {
            'hidden_dim': [32, 64],
            'num_stages': [2, 3]
        }
    }
    
    configs = parse_model_config(config)
    
    print(f"Creating {len(configs)} models...")
    print()
    
    for i, cfg in enumerate(configs, 1):
        model = create_model(cfg['model_type'], cfg)
        num_params = sum(p.numel() for p in model.parameters())
        name = generate_config_name(cfg, 'example')
        
        print(f"{i}. {name}")
        print(f"   Parameters: {num_params:,}")
        print(f"   Config: {cfg}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Grid search examples')
    parser.add_argument('--full', action='store_true',
                       help='Run full examples (generates many configs)')
    args = parser.parse_args()
    
    # Run examples
    example_1_basic()
    
    if args.full:
        example_2_full_search()
        example_3_range_search()
        example_4_model_creation()
    else:
        print("\n" + "="*80)
        print("💡 Tip: Run with --full flag to see more examples:")
        print("   python grid_search_example.py --full")
        print("="*80)
    
    # Show how to use with train.py
    print("\n" + "="*80)
    print("Using Grid Search with train.py")
    print("="*80)
    print("""
To use grid search in training:

1. Define search space in configs/model_configs.yaml:
   
   separator1_my_search:
     model_type: separator1
     fixed_params:
       mlp_depth: 3
     search_space:
       hidden_dim: [32, 64, 128]
       num_stages: [2, 3]

2. Train with the search space:
   
   python train.py \\
     --model_config separator1_my_search \\
     --training_config grid_search_quick

3. Results will be saved for each configuration with descriptive names:
   
   experiments_refactored/
     separator1_my_search_hd32_stages2/
     separator1_my_search_hd32_stages3/
     separator1_my_search_hd64_stages2/
     ...

4. Best configuration will be highlighted in the training summary!
""")


if __name__ == '__main__':
    main()
