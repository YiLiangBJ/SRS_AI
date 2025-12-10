"""
Model Evaluation Script (Refactored)

Evaluates trained models across different SNR values and TDL configurations.

Usage:
    # Evaluate single model
    python evaluate_models.py \
        --model_dir experiments_refactored/separator1_default_default/separator1_default \
        --output evaluation_results
    
    # Evaluate with custom SNR range
    python evaluate_models.py \
        --model_dir experiments_refactored/separator1_default_default/separator1_default \
        --snr_values 0,5,10,15,20,25,30 \
        --tdl_configs A-30,B-100,C-300 \
        --num_samples 1000
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import refactored modules
from models import create_model
from data import generate_training_batch
from training import evaluate_model


def load_trained_model(model_dir: Path):
    """
    Load a trained model from checkpoint
    
    Args:
        model_dir: Directory containing model.pth and config.yaml
    
    Returns:
        model: Loaded model
        config: Model configuration
    """
    model_path = model_dir / 'model.pth'
    config_path = model_dir / 'config.yaml'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load configuration
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            model_config = full_config.get('model_config', {})
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
    
    # Create model
    model_type = model_config.get('model_type', 'separator1')
    model = create_model(model_type, model_config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from {model_dir}")
    print(f"  Type: {model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, model_config


def evaluate_at_snr(
    model,
    model_config: dict,
    snr_db: float,
    tdl_config: str,
    num_samples: int = 1000,
    batch_size: int = 100
):
    """
    Evaluate model at specific SNR
    
    Args:
        model: Model to evaluate
        model_config: Model configuration
        snr_db: SNR in dB
        tdl_config: TDL configuration
        num_samples: Total number of samples
        batch_size: Batch size
    
    Returns:
        dict: Evaluation results
    """
    num_batches = num_samples // batch_size
    seq_len = model_config.get('seq_len', 12)
    pos_values = model_config.get('pos_values', [0, 3, 6, 9])
    num_ports = len(pos_values)
    
    total_mse = 0.0
    total_power = 0.0
    port_mse = np.zeros(num_ports)
    port_power = np.zeros(num_ports)
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate test data
            y, h_targets, _, _, _ = generate_training_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                pos_values=pos_values,
                snr_db=snr_db,
                tdl_config=tdl_config,
                return_complex=False
            )
            
            # Predict
            h_pred = model(y)
            
            # Calculate MSE
            mse = (h_pred - h_targets).pow(2).sum().item()
            power = h_targets.pow(2).sum().item()
            
            total_mse += mse
            total_power += power
            
            # Per-port MSE
            for p in range(num_ports):
                port_mse[p] += (h_pred[:, p] - h_targets[:, p]).pow(2).sum().item()
                port_power[p] += h_targets[:, p].pow(2).sum().item()
    
    # Calculate NMSE
    nmse = total_mse / (total_power + 1e-10)
    nmse_db = 10 * np.log10(nmse) if nmse > 0 else -100
    
    # Per-port NMSE
    port_nmse = port_mse / (port_power + 1e-10)
    port_nmse_db = 10 * np.log10(port_nmse)
    port_nmse_db[np.isinf(port_nmse_db)] = -100
    
    return {
        'snr_db': float(snr_db),
        'tdl_config': tdl_config,
        'nmse': float(nmse),
        'nmse_db': float(nmse_db),
        'per_port_nmse': port_nmse.tolist(),
        'per_port_nmse_db': port_nmse_db.tolist(),
        'num_samples': num_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--snr_values', type=str, default='0,5,10,15,20,25,30',
                       help='SNR values to evaluate (comma-separated)')
    parser.add_argument('--tdl_configs', type=str, default='A-30',
                       help='TDL configurations (comma-separated)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples per SNR')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse arguments
    model_dir = Path(args.model_dir)
    snr_values = [float(x) for x in args.snr_values.split(',')]
    tdl_configs = args.tdl_configs.split(',')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Model Evaluation (Refactored)")
    print("="*80)
    print(f"Model: {model_dir}")
    print(f"SNR values: {snr_values}")
    print(f"TDL configs: {tdl_configs}")
    print(f"Samples per SNR: {args.num_samples}")
    print(f"Output: {output_dir}")
    print()
    
    # Load model
    model, model_config = load_trained_model(model_dir)
    
    # Evaluate across SNRs and TDLs
    results = {
        'model_dir': str(model_dir),
        'model_config': model_config,
        'evaluation_date': datetime.now().isoformat(),
        'snr_values': snr_values,
        'tdl_configs': tdl_configs,
        'num_samples': args.num_samples,
        'results': []
    }
    
    print("\n" + "="*80)
    print("Evaluation Progress")
    print("="*80)
    
    total_evals = len(snr_values) * len(tdl_configs)
    pbar = tqdm(total=total_evals, desc="Evaluating")
    
    for tdl_config in tdl_configs:
        for snr_db in snr_values:
            eval_result = evaluate_at_snr(
                model=model,
                model_config=model_config,
                snr_db=snr_db,
                tdl_config=tdl_config,
                num_samples=args.num_samples,
                batch_size=args.batch_size
            )
            
            results['results'].append(eval_result)
            
            pbar.set_postfix({
                'TDL': tdl_config,
                'SNR': f"{snr_db}dB",
                'NMSE': f"{eval_result['nmse_db']:.2f}dB"
            })
            pbar.update(1)
    
    pbar.close()
    
    # Save results
    model_name = model_dir.name
    
    # Save JSON
    json_path = output_dir / f'{model_name}_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save NPY (for plotting)
    npy_path = output_dir / f'{model_name}_results.npy'
    npy_data = {
        'snr_values': np.array(snr_values),
        'nmse_db': {}
    }
    
    for tdl in tdl_configs:
        tdl_results = [r for r in results['results'] if r['tdl_config'] == tdl]
        npy_data['nmse_db'][tdl] = np.array([r['nmse_db'] for r in tdl_results])
    
    np.save(npy_path, npy_data)
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    
    for tdl in tdl_configs:
        print(f"\n{tdl}:")
        tdl_results = [r for r in results['results'] if r['tdl_config'] == tdl]
        
        print(f"  SNR (dB) | NMSE (dB)")
        print(f"  " + "-"*30)
        for r in tdl_results:
            print(f"  {r['snr_db']:8.1f} | {r['nmse_db']:9.2f}")
    
    print(f"\n✓ Results saved:")
    print(f"  JSON: {json_path}")
    print(f"  NPY:  {npy_path}")


if __name__ == '__main__':
    main()
