# SRS Channel Estimation using AI Methods

This repository implements a modular approach for SRS (Sounding Reference Signal) channel estimation in 3GPP NR systems, combining traditional signal processing with AI-based enhancements.

## Project Structure

- `model.py`: Core implementation of the SRS Channel Estimator and trainable MMSE modules
- `config.py`: Configuration parameters for SRS processing
- `utils.py`: Helper functions for signal generation, visualization, etc.
- `data_generator.py`: Generator for synthetic SRS data
- `train.py`: Training script for AI-based components
- `evaluate.py`: Evaluation script for trained models
- `demo.py`: Demonstration of SRS channel estimation

## Background

SRS (Sounding Reference Signal) is used in 3GPP NR systems for uplink channel estimation. The system supports:
- Multiple users through cyclic shift multiplexing
- Multiple antenna ports per user
- Various configurations (ktc=2, ktc=4)

The processing flow includes:
1. LS (Least Squares) channel estimation in frequency domain
2. Transformation to time domain to identify timing offsets
3. Cyclic shift compensation
4. OCC (Orthogonal Cover Code) demultiplexing
5. Linear interpolation
6. Residual correction
7. MMSE (Minimum Mean Square Error) filtering

## AI-Enhanced Approach

This implementation allows for AI-based enhancements at different stages:
- The primary focus is on the MMSE filter matrices (C and R)
- Traditional implementations are provided as baselines
- The AI-based approach can learn better estimation strategies from data

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- tqdm

Install dependencies with:
```bash
pip install torch numpy matplotlib tqdm
```

## Usage

### Demo

Run a demonstration with default settings:
```bash
python demo.py --snr 10
```

Use a trained model:
```bash
python demo.py --checkpoint ./checkpoints/best_model.pt --snr 15
```

Use custom configuration with multiple users and ports:
```bash
python demo.py --custom_config --snr 20
```

### Training

Train the AI-enhanced MMSE filter:
```bash
python train.py --epochs 50 --train_batches 100 --batch_size 16 --use_trainable_mmse
```

Training options:
- `--epochs`: Number of training epochs
- `--train_batches`: Number of batches per epoch
- `--val_batches`: Number of validation batches per epoch
- `--batch_size`: Batch size
- `--save_dir`: Directory for saving checkpoints
- `--use_trainable_mmse`: Enable trainable MMSE matrices
- `--load_checkpoint`: Path to checkpoint to resume training

### Evaluation

Evaluate a trained model across different SNRs:
```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pt --snr_min 0 --snr_max 30 --snr_step 5
```

Evaluation options:
- `--checkpoint`: Path to model checkpoint
- `--num_samples`: Number of samples to evaluate
- `--snr_min`, `--snr_max`, `--snr_step`: SNR range for evaluation
- `--save_dir`: Directory for saving results

## Customization

The system is designed to be modular, allowing for different parts to be enhanced with AI:

1. To modify the MMSE filter approach:
   - Edit the `TrainableMMSEModule` class in `model.py`

2. To add other trainable components:
   - Extend the `SRSChannelEstimator` class in `model.py`

3. To experiment with different configurations:
   - Create custom `SRSConfig` objects in `config.py`

## References

- 3GPP TS 38.211: Physical channels and modulation
- 3GPP TS 38.215: Physical layer measurements
