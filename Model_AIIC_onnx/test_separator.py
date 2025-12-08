"""
Test script for Channel Separator models using existing SRS data generator

This script integrates the new channel separator models with the existing
SRS data generation framework.
"""

import sys
import os
import argparse

# Parse --cpu_ratio argument BEFORE setting up threading (needed before importing torch)
def parse_cpu_ratio():
    """Quick parse of --cpu_ratio to set thread count before torch import"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cpu_ratio', type=float, default=1.0)
    args, _ = parser.parse_known_args()
    return args.cpu_ratio

cpu_ratio = parse_cpu_ratio()

# CPU optimization: Set thread count BEFORE importing torch/numpy
# Detect if running under numactl or in multi-NUMA system
try:
    # Get CPUs available to this process (respects numactl, cgroup, etc.)
    available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback for systems without sched_getaffinity
    available_cpus = os.cpu_count()

# Use physical cores only, not hyperthreads
# For 2-socket SPR-EE: 224 logical cores = 112 physical cores
# Best practice: use physical cores only for CPU-intensive workloads
num_physical_cores = available_cpus // 2  # Assume 2-way SMT (hyperthreading)

# Apply CPU ratio (user-specified fraction of physical cores)
num_threads = max(1, int(num_physical_cores * cpu_ratio))

# Override with environment variable if set
if 'OMP_NUM_THREADS' in os.environ:
    num_threads = int(os.environ['OMP_NUM_THREADS'])
    print(f"📌 Using OMP_NUM_THREADS from environment: {num_threads}")
else:
    # Set all threading environment variables
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

# Intel MKL optimizations for SPR
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['KMP_HW_SUBSET'] = '1t'  # Use 1 thread per physical core (disable hyperthreading)

print(f"🚀 CPU Optimization:")
print(f"   Available CPUs: {available_cpus}")
print(f"   Physical cores: {num_physical_cores}")
print(f"   CPU ratio: {cpu_ratio:.2f} ({cpu_ratio*100:.0f}%)")
print(f"   Using threads: {num_threads}")
print(f"   NUMA nodes: Run with 'numactl --hardware' to check")

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import traceback
from torch.utils.tensorboard import SummaryWriter

# Set PyTorch inter-op threads globally (can only be set once)
torch.set_num_interop_threads(1)

# No TensorFlow needed - using custom NumPy TDL
# print(f"✅ Using custom TDL channel (pure NumPy, no GIL limitation)")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
except ImportError:
    from channel_separator import ResidualRefinementSeparatorReal
from data_generator import BaseSRSDataGenerator
from user_config import SRSConfig, create_example_config
from system_config import create_default_system_config
from utils import calculate_nmse


# ============================================================================
# SNR-Aware Loss Functions
# ============================================================================

def calculate_loss(h_pred, h_targets, snr_db=None, loss_type='nmse'):
    """
    统一的损失计算函数，支持多种损失类型（实数版本）
    
    **默认行为**: 对每个样本单独计算 NMSE，然后取平均
    
    Args:
        h_pred: 预测的信道 (B, P, L*2) real stacked [real; imag]
        h_targets: 目标信道 (B, P, L*2) real stacked [real; imag]
        snr_db: SNR in dB，可以是:
                - None: 使用原始 NMSE
                - float: 所有样本相同 SNR
                - list/array/tensor: 每个样本不同 SNR
        loss_type: 损失类型
                - 'nmse': 原始 NMSE（默认，每样本单独计算）
                - 'normalized': SNR 归一化损失（推荐用于大范围 SNR）
                - 'log': 对数空间（dB）损失
                - 'weighted': SNR 区间加权损失
    
    Returns:
        loss: 标量损失值
    """
    batch_size = h_pred.shape[0]
    
    # 对每个样本单独计算 NMSE (in real domain)
    # h_pred, h_targets: (B, P, L*2) where [:, :, :L] = real, [:, :, L:] = imag
    nmse_per_sample = []
    for i in range(batch_size):
        # MSE in real domain (both real and imaginary parts)
        mse_i = (h_pred[i] - h_targets[i]).pow(2).mean()
        signal_power_i = h_targets[i].pow(2).mean()
        nmse_i = mse_i / (signal_power_i + 1e-10)
        nmse_per_sample.append(nmse_i)
    
    # 转换为 tensor
    nmse_per_sample = torch.stack(nmse_per_sample)  # (B,)
    
    if loss_type == 'nmse':
        # 原始 NMSE：每个样本单独计算，然后取平均
        return nmse_per_sample.mean()
    
    elif loss_type == 'normalized':
        # SNR 归一化损失（推荐用于 0-30 dB 大范围训练）
        # 核心思想：loss = actual_nmse / theoretical_best_nmse
        
        if snr_db is None:
            # 如果没有提供 SNR，回退到原始 NMSE
            return nmse_per_sample.mean()
        
        # 处理 tuple 类型（范围）
        if isinstance(snr_db, tuple):
            # tuple 表示范围，但这里需要具体值
            # 使用中间值作为估计
            snr_db = (snr_db[0] + snr_db[1]) / 2
        
        # 处理 SNR 输入，支持每个样本不同 SNR
        if isinstance(snr_db, (list, np.ndarray)):
            snr_db_array = np.array(snr_db)
            if len(snr_db_array) == batch_size:
                # 每个样本不同 SNR（推荐）
                snr_linear = 10 ** (snr_db_array / 10)
            else:
                # 所有样本相同 SNR
                snr_linear = 10 ** (np.mean(snr_db_array) / 10)
                snr_linear = np.full(batch_size, snr_linear)
        elif isinstance(snr_db, torch.Tensor):
            if snr_db.numel() == batch_size:
                snr_linear = 10 ** (snr_db.cpu().numpy() / 10)
            else:
                snr_linear = 10 ** (snr_db.mean().item() / 10)
                snr_linear = np.full(batch_size, snr_linear)
        else:
            # 单一 SNR 值
            snr_linear = 10 ** (snr_db / 10)
            snr_linear = np.full(batch_size, snr_linear)
        
        # 理论最优 NMSE（考虑噪声影响）
        # 在高斯噪声下，理论 MMSE ≈ σ²_noise / σ²_signal = 1 / (1 + SNR)
        theoretical_nmse = torch.tensor(1.0 / (1.0 + snr_linear), dtype=torch.float32, device=nmse_per_sample.device)
        
        # 归一化损失：每个样本单独归一化
        normalized_loss_per_sample = nmse_per_sample / (theoretical_nmse + 1e-10)
        
        return normalized_loss_per_sample.mean()
    
    elif loss_type == 'log':
        # 对数空间（dB）损失：每个样本单独计算
        nmse_db_per_sample = 10 * torch.log10(nmse_per_sample + 1e-10)
        
        # 返回绝对值的平均，确保损失为正
        return torch.abs(nmse_db_per_sample).mean()
    
    elif loss_type == 'weighted':
        # SNR 区间加权损失：每个样本单独加权
        
        if snr_db is None:
            # 如果没有提供 SNR，回退到原始 NMSE
            return nmse_per_sample.mean()
        
        # 处理 tuple 类型（范围）
        if isinstance(snr_db, tuple):
            # tuple 表示范围，使用中间值作为估计
            snr_db = (snr_db[0] + snr_db[1]) / 2
        
        # 处理 SNR 输入
        if isinstance(snr_db, (list, np.ndarray)):
            snr_db_array = np.array(snr_db)
            if len(snr_db_array) != batch_size:
                # 如果不是每个样本一个 SNR，用平均值
                snr_db_array = np.full(batch_size, np.mean(snr_db_array))
        elif isinstance(snr_db, torch.Tensor):
            if snr_db.numel() != batch_size:
                snr_db_array = np.full(batch_size, snr_db.mean().item())
            else:
                snr_db_array = snr_db.cpu().numpy()
        else:
            # 单一 SNR 值
            snr_db_array = np.full(batch_size, snr_db)
        
        # 根据每个样本的 SNR 区间设置权重
        weights = np.zeros(batch_size)
        for i in range(batch_size):
            if snr_db_array[i] < 0:
                weights[i] = 0.5  # 低 SNR: 降低权重（接近理论极限）
            elif snr_db_array[i] < 10:
                weights[i] = 0.8  # 中低 SNR
            elif snr_db_array[i] < 20:
                weights[i] = 1.0  # 中 SNR: 标准权重
            else:
                weights[i] = 1.5  # 高 SNR: 增加权重（更有改进空间）
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=nmse_per_sample.device)
        weighted_loss_per_sample = weights_tensor * nmse_per_sample
        
        return weighted_loss_per_sample.mean()
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Choose from: 'nmse', 'normalized', 'log', 'weighted'")


def export_to_onnx(model, save_path: str, seq_len: int = 12, batch_size: int = 1):
    """
    Export PyTorch model to ONNX format
    
    Note: Full ONNX export with complex numbers is not supported by PyTorch.
    This function attempts export but may fail due to complex number operations.
    
    Alternative: Use PyTorch JIT tracing or save as TorchScript instead.
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX file
        seq_len: Sequence length
        batch_size: Batch size for dummy input
    """
    try:
        # Try to export despite complex number limitations
        # This will likely fail, but we attempt it anyway
        import torch.jit
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, dtype=torch.complex64)
        
        # Try tracing first (sometimes works better than export)
        traced = torch.jit.trace(model, dummy_input)
        
        # Save as TorchScript instead (better complex support)
        torchscript_path = save_path.replace('.onnx', '.pt')
        torch.jit.save(traced, torchscript_path)
        
        return True, f"Saved as TorchScript (ONNX not supported for complex models): {torchscript_path}"
        
    except Exception as e:
        error_msg = str(e)
        if "complex" in error_msg.lower() or "ComplexFloat" in error_msg:
            # Known issue: ONNX doesn't support complex numbers
            return False, "ONNX export not supported for complex-valued models. Use TorchScript (.pt) instead or implement real-valued model variant."
        return False, error_msg


def generate_training_data(
    srs_config: SRSConfig = None,
    batch_size: int = 32,
    snr_db = 20.0,  # Can be: scalar, list [snr0, snr1, snr2, snr3], or tuple (min, max) for random
    seq_len: int = 12,
    pos_values: list = [0, 3, 6, 9],  # Port positions, e.g., [0, 3, 6, 9] or [0, 2, 4, 6, 8, 10]
    tdl_config: str = 'A-30',  # Format: 'MODEL-DELAY_NS' e.g., 'A-30', 'B-100', 'C-300'
    snr_sampler = None,  # Optional: SNRSampler instance for smart SNR sampling
    snr_per_sample: bool = False  # ⭐ NEW: If True, each sample in batch gets different SNR (old behavior)
):
    """
    Generate training data with TDL channel and SNR control
    
    Args:
        snr_db: SNR in dB. 
                - Scalar: same SNR for all ports
                - List [snr0, snr1, snr2, snr3]: fixed SNR per port
                - Tuple (min, max): random SNR uniformly sampled per sample
        pos_values: Port positions (shifts). If None, defaults to [0, 3, 6, 9] for 4 ports
                    Examples: [0, 3, 6, 9] (4 ports), [0, 2, 4, 6, 8, 10] (6 ports)
        tdl_config: TDL configuration string 'MODEL-DELAY_NS'
                    e.g., 'A-30' = TDL-A with 30ns delay spread
                          'B-100' = TDL-B with 100ns delay spread
                    Or list of configs for random selection per sample
    
    Returns:
        y: (B, L) received signal with noise
        h_targets: (B, P, L) shifted channel targets (adjusted for SNR)
        pos_values: list of port positions
        h_true: (B, P, L) original channels (adjusted for SNR)
    """
    # num_ports is determined by length of pos_values
    num_ports = len(pos_values)
    
    # Calculate random timing offset for each port
    # ±256*Tc where Tc = 1/(480e3*4096) ≈ 0.509 ns
    scs = 30e3
    Ktc = 4
    Tc = 1.0 / (480e3 * 4096)  # 3GPP basic time unit
    Ts = 1.0 / (scs * Ktc * seq_len)  # Sampling interval
    
    # Generate random timing offsets in units of Tc for each port and each sample
    # Shape: (batch_size, num_ports)
    timing_offset_Tc = np.random.uniform(-256, 256, (batch_size, num_ports))
    
    # Convert to normalized offset (in units of samples)
    # delta = timing_offset_Tc * Tc / Ts
    timing_offset_samples = timing_offset_Tc * Tc / Ts  # Shape: (batch_size, num_ports)
    
    # Parse TDL configuration
    # Support: single config string or list for random selection
    if isinstance(tdl_config, list):
        # Random selection per sample (each sample in batch gets random config)
        tdl_configs = tdl_config
        tdl_config = np.random.choice(tdl_configs)
    
    # Parse: 'A-30' -> model='A', delay_spread=30e-9
    parts = tdl_config.split('-')
    tdl_model = parts[0].upper()
    delay_ns = float(parts[1]) if len(parts) > 1 else 30
    delay_spread = delay_ns * 1e-9
    
    # Use custom TDL channel (pure NumPy, no GIL, independent fading)
    from Model_AIIC.tdl_channel import TDLChannel
    
    # Create TDL channel model
    tdl = TDLChannel(
        model=tdl_model,
        delay_spread=delay_spread,
        carrier_frequency=3.5e9,
        normalize=True
    )
    
    # Generate all channels at once: batch_size x num_ports
    # Each sample has independent random phases (no time correlation)
    sampling_rate = scs * Ktc * seq_len
    
    h_base = tdl.generate_batch_parallel(
        batch_size=batch_size,
        num_ports=num_ports,
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        return_torch=True
    )
    
    # Apply random timing offset via frequency domain phase rotation
    # h_offset[k] = IFFT(FFT(h[k]) * exp(j*2*pi*k*delta/L))
    # where delta is the timing offset in samples
    H_fft = torch.fft.fft(h_base, dim=-1)  # (batch_size, num_ports, seq_len)
    
    # Create phase rotation matrix
    k = torch.arange(seq_len, dtype=torch.float32)  # frequency index
    timing_offset_tensor = torch.from_numpy(timing_offset_samples).float()  # (batch_size, num_ports)
    
    # Phase shift: exp(j * 2*pi * k * delta / L)
    # Broadcast: (batch_size, num_ports, 1) * (1, 1, seq_len) -> (batch_size, num_ports, seq_len)
    phase_shift = torch.exp(1j * 2 * np.pi * k[None, None, :] * timing_offset_tensor[:, :, None] / seq_len)
    
    # Apply phase rotation
    H_shifted = H_fft * phase_shift.to(torch.complex64)
    
    # Convert back to time domain
    h_base = torch.fft.ifft(H_shifted, dim=-1)  # (batch_size, num_ports, seq_len)
    
    # Note: TDL channel already normalized (normalize=True in TDLChannel)
    # Do NOT normalize again here to preserve random fading per sample
    
    # Generate noise with unit power
    noise = (torch.randn(batch_size, seq_len) + 1j * torch.randn(batch_size, seq_len))
    noise = noise / noise.abs().pow(2).mean().sqrt()
    
    # Adjust signal power based on SNR (vectorized - no loops!)
    # ⭐ Two modes: per-batch SNR (default) vs per-sample SNR (old behavior)
    if isinstance(snr_db, tuple) and len(snr_db) == 2:
        snr_min, snr_max = snr_db
        
        if snr_per_sample:
            # ⭐ Mode 2: Per-Sample SNR (old behavior)
            # Each sample in batch gets DIFFERENT random SNR
            # Better for learning SNR-invariant features
            sample_snrs = np.random.uniform(snr_min, snr_max, batch_size)  # (B,)
            signal_powers = torch.tensor([10 ** (snr / 10) for snr in sample_snrs])  # (B,)
            # Broadcast: (B, P, L) * (B, 1, 1) -> (B, P, L)
            h_true = h_base * signal_powers.sqrt().view(batch_size, 1, 1)
            actual_batch_snr = float(np.mean(sample_snrs))  # Average SNR for logging
        else:
            # ⭐ Mode 1: Per-Batch SNR (current/default)
            # All samples in batch get SAME SNR
            # Better for learning specific SNR behaviors
            if snr_sampler is not None:
                batch_snr = snr_sampler.sample()  # Smart sampling (stratified/round-robin/etc)
            else:
                batch_snr = np.random.uniform(snr_min, snr_max)  # Uniform random (can cluster)
            
            signal_power = torch.tensor(10 ** (batch_snr / 10))
            h_true = h_base * signal_power.sqrt()
            actual_batch_snr = batch_snr
    elif isinstance(snr_db, list):
        # List of SNR values: randomly select one for this batch
        # e.g., [5, 10, 15, 20, 25] -> pick one randomly
        if len(snr_db) == num_ports:
            # Special case: if list length matches num_ports, assume per-port SNR
            signal_powers = torch.tensor([10 ** (snr / 10) for snr in snr_db])  # (P,)
            h_true = h_base * signal_powers.sqrt().view(1, num_ports, 1)
            actual_batch_snr = sum(snr_db) / len(snr_db)  # Average SNR
        else:
            # General case: randomly select one SNR from the list
            batch_snr = np.random.choice(snr_db)
            signal_power = torch.tensor(10 ** (batch_snr / 10))
            h_true = h_base * signal_power.sqrt()
            actual_batch_snr = batch_snr
    else:
        # Same SNR for all ports (scalar)
        signal_power = torch.tensor(10 ** (snr_db / 10))
        h_true = h_base * signal_power.sqrt()
        actual_batch_snr = snr_db
    
    # Create mixed signal with shifted channels
    y_clean = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    h_targets = []
    for i, pos in enumerate(pos_values):
        shifted = torch.roll(h_true[:, i], shifts=-pos, dims=-1)
        y_clean += shifted
        h_targets.append(shifted)
    h_targets = torch.stack(h_targets, dim=1)
    
    # Add noise
    y = y_clean + noise
    
    # Convert complex to real stacked format [real; imag] for ONNX-compatible model
    # y: (B, L) complex -> (B, L*2) real
    y_stacked = torch.cat([y.real, y.imag], dim=-1)
    
    # h_targets: (B, P, L) complex -> (B, P, L*2) real
    h_targets_stacked = torch.cat([h_targets.real, h_targets.imag], dim=-1)
    
    # h_true: (B, P, L) complex -> (B, P, L*2) real
    h_true_stacked = torch.cat([h_true.real, h_true.imag], dim=-1)
    
    # Return both complex and stacked versions
    # For ONNX model: use y_stacked, h_targets_stacked
    # For evaluation: can convert back from stacked to complex if needed
    # ⭐ Also return actual batch SNR used
    return y_stacked, h_targets_stacked, pos_values, h_true_stacked, actual_batch_snr


def test_model(
    num_batches=100, 
    batch_size=32, 
    num_stages=3, 
    snr_db=20.0,  # Can be scalar, list, or tuple (min, max)
    share_weights=False,
    pos_values=[0, 3, 6, 9],  # Port positions, e.g., [0, 3, 6, 9] or [0, 2, 4, 6, 8, 10]
    tdl_configs='A-30',  # Can be string or list of strings
    loss_type='nmse',  # Loss function type: 'nmse', 'normalized', 'log', 'weighted'
    activation_type='split_relu',  # Complex activation: 'split_relu', 'mod_relu', 'z_relu', 'cardioid'
    onnx_mode=False,  # ⭐ ONNX Opset 9 compatible mode (~20% slower but MATLAB compatible)
    early_stop_loss=None,  # Stop if loss below this value
    validation_interval=100,  # Validate every N batches
    patience=5,  # Number of validation checks that must pass early stop threshold
    save_dir=None,  # Directory to save model and metrics
    exp_name=None,  # Experiment name for this run
    snr_sampling_strategy='stratified',  # SNR sampling: 'uniform', 'stratified', 'round_robin'
    snr_num_bins=10,  # Number of SNR bins for stratified/round_robin sampling
    snr_per_sample=False,  # ⭐ NEW: Per-sample SNR (True) vs per-batch SNR (False, default)
    progress_callback=None  # Callback function(current_batch, total_batches) for progress reporting
):
    """
    Test Residual Refinement Channel Separator with online training using TDL channels
    
    Args:
        num_batches: Maximum number of training batches
        batch_size: Batch size
        num_stages: Number of refinement stages
        snr_db: SNR configuration (scalar, list, or (min, max) tuple)
        share_weights: Whether to share weights across stages
        pos_values: Port positions. Default [0, 3, 6, 9] for 4 ports.
                    Examples: [0, 3, 6, 9] (4 ports), [0, 2, 4, 6, 8, 10] (6 ports)
        tdl_configs: TDL configuration(s). String like 'A-30' or list like ['A-30', 'B-100', 'C-300']
        loss_type: Loss function type
                   - 'nmse': Original NMSE (default)
                   - 'normalized': SNR-normalized loss (recommended for wide SNR range)
                   - 'log': Log-space (dB) loss
                   - 'weighted': SNR-weighted loss
        early_stop_loss: If not None, stop training when loss < this value (with patience)
        validation_interval: How often to run validation
        patience: Number of consecutive validation passes needed for early stopping
        save_dir: Directory to save model and metrics (None = don't save)
        exp_name: Experiment name (used for subfolder naming)
    """
    # Get thread count from environment (set at module import)
    num_threads = int(os.environ.get('OMP_NUM_THREADS', 56))
    
    # Set PyTorch threading (only if not already set)
    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
    # Note: set_num_interop_threads can only be called once, so we skip it in loop
    
    print("="*80)
    print(f"Residual Refinement Channel Separator - Online Training")
    print("="*80)
    print(f"🔧 Thread Configuration:")
    print(f"   Available CPUs: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}")
    print(f"   PyTorch intra-op threads: {torch.get_num_threads()}")
    print(f"   PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    print(f"   OMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")
    print(f"   Using custom NumPy TDL channel (no GIL limitation)")
    
    # Configuration
    seq_len = 12
    
    # num_ports is determined by length of pos_values
    num_ports = len(pos_values)
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of ports: {num_ports}")
    print(f"  Port positions: {pos_values}")
    print(f"  Num stages: {num_stages}")
    print(f"  Share weights: {share_weights}")
    print(f"  SNR: {snr_db} dB")
    print(f"  Loss type: {loss_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max training batches: {num_batches}")
    print(f"  TDL configs: {tdl_configs}")
    if early_stop_loss is not None:
        print(f"  Early stop loss: {early_stop_loss:.6f} (patience: {patience})")
        print(f"  Validation interval: {validation_interval}")
    
    # ⭐ Create SNR sampler for smart SNR sampling (if SNR range specified)
    from Model_AIIC_onnx.snr_sampler import create_snr_sampler
    snr_sampler = create_snr_sampler(snr_db, strategy=snr_sampling_strategy, num_bins=snr_num_bins)
    
    if snr_sampler is not None:
        print(f"  SNR sampling: {snr_sampling_strategy} ({snr_num_bins} bins)")
        print(f"    → Ensures balanced coverage across SNR range")
        print(f"    → Prevents clustering of SNR values")
    
    # Print SNR mode
    if isinstance(snr_db, tuple):
        if snr_per_sample:
            print(f"  SNR mode: Per-Sample (each sample gets different random SNR)")
            print(f"    → Better for learning SNR-invariant features")
            print(f"    → Gradient targets average SNR ≈ {sum(snr_db)/2:.1f} dB")
        else:
            print(f"  SNR mode: Per-Batch (all samples in batch get same SNR)")
            print(f"    → Better for learning specific SNR behaviors")
            print(f"    → Each gradient update targets a specific SNR point")
    
    # Create model (Real-valued ONNX-compatible version)
    model = ResidualRefinementSeparatorReal(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=64,
        num_stages=num_stages,
        share_weights_across_stages=share_weights,
        activation_type=activation_type,  # New parameter for ONNX version
        onnx_mode=onnx_mode  # ⭐ ONNX Opset 9 compatibility mode
    )
    # Note: Energy normalization is now handled externally (outside the model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Setup TensorBoard
    writer = None
    if save_dir is not None:
        from pathlib import Path
        exp_path = Path(save_dir) / exp_name if exp_name else Path(save_dir)
        log_dir = exp_path / 'tensorboard'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        
        # Log hyperparameters
        hparam_dict = {
            'num_stages': num_stages,
            'share_weights': share_weights,
            'num_ports': num_ports,
            'pos_values': str(pos_values),
            'batch_size': batch_size,
            'snr_db': str(snr_db),
            'tdl_configs': str(tdl_configs),
            'loss_type': loss_type,
            'hidden_dim': 64,
            'num_params': num_params
        }
        print(f"📊 TensorBoard logs: {log_dir}")
        print(f"   Run: tensorboard --logdir {save_dir}")
    
    # Online Training
    print(f"\nOnline Training...")
    print(f"💡 Tip: Use larger batch_size (e.g., 2048-4096) for better CPU utilization")
    model.train()
    losses = []
    val_losses = []
    
    import time
    start_time = time.time()
    data_gen_time = 0
    forward_time = 0
    backward_time = 0
    
    # Early stopping variables
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    # Progress tracking
    training_start_time = time.time()
    last_progress_print = 0
    progress_print_interval = 100  # Print detailed progress every N batches
    
    for batch_idx in range(num_batches):
        # Generate batch on-the-fly using TDL
        t0 = time.time()
        y, h_targets, _, h_true, batch_snr = generate_training_data(
            batch_size=batch_size, 
            snr_db=snr_db, 
            seq_len=seq_len, 
            pos_values=pos_values,
            tdl_config=tdl_configs,
            snr_sampler=snr_sampler,  # ⭐ Use smart SNR sampling
            snr_per_sample=snr_per_sample  # ⭐ Per-sample SNR mode
        )
        data_gen_time += time.time() - t0
        
        # Forward
        t0 = time.time()
        optimizer.zero_grad()
        h_pred = model(y)
        forward_time += time.time() - t0
        
        # Calculate raw NMSE (for skip decision)
        mse = (h_pred - h_targets).abs().pow(2).mean()
        signal_power = h_targets.abs().pow(2).mean()
        nmse = mse / (signal_power + 1e-10)
        
        # Calculate loss using specified loss function (for optimization)
        loss = calculate_loss(h_pred, h_targets, batch_snr, loss_type=loss_type)
        
        # ⭐ SNR-aware training: Skip backward if NMSE already below SNR noise floor
        # Use raw NMSE (not transformed loss) to decide skip
        # Theoretical noise floor: NMSE = 1/SNR (in linear scale)
        # Skip threshold: NMSE < 1/(10^((SNR+5)/10)) (5 dB margin)
        snr_noise_floor_linear = 1.0 / (10 ** ((batch_snr + 5) / 10))
        skip_backward = nmse.item() < snr_noise_floor_linear
        
        if not skip_backward:
            # Backward
            t0 = time.time()
            loss.backward()
            optimizer.step()
            backward_time += time.time() - t0
        else:
            # Skip backward - already converged for this SNR
            pass
        
        losses.append(loss.item())
        
        # Log to TensorBoard
        if writer is not None:
            nmse_db = 10 * torch.log10(nmse)
            loss_db = 10 * torch.log10(loss)
            
            # Log both NMSE (raw metric) and loss (optimization objective)
            writer.add_scalar('NMSE/train', nmse.item(), batch_idx)
            writer.add_scalar('NMSE/train_db', nmse_db.item(), batch_idx)
            writer.add_scalar('Loss/train', loss.item(), batch_idx)
            writer.add_scalar('Loss/train_db', loss_db.item(), batch_idx)
            writer.add_scalar('SNR/batch_snr', batch_snr, batch_idx)
            writer.add_scalar('Skip/backward_skipped', 1.0 if skip_backward else 0.0, batch_idx)
            
            # Log per-port NMSE
            for p in range(num_ports):
                port_mse = (h_pred[:, p] - h_targets[:, p]).abs().pow(2).mean()
                port_power = h_targets[:, p].abs().pow(2).mean()
                port_nmse = port_mse / (port_power + 1e-10)
                port_nmse_db = 10 * torch.log10(port_nmse)
                writer.add_scalar(f'NMSE_per_port/port_{p}_db', port_nmse_db.item(), batch_idx)
        
        # Print progress with throughput and timing breakdown
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * batch_size / elapsed if elapsed > 0 else 0
            
            # Calculate percentage of time spent in each phase
            total_time = data_gen_time + forward_time + backward_time
            if total_time > 0:
                data_pct = 100 * data_gen_time / total_time
                fwd_pct = 100 * forward_time / total_time
                bwd_pct = 100 * backward_time / total_time
                timing_info = f"[Data:{data_pct:.0f}% Fwd:{fwd_pct:.0f}% Bwd:{bwd_pct:.0f}%]"
                
                # Log timing info to TensorBoard
                if writer is not None:
                    writer.add_scalar('Throughput/samples_per_sec', samples_per_sec, batch_idx)
                    writer.add_scalar('Time/data_pct', data_pct, batch_idx)
                    writer.add_scalar('Time/forward_pct', fwd_pct, batch_idx)
                    writer.add_scalar('Time/backward_pct', bwd_pct, batch_idx)
            else:
                timing_info = ""
            
            # Convert NMSE and loss to dB for display
            nmse_db = 10 * torch.log10(nmse)
            loss_db = 10 * torch.log10(loss)
            
            # Show SNR info
            if isinstance(snr_db, (tuple, list)):
                # For tuple/list: show actual SNR used in this batch
                snr_info = f"SNR:{batch_snr:.1f}dB"
            else:
                # For scalar: show fixed SNR
                snr_info = f"SNR:{snr_db}dB"
            
            # Show if backward was skipped
            skip_info = " [SKIP]" if skip_backward else ""
            
            # Show both NMSE and loss (if different)
            if loss_type == 'nmse':
                loss_str = f"NMSE: {nmse.item():.6f} ({nmse_db.item():.2f} dB)"
            else:
                loss_str = f"NMSE: {nmse.item():.6f} ({nmse_db.item():.2f} dB), Loss({loss_type}): {loss.item():.6f}"
            
            print(f"  Batch {batch_idx+1}/{num_batches}, {snr_info}, "
                  f"{loss_str}{skip_info}, "
                  f"Throughput: {samples_per_sec:.0f} samples/s {timing_info}")
        
        # Print detailed progress every N batches
        if (batch_idx + 1) % progress_print_interval == 0:
            training_elapsed = time.time() - training_start_time
            progress_pct = (batch_idx + 1) / num_batches * 100
            
            # Estimate remaining time
            if batch_idx > 0:
                avg_time_per_batch = training_elapsed / (batch_idx + 1)
                remaining_batches = num_batches - (batch_idx + 1)
                est_remaining = avg_time_per_batch * remaining_batches
                est_hours = int(est_remaining // 3600)
                est_mins = int((est_remaining % 3600) // 60)
                est_secs = int(est_remaining % 60)
                
                elapsed_hours = int(training_elapsed // 3600)
                elapsed_mins = int((training_elapsed % 3600) // 60)
                elapsed_secs = int(training_elapsed % 60)
                
                print(f"  ⏱️  Progress: {progress_pct:.1f}% ({batch_idx+1}/{num_batches} batches) | "
                      f"Elapsed: {elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d} | "
                      f"ETA: {est_hours:02d}:{est_mins:02d}:{est_secs:02d}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(batch_idx + 1, num_batches)
        
        # Validation and early stopping check
        if early_stop_loss is not None and (batch_idx + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                # Run validation on multiple batches for stability
                # ⭐ Each validation batch uses different SNR (if range specified)
                val_loss_sum = 0
                val_batches = 5
                val_snrs = []
                for _ in range(val_batches):
                    y_val, h_val, _, _, val_snr = generate_training_data(
                        batch_size=batch_size,
                        snr_db=snr_db,
                        seq_len=seq_len,
                        pos_values=pos_values,
                        tdl_config=tdl_configs,
                        snr_sampler=snr_sampler,  # ⭐ Use smart SNR sampling for validation too
                        snr_per_sample=snr_per_sample  # ⭐ Same mode as training
                    )
                    val_snrs.append(val_snr)
                    h_pred_val = model(y_val)
                    mse_val = (h_pred_val - h_val).abs().pow(2).mean()
                    signal_power_val = h_val.abs().pow(2).mean()
                    nmse_val = mse_val / (signal_power_val + 1e-10)
                    val_loss_sum += nmse_val.item()
                
                avg_val_loss = val_loss_sum / val_batches
                val_losses.append(avg_val_loss)
                avg_val_snr = sum(val_snrs) / len(val_snrs)
                
                # Convert to dB for display
                avg_val_loss_db = 10 * np.log10(avg_val_loss) if avg_val_loss > 0 else -np.inf
                
                # Log validation metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/validation', avg_val_loss, batch_idx)
                    writer.add_scalar('Loss/validation_db', avg_val_loss_db, batch_idx)
                
                print(f"  → Validation Loss: {avg_val_loss:.6f} ({avg_val_loss_db:.2f} dB) [Avg SNR: {avg_val_snr:.1f} dB]")
                
                # Check early stopping condition (user-specified threshold only)
                if avg_val_loss < early_stop_loss:
                    early_stop_counter += 1
                    print(f"  → Early stop progress: {early_stop_counter}/{patience}")
                    
                    if early_stop_counter >= patience:
                        print(f"\n✓ Early stopping triggered! Val loss {avg_val_loss:.6f} < {early_stop_loss:.6f}")
                        print(f"  Stopped at batch {batch_idx+1}/{num_batches}")
                        break
                else:
                    early_stop_counter = 0  # Reset counter
                
                # Track best validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
            
            model.train()  # Back to training mode
    
    # Print SNR sampler statistics
    if snr_sampler is not None:
        snr_sampler.print_stats()
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Batch")
    print(f"{'='*80}")
    
    model.eval()
    final_train_loss = losses[-1] if losses else None
    best_val_loss = min(val_losses) if val_losses else None
    
    with torch.no_grad():
        # Generate test batch (will use random SNR if range specified)
        y_test, h_targets_test, _, h_true_test, test_snr = generate_training_data(
            batch_size=200, 
            snr_db=snr_db, 
            seq_len=seq_len, 
            pos_values=pos_values
        )
        
        # Predict
        h_pred = model(y_test)
        
        # Calculate NMSE (h_targets_test already includes the shift)
        mse = (h_pred - h_targets_test).abs().pow(2).mean()
        signal_power = h_targets_test.abs().pow(2).mean()
        nmse_linear = mse / (signal_power + 1e-10)
        nmse_db = 10 * torch.log10(nmse_linear)
        
        # Port-wise NMSE (linear and dB)
        port_nmse_linear = []
        port_nmse_db = []
        for p in range(num_ports):
            mse_p = (h_pred[:, p] - h_targets_test[:, p]).abs().pow(2).mean()
            power_p = h_targets_test[:, p].abs().pow(2).mean()
            nmse_p_linear = mse_p / (power_p + 1e-10)
            nmse_p_db = 10 * torch.log10(nmse_p_linear)
            port_nmse_linear.append(nmse_p_linear.item())
            port_nmse_db.append(nmse_p_db.item())
        
        # Store for metrics
        test_nmse = nmse_linear.item()
        test_nmse_db = nmse_db.item()
        
        # Log test results to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/test', test_nmse, len(losses))
            writer.add_scalar('Loss/test_db', test_nmse_db, len(losses))
            
            for p in range(num_ports):
                writer.add_scalar(f'NMSE_per_port_test/port_{p}_db', port_nmse_db[p], len(losses))
            
            # Log hyperparameters with final metrics
            metric_dict = {
                'test_nmse_db': test_nmse_db,
                'final_train_loss': final_train_loss if final_train_loss else 0,
                'best_val_loss': best_val_loss if best_val_loss else 0,
            }
            writer.add_hparams(hparam_dict, metric_dict)
            writer.close()
        
        print(f"  Test SNR: {test_snr:.1f} dB")
        print(f"  Test NMSE: {test_nmse:.6f} ({test_nmse_db:.2f} dB)")
        print(f"  Port-wise NMSE (linear): {[f'{x:.6f}' for x in port_nmse_linear]}")
        print(f"  Port-wise NMSE (dB):     {[f'{x:.2f}' for x in port_nmse_db]} dB")
        
        # Print final training loss
        if final_train_loss is not None:
            final_train_loss_db = 10 * np.log10(final_train_loss) if final_train_loss > 0 else -np.inf
            print(f"  Final train loss: {final_train_loss:.6f} ({final_train_loss_db:.2f} dB)")
        
        # Print best validation loss if available
        if best_val_loss is not None:
            best_val_loss_db = 10 * np.log10(best_val_loss) if best_val_loss > 0 else -np.inf
            print(f"  Best val loss: {best_val_loss:.6f} ({best_val_loss_db:.2f} dB)")
        if best_val_loss:
            print(f"  Best val loss:    {best_val_loss:.6f}")
        print(f"  Avg loss (last 10): {sum(losses[-10:])/10:.6f}")
    
    # Save model and metrics if save_dir is provided
    if save_dir is not None:
        from pathlib import Path
        import json
        from datetime import datetime
        
        # Create save directory
        save_path = Path(save_dir)
        if exp_name:
            save_path = save_path / exp_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Saving Model and Metrics")
        print(f"{'='*80}")
        print(f"Save directory: {save_path}")
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'seq_len': seq_len,
                'num_ports': num_ports,
                'hidden_dim': 64,
                'num_stages': num_stages,
                'share_weights': share_weights,
                'onnx_mode': getattr(model, 'onnx_mode', False),  # ⭐ Save onnx_mode as hyperparameter
                'activation_type': activation_type  # Save activation type
            },
            'hyperparameters': {
                'num_stages': num_stages,
                'share_weights': share_weights,
                'num_ports': num_ports,
                'pos_values': pos_values,
                'batch_size': batch_size,
                'snr_db': str(snr_db),
                'tdl_configs': tdl_configs,
                'loss_type': loss_type
            },
            'losses': losses,
            'val_losses': val_losses,
            'final_train_loss': final_train_loss,
            'best_val_loss': best_val_loss,
            'test_nmse': test_nmse,
            'test_nmse_db': test_nmse_db
        }, save_path / 'model.pth')
        print(f"  ✓ Saved PyTorch model: {save_path / 'model.pth'}")
        
        # Try to export to ONNX/TorchScript
        model_cpu = model.cpu()
        success, message = export_to_onnx(model_cpu, str(save_path / 'model.onnx'), seq_len=seq_len)
        if success:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ Export note: {message}")
            
            # Save as TorchScript instead (works with complex numbers)
            try:
                traced = torch.jit.trace(model_cpu, torch.randn(1, seq_len, dtype=torch.complex64))
                torchscript_path = save_path / 'model.pt'
                torch.jit.save(traced, str(torchscript_path))
                print(f"  ✓ Saved TorchScript: {torchscript_path} (use in Python/C++)")
            except Exception as e2:
                print(f"  ✗ TorchScript export also failed: {e2}")
        
        # Save metrics JSON
        metrics = {
            'hyperparameters': {
                'num_stages': num_stages,
                'share_weights': share_weights,
                'num_ports': num_ports,
                'pos_values': pos_values,
                'batch_size': batch_size,
                'max_batches': num_batches,
                'snr_db': str(snr_db),
                'tdl_configs': tdl_configs,
                'loss_type': loss_type,
                'early_stop_loss': early_stop_loss,
                'validation_interval': validation_interval,
                'patience': patience
            },
            'results': {
                'final_train_loss': final_train_loss,
                'final_train_loss_db': 10 * np.log10(final_train_loss) if final_train_loss > 0 else -np.inf,
                'best_val_loss': best_val_loss,
                'best_val_loss_db': 10 * np.log10(best_val_loss) if best_val_loss and best_val_loss > 0 else None,
                'test_nmse': test_nmse,
                'test_nmse_db': test_nmse_db,
                'port_wise_nmse': port_nmse_linear,
                'port_wise_nmse_db': port_nmse_db,
                'num_batches_trained': len(losses),
                'stopped_early': len(losses) < num_batches
            },
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Saved metrics: {save_path / 'metrics.json'}")
        
        # Save loss curves
        np.save(save_path / 'train_losses.npy', np.array(losses))
        if val_losses:
            np.save(save_path / 'val_losses.npy', np.array(val_losses))
        print(f"  ✓ Saved loss curves")
        
        # Generate Markdown report
        report_path = save_path / 'training_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Training Report\n\n")
            f.write(f"**Experiment**: {exp_name if exp_name else 'unnamed'}\n\n")
            f.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
            
            f.write(f"## Configuration\n\n")
            f.write(f"### Model Architecture\n\n")
            f.write(f"| Parameter | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| Sequence Length | {seq_len} |\n")
            f.write(f"| Number of Ports | {num_ports} |\n")
            f.write(f"| Port Positions | {pos_values} |\n")
            f.write(f"| Hidden Dimension | 64 |\n")
            f.write(f"| Number of Stages | {num_stages} |\n")
            f.write(f"| Share Weights | {share_weights} |\n")
            f.write(f"| Normalize Energy | True |\n")
            f.write(f"| Total Parameters | {sum(p.numel() for p in model.parameters()):,} |\n\n")
            
            f.write(f"### Training Configuration\n\n")
            f.write(f"| Parameter | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| Batch Size | {batch_size} |\n")
            f.write(f"| Max Batches | {num_batches} |\n")
            f.write(f"| Batches Trained | {len(losses)} |\n")
            f.write(f"| SNR | {snr_db} |\n")
            f.write(f"| TDL Configs | {tdl_configs} |\n")
            if early_stop_loss:
                f.write(f"| Early Stop Loss | {early_stop_loss} |\n")
                f.write(f"| Validation Interval | {validation_interval} |\n")
                f.write(f"| Patience | {patience} |\n")
            f.write(f"| Stopped Early | {'Yes' if len(losses) < num_batches else 'No'} |\n\n")
            
            f.write(f"---\n\n")
            f.write(f"## Training Results\n\n")
            
            if final_train_loss is not None:
                final_train_loss_db = 10 * np.log10(final_train_loss) if final_train_loss > 0 else -np.inf
                f.write(f"**Final Training Loss**: `{final_train_loss:.6f}` (`{final_train_loss_db:.2f} dB`)\n\n")
            
            if best_val_loss is not None:
                best_val_loss_db = 10 * np.log10(best_val_loss) if best_val_loss > 0 else -np.inf
                f.write(f"**Best Validation Loss**: `{best_val_loss:.6f}` (`{best_val_loss_db:.2f} dB`)\n\n")
            
            f.write(f"**Test NMSE**: `{test_nmse:.6f}` (`{test_nmse_db:.2f} dB`)\n\n")
            
            f.write(f"### Port-wise Performance\n\n")
            f.write(f"| Port | NMSE (Linear) | NMSE (dB) |\n")
            f.write(f"|------|---------------|----------|\n")
            for p in range(num_ports):
                f.write(f"| {p} | {port_nmse_linear[p]:.6f} | {port_nmse_db[p]:.2f} dB |\n")
            f.write(f"\n")
            
            if val_losses:
                f.write(f"### Validation History\n\n")
                f.write(f"```\n")
                for i, vl in enumerate(val_losses):
                    vl_db = 10 * np.log10(vl) if vl > 0 else -np.inf
                    f.write(f"Val {i+1}: {vl:.6f} ({vl_db:.2f} dB)\n")
                f.write(f"```\n\n")
            
            f.write(f"---\n\n")
            f.write(f"## Files\n\n")
            f.write(f"- `model.pth` - PyTorch model weights (state dict)\n")
            if (save_path / 'model.pt').exists():
                f.write(f"- `model.pt` - TorchScript format (Python/C++ compatible)\n")
            if (save_path / 'model.onnx').exists():
                f.write(f"- `model.onnx` - ONNX format (if export succeeded)\n")
            f.write(f"- `metrics.json` - Detailed metrics\n")
            f.write(f"- `train_losses.npy` - Training loss history\n")
            if val_losses:
                f.write(f"- `val_losses.npy` - Validation loss history\n")
            f.write(f"- `training_report.md` - This report\n\n")
            
            # Add model usage instructions
            if (save_path / 'model.pt').exists():
                f.write(f"---\n\n")
                f.write(f"## TorchScript Model Usage\n\n")
                f.write(f"### Python Example\n\n")
                f.write(f"```python\n")
                f.write(f"import torch\n\n")
                f.write(f"# Load TorchScript model\n")
                f.write(f"model = torch.jit.load('model.pt')\n")
                f.write(f"model.eval()\n\n")
                f.write(f"# Prepare input (complex signal)\n")
                f.write(f"y = torch.randn(1, {seq_len}, dtype=torch.complex64)  # [batch, seq_len]\n\n")
                f.write(f"# Run inference\n")
                f.write(f"with torch.no_grad():\n")
                f.write(f"    h = model(y)  # [batch, {num_ports}, {seq_len}]\n")
                f.write(f"```\n\n")
                f.write(f"### MATLAB Usage (via Python Engine)\n\n")
                f.write(f"```matlab\n")
                f.write(f"% Start Python engine\n")
                f.write(f"pe = pyenv('Version', 'path/to/python');\n\n")
                f.write(f"% Load model via Python\n")
                f.write(f"model = py.torch.jit.load('model.pt');\n")
                f.write(f"model.eval();\n\n")
                f.write(f"% Prepare input\n")
                f.write(f"y_complex = randn({seq_len}, 1) + 1i*randn({seq_len}, 1);\n")
                f.write(f"% Convert to PyTorch tensor (requires additional conversion)\n")
                f.write(f"```\n\n")
                f.write(f"**Note**: For MATLAB, consider re-implementing the model natively or using Python Engine.\n\n")
            
            if (save_path / 'model.onnx').exists():
                f.write(f"---\n\n")
                f.write(f"## ONNX Model Usage\n\n")
                f.write(f"### Input Format\n\n")
                f.write(f"- **Shape**: `[batch, 2, {seq_len}]`\n")
                f.write(f"- **Type**: `float32`\n")
                f.write(f"- **Channel 0**: Real part of complex signal\n")
                f.write(f"- **Channel 1**: Imaginary part of complex signal\n\n")
                f.write(f"### Output Format\n\n")
                f.write(f"- **Shape**: `[batch, {num_ports}, 2, {seq_len}]`\n")
                f.write(f"- **Type**: `float32`\n")
                f.write(f"- **Dimension 2, Channel 0**: Real part of estimated channels\n")
                f.write(f"- **Dimension 2, Channel 1**: Imaginary part of estimated channels\n\n")
                f.write(f"### MATLAB Example\n\n")
                f.write(f"```matlab\n")
                f.write(f"% Load ONNX model\n")
                f.write(f"net = importONNXNetwork('model.onnx', 'OutputLayerType', 'regression');\n\n")
                f.write(f"% Prepare input (complex signal -> [real; imag])\n")
                f.write(f"y_complex = randn({seq_len}, 1) + 1i*randn({seq_len}, 1);  % Your signal\n")
                f.write(f"y_input = cat(1, real(y_complex)', imag(y_complex)');  % [2, {seq_len}]\n")
                f.write(f"y_input = reshape(y_input, [1, 2, {seq_len}]);  % Add batch dim\n\n")
                f.write(f"% Run inference\n")
                f.write(f"h_output = predict(net, y_input);  % [{num_ports}, 2, {seq_len}]\n\n")
                f.write(f"% Convert back to complex\n")
                f.write(f"h_real = squeeze(h_output(:, 1, :));  % [{num_ports}, {seq_len}]\n")
                f.write(f"h_imag = squeeze(h_output(:, 2, :));\n")
                f.write(f"h_complex = h_real + 1i*h_imag;  % [{num_ports}, {seq_len}]\n")
                f.write(f"```\n\n")
                f.write(f"### Python Example\n\n")
                f.write(f"```python\n")
                f.write(f"import onnxruntime as ort\n")
                f.write(f"import numpy as np\n\n")
                f.write(f"# Load model\n")
                f.write(f"session = ort.InferenceSession('model.onnx')\n\n")
                f.write(f"# Prepare input\n")
                f.write(f"y_complex = np.random.randn({seq_len}) + 1j*np.random.randn({seq_len})\n")
                f.write(f"y_input = np.stack([y_complex.real, y_complex.imag], axis=0)[None, :, :]  # [1, 2, {seq_len}]\n")
                f.write(f"y_input = y_input.astype(np.float32)\n\n")
                f.write(f"# Run inference\n")
                f.write(f"h_output = session.run(None, {{'input_real_imag': y_input}})[0]  # [1, {num_ports}, 2, {seq_len}]\n\n")
                f.write(f"# Convert back to complex\n")
                f.write(f"h_complex = h_output[0, :, 0, :] + 1j*h_output[0, :, 1, :]  # [{num_ports}, {seq_len}]\n")
                f.write(f"```\n")
            
        print(f"  ✓ Saved training report: {report_path}")
        print(f"{'='*80}\n")
    
    return model, losses


def print_overall_progress(experiments_info, current_exp_idx, current_exp_progress, current_exp_start_time, overall_start_time):
    """
    Print comprehensive progress report for all experiments
    
    Args:
        experiments_info: List of dicts with experiment info
        current_exp_idx: Index of current experiment (0-based)
        current_exp_progress: Progress of current experiment (0.0-1.0)
        current_exp_start_time: Start time of current experiment
        overall_start_time: Start time of all experiments
    """
    print("\n" + "="*100)
    print("🔄 OVERALL TRAINING PROGRESS")
    print("="*100)
    
    total_exps = len(experiments_info)
    completed_exps = [e for e in experiments_info if e['status'] == 'completed']
    failed_exps = [e for e in experiments_info if e['status'] == 'failed']
    
    # Overall statistics
    elapsed_total = time.time() - overall_start_time
    elapsed_h = int(elapsed_total // 3600)
    elapsed_m = int((elapsed_total % 3600) // 60)
    elapsed_s = int(elapsed_total % 60)
    
    print(f"\n📊 Summary:")
    print(f"  Total experiments: {total_exps}")
    print(f"  ✅ Completed: {len(completed_exps)}")
    print(f"  🔄 In progress: {1 if current_exp_idx < total_exps else 0}")
    print(f"  ⏳ Pending: {total_exps - current_exp_idx - 1}")
    print(f"  ❌ Failed: {len(failed_exps)}")
    print(f"  ⏱️  Total elapsed: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}")
    
    # ETA calculation
    if len(completed_exps) > 0:
        avg_duration = sum(e['duration'] for e in completed_exps) / len(completed_exps)
        remaining_exps = total_exps - current_exp_idx - 1
        # Add remaining time for current experiment
        current_exp_elapsed = time.time() - current_exp_start_time if current_exp_start_time else 0
        est_current_remaining = (current_exp_elapsed / max(current_exp_progress, 0.01)) * (1 - current_exp_progress)
        est_remaining = est_current_remaining + (remaining_exps * avg_duration)
        
        eta_h = int(est_remaining // 3600)
        eta_m = int((est_remaining % 3600) // 60)
        eta_s = int(est_remaining % 60)
        print(f"  🎯 Estimated remaining: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}")
    
    # Completed experiments
    if completed_exps:
        print(f"\n✅ Completed ({len(completed_exps)}):")
        for exp in completed_exps[-3:]:  # Show last 3
            dur_h = int(exp['duration'] // 3600)
            dur_m = int((exp['duration'] % 3600) // 60)
            dur_s = int(exp['duration'] % 60)
            print(f"  • {exp['name']:<60} [{dur_h:02d}:{dur_m:02d}:{dur_s:02d}]")
        if len(completed_exps) > 3:
            print(f"  ... and {len(completed_exps) - 3} more")
    
    # Current experiment
    if current_exp_idx < total_exps:
        current_exp = experiments_info[current_exp_idx]
        current_elapsed = time.time() - current_exp_start_time if current_exp_start_time else 0
        curr_h = int(current_elapsed // 3600)
        curr_m = int((current_elapsed % 3600) // 60)
        curr_s = int(current_elapsed % 60)
        
        print(f"\n🔄 In Progress:")
        print(f"  • {current_exp['name']}")
        print(f"    Progress: {current_exp_progress*100:.1f}% | Elapsed: {curr_h:02d}:{curr_m:02d}:{curr_s:02d}")
        
        # Estimate remaining time for current experiment
        if current_exp_progress > 0.01:
            est_total = current_elapsed / current_exp_progress
            est_remain = est_total - current_elapsed
            est_h = int(est_remain // 3600)
            est_m = int((est_remain % 3600) // 60)
            est_s = int(est_remain % 60)
            print(f"    Estimated remaining: {est_h:02d}:{est_m:02d}:{est_s:02d}")
    
    # Pending experiments
    pending_start = current_exp_idx + 1
    pending_exps = experiments_info[pending_start:]
    if pending_exps:
        print(f"\n⏳ Pending ({len(pending_exps)}):")
        for exp in pending_exps[:5]:  # Show first 5
            print(f"  • {exp['name']}")
        if len(pending_exps) > 5:
            print(f"  ... and {len(pending_exps) - 5} more")
    
    # Failed experiments
    if failed_exps:
        print(f"\n❌ Failed ({len(failed_exps)}):")
        for exp in failed_exps:
            print(f"  • {exp['name']}: {exp.get('error', 'Unknown error')}")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=10000,
                       help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size (larger=better CPU utilization, e.g., 2048-4096 for 56 cores)')
    parser.add_argument('--stages', type=str, default='3',
                       help='Number of refinement stages. Single: "3", Multiple: "2,3,4"')
    parser.add_argument('--ports', type=str, default='0,3,6,9',
                       help='Port positions (comma-separated). E.g., "0,3,6,9" (4 ports) or "0,2,4,6,8,10" (6 ports). Default: "0,3,6,9"')
    parser.add_argument('--snr', type=str, default='20.0',
                       help='SNR in dB. Can be: scalar (e.g., "20"), range (e.g., "10,30"), or list (e.g., "[15,18,20,22]")')
    parser.add_argument('--tdl', type=str, default='A-30',
                       help='TDL config(s). Single: "A-30", Multiple: "A-30,B-100,C-300"')
    parser.add_argument('--share_weights', type=str, default='False',
                       help='Share weights across stages. Single: "True", Multiple: "True,False"')
    parser.add_argument('--loss_type', type=str, default='weighted',
                       help='Loss function type: "nmse" (default), "normalized" (SNR-aware), "log" (dB space), "weighted" (SNR-weighted)')
    parser.add_argument('--activation_type', type=str, default='relu',
                       help='Complex activation: "relu" (FASTEST, recommended), "split_relu", "mod_relu" (slow), "z_relu" (very slow), "cardioid" (very slow). Multiple: "relu,split_relu"')
    parser.add_argument('--onnx_mode', action='store_true',
                       help='Use ONNX Opset 9 compatible mode (slower ~20%% but MATLAB compatible)')
    parser.add_argument('--early_stop', type=float, default=None,
                       help='Early stop threshold for loss')
    parser.add_argument('--val_interval', type=int, default=100,
                       help='Validation interval (batches)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default='auto',
                       help='Directory to save models and metrics. "auto" = auto-generate with timestamp (default), "none" = don\'t save, or specify custom path')
    parser.add_argument('--cpu_ratio', type=float, default=1.0,
                       help='Ratio of physical CPU cores to use (0.0-1.0). Default: 1.0 (use all cores)')
    parser.add_argument('--snr_sampling', type=str, default='stratified',
                       help='SNR sampling strategy: "uniform" (random, can cluster), "stratified" (balanced, default), "round_robin" (systematic)')
    parser.add_argument('--snr_bins', type=int, default=10,
                       help='Number of SNR bins for stratified/round_robin sampling. Default: 10')
    parser.add_argument('--snr_per_sample', action='store_true',
                       help='⭐ Per-sample SNR mode: each sample in batch gets different random SNR (old behavior). Default: False (per-batch SNR)')
    
    args = parser.parse_args()
    
    # Parse port positions
    pos_values = [int(x.strip()) for x in args.ports.split(',')]
    
    # Parse SNR argument
    snr_str = args.snr.strip()
    if ',' in snr_str and not snr_str.startswith('['):
        # Range: "10,30" -> (10, 30)
        parts = [float(x.strip()) for x in snr_str.split(',')]
        snr_db = tuple(parts)
    elif snr_str.startswith('['):
        # List: "[15,18,20,22]" -> [15, 18, 20, 22]
        snr_db = eval(snr_str)
    else:
        # Scalar: "20" -> 20.0
        snr_db = float(snr_str)
    
    # Parse TDL configs
    tdl_configs = args.tdl.strip()
    if ',' in tdl_configs:
        # Multiple configs: "A-30,B-100,C-300" -> ['A-30', 'B-100', 'C-300']
        tdl_configs = [x.strip() for x in tdl_configs.split(',')]
    
    # Parse hyperparameter lists
    stages_list = [int(x.strip()) for x in args.stages.split(',')]
    share_weights_list = [x.strip().lower() == 'true' for x in args.share_weights.split(',')]
    loss_type_list = [x.strip() for x in args.loss_type.split(',')]
    activation_type_list = [x.strip() for x in args.activation_type.split(',')]
    
    # Validate loss types
    valid_loss_types = ['nmse', 'normalized', 'log', 'weighted']
    for loss_type in loss_type_list:
        if loss_type not in valid_loss_types:
            print(f"❌ Error: Invalid loss_type '{loss_type}'")
            print(f"   Valid options: {', '.join(valid_loss_types)}")
            sys.exit(1)
    
    # Validate activation types
    valid_activation_types = ['relu', 'split_relu', 'mod_relu', 'z_relu', 'cardioid']
    for activation_type in activation_type_list:
        if activation_type not in valid_activation_types:
            print(f"❌ Error: Invalid activation_type '{activation_type}'")
            print(f"   Valid options: {', '.join(valid_activation_types)}")
            print(f"   💡 Recommended: 'relu' (fastest, 10-100x faster than others)")
            sys.exit(1)
    
    # Generate all hyperparameter combinations
    from itertools import product
    hyperparameter_combinations = list(product(stages_list, share_weights_list, loss_type_list, activation_type_list))
    
    # Auto-generate save directory if needed
    if args.save_dir == 'auto':
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        num_ports = len(pos_values)
        
        # Format SNR string for filename
        if isinstance(snr_db, tuple):
            snr_str = f"snr{int(snr_db[0])}-{int(snr_db[1])}"
        elif isinstance(snr_db, list):
            snr_str = f"snr{'_'.join(map(str, snr_db))}"
        else:
            snr_str = f"snr{int(snr_db)}"
        
        args.save_dir = f"./experiments/{timestamp}_batch{args.batches}_bs{args.batch_size}_ports{num_ports}_{snr_str}"
        print(f"📁 Auto-generated save directory: {args.save_dir}")
    elif args.save_dir.lower() == 'none':
        args.save_dir = None
    
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search Configuration")
    print(f"{'='*80}")
    print(f"Total combinations: {len(hyperparameter_combinations)}")
    print(f"  stages: {stages_list}")
    print(f"  share_weights: {share_weights_list}")
    print(f"  loss_type: {loss_type_list}")
    print(f"  activation_type: {activation_type_list}")
    print(f"Common settings:")
    if pos_values:
        print(f"  Port positions: {pos_values} ({len(pos_values)} ports)")
    else:
        print(f"  Port positions: Auto-generated (default)")
    print(f"  SNR: {snr_db}")
    print(f"  TDL: {tdl_configs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batches: {args.batches}")
    if args.early_stop:
        print(f"  Early stop: {args.early_stop}")
    if args.save_dir:
        print(f"  Save directory: {args.save_dir}")
    print(f"{'='*80}\n")
    
    # Run all combinations
    results = []
    overall_start_time = time.time()
    
    # Initialize experiments info for progress tracking
    experiments_info = []
    for num_stages, share_weights, loss_type, activation_type in hyperparameter_combinations:
        exp_name = f"stages={num_stages}_share={share_weights}_loss={loss_type}_act={activation_type}"
        experiments_info.append({
            'name': exp_name,
            'status': 'pending',
            'duration': 0,
            'num_stages': num_stages,
            'share_weights': share_weights,
            'loss_type': loss_type,
            'activation_type': activation_type
        })
    
    for idx, (num_stages, share_weights, loss_type, activation_type) in enumerate(hyperparameter_combinations):
        exp_name = experiments_info[idx]['name']
        experiments_info[idx]['status'] = 'in_progress'
        
        # Print initial experiment header
        print(f"\n{'#'*100}")
        print(f"# Experiment {idx+1}/{len(hyperparameter_combinations)}: {exp_name}")
        print(f"# Overall Progress: {idx}/{len(hyperparameter_combinations)} completed ({idx/len(hyperparameter_combinations)*100:.1f}%)")
        print(f"{'#'*100}\n")
        
        exp_start_time = time.time()
        
        # Define progress callback to print overall progress periodically
        last_overall_progress_print = [0]  # Use list to make it mutable in closure
        
        def progress_callback(current_batch, total_batches):
            # Print overall progress every 500 batches
            if current_batch - last_overall_progress_print[0] >= 500:
                last_overall_progress_print[0] = current_batch
                progress = current_batch / total_batches
                print_overall_progress(experiments_info, idx, progress, exp_start_time, overall_start_time)
        
        try:
            model, losses = test_model(
                num_batches=args.batches,
                batch_size=args.batch_size,
                num_stages=num_stages,
                snr_db=snr_db,
                share_weights=share_weights,
                pos_values=pos_values,
                tdl_configs=tdl_configs,
                activation_type=activation_type,  # New parameter
                onnx_mode=args.onnx_mode,  # ⭐ ONNX compatibility mode
                loss_type=loss_type,
                early_stop_loss=args.early_stop,
                validation_interval=args.val_interval,
                patience=args.patience,
                save_dir=args.save_dir,
                exp_name=exp_name,
                snr_sampling_strategy=args.snr_sampling,  # ⭐ Smart SNR sampling
                snr_num_bins=args.snr_bins,
                snr_per_sample=args.snr_per_sample,  # ⭐ Per-sample SNR mode
                progress_callback=progress_callback  # ⭐ Pass progress callback
            )
            
            exp_duration = time.time() - exp_start_time
            exp_hours = int(exp_duration // 3600)
            exp_mins = int((exp_duration % 3600) // 60)
            exp_secs = int(exp_duration % 60)
            
            # Update experiment status
            experiments_info[idx]['status'] = 'completed'
            experiments_info[idx]['duration'] = exp_duration
            
            print(f"\n{'='*100}")
            print(f"✓ Experiment {idx+1}/{len(hyperparameter_combinations)} completed in {exp_hours:02d}:{exp_mins:02d}:{exp_secs:02d}")
            print(f"{'='*100}")
            
            results.append({
                'experiment': exp_name,
                'num_stages': num_stages,
                'share_weights': share_weights,
                'loss_type': loss_type,
                'activation_type': activation_type,
                'final_loss': losses[-1] if losses else None,
                'min_loss': min(losses) if losses else None,
                'num_batches_trained': len(losses),
                'duration_seconds': exp_duration,
                'status': 'success'
            })
            
            # Print overall progress after completion
            print_overall_progress(experiments_info, idx + 1, 0, None, overall_start_time)
            
        except Exception as e:
            print(f"\n✗ Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Update experiment status
            experiments_info[idx]['status'] = 'failed'
            experiments_info[idx]['error'] = str(e)
            
            results.append({
                'experiment': exp_name,
                'num_stages': num_stages,
                'share_weights': share_weights,
                'loss_type': loss_type,
                'status': 'failed',
                'error': str(e)
            })
            
            # Print overall progress after failure
            print_overall_progress(experiments_info, idx + 1, 0, None, overall_start_time)
    
    # Print summary
    total_elapsed = time.time() - overall_start_time
    total_hours = int(total_elapsed // 3600)
    total_mins = int((total_elapsed % 3600) // 60)
    total_secs = int(total_elapsed % 60)
    
    print(f"\n{'='*80}")
    print(f"All Experiments Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_hours:02d}:{total_mins:02d}:{total_secs:02d}")
    print(f"Total experiments: {len(hyperparameter_combinations)}")
    print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results if r['status'] == 'failed'])}")
    print(f"\nResults Summary:")
    print(f"{'-'*80}")
    
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        # Sort by min_loss
        sorted_results = sorted(successful_results, key=lambda x: x['min_loss'])
        
        print(f"\n{'Rank':<6} {'Experiment':<50} {'Min Loss':<15} {'Final Loss':<15} {'Duration':<12} {'Batches':<10}")
        print(f"{'-'*120}")
        for i, result in enumerate(sorted_results):
            min_loss_db = 10 * np.log10(result['min_loss']) if result['min_loss'] > 0 else -np.inf
            final_loss_db = 10 * np.log10(result['final_loss']) if result['final_loss'] > 0 else -np.inf
            
            duration = result.get('duration_seconds', 0)
            dur_hours = int(duration // 3600)
            dur_mins = int((duration % 3600) // 60)
            dur_secs = int(duration % 60)
            duration_str = f"{dur_hours:02d}:{dur_mins:02d}:{dur_secs:02d}"
            
            print(f"{i+1:<6} {result['experiment']:<50} "
                  f"{result['min_loss']:.4f} ({min_loss_db:>6.2f}dB) "
                  f"{result['final_loss']:.4f} ({final_loss_db:>6.2f}dB) "
                  f"{duration_str:<12} "
                  f"{result['num_batches_trained']:<10}")
        
        best_loss_db = 10 * np.log10(sorted_results[0]['min_loss']) if sorted_results[0]['min_loss'] > 0 else -np.inf
        print(f"\nBest configuration: {sorted_results[0]['experiment']}")
        print(f"  Min Loss: {sorted_results[0]['min_loss']:.6f} ({best_loss_db:.2f} dB)")
        
        # Show average time per experiment
        avg_duration = sum(r.get('duration_seconds', 0) for r in successful_results) / len(successful_results)
        avg_hours = int(avg_duration // 3600)
        avg_mins = int((avg_duration % 3600) // 60)
        avg_secs = int(avg_duration % 60)
        print(f"  Average time per experiment: {avg_hours:02d}:{avg_mins:02d}:{avg_secs:02d}")
    
    failed_results = [r for r in results if r['status'] == 'failed']
    if failed_results:
        print(f"\nFailed experiments: {len(failed_results)}")
        for result in failed_results:
            print(f"  - {result['experiment']}: {result['error']}")
    
    # Save summary if save_dir provided
    if args.save_dir:
        from pathlib import Path
        import json
        
        summary_path = Path(args.save_dir) / 'search_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'hyperparameters': {
                    'stages': stages_list,
                    'share_weights': share_weights_list,
                    'snr_db': str(snr_db),
                    'tdl_configs': tdl_configs,
                    'batch_size': args.batch_size,
                    'max_batches': args.batches
                },
                'results': results
            }, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✓ Test completed successfully!")
    print(f"{'='*80}")
