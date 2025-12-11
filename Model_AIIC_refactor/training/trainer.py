"""
Unified trainer for all channel separator models.

Supports any model inheriting from BaseSeparatorModel.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Tuple, List
import time

from .loss_functions import calculate_loss
from .metrics import evaluate_model

try:
    from ..data import generate_training_batch
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import generate_training_batch

# ✅ Mixed Precision Training support
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class Trainer:
    """
    Unified model trainer
    
    Supports any model inheriting from BaseSeparatorModel with consistent
    training, evaluation, and checkpointing.
    
    Example:
        >>> from models import create_model
        >>> from training import Trainer
        >>> 
        >>> config = {'seq_len': 12, 'num_ports': 4, 'hidden_dim': 64}
        >>> model = create_model('separator1', config)
        >>> 
        >>> trainer = Trainer(model, learning_rate=0.01, loss_type='weighted')
        >>> losses = trainer.train(
        ...     num_batches=10000,
        ...     batch_size=2048,
        ...     snr_db=(0, 30),
        ...     pos_values=[0, 3, 6, 9]
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        loss_type: str = 'nmse',
        device: Union[str, torch.device] = 'auto',
        use_amp: bool = True,  # ✅ NEW: Mixed precision
        compile_model: bool = True  # ✅ NEW: Model compilation
    ):
        """
        Initialize trainer
        
        Args:
            model: Model instance (must inherit from BaseSeparatorModel)
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ('nmse', 'weighted', 'log', 'normalized')
            device: Device ('auto', 'cpu', 'cuda', or torch.device)
            use_amp: Use automatic mixed precision (GPU only) ✅ NEW
            compile_model: Compile model with torch.compile (GPU only, PyTorch 2.0+) ✅ NEW
        """
        self.model = model
        self.loss_type = loss_type
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # ✅ Model compilation (GPU only, PyTorch 2.0+)
        self.compiled = False
        if compile_model and self.device.type == 'cuda':
            try:
                if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
                    print("🚀 Compiling model with torch.compile...")
                    self.model = torch.compile(self.model)
                    self.compiled = True
                    print("   ✓ Model compiled successfully")
                else:
                    print("⚠️  torch.compile not available (requires PyTorch 2.0+)")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
        elif compile_model and self.device.type == 'cpu':
            print("ℹ️  Model compilation skipped (CPU mode, limited benefit)")
        
        # ✅ Mixed precision training (GPU only)
        self.use_amp = use_amp and self.device.type == 'cuda' and AMP_AVAILABLE
        if self.use_amp:
            self.scaler = GradScaler()
            print("⚡ Mixed precision training enabled (FP16)")
        elif use_amp and self.device.type == 'cpu':
            print("ℹ️  Mixed precision skipped (CPU mode not supported)")
        elif use_amp and not AMP_AVAILABLE:
            print("⚠️  Mixed precision not available (update PyTorch)")
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training state
        self.losses = []
        self.val_losses = []
        self.training_start_time = None
        self.current_batch = 0
        
        # Timing breakdown
        self.data_gen_time = 0
        self.forward_time = 0
        self.backward_time = 0
    
    def train(
        self,
        num_batches: int,
        batch_size: int,
        snr_config,  # SNRConfig object or legacy format
        pos_values: List[int] = None,
        tdl_config: Union[str, List[str]] = 'A-30',
        seq_len: int = None,
        print_interval: int = 100,
        val_interval: int = None,
        early_stop_loss: float = None,
        patience: int = 3,
        progress_tracker = None  # Optional progress tracker for multi-task training
    ) -> List[float]:
        """
        Train model
        
        Args:
            num_batches: Number of training batches
            batch_size: Batch size
            snr_config: SNRConfig object (or legacy snr_db for backward compat)
            pos_values: Port positions (default: [0, 3, 6, 9])
            tdl_config: TDL configuration (default: 'A-30')
            seq_len: Sequence length (default: from model)
            print_interval: Print progress every N batches
            val_interval: Validate every N batches (optional)
            early_stop_loss: Stop if loss below this value
            patience: Number of validations that must meet early stop
        
        Returns:
            losses: List of training losses
        """
        # Handle legacy snr_db format
        try:
            from ..utils import SNRConfig
            if not isinstance(snr_config, SNRConfig):
                # Legacy format: convert to SNRConfig
                if isinstance(snr_config, tuple):
                    snr_config = SNRConfig({'type': 'range', 'min': snr_config[0], 'max': snr_config[1]})
                elif isinstance(snr_config, (int, float)):
                    snr_config = SNRConfig({'type': 'discrete', 'values': [snr_config]})
        except:
            pass
        if pos_values is None:
            pos_values = [0, 3, 6, 9]
        
        if seq_len is None:
            seq_len = self.model.seq_len
        
        print(f"🚀 Starting training on {self.device}")
        print(f"   Model: {self.model.__class__.__name__}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Loss type: {self.loss_type}")
        
        self.model.train()
        self.training_start_time = time.time()
        self.losses = []
        
        # Reset timing counters
        self.data_gen_time = 0
        self.forward_time = 0
        self.backward_time = 0
        
        early_stop_counter = 0
        
        for batch_idx in range(num_batches):
            self.current_batch = batch_idx
            
            # Get SNR for this batch
            snr_for_batch = snr_config.get_snr_for_data_generator()
            
            # Generate data ✅ Directly on device
            t0_data = time.time()
            y, h_targets, _, _, actual_snr = generate_training_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                pos_values=pos_values,
                snr_db=snr_for_batch,
                tdl_config=tdl_config,
                snr_per_sample=snr_config.per_sample if hasattr(snr_config, 'per_sample') else False,
                return_complex=False,  # Always use real stacked format
                device=self.device  # ✅ Generate directly on device (GPU/CPU)
            )
            self.data_gen_time += time.time() - t0_data
            
            # ✅ No need to move to device - already there!
            # y and h_targets are already on self.device
            
            # Forward + Backward with optional mixed precision
            t0_fwd = time.time()
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # ✅ Mixed precision training (FP16)
                with autocast():
                    h_pred = self.model(y)
                    loss = calculate_loss(h_pred, h_targets, actual_snr, self.loss_type)
                self.forward_time += time.time() - t0_fwd
                
                # Backward with gradient scaling
                t0_bwd = time.time()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.backward_time += time.time() - t0_bwd
            else:
                # ✅ Standard FP32 training
                h_pred = self.model(y)
                loss = calculate_loss(h_pred, h_targets, actual_snr, self.loss_type)
                self.forward_time += time.time() - t0_fwd
                
                # Backward
                t0_bwd = time.time()
                loss.backward()
                self.optimizer.step()
                self.backward_time += time.time() - t0_bwd
            
            loss_value = loss.item()
            self.losses.append(loss_value)
            
            # Check if progress tracker should report (every 5 minutes)
            if progress_tracker:
                progress_tracker.check_and_report()
            
            # Print simple progress every 20 batches
            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                nmse = ((h_pred - h_targets).pow(2).mean() / 
                       h_targets.pow(2).mean()).item()
                nmse_db = 10 * torch.log10(torch.tensor(nmse))
                elapsed = time.time() - self.training_start_time
                
                # Calculate throughput
                samples_per_sec = (batch_idx + 1) * batch_size / elapsed if elapsed > 0 else 0
                
                # Calculate timing breakdown
                total_time = self.data_gen_time + self.forward_time + self.backward_time
                if total_time > 0:
                    data_pct = 100 * self.data_gen_time / total_time
                    fwd_pct = 100 * self.forward_time / total_time
                    bwd_pct = 100 * self.backward_time / total_time
                    timing_info = f"[Data:{data_pct:.0f}% Fwd:{fwd_pct:.0f}% Bwd:{bwd_pct:.0f}%]"
                else:
                    timing_info = ""
                
                print(f"  Batch {batch_idx+1}/{num_batches}, "
                      f"SNR:{actual_snr:.1f}dB, "
                      f"Loss:{loss_value:.6f}, "
                      f"NMSE:{nmse_db:.2f}dB, "
                      f"Throughput:{samples_per_sec:,.0f} samples/s {timing_info}")
            
            # Validation
            if val_interval and (batch_idx + 1) % val_interval == 0:
                val_loss = self.validate(
                    batch_size=batch_size,
                    snr_db=actual_snr,
                    pos_values=pos_values,
                    tdl_config=tdl_config,
                    seq_len=seq_len
                )
                self.val_losses.append(val_loss)
                
                # Early stopping
                if early_stop_loss and val_loss < early_stop_loss:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"\n✓ Early stopping: loss {val_loss:.6f} < {early_stop_loss}")
                        break
                else:
                    early_stop_counter = 0
        
        training_duration = time.time() - self.training_start_time
        print(f"\n✓ Training completed in {training_duration:.1f}s")
        print(f"  Final loss: {self.losses[-1]:.6f}")
        print(f"  Min loss: {min(self.losses):.6f}")
        
        return self.losses
    
    def validate(
        self,
        batch_size: int,
        snr_db: float,
        pos_values: List[int],
        tdl_config: str,
        seq_len: int
    ) -> float:
        """
        Validate model on a test batch
        
        Returns:
            val_loss: Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            y, h_targets, _, _, actual_snr = generate_training_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                pos_values=pos_values,
                snr_db=snr_db,
                tdl_config=tdl_config,
                return_complex=False,
                device=self.device  # ✅ Generate on device
            )
            
            # ✅ No need to move - already on device
            
            h_pred = self.model(y)
            loss = calculate_loss(h_pred, h_targets, actual_snr, self.loss_type)
        
        self.model.train()
        return loss.item()
    
    def evaluate(
        self,
        batch_size: int = 200,
        snr_db: float = 20.0,
        pos_values: List[int] = None,
        tdl_config: str = 'A-30',
        seq_len: int = None
    ) -> dict:
        """
        Comprehensive evaluation
        
        Returns:
            results: Dictionary with evaluation metrics
        """
        if pos_values is None:
            pos_values = [0, 3, 6, 9]
        
        if seq_len is None:
            seq_len = self.model.seq_len
        
        self.model.eval()
        
        with torch.no_grad():
            y, h_targets, _, _, actual_snr = generate_training_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                pos_values=pos_values,
                snr_db=snr_db,
                tdl_config=tdl_config,
                return_complex=False,
                device=self.device  # ✅ Generate on device
            )
            
            # ✅ No need to move - already on device
            
            h_pred = self.model(y)
            
            results = evaluate_model(h_pred, h_targets, actual_snr)
        
        self.model.train()
        return results
    
    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        additional_info: dict = None
    ):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save checkpoint
            additional_info: Additional info to save (optional)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_info': self.model.get_model_info(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'val_losses': self.val_losses,
            'loss_type': self.loss_type
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint.get('losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        return checkpoint


__all__ = ['Trainer']
