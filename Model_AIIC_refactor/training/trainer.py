"""
Unified trainer for all channel separator models.

Supports any model inheriting from BaseSeparatorModel.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

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

# ✅ TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


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
        compile_model: bool = True,  # ✅ NEW: Model compilation
        tensorboard_dir: Optional[Union[str, Path]] = None,  # ✅ NEW: TensorBoard logging
        scheduler_config: Optional[Dict[str, Any]] = None,
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
            tensorboard_dir: Directory for TensorBoard logs (None to disable) ✅ NEW
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
        
        # ✅ CRITICAL: Set CUDA device explicitly before any CUDA operations
        # This ensures all subsequent CUDA operations use the correct GPU
        if self.device.type == 'cuda':
            if self.device.index is not None:
                torch.cuda.set_device(self.device)
                print(f"🎯 Using GPU: {self.device} ({torch.cuda.get_device_name(self.device)})")
            else:
                torch.cuda.set_device(0)
                self.device = torch.device('cuda:0')  # Make explicit
                print(f"🎯 Using GPU: cuda:0 ({torch.cuda.get_device_name(0)})")
        else:
            print(f"🎯 Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # ✅ TensorBoard setup
        self.writer = None
        if tensorboard_dir is not None and TENSORBOARD_AVAILABLE:
            tensorboard_dir = Path(tensorboard_dir)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            print(f"📊 TensorBoard logging enabled: {tensorboard_dir}")
            print(f"   Run: tensorboard --logdir {tensorboard_dir}")
        elif tensorboard_dir is not None and not TENSORBOARD_AVAILABLE:
            print("⚠️  TensorBoard not available. Install: pip install tensorboard")
        
        # ✅ Model compilation (GPU only, PyTorch 2.0+)
        self.compiled = False
        if compile_model and self.device.type == 'cuda':
            try:
                if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
                    print("🚀 Compiling model with torch.compile...")
                    
                    # ✅ Enable TF32 for better performance (Ampere+ GPUs)
                    if torch.cuda.is_available():
                        torch.set_float32_matmul_precision('high')  # Enable TF32
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        print("   ⚡ TensorFloat32 (TF32) enabled for faster matrix operations")
                    
                    # ✅ Compile with optimizations
                    # mode='reduce-overhead': faster compilation, good performance
                    # mode='max-autotune': slower compilation, best performance
                    self.model = torch.compile(
                        self.model,
                        mode='reduce-overhead',  # Faster compilation
                        fullgraph=False  # Allow graph breaks for flexibility
                    )
                    self.compiled = True
                    print("   ✓ Model compiled successfully")
                    print("   ℹ️  First few batches will be slower (JIT compilation)")
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
        
        # ✅ Setup learning rate scheduler (adaptive, but smoother by default)
        default_scheduler_config = {
            'enabled': True,
            'factor': 0.8,
            'patience': 30,
            'threshold': 5e-3,
            'threshold_mode': 'abs',
            'cooldown': 10,
            'min_lr': 1e-6,
        }
        self.scheduler_config = {**default_scheduler_config, **(scheduler_config or {})}
        self.scheduler = None
        if self.scheduler_config.get('enabled', True):
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.scheduler_config['factor'],
                patience=self.scheduler_config['patience'],
                threshold=self.scheduler_config['threshold'],
                threshold_mode=self.scheduler_config['threshold_mode'],
                cooldown=self.scheduler_config['cooldown'],
                min_lr=self.scheduler_config['min_lr'],
                verbose=True,
            )
            print("📉 Adaptive learning rate scheduler enabled")
            print(f"   Initial LR: {learning_rate}")
            print(
                f"   factor={self.scheduler_config['factor']}, patience={self.scheduler_config['patience']}, "
                f"threshold={self.scheduler_config['threshold']} ({self.scheduler_config['threshold_mode']}), "
                f"cooldown={self.scheduler_config['cooldown']}, min_lr={self.scheduler_config['min_lr']}"
            )
        else:
            print("ℹ️  Learning rate scheduler disabled")
        
        # Training state
        self.losses = []
        self.val_losses = []
        self.training_start_time = None
        self.current_batch = 0
        
        # Timing breakdown
        self.data_gen_time = 0
        self.forward_time = 0
        self.backward_time = 0

    def _generate_batch(
        self,
        batch_size: int,
        seq_len: int,
        pos_values: List[int],
        snr_for_batch,
        tdl_config: Union[str, List[str]],
        snr_per_sample: bool = False,
    ):
        """Generate one batch and return both logging SNR and loss SNR inputs."""
        y, h_targets, _, _, actual_snr, snr_tensor = generate_training_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            pos_values=pos_values,
            snr_db=snr_for_batch,
            tdl_config=tdl_config,
            snr_per_sample=snr_per_sample,
            return_complex=False,
            device=self.device,
            return_snr_tensor=True,
        )
        loss_snr = snr_tensor if snr_per_sample else actual_snr
        return y, h_targets, actual_snr, loss_snr
    
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
        validation_batches: int = 4,
        early_stop_loss: float = None,
        patience: int = 3,
        progress_tracker = None,  # Optional progress tracker for multi-task training
        save_interval: int = None,  # ✅ NEW: Save checkpoint every N batches
        save_dir: Union[str, Path] = None,  # ✅ NEW: Directory for periodic checkpoints
        keep_last_n: int = 2  # ✅ NEW: Keep only last N checkpoints
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
            validation_batches: Number of batches to average for validation
            early_stop_loss: Stop if loss below this value
            patience: Number of validations that must meet early stop
            progress_tracker: Optional progress tracker for multi-task training
            save_interval: Save checkpoint every N batches (optional) ✅ NEW
            save_dir: Directory for periodic checkpoints (optional) ✅ NEW
            keep_last_n: Keep only last N checkpoints (default: 2) ✅ NEW
        
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
        
        # ✅ Track saved checkpoints for cleanup
        saved_checkpoints = []
        
        # ✅ Adaptive print interval
        if print_interval is None:
            if num_batches <= 100:
                print_interval = 10   # Small task: every 10 batches
            elif num_batches <= 1000:
                print_interval = 100  # Medium task: every 100 batches
            else:
                print_interval = 200  # Large task: every 200 batches
        
        # ✅ For accurate throughput calculation (only between prints)
        last_print_batch = -1
        last_print_time = time.time()
        
        early_stop_counter = 0
        snr_per_sample = snr_config.per_sample if hasattr(snr_config, 'per_sample') else False
        
        for batch_idx in range(num_batches):
            self.current_batch = batch_idx
            
            # Get SNR for this batch
            snr_for_batch = snr_config.get_snr_for_data_generator()
            
            # Generate data ✅ Directly on device
            t0_data = time.time()
            y, h_targets, actual_snr, loss_snr = self._generate_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                pos_values=pos_values,
                snr_for_batch=snr_for_batch,
                tdl_config=tdl_config,
                snr_per_sample=snr_per_sample,
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
                    loss = calculate_loss(h_pred, h_targets, loss_snr, self.loss_type)
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
                loss = calculate_loss(h_pred, h_targets, loss_snr, self.loss_type)
                self.forward_time += time.time() - t0_fwd
                
                # Backward
                t0_bwd = time.time()
                loss.backward()
                self.optimizer.step()
                self.backward_time += time.time() - t0_bwd
            
            loss_value = loss.item()
            self.losses.append(loss_value)
            
            # ✅ Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss_value, batch_idx)
                self.writer.add_scalar('SNR/train', actual_snr, batch_idx)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], batch_idx)
            
            # Check if progress tracker should report (every 5 minutes)
            if progress_tracker:
                progress_tracker.check_and_report()
            
            # ✅ Print progress at adaptive intervals
            should_print = (
                (batch_idx + 1) % print_interval == 0 or  # Regular interval
                batch_idx == 0 or                          # First batch
                batch_idx == num_batches - 1              # Last batch
            )
            
            if should_print:
                metrics = evaluate_model(h_pred, h_targets, actual_snr)
                nmse = metrics['nmse']
                nmse_db = metrics['nmse_db']
                
                # ✅ Log NMSE to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('NMSE/train', nmse, batch_idx)
                    self.writer.add_scalar('NMSE_dB/train', nmse_db, batch_idx)
                
                # ✅ Calculate throughput: only samples since last print
                current_time = time.time()
                batches_since_print = (batch_idx - last_print_batch)
                time_since_print = current_time - last_print_time
                
                if time_since_print > 0 and batches_since_print > 0:
                    samples_per_sec = batches_since_print * batch_size / time_since_print
                else:
                    samples_per_sec = 0
                
                # ✅ Log throughput to TensorBoard
                if self.writer is not None and samples_per_sec > 0:
                    self.writer.add_scalar('Throughput/samples_per_sec', samples_per_sec, batch_idx)
                
                # Update for next print
                last_print_batch = batch_idx
                last_print_time = current_time
                
                # Calculate timing breakdown (cumulative, for reference)
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
            
            # ✅ Periodic checkpoint saving (keep only last N)
            if save_interval and save_dir and (batch_idx + 1) % save_interval == 0:
                checkpoint_path = Path(save_dir) / f'checkpoint_batch_{batch_idx+1}.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save new checkpoint
                self.save_checkpoint(
                    checkpoint_path,
                    additional_info={
                        'batch_idx': batch_idx + 1,
                        'num_batches': num_batches,
                        'training_time': time.time() - self.training_start_time
                    }
                )
                saved_checkpoints.append(checkpoint_path)
                print(f"  💾 Checkpoint saved: {checkpoint_path.name}")
                
                # Remove old checkpoints (keep only last N)
                while len(saved_checkpoints) > keep_last_n:
                    old_checkpoint = saved_checkpoints.pop(0)
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                        print(f"  🗑️  Removed old checkpoint: {old_checkpoint.name}")
            
            # Validation
            if val_interval and (batch_idx + 1) % val_interval == 0:
                val_loss, val_nmse_db = self.validate(
                    batch_size=batch_size,
                    snr_config=snr_config,
                    pos_values=pos_values,
                    tdl_config=tdl_config,
                    seq_len=seq_len,
                    num_batches=validation_batches,
                )
                self.val_losses.append(val_loss)
                
                # ✅ Update learning rate scheduler based on validation loss
                if self.scheduler is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Log LR change
                    if new_lr != old_lr:
                        print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                print(f"  Validation ({validation_batches} batches): Loss:{val_loss:.6f}, NMSE:{val_nmse_db:.2f}dB")
                
                # ✅ Log validation loss to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/validation', val_loss, batch_idx)
                    self.writer.add_scalar('NMSE_dB/validation', val_nmse_db, batch_idx)
                
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
        
        # ✅ Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            print(f"  📊 TensorBoard logs saved")
        
        return self.losses
    
    def validate(
        self,
        batch_size: int,
        snr_config,
        pos_values: List[int],
        tdl_config: Union[str, List[str]],
        seq_len: int,
        num_batches: int = 4,
    ) -> tuple[float, float]:
        """
        Validate model on several batches sampled from the training distribution.
        
        Returns:
            val_loss: Average validation loss
            val_nmse_db: Average validation NMSE in dB
        """
        self.model.eval()
        total_loss = 0.0
        total_nmse_db = 0.0
        snr_per_sample = snr_config.per_sample if hasattr(snr_config, 'per_sample') else False
        
        with torch.no_grad():
            for _ in range(num_batches):
                snr_for_batch = snr_config.get_snr_for_data_generator() if hasattr(snr_config, 'get_snr_for_data_generator') else snr_config
                y, h_targets, actual_snr, loss_snr = self._generate_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    pos_values=pos_values,
                    snr_for_batch=snr_for_batch,
                    tdl_config=tdl_config,
                    snr_per_sample=snr_per_sample,
                )

                h_pred = self.model(y)
                loss = calculate_loss(h_pred, h_targets, loss_snr, self.loss_type)
                metrics = evaluate_model(h_pred, h_targets, actual_snr)
                total_loss += loss.item()
                total_nmse_db += metrics['nmse_db']
        
        self.model.train()
        return total_loss / num_batches, total_nmse_db / num_batches
    
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
        
        # ✅ Get original model (handle torch.compile() wrapper)
        if hasattr(self.model, '_orig_mod'):
            # Model is compiled with torch.compile(), use the original
            original_model = self.model._orig_mod
        else:
            # Model is not compiled
            original_model = self.model
        
        # ✅ Save original model's state_dict (without _orig_mod prefix)
        checkpoint = {
            'model_state_dict': original_model.state_dict(),
            'model_info': original_model.get_model_info() if hasattr(original_model, 'get_model_info') else {},
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
