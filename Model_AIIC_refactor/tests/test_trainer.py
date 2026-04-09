"""
Unit tests for training components.
"""

import unittest
import torch
from models import create_model
from training import Trainer, calculate_loss, evaluate_model
from utils import parse_snr_config


class TestTraining(unittest.TestCase):
    """Test training components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 32,
            'num_stages': 2,
            'mlp_depth': 3
        }
        self.model = create_model('separator1', self.config)
        self.batch_size = 16
    
    def test_trainer_creation(self):
        """Test Trainer initialization"""
        trainer = Trainer(
            self.model,
            learning_rate=0.01,
            loss_type='nmse',
            device='cpu'
        )
        
        self.assertEqual(trainer.loss_type, 'nmse')
        self.assertEqual(trainer.device.type, 'cpu')
    
    def test_calculate_loss_nmse(self):
        """Test NMSE loss calculation"""
        pred = torch.randn(self.batch_size, 4, 24)
        target = torch.randn(self.batch_size, 4, 24)
        
        loss = calculate_loss(pred, target, snr_db=20.0, loss_type='nmse')
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertGreater(loss.item(), 0)
    
    def test_calculate_loss_weighted(self):
        """Test weighted loss calculation"""
        pred = torch.randn(self.batch_size, 4, 24)
        target = torch.randn(self.batch_size, 4, 24)
        
        loss = calculate_loss(pred, target, snr_db=10.0, loss_type='weighted')
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        pred = torch.randn(self.batch_size, 4, 24)
        target = torch.randn(self.batch_size, 4, 24)
        
        metrics = evaluate_model(pred, target, snr_db=20.0)
        
        self.assertIn('nmse', metrics)
        self.assertIn('nmse_db', metrics)
        self.assertIn('per_port_nmse', metrics)
        self.assertIn('snr_db', metrics)
        self.assertEqual(len(metrics['per_port_nmse']), 4)
    
    def test_trainer_train_short(self):
        """Test short training run"""
        trainer = Trainer(
            self.model,
            learning_rate=0.01,
            loss_type='nmse',
            device='cpu'
        )
        
        losses = trainer.train(
            num_batches=2,
            batch_size=16,
            snr_config=parse_snr_config({'type': 'range', 'min': 10, 'max': 20}),
            pos_values=[0, 3, 6, 9],
            print_interval=1
        )
        
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))


if __name__ == '__main__':
    unittest.main()
