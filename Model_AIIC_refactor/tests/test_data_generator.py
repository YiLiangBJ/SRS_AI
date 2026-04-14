"""
Unit tests for data generation.
"""

import unittest
import torch
from data import generate_training_batch


class TestDataGenerator(unittest.TestCase):
    """Test data generation functions"""
    
    def test_generate_batch_basic(self):
        """Test basic batch generation"""
        y, h_targets, pos, h_true, snr = generate_training_batch(
            batch_size=32,
            snr_db=20.0,
            tdl_config='A-30'
        )
        
        # Check shapes
        self.assertEqual(y.shape, (32, 24))  # (B, L*2)
        self.assertEqual(h_targets.shape, (32, 4, 24))  # (B, P, L*2)
        self.assertEqual(h_true.shape, (32, 4, 24))
        
        # Check SNR
        self.assertAlmostEqual(snr, 20.0, places=1)
        
        # Check pos_values
        self.assertEqual(len(pos), 4)
    
    def test_generate_batch_snr_range(self):
        """Test batch generation with SNR range"""
        y, h_targets, pos, h_true, snr = generate_training_batch(
            batch_size=32,
            snr_db=(0, 30),
            tdl_config='A-30'
        )
        
        # Check SNR is in range
        self.assertGreaterEqual(snr, 0)
        self.assertLessEqual(snr, 30)
    
    def test_generate_batch_custom_ports(self):
        """Test batch generation with custom port positions"""
        pos_values = [0, 2, 4, 6, 8, 10]
        y, h_targets, pos, h_true, snr = generate_training_batch(
            batch_size=16,
            pos_values=pos_values,
            snr_db=15.0
        )
        
        # Check number of ports
        self.assertEqual(h_targets.shape[1], 6)  # 6 ports
        self.assertEqual(pos, pos_values)
    
    def test_generate_batch_complex_output(self):
        """Test generation with complex output"""
        y, h_targets, pos, h_true, snr = generate_training_batch(
            batch_size=32,
            snr_db=20.0,
            return_complex=True
        )
        
        # Check complex format
        self.assertEqual(y.shape, (32, 12))  # (B, L) complex
        self.assertTrue(y.dtype in [torch.complex64, torch.complex128])
        self.assertEqual(h_targets.shape, (32, 4, 12))

    def test_generate_batch_returns_per_sample_snr_tensor(self):
        """Test optional per-sample SNR tensor output for loss weighting."""
        _, _, _, _, snr_mean, snr_tensor = generate_training_batch(
            batch_size=32,
            snr_db=(0, 30),
            snr_per_sample=True,
            return_snr_tensor=True,
        )

        self.assertEqual(snr_tensor.shape, (32,))
        self.assertGreaterEqual(float(snr_tensor.min()), 0.0)
        self.assertLessEqual(float(snr_tensor.max()), 30.0)
        self.assertAlmostEqual(float(snr_tensor.mean()), snr_mean, places=4)
    
    def test_different_tdl_configs(self):
        """Test different TDL configurations"""
        for tdl_config in ['A-30', 'B-100', 'C-300']:
            y, h_targets, pos, h_true, snr = generate_training_batch(
                batch_size=16,
                snr_db=20.0,
                tdl_config=tdl_config
            )
            
            self.assertEqual(y.shape, (16, 24))
            self.assertEqual(h_targets.shape, (16, 4, 24))


if __name__ == '__main__':
    unittest.main()
