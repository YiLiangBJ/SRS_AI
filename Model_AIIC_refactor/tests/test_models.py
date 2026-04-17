"""
Unit tests for models.
"""

import unittest
import torch
from models import create_model, list_models, Separator1, Separator2


class TestModels(unittest.TestCase):
    """Test model creation and forward pass"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 32,
            'num_stages': 2,
            'mlp_depth': 3,
            'share_weights_across_stages': False
        }
        self.batch_size = 16
    
    def test_list_models(self):
        """Test listing available models"""
        models = list_models()
        self.assertIn('separator1', models)
        self.assertIn('separator2', models)
        self.assertGreater(len(models), 0)
    
    def test_create_separator1(self):
        """Test Separator1 creation"""
        model = create_model('separator1', self.config)
        self.assertIsInstance(model, Separator1)
        self.assertTrue(model.normalize_energy)
        
        # Test parameter count
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)

    def test_separator1_layer_norm_defaults_disabled(self):
        """Test Separator1 disables hidden LayerNorm unless explicitly enabled."""
        model = create_model('separator1', self.config)
        self.assertFalse(model.use_hidden_layer_norm)

        model_with_layer_norm = create_model(
            'separator1',
            {**self.config, 'use_hidden_layer_norm': True},
        )
        self.assertTrue(model_with_layer_norm.use_hidden_layer_norm)
    
    def test_create_separator2(self):
        """Test Separator2 creation"""
        config = {**self.config, 'activation_type': 'relu', 'onnx_mode': False}
        model = create_model('separator2', config)
        self.assertIsInstance(model, Separator2)
        
        # Test parameter count
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)
    
    def test_separator1_forward_real(self):
        """Test Separator1 forward pass with real stacked input"""
        model = create_model('separator1', self.config)
        
        # Real stacked input: (B, L*2)
        y = torch.randn(self.batch_size, self.config['seq_len'] * 2)
        
        # Forward pass
        h = model(y)
        
        # Check output shape: (B, P, L*2)
        expected_shape = (self.batch_size, self.config['num_ports'], self.config['seq_len'] * 2)
        self.assertEqual(h.shape, expected_shape)
    
    def test_separator1_forward_complex(self):
        """Test Separator1 forward pass with complex input"""
        model = create_model('separator1', self.config)
        
        # Complex input: (B, L)
        y = torch.randn(self.batch_size, self.config['seq_len'], dtype=torch.complex64)
        
        # Forward pass
        h = model(y)
        
        # Check output shape: (B, P, L) complex
        expected_shape = (self.batch_size, self.config['num_ports'], self.config['seq_len'])
        self.assertEqual(h.shape, expected_shape)
        self.assertTrue(h.dtype in [torch.complex64, torch.complex128])
    
    def test_separator2_forward(self):
        """Test Separator2 forward pass"""
        config = {**self.config, 'activation_type': 'relu', 'onnx_mode': False}
        model = create_model('separator2', config)
        
        # Real stacked input: (B, L*2)
        y = torch.randn(self.batch_size, self.config['seq_len'] * 2)
        
        # Forward pass
        h = model(y)
        
        # Check output shape: (B, P, L*2)
        expected_shape = (self.batch_size, self.config['num_ports'], self.config['seq_len'] * 2)
        self.assertEqual(h.shape, expected_shape)
    
    def test_model_info(self):
        """Test get_model_info method"""
        model = create_model('separator1', self.config)
        info = model.get_model_info()
        
        self.assertIn('model_class', info)
        self.assertIn('num_params', info)
        self.assertIn('normalize_energy', info)
        self.assertIn('seq_len', info)
        self.assertIn('num_ports', info)
        self.assertEqual(info['seq_len'], self.config['seq_len'])
        self.assertEqual(info['num_ports'], self.config['num_ports'])

    def test_separator1_is_scale_equivariant_with_internal_normalization(self):
        model = create_model('separator1', self.config)
        model.eval()

        y = torch.randn(self.batch_size, self.config['seq_len'] * 2)
        with torch.no_grad():
            reference = model(y)
            scaled = model(y * 7.5)

        self.assertTrue(torch.allclose(scaled, reference * 7.5, atol=1e-4, rtol=1e-4))

    def test_separator2_is_scale_equivariant_with_internal_normalization(self):
        config = {**self.config, 'activation_type': 'relu', 'onnx_mode': False}
        model = create_model('separator2', config)
        model.eval()

        y = torch.randn(self.batch_size, self.config['seq_len'] * 2)
        with torch.no_grad():
            reference = model(y)
            scaled = model(y * 3.0)

        self.assertTrue(torch.allclose(scaled, reference * 3.0, atol=1e-4, rtol=1e-4))
    
    def test_from_config(self):
        """Test from_config class method"""
        model = Separator1.from_config(self.config)
        self.assertIsInstance(model, Separator1)
        self.assertEqual(model.seq_len, self.config['seq_len'])
        self.assertEqual(model.num_ports, self.config['num_ports'])


if __name__ == '__main__':
    unittest.main()
