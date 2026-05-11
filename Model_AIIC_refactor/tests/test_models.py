"""
Unit tests for models.
"""

import unittest
import torch
from models import create_model, list_models, FullMLP, Separator1, Separator2, Separator3


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
        self.assertIn('full_mlp', models)
        self.assertIn('separator1', models)
        self.assertIn('separator2', models)
        self.assertIn('separator3', models)
        self.assertGreater(len(models), 0)

    def test_create_full_mlp(self):
        """Test FullMLP creation."""
        model = create_model('full_mlp', self.config)
        self.assertIsInstance(model, FullMLP)
        self.assertTrue(model.normalize_energy)
        self.assertEqual(model.residual_correction_mode, 'none')

        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)

    def test_full_mlp_depth2_uses_hidden_dim(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'mlp_depth': 2,
            'normalize_energy': False,
        }
        model = create_model('full_mlp', config)
        linear_layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]

        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(linear_layers[0].in_features, 24)
        self.assertEqual(linear_layers[0].out_features, 8)
        self.assertEqual(linear_layers[1].in_features, 8)
        self.assertEqual(linear_layers[1].out_features, 96)

    def test_full_mlp_depth2_parameter_count_depends_on_hidden_dim(self):
        small = create_model('full_mlp', {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'mlp_depth': 2,
        })
        large = create_model('full_mlp', {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 16,
            'mlp_depth': 2,
        })

        self.assertNotEqual(
            sum(parameter.numel() for parameter in small.parameters()),
            sum(parameter.numel() for parameter in large.parameters()),
        )

    def test_full_mlp_masked_residual_uses_pos_values_for_selected_taps(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'pos_values': [0, 3, 6, 9],
            'hidden_dim': 8,
            'mlp_depth': 2,
            'normalize_energy': False,
            'residual_correction_mode': 'masked',
        }
        model = create_model('full_mlp', config)
        for parameter in model.parameters():
            parameter.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = torch.zeros_like(h)
        for branch_idx, pos_value in enumerate(config['pos_values']):
            expected[0, branch_idx, pos_value] = y[0, pos_value]
            expected[0, branch_idx, pos_value + config['seq_len']] = y[0, pos_value + config['seq_len']]
        self.assertTrue(torch.equal(h, expected))

    def test_full_mlp_masked_residual_supports_six_port_pos_values(self):
        config = {
            'seq_len': 12,
            'num_ports': 6,
            'pos_values': [0, 2, 4, 6, 8, 10],
            'hidden_dim': 8,
            'mlp_depth': 2,
            'normalize_energy': False,
            'residual_correction_mode': 'masked',
        }
        model = create_model('full_mlp', config)
        for parameter in model.parameters():
            parameter.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = torch.zeros_like(h)
        for branch_idx, pos_value in enumerate(config['pos_values']):
            expected[0, branch_idx, pos_value] = y[0, pos_value]
            expected[0, branch_idx, pos_value + config['seq_len']] = y[0, pos_value + config['seq_len']]
        self.assertTrue(torch.equal(h, expected))

    def test_full_mlp_learned_dense_residual_zero_mask_blocks_residual(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'mlp_depth': 2,
            'normalize_energy': False,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('full_mlp', config)
        for parameter in model.parameters():
            if parameter is not model.learned_residual_mask:
                parameter.data.zero_()
        model.learned_residual_mask.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        self.assertTrue(torch.equal(h, torch.zeros_like(h)))

    def test_full_mlp_learned_dense_residual_unit_mask_broadcasts_full_residual(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'mlp_depth': 2,
            'normalize_energy': False,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('full_mlp', config)
        for parameter in model.parameters():
            if parameter is not model.learned_residual_mask:
                parameter.data.zero_()
        model.learned_residual_mask.data.fill_(1.0)

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = y.unsqueeze(1).repeat(1, config['num_ports'], 1)
        self.assertTrue(torch.equal(h, expected))

    def test_full_mlp_learned_dense_residual_receives_gradients(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'mlp_depth': 3,
            'normalize_energy': False,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('full_mlp', config)
        y = torch.randn(2, config['seq_len'] * 2)

        loss = model(y).sum()
        loss.backward()

        self.assertIsNotNone(model.learned_residual_mask.grad)
    
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

    def test_separator1_residual_correction_mode_defaults_to_global(self):
        model = create_model('separator1', self.config)
        self.assertEqual(model.residual_correction_mode, 'global')

    def test_separator1_masked_residual_uses_pos_values_for_selected_taps(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'pos_values': [0, 3, 6, 9],
            'hidden_dim': 8,
            'num_stages': 1,
            'mlp_depth': 2,
            'share_weights_across_stages': False,
            'normalize_energy': False,
            'residual_correction_mode': 'masked',
        }
        model = create_model('separator1', config)
        for parameter in model.parameters():
            parameter.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = torch.zeros_like(h)
        for branch_idx, pos_value in enumerate(config['pos_values']):
            expected[0, branch_idx, pos_value] = y[0, pos_value]
            expected[0, branch_idx, pos_value + config['seq_len']] = y[0, pos_value + config['seq_len']]
        self.assertTrue(torch.equal(h, expected))

    def test_separator1_masked_residual_supports_six_port_pos_values(self):
        config = {
            'seq_len': 12,
            'num_ports': 6,
            'pos_values': [0, 2, 4, 6, 8, 10],
            'hidden_dim': 8,
            'num_stages': 1,
            'mlp_depth': 2,
            'share_weights_across_stages': False,
            'normalize_energy': False,
            'residual_correction_mode': 'masked',
        }
        model = create_model('separator1', config)
        for parameter in model.parameters():
            parameter.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = torch.zeros_like(h)
        for branch_idx, pos_value in enumerate(config['pos_values']):
            expected[0, branch_idx, pos_value] = y[0, pos_value]
            expected[0, branch_idx, pos_value + config['seq_len']] = y[0, pos_value + config['seq_len']]
        self.assertTrue(torch.equal(h, expected))

    def test_separator1_learned_dense_residual_zero_mask_blocks_residual(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'num_stages': 1,
            'mlp_depth': 2,
            'share_weights_across_stages': False,
            'normalize_energy': False,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator1', config)
        for parameter in model.parameters():
            if parameter is not model.learned_residual_mask:
                parameter.data.zero_()
        model.learned_residual_mask.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        self.assertTrue(torch.equal(h, torch.zeros_like(h)))

    def test_separator1_learned_dense_residual_unit_mask_matches_global_for_single_stage(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'hidden_dim': 8,
            'num_stages': 1,
            'mlp_depth': 2,
            'share_weights_across_stages': False,
            'normalize_energy': False,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator1', config)
        for parameter in model.parameters():
            if parameter is not model.learned_residual_mask:
                parameter.data.zero_()
        model.learned_residual_mask.data.fill_(1.0)

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = y.unsqueeze(1).repeat(1, config['num_ports'], 1)
        self.assertTrue(torch.equal(h, expected))

    def test_separator1_learned_dense_residual_respects_stage_sharing_shape(self):
        shared_model = create_model(
            'separator1',
            {**self.config, 'residual_correction_mode': 'learned_dense', 'share_weights_across_stages': True},
        )
        self.assertEqual(shared_model.learned_residual_mask.shape, (self.config['num_ports'], self.config['seq_len'] * 2))

        unshared_model = create_model(
            'separator1',
            {**self.config, 'residual_correction_mode': 'learned_dense', 'share_weights_across_stages': False},
        )
        self.assertEqual(
            unshared_model.learned_residual_mask.shape,
            (self.config['num_stages'], self.config['num_ports'], self.config['seq_len'] * 2),
        )

    def test_separator1_learned_dense_residual_receives_gradients(self):
        config = {**self.config, 'residual_correction_mode': 'learned_dense', 'normalize_energy': False}
        model = create_model('separator1', config)
        y = torch.randn(2, config['seq_len'] * 2)

        loss = model(y).sum()
        loss.backward()

        self.assertIsNotNone(model.learned_residual_mask.grad)
    
    def test_create_separator2(self):
        """Test Separator2 creation"""
        config = {**self.config, 'activation_type': 'relu', 'onnx_mode': False}
        model = create_model('separator2', config)
        self.assertIsInstance(model, Separator2)
        
        # Test parameter count
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)

    def test_create_separator3(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': True,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)
        self.assertIsInstance(model, Separator3)
        self.assertEqual(model.residual_correction_mode, 'learned_dense')
        self.assertEqual(model.num_stages, 2)
        self.assertEqual(model.stage_hidden_dims, [64, 64])
        self.assertEqual(model.learned_residual_masks.shape, (2, 4, 24))

    def test_separator3_supports_stage_specific_hidden_dims(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': True,
            'stage_hidden_dims': [128, 64, 64],
            'num_stages': 3,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)

        self.assertEqual(model.stage_hidden_dims, [128, 64, 64])
        self.assertEqual(model.stages[0].network[0].in_features, 24)
        self.assertEqual(model.stages[0].network[0].out_features, 128)
        self.assertEqual(model.stages[1].network[0].in_features, 96)
        self.assertEqual(model.stages[1].network[0].out_features, 64)

    def test_separator3_forward_real(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': True,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)
        y = torch.randn(self.batch_size, config['seq_len'] * 2)
        h = model(y)
        self.assertEqual(h.shape, (self.batch_size, config['num_ports'], config['seq_len'] * 2))

    def test_separator3_forward_complex(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': True,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)
        y = torch.randn(self.batch_size, config['seq_len'], dtype=torch.complex64)
        h = model(y)
        self.assertEqual(h.shape, (self.batch_size, config['num_ports'], config['seq_len']))
        self.assertTrue(h.dtype in [torch.complex64, torch.complex128])

    def test_separator3_learned_dense_zero_masks_block_residual(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': False,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)
        for stage in model.stages:
            for parameter in stage.parameters():
                parameter.data.zero_()
        model.learned_residual_masks.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)
        self.assertTrue(torch.equal(h, torch.zeros_like(h)))

    def test_separator3_learned_dense_unit_masks_broadcast_full_residual(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': False,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)
        for stage in model.stages:
            for parameter in stage.parameters():
                parameter.data.zero_()
        model.learned_residual_masks.data.fill_(1.0)

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)
        expected = y.unsqueeze(1).repeat(1, config['num_ports'], 1)
        self.assertTrue(torch.equal(h, expected))

    def test_separator3_learned_dense_residual_receives_gradients(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': False,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'learned_dense',
        }
        model = create_model('separator3', config)
        y = torch.randn(2, config['seq_len'] * 2)
        loss = model(y).sum()
        loss.backward()
        self.assertIsNotNone(model.learned_residual_masks.grad)

    def test_separator3_generated_dense_creation(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': False,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'generated_dense',
            'pos_values': [0, 3, 6, 9],
        }
        model = create_model('separator3', config)
        self.assertIsInstance(model, Separator3)
        self.assertEqual(model.residual_correction_mode, 'generated_dense')
        self.assertIsNone(model.learned_residual_masks)
        self.assertEqual(len(model.mask_generators), 2)

    def test_separator3_generated_dense_zero_delta_uses_base_mask(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': False,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'generated_dense',
            'pos_values': [0, 3, 6, 9],
        }
        model = create_model('separator3', config)
        for stage in model.stages:
            for parameter in stage.parameters():
                parameter.data.zero_()
        for generator in model.mask_generators:
            for parameter in generator.parameters():
                parameter.data.zero_()

        y = torch.arange(1.0, 25.0).unsqueeze(0)
        h = model(y)

        expected = torch.zeros_like(h)
        for branch_idx, pos_value in enumerate(config['pos_values']):
            expected[0, branch_idx, pos_value] = y[0, pos_value]
            expected[0, branch_idx, pos_value + config['seq_len']] = y[0, pos_value + config['seq_len']]
        self.assertTrue(torch.equal(h, expected))

    def test_separator3_generated_dense_receives_gradients(self):
        config = {
            'seq_len': 12,
            'num_ports': 4,
            'normalize_energy': False,
            'hidden_dim': 64,
            'num_stages': 2,
            'mlp_depth': 2,
            'residual_correction_mode': 'generated_dense',
            'pos_values': [0, 3, 6, 9],
        }
        model = create_model('separator3', config)
        y = torch.randn(2, config['seq_len'] * 2)
        loss = model(y).sum()
        loss.backward()
        for generator in model.mask_generators:
            for parameter in generator.parameters():
                self.assertIsNotNone(parameter.grad)
    
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

    def test_full_mlp_forward_real(self):
        """Test FullMLP forward pass with real stacked input."""
        model = create_model('full_mlp', self.config)
        y = torch.randn(self.batch_size, self.config['seq_len'] * 2)

        h = model(y)

        expected_shape = (self.batch_size, self.config['num_ports'], self.config['seq_len'] * 2)
        self.assertEqual(h.shape, expected_shape)

    def test_full_mlp_forward_complex(self):
        """Test FullMLP forward pass with complex input."""
        model = create_model('full_mlp', self.config)
        y = torch.randn(self.batch_size, self.config['seq_len'], dtype=torch.complex64)

        h = model(y)

        expected_shape = (self.batch_size, self.config['num_ports'], self.config['seq_len'])
        self.assertEqual(h.shape, expected_shape)
        self.assertTrue(h.dtype in [torch.complex64, torch.complex128])
    
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

    def test_full_mlp_is_scale_equivariant_with_internal_normalization(self):
        model = create_model('full_mlp', self.config)
        model.eval()

        y = torch.randn(self.batch_size, self.config['seq_len'] * 2)
        with torch.no_grad():
            reference = model(y)
            scaled = model(y * 5.0)

        self.assertTrue(torch.allclose(scaled, reference * 5.0, atol=1e-4, rtol=1e-4))
    
    def test_from_config(self):
        """Test from_config class method"""
        model = Separator1.from_config(self.config)
        self.assertIsInstance(model, Separator1)
        self.assertEqual(model.seq_len, self.config['seq_len'])
        self.assertEqual(model.num_ports, self.config['num_ports'])


if __name__ == '__main__':
    unittest.main()
