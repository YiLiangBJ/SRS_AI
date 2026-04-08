"""
Unit tests for config_parser.
"""

import unittest
from utils.config_parser import (
    parse_search_space_value,
    expand_search_space,
    parse_config_variants,
    generate_config_name
)


class TestConfigParser(unittest.TestCase):
    """Test configuration parser functions"""
    
    def test_parse_single_value(self):
        """Test parsing single value"""
        result = parse_search_space_value(64)
        self.assertEqual(result, [64])
    
    def test_parse_list_values(self):
        """Test parsing list of values"""
        result = parse_search_space_value([32, 64, 128])
        self.assertEqual(result, [32, 64, 128])
    
    def test_parse_choice(self):
        """Test parsing choice type"""
        config = {'type': 'choice', 'values': [32, 64, 128]}
        result = parse_search_space_value(config)
        self.assertEqual(result, [32, 64, 128])
    
    def test_parse_range(self):
        """Test parsing range type"""
        config = {'type': 'range', 'min': 2, 'max': 5, 'step': 1}
        result = parse_search_space_value(config)
        self.assertEqual(result, [2, 3, 4, 5])
    
    def test_expand_search_space_simple(self):
        """Test expanding simple search space"""
        search_space = {
            'hidden_dim': [32, 64],
            'num_stages': [2, 3]
        }
        configs = expand_search_space(search_space)
        
        self.assertEqual(len(configs), 4)  # 2 x 2
        
        # Check all combinations exist
        expected = [
            {'hidden_dim': 32, 'num_stages': 2},
            {'hidden_dim': 32, 'num_stages': 3},
            {'hidden_dim': 64, 'num_stages': 2},
            {'hidden_dim': 64, 'num_stages': 3}
        ]
        self.assertEqual(sorted(configs, key=str), sorted(expected, key=str))
    
    def test_expand_search_space_three_params(self):
        """Test expanding search space with three parameters"""
        search_space = {
            'hidden_dim': [32, 64],
            'num_stages': [2, 3],
            'mlp_depth': [2, 3]
        }
        configs = expand_search_space(search_space)
        
        self.assertEqual(len(configs), 8)  # 2 x 2 x 2
    
    def test_parse_model_config_single(self):
        """Test parsing single configuration (backward compatible)"""
        config = {
            'model_type': 'separator1',
            'hidden_dim': 64,
            'num_stages': 3
        }
        result = parse_config_variants(config)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], config)
    
    def test_parse_model_config_search_space(self):
        """Test parsing search space configuration"""
        config = {
            'model_type': 'separator1',
            'search_space': {
                'hidden_dim': [32, 64],
                'num_stages': [2, 3]
            }
        }
        result = parse_config_variants(config)
        
        self.assertEqual(len(result), 4)  # 2 x 2
        
        # All should have model_type
        for cfg in result:
            self.assertEqual(cfg['model_type'], 'separator1')
    
    def test_parse_model_config_fixed_and_search(self):
        """Test parsing fixed + search parameters"""
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
        result = parse_config_variants(config)
        
        self.assertEqual(len(result), 4)  # 2 x 2
        
        # All should have fixed params
        for cfg in result:
            self.assertEqual(cfg['model_type'], 'separator1')
            self.assertEqual(cfg['mlp_depth'], 3)
            self.assertEqual(cfg['share_weights_across_stages'], False)
            self.assertIn('hidden_dim', cfg)
            self.assertIn('num_stages', cfg)
    
    def test_generate_config_name(self):
        """Test generating configuration name"""
        config = {
            'model_type': 'separator1',
            'hidden_dim': 64,
            'num_stages': 3
        }
        name = generate_config_name(config)
        
        self.assertIn('separator1', name)
        self.assertIn('64', name)
        self.assertIn('3', name)
    
    def test_generate_config_name_with_base(self):
        """Test generating configuration name with base name"""
        config = {
            'model_type': 'separator1',
            'hidden_dim': 64,
            'num_stages': 3
        }
        name = generate_config_name(config, base_name='test_config')
        
        self.assertIn('test_config', name)
        self.assertIn('64', name)


if __name__ == '__main__':
    unittest.main()
