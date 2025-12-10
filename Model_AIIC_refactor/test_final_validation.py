"""
Comprehensive test for pos_values and SNR refactoring
"""

import yaml
from pathlib import Path
import sys

print("="*80)
print("Configuration Refactoring Validation")
print("="*80)

passed = 0
failed = 0

# Test 1: pos_values in model_configs
print("\n[Test 1] pos_values in model_configs")
try:
    with open('configs/model_configs.yaml', 'r', encoding='utf-8') as f:
        model_configs = yaml.safe_load(f)
    
    # Check common config
    common = model_configs.get('common', {})
    assert 'pos_values' in common, "pos_values missing in common"
    assert common['pos_values'] == [0, 3, 6, 9], "Wrong pos_values in common"
    
    # Check 6-port config
    sep1_6ports = model_configs['models'].get('separator1_6ports')
    assert sep1_6ports is not None, "separator1_6ports missing"
    assert sep1_6ports.get('num_ports') == 6, "Wrong num_ports"
    assert sep1_6ports.get('pos_values') == [0, 2, 4, 6, 8, 10], "Wrong 6-port pos_values"
    
    print("  ✓ pos_values correctly placed in model_configs")
    passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failed += 1

# Test 2: pos_values removed from training_configs
print("\n[Test 2] pos_values removed from training_configs")
try:
    with open('configs/training_configs.yaml', 'r', encoding='utf-8') as f:
        training_configs = yaml.safe_load(f)
    
    # Check that pos_values is NOT in any training config
    for config_name, config in training_configs.items():
        if isinstance(config, dict):
            assert 'pos_values' not in config, f"pos_values found in {config_name}"
    
    print("  ✓ pos_values correctly removed from training_configs")
    passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failed += 1

# Test 3: SNR config in new format
print("\n[Test 3] SNR config in new format")
try:
    default_config = training_configs.get('default', {})
    snr_config = default_config.get('snr_config')
    
    assert snr_config is not None, "snr_config missing"
    assert 'type' in snr_config, "type missing in snr_config"
    assert snr_config['type'] in ['range', 'discrete'], "Invalid SNR type"
    
    # Check range type
    if snr_config['type'] == 'range':
        assert 'min' in snr_config, "min missing"
        assert 'max' in snr_config, "max missing"
    
    # Check discrete type exists
    discrete_config = training_configs.get('discrete_snr')
    assert discrete_config is not None, "discrete_snr config missing"
    discrete_snr = discrete_config.get('snr_config')
    assert discrete_snr['type'] == 'discrete', "Wrong type"
    assert 'values' in discrete_snr, "values missing in discrete SNR"
    
    print("  ✓ SNR config in correct new format")
    passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failed += 1

# Test 4: SNRConfig class
print("\n[Test 4] SNRConfig class functionality")
try:
    from utils import SNRConfig, parse_snr_config
    
    # Test range
    config1 = {'type': 'range', 'min': 0, 'max': 30}
    snr1 = SNRConfig(config1)
    sample1 = snr1.sample()
    assert isinstance(sample1, (int, float)), "Sample should be number"
    assert 0 <= sample1 <= 30, "Sample out of range"
    
    # Test discrete
    config2 = {'type': 'discrete', 'values': [0, 10, 20, 30]}
    snr2 = SNRConfig(config2)
    sample2 = snr2.sample()
    assert sample2 in [0, 10, 20, 30], "Sample not in discrete values"
    
    # Test parsing
    snr3 = parse_snr_config(default_config.get('snr_config'))
    assert isinstance(snr3, SNRConfig), "Parsing failed"
    
    print("  ✓ SNRConfig class working correctly")
    passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failed += 1

# Test 5: Config name generation
print("\n[Test 5] Config name generation")
try:
    from utils import generate_config_name
    
    config = {
        'model_type': 'separator1',
        'hidden_dim': 64,
        'num_stages': 3,
        'mlp_depth': 3,
        'num_ports': 4,
        'pos_values': [0, 3, 6, 9],  # Should NOT appear in name
        'seq_len': 12                 # Should NOT appear in name
    }
    
    name = generate_config_name(config)
    
    # Check important params are included
    assert 'hd64' in name, "hidden_dim missing from name"
    assert 'stages3' in name, "num_stages missing from name"
    
    # Check unimportant params are excluded
    assert 'pos_values' not in name, "pos_values should not be in name"
    assert 'seq_len' not in name, "seq_len should not be in name"
    assert '[0, 3, 6, 9]' not in name, "pos_values array should not be in name"
    
    print(f"  ✓ Config name generation correct: {name}")
    passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failed += 1

# Test 6: Hierarchical naming
print("\n[Test 6] Hierarchical naming structure")
try:
    test_dir = Path('./test_final/separator1_small_quick_test')
    
    # Check experiment directory exists
    assert test_dir.exists(), "Experiment directory missing"
    
    # Check model instance directory exists
    model_dirs = list(test_dir.iterdir())
    assert len(model_dirs) > 0, "No model instance directories"
    
    model_dir = model_dirs[0]
    assert model_dir.is_dir(), "Model instance is not a directory"
    
    # Check required files
    assert (model_dir / 'model.pth').exists(), "model.pth missing"
    assert (model_dir / 'config.yaml').exists(), "config.yaml missing"
    
    print(f"  ✓ Hierarchical naming working: {model_dir.name}")
    passed += 1
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    failed += 1

# Summary
print("\n" + "="*80)
print("Test Summary")
print("="*80)
print(f"Passed: {passed}/6")
print(f"Failed: {failed}/6")

if failed == 0:
    print("\n🎉 All tests passed! Refactoring successful!")
    sys.exit(0)
else:
    print(f"\n❌ {failed} test(s) failed. Please review.")
    sys.exit(1)
