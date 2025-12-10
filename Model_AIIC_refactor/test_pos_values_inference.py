"""
Test pos_values and num_ports inference
"""

import yaml
from pathlib import Path
from utils import parse_model_config, generate_config_name

print("="*80)
print("Test: pos_values and num_ports inference")
print("="*80)

# Load model configs
config_file = Path(__file__).parent / 'configs' / 'model_configs.yaml'
with open(config_file, 'r', encoding='utf-8') as f:
    all_configs = yaml.safe_load(f)

# Test 1: Simple 4-port configuration
print("\n[Test 1] separator1_default (4-port)")
config = all_configs['models']['separator1_default']
parsed = parse_model_config(config)
print(f"  pos_values: {parsed[0].get('pos_values')}")
print(f"  num_ports: {parsed[0].get('num_ports')} (inferred)")
assert parsed[0]['num_ports'] == 4, "Should infer 4 ports"
assert len(parsed[0]['pos_values']) == 4, "Should have 4 pos_values"
print("  ✓ Passed")

# Test 2: 6-port configuration
print("\n[Test 2] separator1_6ports (6-port)")
config = all_configs['models']['separator1_6ports']
parsed = parse_model_config(config)
print(f"  pos_values: {parsed[0].get('pos_values')}")
print(f"  num_ports: {parsed[0].get('num_ports')} (inferred)")
assert parsed[0]['num_ports'] == 6, "Should infer 6 ports"
assert len(parsed[0]['pos_values']) == 6, "Should have 6 pos_values"
print("  ✓ Passed")

# Test 3: Grid search with fixed pos_values
print("\n[Test 3] separator1_grid_search_basic")
config = all_configs['models']['separator1_grid_search_basic']
parsed = parse_model_config(config)
print(f"  Number of configs: {len(parsed)}")
print(f"  All have pos_values: {all('pos_values' in c for c in parsed)}")
print(f"  All have num_ports: {all('num_ports' in c for c in parsed)}")
print(f"  Sample config:")
print(f"    pos_values: {parsed[0]['pos_values']}")
print(f"    num_ports: {parsed[0]['num_ports']} (inferred)")
assert all(c['num_ports'] == 4 for c in parsed), "All should be 4-port"
print("  ✓ Passed")

# Test 4: 6-port grid search
print("\n[Test 4] separator1_6ports_search")
config = all_configs['models']['separator1_6ports_search']
parsed = parse_model_config(config)
print(f"  Number of configs: {len(parsed)}")
print(f"  Sample config:")
print(f"    pos_values: {parsed[0]['pos_values']}")
print(f"    num_ports: {parsed[0]['num_ports']} (inferred)")
assert all(c['num_ports'] == 6 for c in parsed), "All should be 6-port"
assert all(len(c['pos_values']) == 6 for c in parsed), "All should have 6 pos_values"
print("  ✓ Passed")

# Test 5: Config name generation (should NOT include ports)
print("\n[Test 5] Config name generation")
test_config = {
    'model_type': 'separator1',
    'pos_values': [0, 3, 6, 9],
    'num_ports': 4,  # This should be excluded from name
    'hidden_dim': 64,
    'num_stages': 3,
    'mlp_depth': 3,
    'share_weights_across_stages': False
}
name = generate_config_name(test_config)
print(f"  Generated name: {name}")
assert 'ports' not in name, "Name should NOT include 'ports' (it's derived)"
assert 'hd64' in name, "Name should include hidden_dim"
assert 'stages3' in name, "Name should include num_stages"
print("  ✓ Passed (ports not in name)")

# Test 6: Verify common config no longer has num_ports
print("\n[Test 6] Common config")
common_config = all_configs.get('common', {})
print(f"  seq_len: {common_config.get('seq_len')}")
print(f"  num_ports: {common_config.get('num_ports', 'NOT PRESENT')}")
assert 'num_ports' not in common_config, "common should not have num_ports"
print("  ✓ Passed (num_ports not in common)")

# Summary
print("\n" + "="*80)
print("Summary")
print("="*80)
print("✓ All tests passed!")
print()
print("Key points:")
print("  ✅ pos_values is a model parameter (defines which ports)")
print("  ✅ num_ports is inferred from len(pos_values)")
print("  ✅ num_ports not in common config")
print("  ✅ num_ports not in generated names")
print("  ✅ Works for both 4-port and 6-port models")
print("  ✅ Works for single configs and grid search")
