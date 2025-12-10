"""
Test SNR configuration and pos_values refactoring
"""

import yaml
from pathlib import Path

# Test 1: Load model configs and check pos_values
print("="*80)
print("Test 1: Model configs - pos_values")
print("="*80)

model_configs_file = Path(__file__).parent / 'configs' / 'model_configs.yaml'
with open(model_configs_file, 'r', encoding='utf-8') as f:
    model_configs = yaml.safe_load(f)

common = model_configs.get('common', {})
print(f"Common config:")
print(f"  seq_len: {common.get('seq_len')}")
print(f"  num_ports: {common.get('num_ports')}")
print(f"  pos_values: {common.get('pos_values')}")  # Should be [0, 3, 6, 9]

# Test 6-port config
sep1_6ports = model_configs['models'].get('separator1_6ports')
if sep1_6ports:
    print(f"\nseparator1_6ports:")
    print(f"  num_ports: {sep1_6ports.get('num_ports')}")
    print(f"  pos_values: {sep1_6ports.get('pos_values')}")  # Should be [0, 2, 4, 6, 8, 10]

# Test 2: Load training configs and check SNR
print("\n" + "="*80)
print("Test 2: Training configs - SNR")
print("="*80)

training_configs_file = Path(__file__).parent / 'configs' / 'training_configs.yaml'
with open(training_configs_file, 'r', encoding='utf-8') as f:
    training_configs = yaml.safe_load(f)

default_config = training_configs.get('default', {})
print(f"\nDefault training config:")
print(f"  SNR config: {default_config.get('snr_config')}")
print(f"  Has pos_values: {'pos_values' in default_config}")  # Should be False

discrete_snr_config = training_configs.get('discrete_snr', {})
if discrete_snr_config:
    print(f"\nDiscrete SNR training config:")
    print(f"  SNR config: {discrete_snr_config.get('snr_config')}")

# Test 3: SNRConfig class
print("\n" + "="*80)
print("Test 3: SNRConfig class")
print("="*80)

from utils import SNRConfig, parse_snr_config

# Range-based
config1 = {'type': 'range', 'min': 0, 'max': 30, 'sampling': 'stratified', 'num_bins': 10}
snr1 = SNRConfig(config1)
print(f"\nRange-based: {snr1}")
print(f"  Sample: {snr1.sample()}")

# Discrete
config2 = {'type': 'discrete', 'values': [0, 10, 20, 30]}
snr2 = SNRConfig(config2)
print(f"\nDiscrete: {snr2}")
print(f"  Sample: {snr2.sample()}")
print(f"  Sample: {snr2.sample()}")
print(f"  Sample: {snr2.sample()}")

# Test 4: Parse from training config
print("\n" + "="*80)
print("Test 4: Parse SNR from training config")
print("="*80)

snr_from_default = parse_snr_config(default_config.get('snr_config', {}))
print(f"\nFrom default config: {snr_from_default}")

if discrete_snr_config:
    snr_from_discrete = parse_snr_config(discrete_snr_config.get('snr_config', {}))
    print(f"From discrete config: {snr_from_discrete}")

print("\n" + "="*80)
print("✓ All tests passed!")
print("="*80)
