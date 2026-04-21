"""Unit tests for v2 component-based experiment plan helpers."""

import tempfile
import unittest
from pathlib import Path

import yaml

from utils.experiment_plan import build_experiment_suite


class TestExperimentPlan(unittest.TestCase):
    """Test v2 experiment plan construction."""

    def test_build_experiment_suite_from_component_definition(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_separator1_v2',
        )

        self.assertEqual(suite.schema_version, 'v2')
        self.assertEqual(suite.task_recipe_name, 'channel_separator_4port_quick')
        self.assertEqual(suite.training_recipe_name, 'quick_supervised')
        self.assertEqual(suite.model_recipe_names, ['separator1_small'])
        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].task_recipe_name, 'channel_separator_4port_quick')
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'separator1')
        self.assertEqual(suite.plan[0].model_spec['seq_len'], 12)
        self.assertEqual(suite.plan[0].training_spec['loss_type'], 'nmse')
        self.assertEqual(suite.plan[0].training_spec['strategy_type'], 'standard_supervised')
        self.assertEqual(suite.plan[0].task_spec['params']['tdl_config'], 'A-30')

    def test_build_full_mlp_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_full_mlp_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'full_mlp')
        self.assertEqual(suite.plan[0].training_spec['loss_type'], 'nmse')
        self.assertEqual(suite.plan[0].model_spec['mlp_depth'], 3)

    def test_build_v2_experiment_suite_supports_nested_experiment_sweeps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / 'v2').mkdir(parents=True, exist_ok=True)

            with open(config_dir / 'v2' / 'tasks.yaml', 'w', encoding='utf-8') as config_file:
                yaml.safe_dump({
                    'tasks': {
                        'task_a': {
                            'type': 'channel_separator',
                            'params': {
                                'seq_len': 12,
                                'pos_values': [0, 3, 6, 9],
                                'snr_config': {'type': 'range', 'min': 0, 'max': 30},
                                'tdl_config': 'A-30',
                            },
                        }
                    }
                }, config_file, sort_keys=False)
            with open(config_dir / 'v2' / 'models.yaml', 'w', encoding='utf-8') as config_file:
                yaml.safe_dump({
                    'models': {
                        'full_mlp_base': {
                            'type': 'full_mlp',
                            'params': {
                                'hidden_dim': 32,
                                'mlp_depth': 3,
                            },
                        }
                    }
                }, config_file, sort_keys=False)
            with open(config_dir / 'v2' / 'training_strategies.yaml', 'w', encoding='utf-8') as config_file:
                yaml.safe_dump({
                    'training_strategies': {
                        'standard': {
                            'type': 'standard_supervised',
                            'params': {
                                'batch_size': 64,
                                'num_batches': 128,
                                'optimizer': {
                                    'type': 'adam',
                                    'params': {'learning_rate': 0.01},
                                },
                                'loss': {'type': 'nmse'},
                            },
                        }
                    }
                }, config_file, sort_keys=False)
            with open(config_dir / 'v2' / 'experiments.yaml', 'w', encoding='utf-8') as config_file:
                yaml.safe_dump({
                    'experiments': {
                        'nested_sweep': {
                            'task': 'task_a',
                            'model': 'full_mlp_base',
                            'training_strategy': 'standard',
                            'sweeps': [
                                {
                                    'target': 'model.params.hidden_dim',
                                    'alias': 'hd',
                                    'values': [32, 64],
                                },
                                {
                                    'target': 'training_strategy.params.loss.type',
                                    'alias': 'loss',
                                    'values': ['nmse', 'log'],
                                },
                            ],
                        }
                    }
                }, config_file, sort_keys=False)

            suite = build_experiment_suite(
                config_dir=config_dir,
                experiment_name='nested_sweep',
            )

            self.assertEqual(suite.schema_version, 'v2')
            self.assertEqual(len(suite.plan), 4)
            self.assertTrue(any('hd32' in item.run_name for item in suite.plan))
            self.assertTrue(any('losslog' in item.run_name for item in suite.plan))
            self.assertEqual({item.model_spec['hidden_dim'] for item in suite.plan}, {32, 64})
            self.assertEqual({item.training_spec['loss_type'] for item in suite.plan}, {'nmse', 'log'})

    def test_full_mlp_local_model_sweeps_expand(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='full_mlp_arch_search_v2',
        )

        self.assertEqual(len(suite.plan), 9)
        self.assertEqual({item.model_spec['hidden_dim'] for item in suite.plan}, {64, 128, 256})
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3, 4})
        self.assertTrue(all(item.training_spec['loss_type'] == 'nmse' for item in suite.plan))


if __name__ == '__main__':
    unittest.main()
