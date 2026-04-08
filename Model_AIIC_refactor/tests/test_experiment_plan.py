"""Unit tests for experiment plan helpers."""

import unittest

from utils.experiment_plan import (
    build_experiment_plan,
    prepare_model_config_variants,
    prepare_training_config_variants,
)


class TestExperimentPlan(unittest.TestCase):
    """Test experiment plan construction."""

    def test_prepare_training_variants_with_search_space(self):
        all_training_configs = {
            'loss_lr_search': {
                'fixed_params': {
                    'batch_size': 256,
                    'num_batches': 1000,
                },
                'search_space': {
                    'loss_type': ['nmse', 'log'],
                    'learning_rate': [0.01, 0.001],
                }
            }
        }

        variants = prepare_training_config_variants(all_training_configs, 'loss_lr_search')

        self.assertEqual(len(variants), 4)
        self.assertTrue(all(variant.variant_name.startswith('loss_lr_search_') for variant in variants))
        self.assertTrue(any('lossnmse' in variant.variant_name for variant in variants))
        self.assertTrue(any('lr0.001' in variant.variant_name for variant in variants))

    def test_prepare_training_variants_applies_overrides(self):
        all_training_configs = {
            'quick': {
                'batch_size': 32,
                'num_batches': 100,
            }
        }

        variants = prepare_training_config_variants(
            all_training_configs,
            'quick',
            batch_size_override=128,
            num_batches_override=500,
        )

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].config['batch_size'], 128)
        self.assertEqual(variants[0].config['num_batches'], 500)

    def test_build_experiment_plan_cartesian_product(self):
        all_model_configs = {
            'common': {'seq_len': 12},
            'models': {
                'separator1_test': {
                    'model_type': 'separator1',
                    'fixed_params': {'pos_values': [0, 3, 6, 9]},
                    'search_space': {'hidden_dim': [32, 64]}
                }
            }
        }
        all_training_configs = {
            'loss_search': {
                'fixed_params': {'num_batches': 100},
                'search_space': {'loss_type': ['nmse', 'log']}
            }
        }

        model_variants_by_name, missing = prepare_model_config_variants(
            all_model_configs,
            ['separator1_test']
        )
        training_variants = prepare_training_config_variants(all_training_configs, 'loss_search')
        plan = build_experiment_plan(['separator1_test'], model_variants_by_name, training_variants)

        self.assertEqual(missing, [])
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[0].task_index, 1)
        self.assertEqual(plan[-1].task_index, 4)
        self.assertTrue(all('loss_search_' in item.run_name for item in plan))


if __name__ == '__main__':
    unittest.main()