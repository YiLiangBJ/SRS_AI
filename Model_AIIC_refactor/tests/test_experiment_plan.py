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
        self.assertEqual(suite.task_recipe_name, 'channel_separator_6port_quick')
        self.assertEqual(suite.training_recipe_name, 'quick_supervised')
        self.assertEqual(suite.model_recipe_names, ['separator1_small'])
        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].task_recipe_name, 'channel_separator_6port_quick')
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'separator1')
        self.assertEqual(suite.plan[0].model_spec['seq_len'], 12)
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)
        self.assertTrue(suite.plan[0].model_spec['normalize_energy'])
        self.assertEqual(suite.plan[0].training_spec['loss_type'], 'nmse')
        self.assertEqual(suite.plan[0].training_spec['strategy_type'], 'standard_supervised')
        self.assertEqual(suite.plan[0].task_spec['params']['tdl_config'], 'A-30')

    def test_build_masked_separator1_quick_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_separator1_masked_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'separator1')
        self.assertEqual(suite.plan[0].model_spec['residual_correction_mode'], 'masked')
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)

    def test_build_learned_dense_separator1_quick_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_separator1_learned_dense_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'separator1')
        self.assertEqual(suite.plan[0].model_spec['residual_correction_mode'], 'learned_dense')
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)

    def test_build_separator3_quick_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_separator3_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'separator3')
        self.assertEqual(suite.plan[0].model_spec['residual_correction_mode'], 'learned_dense')
        self.assertEqual(suite.plan[0].model_spec['hidden_dim'], 128)
        self.assertEqual(suite.plan[0].model_spec['num_stages'], 2)
        self.assertEqual(suite.plan[0].model_spec['mlp_depth'], 2)
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)

    def test_build_full_mlp_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_full_mlp_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'full_mlp')
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)
        self.assertEqual(suite.plan[0].training_spec['loss_type'], 'nmse')
        self.assertEqual(suite.plan[0].model_spec['mlp_depth'], 3)
        self.assertTrue(suite.plan[0].model_spec['normalize_energy'])

    def test_build_masked_full_mlp_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_full_mlp_masked_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'full_mlp')
        self.assertEqual(suite.plan[0].model_spec['residual_correction_mode'], 'masked')
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)

    def test_build_learned_dense_full_mlp_experiment_suite(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_full_mlp_learned_dense_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'full_mlp')
        self.assertEqual(suite.plan[0].model_spec['residual_correction_mode'], 'learned_dense')
        self.assertEqual(suite.plan[0].model_spec['num_ports'], 6)
        self.assertAlmostEqual(suite.plan[0].training_spec['learned_dense_mask_regularization'], 1.0e-5)

    def test_compare_default_models_includes_three_models_on_6port(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='compare_default_models_v2',
        )

        self.assertEqual(suite.task_recipe_name, 'channel_separator_6port_standard')
        self.assertEqual(len(suite.plan), 3)
        self.assertEqual(
            {item.model_spec['model_type'] for item in suite.plan},
            {'full_mlp', 'separator1', 'separator2'},
        )
        self.assertTrue(all(item.model_spec['num_ports'] == 6 for item in suite.plan))

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
                                'normalize_energy': True,
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
        depth2_items = [item for item in suite.plan if item.model_spec['mlp_depth'] == 2]
        self.assertEqual(len(depth2_items), 3)
        self.assertEqual({item.model_spec['hidden_dim'] for item in depth2_items}, {64, 128, 256})
        self.assertTrue(all('hd' in item.run_name for item in depth2_items))
        self.assertTrue(all(item.training_spec['loss_type'] == 'nmse' for item in suite.plan))

    def test_full_mlp_capacity_search_preserves_depth2_hidden_dim_variants(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='full_mlp_capacity_search_v2',
        )

        self.assertEqual(len(suite.plan), 20)
        depth2_items = [item for item in suite.plan if item.model_spec['mlp_depth'] == 2]
        self.assertEqual(len(depth2_items), 5)
        self.assertEqual({item.model_spec['hidden_dim'] for item in depth2_items}, {32, 64, 128, 256, 512})
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3, 4, 5})

    def test_masked_full_mlp_capacity_search_preserves_sweep_shape(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='full_mlp_capacity_search_masked_v2',
        )

        self.assertEqual(len(suite.plan), 20)
        self.assertTrue(all(item.model_spec['residual_correction_mode'] == 'masked' for item in suite.plan))
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3, 4, 5})

    def test_learned_dense_full_mlp_capacity_search_preserves_sweep_shape(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='full_mlp_capacity_search_learned_dense_v2',
        )

        self.assertEqual(len(suite.plan), 20)
        self.assertTrue(all(item.model_spec['residual_correction_mode'] == 'learned_dense' for item in suite.plan))
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3, 4, 5})

    def test_separator1_grid_search_sweeps_depth_stage_share_and_deduplicates_depth2_hidden_dim(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator1_v2',
        )

        self.assertEqual(len(suite.plan), 20)
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3})
        self.assertEqual({item.model_spec['num_stages'] for item in suite.plan}, {1, 2})
        self.assertEqual({item.model_spec['share_weights_across_stages'] for item in suite.plan}, {False, True})

        depth2_items = [item for item in suite.plan if item.model_spec['mlp_depth'] == 2]
        self.assertEqual(len(depth2_items), 4)
        self.assertEqual({item.model_spec['hidden_dim'] for item in depth2_items}, {64})
        self.assertTrue(all('hd' not in item.run_name for item in depth2_items))

        depth3_items = [item for item in suite.plan if item.model_spec['mlp_depth'] == 3]
        self.assertEqual(len(depth3_items), 16)
        self.assertEqual({item.model_spec['hidden_dim'] for item in depth3_items}, {16, 32, 64, 128})

    def test_masked_separator1_grid_search_preserves_sweep_shape(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator1_masked_v2',
        )

        self.assertEqual(len(suite.plan), 20)
        self.assertTrue(all(item.model_spec['residual_correction_mode'] == 'masked' for item in suite.plan))
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3})
        self.assertEqual({item.model_spec['num_stages'] for item in suite.plan}, {1, 2})
        self.assertEqual({item.model_spec['share_weights_across_stages'] for item in suite.plan}, {False, True})

    def test_learned_dense_separator1_grid_search_preserves_sweep_shape(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator1_learned_dense_v2',
        )

        self.assertEqual(len(suite.plan), 20)
        self.assertTrue(all(item.model_spec['residual_correction_mode'] == 'learned_dense' for item in suite.plan))
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3})
        self.assertEqual({item.model_spec['num_stages'] for item in suite.plan}, {1, 2})
        self.assertEqual({item.model_spec['share_weights_across_stages'] for item in suite.plan}, {False, True})

    def test_separator3_default_experiment_builds_one_run(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator3_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['model_type'], 'separator3')
        self.assertEqual(suite.plan[0].model_spec['hidden_dim'], 128)
        self.assertEqual(suite.plan[0].model_spec['num_stages'], 2)
        self.assertEqual(suite.plan[0].model_spec['mlp_depth'], 2)

    def test_learned_dense_separator3_standard_grid_search_builds_depth_stage_width_variants(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator3_learned_dense_v2',
        )

        self.assertEqual(len(suite.plan), 32)
        self.assertEqual({item.model_spec['model_type'] for item in suite.plan}, {'separator3'})
        self.assertTrue(all(item.model_spec['residual_correction_mode'] == 'learned_dense' for item in suite.plan))
        self.assertEqual({item.model_spec['hidden_dim'] for item in suite.plan}, {32, 64, 128, 256})
        self.assertEqual({item.model_spec['mlp_depth'] for item in suite.plan}, {2, 3})
        self.assertEqual({item.model_spec['num_stages'] for item in suite.plan}, {2, 3, 4, 5})
        self.assertTrue(any('hd32' in item.run_name for item in suite.plan))
        self.assertTrue(any('hd256' in item.run_name for item in suite.plan))
        self.assertTrue(any('depth3' in item.run_name for item in suite.plan))
        self.assertTrue(any('stages5' in item.run_name for item in suite.plan))

    def test_separator3_stage_templates_experiment_builds_hand_designed_variants(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='separator3_stage_templates_learned_dense_v1',
        )

        self.assertEqual(len(suite.plan), 4)
        self.assertEqual({item.model_spec['model_type'] for item in suite.plan}, {'separator3'})
        self.assertEqual(
            {tuple(item.model_spec['stage_hidden_dims']) for item in suite.plan},
            {(128, 64), (128, 64, 64), (128, 128, 64), (64, 64, 64)},
        )
        self.assertEqual({item.model_spec['num_stages'] for item in suite.plan}, {2, 3})
        self.assertTrue(all(item.model_spec['residual_correction_mode'] == 'learned_dense' for item in suite.plan))

    def test_build_experiment_suite_can_filter_to_requested_runs(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        target_run = 'separator1_grid_search_6ports_learned_dense_depth2_stages2_share0'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator1_learned_dense_v2',
            run_names=[target_run],
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].run_name, target_run)
        self.assertEqual(suite.model_recipe_names, ['separator1_grid_search_6ports_learned_dense'])

    def test_build_experiment_suite_can_apply_model_and_training_overrides(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='default_6port_separator1_learned_dense_v2',
            run_names=['separator1_grid_search_6ports_learned_dense_depth2_stages2_share0'],
            model_overrides={'hidden_dim': 16, 'num_stages': 1},
            training_overrides={'batch_size': 8},
        )

        self.assertEqual(len(suite.plan), 1)
        self.assertEqual(suite.plan[0].model_spec['hidden_dim'], 16)
        self.assertEqual(suite.plan[0].model_spec['num_stages'], 1)
        self.assertEqual(suite.plan[0].training_spec['batch_size'], 8)

    def test_multi_stage_training_strategy_compiles(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_full_mlp_two_stage_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        training_spec = suite.plan[0].training_spec
        self.assertEqual(training_spec['strategy_type'], 'multi_stage_supervised')
        self.assertEqual(training_spec['num_stages'], 2)
        self.assertEqual(training_spec['stage_names'], ['warmup_nmse', 'finetune_log'])
        self.assertEqual(training_spec['stages'][0]['loss_type'], 'nmse')
        self.assertEqual(training_spec['stages'][1]['loss_type'], 'log')

    def test_three_stage_training_strategy_compiles(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        suite = build_experiment_suite(
            config_dir=config_dir,
            experiment_name='quick_full_mlp_three_stage_v2',
        )

        self.assertEqual(len(suite.plan), 1)
        training_spec = suite.plan[0].training_spec
        self.assertEqual(training_spec['strategy_type'], 'multi_stage_supervised')
        self.assertEqual(training_spec['num_stages'], 3)
        self.assertEqual(training_spec['stage_names'], ['warmup_nmse', 'finetune_log', 'polish_weighted'])
        self.assertEqual([stage['loss_type'] for stage in training_spec['stages']], ['nmse', 'log', 'weighted'])


if __name__ == '__main__':
    unittest.main()
