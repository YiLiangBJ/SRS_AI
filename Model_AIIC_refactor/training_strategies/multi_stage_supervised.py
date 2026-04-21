"""Sequential multi-stage supervised training strategy."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .base import BaseTrainingStrategy
from .standard_supervised import StandardSupervisedStrategy


class MultiStageSupervisedStrategy(BaseTrainingStrategy):
    """Run multiple supervised stages back-to-back on the same model weights."""

    strategy_type = 'multi_stage_supervised'

    @classmethod
    def compile_runtime_spec(
        cls,
        task_spec: Mapping[str, Any],
        strategy_spec: Mapping[str, Any],
        default_training_config: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        strategy_params = deepcopy(dict(strategy_spec or {}).get('params', {}))
        stages = strategy_params.get('stages') or []
        if not stages:
            raise ValueError('multi_stage_supervised requires params.stages')

        batch_size_override = strategy_params.get('batch_size')
        num_batches_override = strategy_params.get('num_batches')

        compiled_stages = []
        total_batches = 0
        for stage_index, stage in enumerate(stages, 1):
            stage_name = stage.get('name') or f'stage_{stage_index}'
            stage_params = deepcopy(stage.get('params', {}))
            if batch_size_override is not None:
                stage_params['batch_size'] = batch_size_override
            if num_batches_override is not None:
                stage_params['num_batches'] = num_batches_override
            stage_runtime = StandardSupervisedStrategy.compile_runtime_spec(
                task_spec=task_spec,
                strategy_spec={'type': 'standard_supervised', 'params': stage_params},
                default_training_config=default_training_config,
            )
            stage_runtime['stage_name'] = stage_name
            stage_runtime['stage_index'] = stage_index
            compiled_stages.append(stage_runtime)
            total_batches += int(stage_runtime['num_batches'])

        return {
            'strategy_type': cls.strategy_type,
            'num_stages': len(compiled_stages),
            'stage_names': [stage['stage_name'] for stage in compiled_stages],
            'stages': compiled_stages,
            'batch_size': compiled_stages[0]['batch_size'],
            'num_batches': total_batches,
            'loss_type': compiled_stages[-1]['loss_type'],
        }

    def create_trainer(self, model, training_spec: Mapping[str, Any], request, device, tensorboard_dir: Path):
        self._shared_model = self._unwrap_model(model)
        self._request = request
        self._device = device
        self._tensorboard_dir = Path(tensorboard_dir)
        self._final_trainer = None
        self._stage_summaries = []
        return model

    @staticmethod
    def _unwrap_model(model):
        if hasattr(model, '_orig_mod'):
            return model._orig_mod
        return model

    def run(self, trainer, task, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any], experiment_dir: Path, progress_tracker):
        del trainer
        helper = StandardSupervisedStrategy({'type': 'standard_supervised', 'params': {}})
        all_losses = []
        self._stage_summaries = []

        for stage_index, stage_spec in enumerate(training_spec['stages'], 1):
            stage_name = stage_spec['stage_name']
            print(f"\n{'='*80}")
            print(f"Stage {stage_index}/{training_spec['num_stages']}: {stage_name}")
            print(f"{'='*80}")
            print(f"  Runtime params: {stage_spec}")

            stage_tensorboard_dir = self._tensorboard_dir / f'{stage_index:02d}_{stage_name}'
            stage_checkpoint_dir = experiment_dir / 'stage_artifacts' / f'{stage_index:02d}_{stage_name}'
            stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stage_trainer = helper.create_trainer(
                model=self._unwrap_model(self._shared_model),
                training_spec=stage_spec,
                request=self._request,
                device=self._device,
                tensorboard_dir=stage_tensorboard_dir,
            )
            stage_losses = helper.run(
                trainer=stage_trainer,
                task=task,
                model_spec=model_spec,
                training_spec=stage_spec,
                experiment_dir=stage_checkpoint_dir,
                progress_tracker=progress_tracker,
            )
            self._shared_model = self._unwrap_model(stage_trainer.model)
            self._final_trainer = stage_trainer
            all_losses.extend(stage_losses)
            self._stage_summaries.append({
                'stage_name': stage_name,
                'stage_index': stage_index,
                'num_batches': stage_spec['num_batches'],
                'batch_size': stage_spec['batch_size'],
                'loss_type': stage_spec['loss_type'],
                'learning_rate': stage_spec['learning_rate'],
                'final_loss': stage_losses[-1] if stage_losses else None,
                'min_loss': min(stage_losses) if stage_losses else None,
            })
            with open(stage_checkpoint_dir / 'stage_summary.json', 'w', encoding='utf-8') as output_file:
                json.dump(
                    {
                        'stage_name': stage_name,
                        'stage_index': stage_index,
                        'runtime_spec': stage_spec,
                        'final_loss': stage_losses[-1] if stage_losses else None,
                        'min_loss': min(stage_losses) if stage_losses else None,
                        'num_losses': len(stage_losses),
                    },
                    output_file,
                    indent=2,
                    ensure_ascii=False,
                )

        return all_losses

    def final_evaluate(self, trainer, task, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        del trainer, training_spec
        if self._final_trainer is None:
            raise RuntimeError('No final trainer available for multi-stage evaluation')
        return task.evaluate_with_trainer(self._final_trainer, model_spec=model_spec, training_spec={})


__all__ = ['MultiStageSupervisedStrategy']