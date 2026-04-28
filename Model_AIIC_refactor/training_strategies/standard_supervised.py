"""Standard supervised training strategy backed by the existing Trainer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

from .base import BaseTrainingStrategy


class StandardSupervisedStrategy(BaseTrainingStrategy):
    """Current supervised training loop packaged as a pluggable strategy."""

    strategy_type = 'standard_supervised'

    @classmethod
    def compile_runtime_spec(
        cls,
        task_spec: Mapping[str, Any],
        strategy_spec: Mapping[str, Any],
        default_training_config: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        del task_spec
        runtime = deepcopy(dict(default_training_config or {}))
        runtime.pop('snr_config', None)
        runtime.pop('tdl_config', None)

        strategy_params = deepcopy(dict(strategy_spec or {}).get('params', {}))
        runtime['batch_size'] = strategy_params.get('batch_size', runtime.get('batch_size'))
        runtime['num_batches'] = strategy_params.get('num_batches', runtime.get('num_batches'))

        optimizer_type = ((strategy_params.get('optimizer') or {}).get('type')) or 'adam'
        if optimizer_type != 'adam':
            raise NotImplementedError(f"Unsupported optimizer.type '{optimizer_type}'. Supported: adam")
        runtime['optimizer'] = {'type': optimizer_type}
        runtime['learning_rate'] = (((strategy_params.get('optimizer') or {}).get('params') or {}).get('learning_rate'))
        if runtime['learning_rate'] is None:
            runtime['learning_rate'] = default_training_config.get('learning_rate') if default_training_config else None

        loss_type = ((strategy_params.get('loss') or {}).get('type'))
        runtime['loss_type'] = loss_type or runtime.get('loss_type')

        regularization = strategy_params.get('regularization') or {}
        if 'learned_dense_mask_l2_to_one' in regularization:
            runtime['learned_dense_mask_regularization'] = regularization['learned_dense_mask_l2_to_one']

        validation = strategy_params.get('validation') or {}
        if 'interval' in validation:
            runtime['validation_interval'] = validation['interval']
        if 'batches' in validation:
            runtime['validation_batches'] = validation['batches']

        early_stop = strategy_params.get('early_stop') or {}
        if 'loss' in early_stop or 'early_stop' in strategy_params:
            runtime['early_stop_loss'] = early_stop.get('loss')
        if 'patience' in early_stop:
            runtime['patience'] = early_stop['patience']

        logging = strategy_params.get('logging') or {}
        if 'print_interval' in logging:
            runtime['print_interval'] = logging['print_interval']

        checkpoints = strategy_params.get('checkpoints') or {}
        if 'save_interval' in checkpoints:
            runtime['save_interval'] = checkpoints['save_interval']
        if 'keep_last_n' in checkpoints:
            runtime['keep_last_n_checkpoints'] = checkpoints['keep_last_n']

        scheduler = strategy_params.get('scheduler') or {}
        scheduler_type = scheduler.get('type')
        if scheduler_type in {'disabled', 'none'}:
            runtime['lr_scheduler'] = {'enabled': False}
        elif scheduler_type == 'reduce_on_plateau' or scheduler.get('params') is not None:
            runtime['lr_scheduler'] = {
                'enabled': True,
                **deepcopy(runtime.get('lr_scheduler', {})),
                **deepcopy(scheduler.get('params', {}) or {}),
            }

        runtime['strategy_type'] = cls.strategy_type
        return runtime

    def create_trainer(self, model, training_spec: Mapping[str, Any], request, device, tensorboard_dir: Path):
        from training import Trainer

        return Trainer(
            model=model,
            learning_rate=training_spec['learning_rate'],
            loss_type=training_spec['loss_type'],
            device=device,
            use_amp=request.use_amp,
            compile_model=request.compile_model,
            tensorboard_dir=tensorboard_dir,
            scheduler_config=training_spec.get('lr_scheduler'),
            learned_dense_mask_regularization=training_spec.get('learned_dense_mask_regularization'),
        )

    def run(self, trainer, task, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any], experiment_dir: Path, progress_tracker):
        task_context = task.get_training_run_context(model_spec=model_spec, training_spec=training_spec)

        num_batches = training_spec['num_batches']
        save_interval = training_spec.get('save_interval')
        if save_interval is None and num_batches >= 1000:
            save_interval = max(1000, num_batches // 20)
            print(f"  💾 Auto checkpoint: every {save_interval} batches (~{num_batches // save_interval} saves)")
        elif save_interval:
            print(f"  💾 Manual checkpoint: every {save_interval} batches (~{num_batches // save_interval} saves)")

        return trainer.train(
            num_batches=num_batches,
            batch_size=training_spec['batch_size'],
            progress_tracker=progress_tracker,
            save_interval=save_interval,
            save_dir=experiment_dir if save_interval is not None else None,
            keep_last_n=training_spec['keep_last_n_checkpoints'],
            **task_context,
        )

    def final_evaluate(self, trainer, task, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        return task.evaluate_with_trainer(trainer, model_spec=model_spec, training_spec=training_spec)