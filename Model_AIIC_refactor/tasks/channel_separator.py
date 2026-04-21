"""Channel-separation task adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import numpy as np
import torch

from .base import BaseTask


class ChannelSeparatorTask(BaseTask):
    """Task adapter for the current channel-separation problem family."""

    task_type = 'channel_separator'

    @classmethod
    def compile_model_spec(cls, task_spec: Mapping[str, Any], model_spec: Mapping[str, Any]) -> Dict[str, Any]:
        task_params = deepcopy(dict(task_spec or {}).get('params', {}))
        model_params = deepcopy(dict(model_spec or {}).get('params', {}))

        model_context = deepcopy(task_params.get('model_context', {}))
        for key in ('seq_len', 'pos_values', 'normalize_energy'):
            if key in task_params and key not in model_context:
                model_context[key] = deepcopy(task_params[key])

        compiled = {
            **model_context,
            **model_params,
            'model_type': dict(model_spec or {}).get('type'),
        }
        if 'pos_values' in compiled and 'num_ports' not in compiled:
            compiled['num_ports'] = len(compiled['pos_values'])
        return compiled

    def _parse_snr_config(self):
        from utils.snr_config import parse_snr_config

        return parse_snr_config(deepcopy(self.params['snr_config']))

    def _resolve_eval_snr(self) -> float:
        snr_config = self._parse_snr_config()
        if snr_config.config_type == 'range':
            return float((snr_config.min_snr + snr_config.max_snr) / 2)
        return float(np.mean(snr_config.snr_values))

    def build_dummy_input(self, model_spec: Mapping[str, Any], batch_size: int = 1) -> torch.Tensor:
        seq_len = int(model_spec['seq_len'])
        return torch.randn(batch_size, seq_len * 2, dtype=torch.float32)

    def get_training_run_context(self, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            'snr_config': self._parse_snr_config(),
            'pos_values': deepcopy(model_spec['pos_values']),
            'tdl_config': deepcopy(self.params['tdl_config']),
            'seq_len': int(model_spec['seq_len']),
            'print_interval': training_spec['print_interval'],
            'val_interval': training_spec.get('validation_interval'),
            'validation_batches': training_spec.get('validation_batches', 4),
            'early_stop_loss': training_spec.get('early_stop_loss'),
            'patience': training_spec['patience'],
        }

    def evaluate_with_trainer(self, trainer, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        return trainer.evaluate(
            batch_size=self.get_default_eval_batch_size(),
            snr_db=self._resolve_eval_snr(),
            pos_values=deepcopy(model_spec['pos_values']),
            tdl_config=deepcopy(self.params['tdl_config']),
            seq_len=int(model_spec['seq_len']),
        )

    def evaluate_at_snr(
        self,
        model,
        model_spec: Mapping[str, Any],
        snr_db: float,
        tdl_config: str,
        num_batches: int = 100,
        batch_size: int = 2048,
        device: str | torch.device = 'cpu',
        use_amp: bool = False,
    ) -> Dict[str, Any]:
        from data import generate_training_batch
        from training.metrics import evaluate_model

        resolved_device = self.resolve_device(device)
        seq_len = int(model_spec['seq_len'])
        pos_values = model_spec['pos_values']
        num_ports = len(pos_values)

        total_mse = torch.tensor(0.0, device=resolved_device)
        total_power = torch.tensor(0.0, device=resolved_device)
        port_mse = torch.zeros(num_ports, device=resolved_device)
        port_power = torch.zeros(num_ports, device=resolved_device)
        autocast_context = torch.cuda.amp.autocast if use_amp and resolved_device.type == 'cuda' else None

        with torch.no_grad():
            for _ in range(num_batches):
                y, h_targets, _, _, _ = generate_training_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    pos_values=pos_values,
                    snr_db=snr_db,
                    tdl_config=tdl_config,
                    return_complex=False,
                    device=resolved_device,
                )

                if autocast_context is not None:
                    with autocast_context():
                        h_pred = model(y)
                else:
                    h_pred = model(y)

                diff = h_pred - h_targets
                total_mse += diff.pow(2).sum()
                total_power += h_targets.pow(2).sum()
                port_mse += diff.pow(2).sum(dim=(0, 2))
                port_power += h_targets.pow(2).sum(dim=(0, 2))

        nmse = (total_mse / (total_power + 1e-10)).cpu().item()
        nmse_db = 10 * np.log10(nmse) if nmse > 0 else -100
        port_nmse = (port_mse / (port_power + 1e-10)).cpu().numpy()
        port_nmse_db = 10 * np.log10(port_nmse)
        port_nmse_db[np.isinf(port_nmse_db)] = -100

        return {
            'snr_db': float(snr_db),
            'tdl_config': tdl_config,
            'nmse': float(nmse),
            'nmse_db': float(nmse_db),
            'per_port_nmse': port_nmse.tolist(),
            'per_port_nmse_db': port_nmse_db.tolist(),
            'num_samples': num_batches * batch_size,
        }