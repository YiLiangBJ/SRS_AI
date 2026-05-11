"""Separator3: Multi-stage joint MLP separator with learned-dense residual correction."""

import torch
import torch.nn as nn

from .base_model import BaseSeparatorModel


class Separator3(BaseSeparatorModel):
    """Multi-stage joint separator with per-stage learned-dense residual correction.

    Each stage produces a full joint estimate for all ports at once:
    - Stage 1: input_dim -> hidden_dim -> ... -> expanded_dim
    - Stage 2+: expanded_dim -> hidden_dim -> ... -> expanded_dim

    After every stage, the model reshapes the joint estimate into per-port features,
    computes the residual against the original mixed signal, and applies one learned
    dense residual correction mask for that stage.
    """

    class JointStage(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, mlp_depth):
            super().__init__()
            if mlp_depth < 2:
                raise ValueError(f"mlp_depth must be >= 2 (got {mlp_depth})")

            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for _ in range(mlp_depth - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class DynamicMaskGenerator(nn.Module):
        """Generate one residual delta-mask per port from current features and residual."""

        def __init__(self, feature_dim, hidden_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
                nn.Tanh(),
            )
            final_linear = self.network[2]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        def forward(self, port_features, residual):
            residual_expanded = residual.unsqueeze(1).expand(-1, port_features.shape[1], -1)
            combined = torch.cat([port_features, residual_expanded], dim=-1)
            batch_size, num_ports, combined_dim = combined.shape
            return self.network(combined.reshape(batch_size * num_ports, combined_dim)).reshape(
                batch_size, num_ports, -1
            )

    @staticmethod
    def _resolve_stage_hidden_dims(hidden_dim, stage_hidden_dims, num_stages):
        if stage_hidden_dims is None:
            return [int(hidden_dim)] * int(num_stages)

        resolved = [int(value) for value in stage_hidden_dims]
        if len(resolved) != int(num_stages):
            raise ValueError(
                f"stage_hidden_dims length must equal num_stages={num_stages} (got {len(resolved)})"
            )
        return resolved

    def __init__(
        self,
        seq_len=12,
        num_ports=4,
        hidden_dim=128,
        stage_hidden_dims=None,
        num_stages=2,
        mlp_depth=2,
        normalize_energy=True,
        residual_correction_mode='learned_dense',
        pos_values=None,
    ):
        super().__init__(seq_len, num_ports, normalize_energy=normalize_energy)

        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1 (got {num_stages})")
        if mlp_depth < 2:
            raise ValueError(f"mlp_depth must be >= 2 (got {mlp_depth})")

        self.residual_correction_mode = residual_correction_mode
        self.pos_values = None if pos_values is None else [int(value) for value in pos_values]
        self.hidden_dim = int(hidden_dim)
        self.num_stages = int(num_stages)
        self.mlp_depth = int(mlp_depth)

        if self.residual_correction_mode not in {'global', 'masked', 'learned_dense', 'generated_dense'}:
            raise ValueError(
                f"Unsupported residual_correction_mode {residual_correction_mode!r}; "
                "expected 'global', 'masked', 'learned_dense', or 'generated_dense'"
            )
        if self.residual_correction_mode == 'masked':
            if self.pos_values is None:
                raise ValueError("residual_correction_mode='masked' requires pos_values in model config")
            if len(self.pos_values) != num_ports:
                raise ValueError(
                    f"pos_values must have length num_ports={num_ports} when residual_correction_mode='masked' "
                    f"(got {len(self.pos_values)})"
                )
            for pos_value in self.pos_values:
                if pos_value < 0 or pos_value >= self.seq_len:
                    raise ValueError(
                        f"pos_values entry {pos_value} is out of range for seq_len={self.seq_len}"
                    )

        self.input_dim = self.seq_len * 2
        self.expanded_dim = self.num_ports * self.input_dim
        self.stage_hidden_dims = self._resolve_stage_hidden_dims(
            hidden_dim=self.hidden_dim,
            stage_hidden_dims=stage_hidden_dims,
            num_stages=self.num_stages,
        )
        self.stages = nn.ModuleList([
            self.JointStage(
                in_dim=self.input_dim if stage_idx == 0 else self.expanded_dim,
                hidden_dim=self.stage_hidden_dims[stage_idx],
                out_dim=self.expanded_dim,
                mlp_depth=self.mlp_depth,
            )
            for stage_idx in range(self.num_stages)
        ])

        residual_mask = torch.ones(self.num_ports, self.input_dim, dtype=torch.float32)
        if self.residual_correction_mode in {'masked', 'generated_dense'} and self.pos_values is not None:
            residual_mask.zero_()
            for branch_idx, pos_value in enumerate(self.pos_values):
                residual_mask[branch_idx, pos_value] = 1.0
                residual_mask[branch_idx, pos_value + self.seq_len] = 1.0
        self.register_buffer('residual_port_mask', residual_mask, persistent=False)

        if self.residual_correction_mode == 'learned_dense':
            self.learned_residual_masks = nn.Parameter(
                torch.ones(self.num_stages, self.num_ports, self.input_dim, dtype=torch.float32)
            )
        else:
            self.register_parameter('learned_residual_masks', None)

        if self.residual_correction_mode == 'generated_dense':
            generator_hidden_dim = max(16, min(64, self.hidden_dim // 2))
            self.mask_generators = nn.ModuleList([
                self.DynamicMaskGenerator(self.input_dim, generator_hidden_dim)
                for _ in range(self.num_stages)
            ])
        else:
            self.mask_generators = None

    def _apply_residual_correction(self, mixed_signal, features, stage_idx):
        y_recon = features.sum(dim=1)
        residual = mixed_signal - y_recon
        if self.residual_correction_mode == 'global':
            return features + residual.unsqueeze(1)
        if self.residual_correction_mode == 'masked':
            masked_residual = residual.unsqueeze(1) * self.residual_port_mask.unsqueeze(0).to(dtype=features.dtype)
            return features + masked_residual
        if self.residual_correction_mode == 'generated_dense':
            base_mask = self.residual_port_mask.unsqueeze(0).to(dtype=features.dtype)
            delta_mask = self.mask_generators[stage_idx](features, residual).to(dtype=features.dtype)
            generated_mask = torch.clamp(base_mask + delta_mask, min=0.0, max=2.0)
            masked_residual = residual.unsqueeze(1) * generated_mask
            return features + masked_residual
        residual_mask = self.learned_residual_masks[stage_idx]
        masked_residual = residual.unsqueeze(1) * residual_mask.unsqueeze(0).to(dtype=features.dtype)
        return features + masked_residual

    def forward(self, y):
        return_complex = torch.is_complex(y)
        y, input_rms = self.normalize_input_energy(y)
        if return_complex:
            y = torch.cat([y.real, y.imag], dim=-1)
        elif not torch.jit.is_tracing() and y.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected real-stacked input with last dim {self.input_dim}, got {tuple(y.shape)}"
            )

        stage_input = y
        output_features = None
        for stage_idx, stage in enumerate(self.stages):
            stage_output = stage(stage_input)
            output_features = stage_output.view(-1, self.num_ports, self.input_dim)
            output_features = self._apply_residual_correction(
                mixed_signal=y,
                features=output_features,
                stage_idx=stage_idx,
            )
            stage_input = output_features.reshape(-1, self.expanded_dim)

        if return_complex:
            output_features = torch.complex(
                output_features[..., :self.seq_len],
                output_features[..., self.seq_len:],
            )
        return self.restore_output_energy(output_features, input_rms)

    @classmethod
    def from_config(cls, config):
        stage_hidden_dims = config.get('stage_hidden_dims')
        default_num_stages = len(stage_hidden_dims) if stage_hidden_dims is not None else 2
        default_hidden_dim = stage_hidden_dims[0] if stage_hidden_dims is not None else 128
        return cls(
            seq_len=config['seq_len'],
            num_ports=config['num_ports'],
            hidden_dim=config.get('hidden_dim', default_hidden_dim),
            stage_hidden_dims=stage_hidden_dims,
            num_stages=config.get('num_stages', default_num_stages),
            mlp_depth=config.get('mlp_depth', 2),
            normalize_energy=config.get('normalize_energy', True),
            residual_correction_mode=config.get('residual_correction_mode', 'learned_dense'),
            pos_values=config.get('pos_values'),
        )