"""Full-MLP separator that predicts all ports jointly from the mixed input."""

import torch
import torch.nn as nn

from .base_model import BaseSeparatorModel


class FullMLP(BaseSeparatorModel):
    """Joint MLP baseline for channel separation.

    Input is the same real-stacked mixed signal of length ``2 * seq_len``.
    The network predicts a flat vector of length ``num_ports * 2 * seq_len`` and
    reshapes it into ``(B, num_ports, 2 * seq_len)``.
    """

    def __init__(
        self,
        seq_len: int = 12,
        num_ports: int = 4,
        hidden_dim: int = 128,
        mlp_depth: int = 3,
        normalize_energy: bool = True,
        residual_correction_mode: str = 'none',
        pos_values=None,
    ):
        super().__init__(seq_len, num_ports, normalize_energy=normalize_energy)

        if mlp_depth < 2:
            raise ValueError(f"mlp_depth must be >= 2 (got {mlp_depth})")

        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        self.input_dim = seq_len * 2
        self.output_dim = self.input_dim * num_ports
        self.residual_correction_mode = residual_correction_mode
        self.pos_values = None if pos_values is None else [int(value) for value in pos_values]

        if self.residual_correction_mode not in {'none', 'masked', 'learned_dense'}:
            raise ValueError(
                f"Unsupported residual_correction_mode {residual_correction_mode!r}; expected 'none', 'masked', or 'learned_dense'"
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

        residual_mask = torch.ones(self.num_ports, self.input_dim, dtype=torch.float32)
        if self.residual_correction_mode == 'masked':
            residual_mask.zero_()
            for branch_idx, pos_value in enumerate(self.pos_values):
                residual_mask[branch_idx, pos_value] = 1.0
                residual_mask[branch_idx, pos_value + self.seq_len] = 1.0
        self.register_buffer('residual_port_mask', residual_mask, persistent=False)
        if self.residual_correction_mode == 'learned_dense':
            self.learned_residual_mask = nn.Parameter(torch.ones(self.num_ports, self.input_dim, dtype=torch.float32))
        else:
            self.register_parameter('learned_residual_mask', None)
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers = []
        if self.mlp_depth == 2:
            layers.append(nn.Linear(self.input_dim, self.output_dim))
            return nn.Sequential(*layers)

        layers.extend([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        ])

        for _ in range(self.mlp_depth - 3):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return_complex = torch.is_complex(y)
        y, input_rms = self.normalize_input_energy(y)

        if return_complex:
            y = torch.cat([y.real, y.imag], dim=-1)
        elif not torch.jit.is_tracing() and y.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected real-stacked input with last dim {self.input_dim}, got {tuple(y.shape)}"
            )

        features = self.network(y).view(-1, self.num_ports, self.input_dim)
        if self.residual_correction_mode in {'masked', 'learned_dense'}:
            y_recon = features.sum(dim=1)
            residual = y - y_recon
            if self.residual_correction_mode == 'masked':
                residual_mask = self.residual_port_mask
            else:
                residual_mask = self.learned_residual_mask
            masked_residual = residual.unsqueeze(1) * residual_mask.unsqueeze(0).to(dtype=features.dtype)
            features = features + masked_residual

        if return_complex:
            features = torch.complex(features[..., :self.seq_len], features[..., self.seq_len:])

        return self.restore_output_energy(features, input_rms)

    @classmethod
    def from_config(cls, config):
        return cls(
            seq_len=config['seq_len'],
            num_ports=config['num_ports'],
            hidden_dim=config.get('hidden_dim', 128),
            mlp_depth=config.get('mlp_depth', 3),
            normalize_energy=config.get('normalize_energy', True),
            residual_correction_mode=config.get('residual_correction_mode', 'none'),
            pos_values=config.get('pos_values'),
        )