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
    ):
        super().__init__(seq_len, num_ports, normalize_energy=normalize_energy)

        if mlp_depth < 2:
            raise ValueError(f"mlp_depth must be >= 2 (got {mlp_depth})")

        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        self.input_dim = seq_len * 2
        self.output_dim = self.input_dim * num_ports
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
        )