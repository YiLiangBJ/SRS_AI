"""
Separator3: Fixed two-block real MLP separator with learned-dense residual correction.

Architecture:
- Normalize one mixed real-stacked input [real, imag]
- Linear projection from input width 2L to expanded width P * 2L
- Optional hidden ReLU before hidden learned-dense residual correction
- Second linear projection from expanded width P * 2L back to P * 2L
- Output learned-dense residual correction

The model intentionally keeps the control surface small:
- fixed two-block structure
- no stage loop
- no stage weight sharing
"""

import torch
import torch.nn as nn

from .base_model import BaseSeparatorModel


class Separator3(BaseSeparatorModel):
    """Fixed two-block separator with learned-dense residual correction."""

    def __init__(
        self,
        seq_len=12,
        num_ports=4,
        normalize_energy=True,
        use_hidden_relu=False,
        residual_correction_mode='learned_dense',
        pos_values=None,
    ):
        super().__init__(seq_len, num_ports, normalize_energy=normalize_energy)

        self.use_hidden_relu = use_hidden_relu
        self.residual_correction_mode = residual_correction_mode
        self.pos_values = None if pos_values is None else [int(value) for value in pos_values]

        if self.residual_correction_mode not in {'global', 'masked', 'learned_dense'}:
            raise ValueError(
                f"Unsupported residual_correction_mode {residual_correction_mode!r}; "
                "expected 'global', 'masked', or 'learned_dense'"
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
        self.hidden_linear = nn.Linear(self.input_dim, self.expanded_dim)
        self.output_linear = nn.Linear(self.expanded_dim, self.expanded_dim)
        self.hidden_activation = nn.ReLU() if self.use_hidden_relu else None

        residual_mask = torch.ones(self.num_ports, self.input_dim, dtype=torch.float32)
        if self.residual_correction_mode == 'masked':
            residual_mask.zero_()
            for branch_idx, pos_value in enumerate(self.pos_values):
                residual_mask[branch_idx, pos_value] = 1.0
                residual_mask[branch_idx, pos_value + self.seq_len] = 1.0
        self.register_buffer('residual_port_mask', residual_mask, persistent=False)

        if self.residual_correction_mode == 'learned_dense':
            self.hidden_residual_mask = nn.Parameter(
                torch.ones(self.num_ports, self.input_dim, dtype=torch.float32)
            )
            self.output_residual_mask = nn.Parameter(
                torch.ones(self.num_ports, self.input_dim, dtype=torch.float32)
            )
        else:
            self.register_parameter('hidden_residual_mask', None)
            self.register_parameter('output_residual_mask', None)

    def _apply_residual_correction(self, mixed_signal, features, residual_mask=None):
        y_recon = features.sum(dim=1)
        residual = mixed_signal - y_recon
        if self.residual_correction_mode == 'global':
            return features + residual.unsqueeze(1)
        if self.residual_correction_mode == 'masked':
            masked_residual = residual.unsqueeze(1) * self.residual_port_mask.unsqueeze(0).to(dtype=features.dtype)
            return features + masked_residual
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

        hidden = self.hidden_linear(y)
        if self.hidden_activation is not None:
            hidden = self.hidden_activation(hidden)
        hidden_features = hidden.view(-1, self.num_ports, self.input_dim)
        hidden_features = self._apply_residual_correction(
            mixed_signal=y,
            features=hidden_features,
            residual_mask=self.hidden_residual_mask,
        )

        output = self.output_linear(hidden_features.reshape(-1, self.expanded_dim))
        output_features = output.view(-1, self.num_ports, self.input_dim)
        output_features = self._apply_residual_correction(
            mixed_signal=y,
            features=output_features,
            residual_mask=self.output_residual_mask,
        )

        if return_complex:
            output_features = torch.complex(
                output_features[..., :self.seq_len],
                output_features[..., self.seq_len:],
            )
        return self.restore_output_energy(output_features, input_rms)

    @classmethod
    def from_config(cls, config):
        return cls(
            seq_len=config['seq_len'],
            num_ports=config['num_ports'],
            normalize_energy=config.get('normalize_energy', True),
            use_hidden_relu=config.get('use_hidden_relu', False),
            residual_correction_mode=config.get('residual_correction_mode', 'learned_dense'),
            pos_values=config.get('pos_values'),
        )