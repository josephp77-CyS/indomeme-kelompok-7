"""EXPERIMENTAL synthetic-tensor reference for CV/NLP integration review.

NOT the official, pretrained, trained CP2 architecture. CV team must approve shapes,
heads, gate conditions, pooling and baseline parity before wiring encoder outputs.
"""
import torch
from torch import nn


class CrossAttentionReference(nn.Module):
    """Input: t[B,768], H[B,L,768], P[B,49,768], mask[B,L]."""
    MODES = {
        "xa_concat", "xa_residual", "xa_shared_scalar", "xa_task_scalar",
        "xa_shared_vector", "xa_task_vector", "xa_shared_scalar_dropout",
        "xa_task_scalar_dropout",
    }

    def __init__(self, mode: str, dim: int = 768, heads: int = 8,
                 dropout: float = .3, visual_dropout_p: float = .1):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if dim % heads:
            raise ValueError("dim must be divisible by number of heads")
        self.mode = mode
        self.dim = dim
        self.visual_dropout_p = visual_dropout_p
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.concat_projection = nn.Linear(2 * dim, dim) if mode == "xa_concat" else None
        gate_dim = dim if "vector" in mode else 1
        if "task" in mode:
            self.task_gates = nn.ModuleDict({t: nn.Linear(dim * 2, gate_dim) for t in ("hate", "appropriate")})
            self.shared_gate = None
        elif "shared" in mode:
            self.task_gates = None
            self.shared_gate = nn.Linear(dim * 2, gate_dim)
        else:
            self.task_gates = None
            self.shared_gate = None
        self.classifiers = nn.ModuleDict({t: nn.Linear(dim, 2) for t in ("hate", "appropriate")})

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError("values [B,L,D] and mask [B,L] expected")
        m = mask.to(dtype=values.dtype).unsqueeze(-1)
        return (values * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)

    def forward(self, t: torch.Tensor, tokens: torch.Tensor,
                patches: torch.Tensor, content_mask: torch.Tensor,
                visual_off: bool = False, forced_gate=None):
        if t.ndim != 2 or tokens.ndim != 3 or patches.ndim != 3:
            raise ValueError("t [B,D], tokens [B,L,D], patches [B,49,D]")
        if patches.shape[1] != 49 or tokens.shape[2] != self.dim or patches.shape[2] != self.dim:
            raise ValueError("Expected ViT base patch32-224 with 49 patches and aligned hidden dims")
        attended, _ = self.attention(tokens, patches, patches, need_weights=False)
        u = self.masked_mean(attended, content_mask)
        if visual_off:
            m = torch.zeros(t.size(0), 1, dtype=t.dtype, device=t.device)
        elif self.training and "dropout" in self.mode:
            m = (torch.rand(t.size(0), 1, device=t.device) >= self.visual_dropout_p).to(t.dtype)
        else:
            m = torch.ones(t.size(0), 1, dtype=t.dtype, device=t.device)
        effective_u = m * u  # no inverse keep scaling
        gate_input = torch.cat([t, u], dim=-1)
        if self.mode == "xa_concat":
            h = self.concat_projection(torch.cat([t, effective_u], dim=-1))
            return {key: head(self.dropout(h)) for key, head in self.classifiers.items()}
        if self.mode == "xa_residual":
            h = t + effective_u
            return {key: head(self.dropout(h)) for key, head in self.classifiers.items()}
        outputs = {}
        for task, head in self.classifiers.items():
            if forced_gate is None:
                gate_layer = self.task_gates[task] if self.task_gates is not None else self.shared_gate
                gate = torch.sigmoid(gate_layer(gate_input))
            else:
                gate = torch.as_tensor(forced_gate, device=t.device, dtype=t.dtype)
            h_task = t + gate * effective_u
            outputs[task] = head(self.dropout(h_task))
        return outputs
