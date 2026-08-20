"""Muon optimizer (Jordan et al.) — momentum + Newton-Schulz orthogonalization
of 2D weight-matrix gradients, as used by PufferLib's trainer.

The orthogonalization step normalizes the update's singular values, which
empirically improves sample efficiency on matrix-shaped parameters. Non-2D
parameters (biases, embeddings, gains) and output heads should use AdamW —
`build_muon_hybrid` handles the split.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def _newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalize g via quintic Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                update = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = _newton_schulz(update)
                # Scale per Muon convention so lr transfers across shapes.
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


class HybridOptimizer:
    """Muon for hidden 2D weights, AdamW for everything else. Mimics the
    torch.optim API surface the trainer uses (step / zero_grad / state_dict)."""

    def __init__(self, model: torch.nn.Module, muon_lr: float, adam_lr: float) -> None:
        # Heads (logit/value scale is signal) and embedding-like params (row
        # norms are signal) stay on AdamW; only hidden transform matrices get
        # orthogonalized updates.
        head_names = ("dir_head", "op_head", "value_head")
        muon_params, adam_params = [], []
        for name, p in model.named_parameters():
            is_embedding = "emb" in name or "queries" in name
            if p.ndim == 2 and not name.startswith(head_names) and not is_embedding:
                muon_params.append(p)
            else:
                adam_params.append(p)
        self.muon = Muon(muon_params, lr=muon_lr)
        self.adam = torch.optim.AdamW(adam_params, lr=adam_lr, eps=1e-5, weight_decay=0.0)

    def step(self) -> None:
        self.muon.step()
        self.adam.step()

    def zero_grad(self) -> None:
        self.muon.zero_grad()
        self.adam.zero_grad()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])
