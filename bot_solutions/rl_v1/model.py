"""Recurrent entity-attention policy for the agario agent (v2).

Architecture:
  - per-entity-type encoders (own cells, enemy cells, viruses) producing
    tokens, with learned type embeddings
  - one multi-head attention pooling layer (learned queries attend over all
    entity tokens) so the net reasons over cells as a set
  - a context MLP over the self + food-sector features
  - a GRU carrying memory across decisions, so the policy can hold intent
    (chase, flee, ambush) instead of re-deciding from scratch every 80 ms
  - categorical heads for direction / op, plus the value head

The recurrent interface:
  h = policy.initial_state(batch)            # (B, HIDDEN)
  dir, op, logp, value, h = policy.act(obs, h, done)
  logp, ent, value = policy.evaluate_sequence(obs_seq, dir_seq, op_seq,
                                              done_seq, h0)
`done` flags mark observations that start a fresh episode; the hidden state
is zeroed there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .obs import (
    ENEMY_F,
    FOOD_F,
    K_ENEMY,
    K_FOOD,
    K_OWN,
    K_VIRUS,
    N_OPS,
    N_SECTORS,
    N_TURNS,
    OBS_DIM,
    OWN_F,
    SELF_DIM,
)

HIDDEN = 256
D_MODEL = 64
N_QUERIES = 4

_OWN_START = SELF_DIM
_ENEMY_START = _OWN_START + K_OWN * OWN_F
_SECTOR_START = _ENEMY_START + K_ENEMY * ENEMY_F
_PELLET_START = _SECTOR_START + N_SECTORS * 2
_VIRUS_START = _PELLET_START + K_FOOD * FOOD_F
VIRUS_F = 4


def _layer_init(layer: nn.Linear, std: float = 2.0**0.5, bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        _layer_init(nn.Linear(in_dim, out_dim)),
        nn.Tanh(),
        _layer_init(nn.Linear(out_dim, out_dim)),
        nn.Tanh(),
    )


class Highway(nn.Module):
    """Highway residual (PufferNet-style): y = t * H(x) + (1 - t) * x.
    A learnable gate that lets gradients skip the transform — replaces the
    role of normalization layers at a fraction of the cost."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.transform = _layer_init(nn.Linear(dim, dim))
        self.gate = _layer_init(nn.Linear(dim, dim), bias=-1.0)  # start mostly-skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(x))
        return t * torch.tanh(self.transform(x)) + (1.0 - t) * x


class MinGRU(nn.Module):
    """Minimal GRU (Feng et al. 2024), as popularized by PufferNet.

    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t, where z_t and h~_t depend only on
    x_t — which makes the recurrence a linear scan h_t = a_t * h_{t-1} + b_t
    that can be evaluated for a whole sequence in O(log T) vectorized passes
    (Hillis–Steele inclusive scan) instead of T sequential kernel calls.

    Episode resets fold in exactly: zeroing h before consuming x_t is just
    a_t *= (1 - done_t). No segment splitting needed.
    """

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.z_proj = _layer_init(nn.Linear(in_dim, hidden))
        self.h_proj = _layer_init(nn.Linear(in_dim, hidden))

    def _gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self.z_proj(x))
        h_tilde = torch.tanh(self.h_proj(x))
        return 1.0 - z, z * h_tilde  # (a, b) of h_t = a*h_{t-1} + b

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        a, b = self._gates(x)
        return a * h + b

    def sequence(
        self, x: torch.Tensor, h0: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        """x (T, B, in), h0 (B, H), done (T, B) -> hidden states (T, B, H)."""
        a, b = self._gates(x)
        a = a * (1.0 - done.float()).unsqueeze(-1)
        b = torch.cat([(b[:1] + a[:1] * h0.unsqueeze(0)), b[1:]], dim=0)

        t_len = a.shape[0]
        offset = 1
        while offset < t_len:
            pad_a = torch.ones_like(a[:offset])
            pad_b = torch.zeros_like(b[:offset])
            a_sh = torch.cat([pad_a, a[:-offset]], dim=0)
            b_sh = torch.cat([pad_b, b[:-offset]], dim=0)
            b = b + a * b_sh
            a = a * a_sh
            offset *= 2
        return b


class PolicyNet(nn.Module):
    version = 4  # v4: control-state obs (heading/prev action) + speed head

    def __init__(self, rnn_type: str = "gru") -> None:
        super().__init__()
        assert rnn_type in ("gru", "mingru")
        self.rnn_type = rnn_type
        self.model_kwargs = {"rnn_type": rnn_type}

        self.own_enc = _mlp(OWN_F, D_MODEL)
        self.enemy_enc = _mlp(ENEMY_F, D_MODEL)
        self.virus_enc = _mlp(VIRUS_F, D_MODEL)
        self.pellet_enc = _mlp(FOOD_F, D_MODEL)
        self.type_emb = nn.Parameter(torch.zeros(4, D_MODEL))
        self.attn = nn.MultiheadAttention(D_MODEL, num_heads=4, batch_first=True)
        self.queries = nn.Parameter(torch.randn(1, N_QUERIES, D_MODEL) * 0.1)
        self.ctx_enc = _mlp(SELF_DIM + N_SECTORS * 2, D_MODEL)

        gru_in = N_QUERIES * D_MODEL + D_MODEL
        self.pre = nn.Sequential(_layer_init(nn.Linear(gru_in, HIDDEN)), nn.Tanh())
        if rnn_type == "mingru":
            self.rnn = MinGRU(HIDDEN, HIDDEN)
            self.post = Highway(HIDDEN)
        else:
            # nn.GRU (not GRUCell) so sequence replay during PPO updates runs
            # as one fused kernel per episode segment.
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
                else:
                    nn.init.constant_(param, 0.0)
            self.post = nn.Identity()

        self.dir_head = _layer_init(nn.Linear(HIDDEN, N_TURNS), std=0.01)
        self.op_head = _layer_init(nn.Linear(HIDDEN, N_OPS), std=0.01)
        # Continuous movement speed in (0, 1): bias +2 so the untrained policy
        # starts near full speed rather than hovering.
        self.speed_head = _layer_init(nn.Linear(HIDDEN, 1), std=0.01, bias=2.0)
        self.value_head = _layer_init(nn.Linear(HIDDEN, 1), std=1.0)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, HIDDEN, device=device)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, OBS_DIM) -> pre-GRU features (B, HIDDEN)."""
        b = obs.shape[0]
        own = obs[:, _OWN_START:_ENEMY_START].view(b, K_OWN, OWN_F)
        enemy = obs[:, _ENEMY_START:_SECTOR_START].view(b, K_ENEMY, ENEMY_F)
        pellet = obs[:, _PELLET_START:_VIRUS_START].view(b, K_FOOD, FOOD_F)
        virus = obs[:, _VIRUS_START:].view(b, K_VIRUS, VIRUS_F)

        tokens = torch.cat(
            [
                self.own_enc(own) + self.type_emb[0],
                self.enemy_enc(enemy) + self.type_emb[1],
                self.virus_enc(virus) + self.type_emb[2],
                self.pellet_enc(pellet) + self.type_emb[3],
            ],
            dim=1,
        )
        queries = self.queries.expand(b, -1, -1)
        pooled, _ = self.attn(queries, tokens, tokens, need_weights=False)

        ctx = self.ctx_enc(
            torch.cat([obs[:, :SELF_DIM], obs[:, _SECTOR_START:_PELLET_START]], dim=-1)
        )
        return self.pre(torch.cat([pooled.flatten(1), ctx], dim=-1))

    def _step(self, obs: torch.Tensor, h: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        h = h * (1.0 - done.float()).unsqueeze(-1)
        x = self.encode(obs)
        if self.rnn_type == "mingru":
            return self.rnn.step(x, h)
        _, h_next = self.gru(x.unsqueeze(0), h.unsqueeze(0))
        return h_next.squeeze(0)

    def act(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        done: torch.Tensor,
        *,
        temperature: float = 1.0,
        deterministic: bool = False,
        op_deterministic: bool = False,
    ):
        """Returns (turn, op, speed, log_prob, value, h). Speed is the
        deterministic sigmoid output (supervised channel, not in log_prob).
        op_deterministic=True takes argmax for split/eject only — fire on
        confidence, not by lottery — while turns stay sampled."""
        h = self._step(obs, h, done)
        features = self.post(h)
        dir_logits = self.dir_head(features)
        op_logits = self.op_head(features)
        speed = torch.sigmoid(self.speed_head(features)).squeeze(-1)
        value = self.value_head(features).squeeze(-1)
        if temperature != 1.0:
            dir_logits = dir_logits / temperature
            op_logits = op_logits / temperature
        dir_dist = Categorical(logits=dir_logits)
        op_dist = Categorical(logits=op_logits)
        if deterministic:
            direction = dir_logits.argmax(dim=-1)
        else:
            direction = dir_dist.sample()
        if deterministic or op_deterministic:
            op = op_logits.argmax(dim=-1)
        else:
            op = op_dist.sample()
        log_prob = dir_dist.log_prob(direction) + op_dist.log_prob(op)
        return direction, op, speed, log_prob, value, h

    def sequence_features(
        self,
        obs_seq: torch.Tensor,  # (T, B, OBS_DIM)
        done_seq: torch.Tensor,  # (T, B) — done[t] marks obs[t] starting fresh
        h0: torch.Tensor,  # (B, HIDDEN)
    ) -> torch.Tensor:
        t_len, b = obs_seq.shape[0], obs_seq.shape[1]
        # Encode every (t, agent) observation in one fully-batched pass —
        # attention over T*B rows at once instead of T small calls.
        x = self.encode(obs_seq.reshape(t_len * b, -1)).reshape(t_len, b, -1)

        if self.rnn_type == "mingru":
            # Whole sequence in O(log T) vectorized passes; episode resets
            # fold into the scan coefficients directly.
            hidden = self.rnn.sequence(x, h0, done_seq)
        else:
            # The GRU must reset where done[t] == 1, which only happens at
            # episode boundaries (rare within a rollout). Run fused over the
            # segments between boundaries.
            done_f = done_seq.float()
            boundaries = [
                t for t in range(t_len) if bool(done_f[t].any()) and t > 0
            ]
            h = h0.unsqueeze(0) * (1.0 - done_f[0]).view(1, b, 1)
            outs = []
            seg_start = 0
            for t in boundaries:
                out, h = self.gru(x[seg_start:t], h)
                outs.append(out)
                h = h * (1.0 - done_f[t]).view(1, b, 1)
                seg_start = t
            out, _ = self.gru(x[seg_start:], h)
            outs.append(out)
            hidden = torch.cat(outs, dim=0)  # (T, B, HIDDEN)

        return self.post(hidden)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,  # (T, B, OBS_DIM)
        dir_seq: torch.Tensor,  # (T, B)
        op_seq: torch.Tensor,  # (T, B)
        done_seq: torch.Tensor,  # (T, B)
        h0: torch.Tensor,  # (B, HIDDEN)
    ):
        """Returns (log_prob, entropy, value, dir_logits, op_logits, speed).
        The logits/speed are exposed so a KL leash to a reference policy can
        be computed on the same states."""
        features = self.sequence_features(obs_seq, done_seq, h0)
        dir_logits = self.dir_head(features)
        op_logits = self.op_head(features)
        dir_dist = Categorical(logits=dir_logits)
        op_dist = Categorical(logits=op_logits)
        log_prob = dir_dist.log_prob(dir_seq) + op_dist.log_prob(op_seq)
        entropy = dir_dist.entropy() + op_dist.entropy()
        value = self.value_head(features).squeeze(-1)
        speed = torch.sigmoid(self.speed_head(features)).squeeze(-1)
        return log_prob, entropy, value, dir_logits, op_logits, speed
