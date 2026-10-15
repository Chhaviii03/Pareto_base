"""
model.py — ParetoComm neural network components.

All agents share parameters (parameter sharing MARL).
Architecture per agent:
  ObsEncoder  → hidden representation h_t
  GatingNet   → gate probability p_t
  MsgEncoder  → message vector m_t
  PolicyNet   → action distribution
  ValueNet    → scalar value estimate (for AC baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ─── Utility ──────────────────────────────────────────────────────────────────

def init_weights(m):
    """Orthogonal initialisation for linear layers (better for RL)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.zeros_(m.bias)


# ─── Observation Encoder ──────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """
    2-layer MLP: obs_dim → hidden_dim.
    Shared across all agents.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            h: (batch, hidden_dim)
        """
        return self.net(obs)


# ─── Gating Network ───────────────────────────────────────────────────────────

class GatingNetwork(nn.Module):
    """
    Maps hidden representation → gate probability p_t = σ(f_θ(h_t)).
    Single output neuron + sigmoid.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.apply(init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim)
        Returns:
            p: (batch, 1)  gate probabilities in (0,1)
        """
        logit = self.net(h)             # (batch, 1)
        return torch.sigmoid(logit)     # gate probability


# ─── Message Encoder ──────────────────────────────────────────────────────────

class MessageEncoder(nn.Module):
    """
    Maps hidden representation → message vector.
    Enc_ψ: R^{d_h} → R^{d_m}
    """
    def __init__(self, hidden_dim: int = 64, msg_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, msg_dim),
            nn.Tanh(),  # tanh bounds messages; prevents exploding gradients
        )
        self.apply(init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim)
        Returns:
            msg: (batch, msg_dim)
        """
        return self.net(h)


# ─── Policy Network ───────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Actor: (h_t, aggregated_msg) → categorical action distribution.
    Input: concat of hidden repr and received message context.
    """
    def __init__(self, hidden_dim: int = 64, msg_dim: int = 16, n_actions: int = 5):
        super().__init__()
        input_dim = hidden_dim + msg_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        self.apply(init_weights)
        # Small final layer init for stable initial policy
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, h: torch.Tensor, msg_ctx: torch.Tensor) -> Categorical:
        """
        Args:
            h:       (batch, hidden_dim)
            msg_ctx: (batch, msg_dim)  — mean-pooled messages from others
        Returns:
            dist: Categorical distribution over n_actions
        """
        x = torch.cat([h, msg_ctx], dim=-1)   # (batch, hidden_dim + msg_dim)
        logits = self.net(x)
        return Categorical(logits=logits)


# ─── Value Network ────────────────────────────────────────────────────────────

class ValueNetwork(nn.Module):
    """
    Critic: (h_t, aggregated_msg) → scalar value estimate V(o, m̄).
    Used only during training for advantage estimation.
    """
    def __init__(self, hidden_dim: int = 64, msg_dim: int = 16):
        super().__init__()
        input_dim = hidden_dim + msg_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.apply(init_weights)

    def forward(self, h: torch.Tensor, msg_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:       (batch, hidden_dim)
            msg_ctx: (batch, msg_dim)
        Returns:
            value: (batch, 1)
        """
        x = torch.cat([h, msg_ctx], dim=-1)
        return self.net(x)


# ─── ParetoComm Agent (assembled model) ───────────────────────────────────────

class ParetoCommAgent(nn.Module):
    """
    Complete per-agent model for ParetoComm.
    All agents share these parameters (parameter sharing).

    Components:
        encoder:   ObsEncoder
        gate:      GatingNetwork
        msg_enc:   MessageEncoder
        policy:    PolicyNetwork
        value:     ValueNetwork
    """
    def __init__(
        self,
        obs_dim:    int = 13,
        hidden_dim: int = 64,
        msg_dim:    int = 16,
        n_actions:  int = 5,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim
        self.msg_dim    = msg_dim
        self.n_actions  = n_actions

        self.encoder  = ObsEncoder(obs_dim, hidden_dim)
        self.gate     = GatingNetwork(hidden_dim)
        self.msg_enc  = MessageEncoder(hidden_dim, msg_dim)
        self.policy   = PolicyNetwork(hidden_dim, msg_dim, n_actions)
        self.value    = ValueNetwork(hidden_dim, msg_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation → hidden representation."""
        return self.encoder(obs)

    def gate_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Compute gate probability p_t = σ(f_θ(h_t))."""
        return self.gate(h)

    def encode_message(self, h: torch.Tensor) -> torch.Tensor:
        """Encode hidden representation → message vector."""
        return self.msg_enc(h)

    def act(self, h: torch.Tensor, msg_ctx: torch.Tensor) -> Categorical:
        """Get action distribution given h and aggregated messages."""
        return self.policy(h, msg_ctx)

    def get_value(self, h: torch.Tensor, msg_ctx: torch.Tensor) -> torch.Tensor:
        """Estimate scalar value."""
        return self.value(h, msg_ctx)

    def zero_message(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero message vector (used when gate = 0)."""
        return torch.zeros(batch_size, self.msg_dim, device=device)
