"""
gating.py — Straight-Through Gumbel-Sigmoid implementation.

Implements Eq. 4–6 from the paper:
  - Gumbel-Sigmoid relaxation (differentiable during training)
  - Straight-Through estimator (hard gate in forward, soft gradient in backward)
  - Temperature annealing schedule

References:
  Jang et al. (2017) "Categorical Reparameterization with Gumbel-Softmax"
  Maddison et al. (2017) "The Concrete Distribution"
"""

import torch
import torch.nn.functional as F
import math


# ─── Gumbel noise ─────────────────────────────────────────────────────────────

def sample_gumbel(shape: torch.Size, device: torch.device,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Sample from Gumbel(0, 1) distribution.
    Gumbel(0,1) = -log(-log(U))  where U ~ Uniform(0,1).
    """
    U = torch.rand(shape, device=device).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(U)).clamp(-10, 10)  # prevent inf


# ─── Gumbel-Sigmoid (binary Gumbel-Softmax) ───────────────────────────────────

def gumbel_sigmoid(log_p: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Gumbel-Sigmoid relaxation for binary gating.

    Given log_p = log(p_t) where p_t = σ(f_θ(h_t)), computes:

        g̃_t = σ( (log p_t - log(1-p_t) + ε₁ - ε₂) / τ )

    where ε₁, ε₂ ~ Gumbel(0,1).

    This is equivalent to:
        g̃_t = σ( (logit(p_t) + ε₁ - ε₂) / τ )

    As τ → 0, g̃_t → Bernoulli(p_t).
    As τ → ∞, g̃_t → 0.5 (uniform, maximum entropy).

    Args:
        log_p:  (batch, 1)  log gate probabilities, i.e., log(σ(f_θ(h)))
                NOTE: pass log_p = torch.log(p) where p = gate_network(h)
        tau:    temperature scalar > 0

    Returns:
        g_soft: (batch, 1) soft gate in (0, 1)
    """
    # ε₁, ε₂ ~ Gumbel(0,1)
    eps1 = sample_gumbel(log_p.shape, log_p.device)
    eps2 = sample_gumbel(log_p.shape, log_p.device)

    # logit of p: log(p/(1-p)) = log p - log(1-p)
    log_1mp = torch.log(1 - log_p.exp() + 1e-8)
    logit_p = log_p - log_1mp

    # Gumbel-perturbed logit
    perturbed = (logit_p + eps1 - eps2) / tau

    return torch.sigmoid(perturbed)


# ─── Straight-Through Gumbel estimator ────────────────────────────────────────

def straight_through_gumbel(
    p: torch.Tensor,
    tau: float = 1.0,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Straight-Through Gumbel-Sigmoid estimator.

    Forward pass:  g_t  = g_hard  (hard binary {0, 1})
    Backward pass: grad flows through g_soft (differentiable)

    Implementation:
        g_hard = (g_soft > 0.5).float()
        g_t = g_hard + sg(g_soft - g_hard)
            = g_soft + sg(g_hard - g_soft)  [equivalent]
    where sg(·) = stop-gradient (detach in PyTorch).

    In PyTorch:
        g_t = g_hard + (g_soft - g_soft.detach())  [one-liner]

    This is mathematically equivalent to:
        forward:   g_t = g_hard
        backward:  ∂g_t/∂θ = ∂g_soft/∂θ

    Args:
        p:         (batch, 1)  gate probabilities in (0, 1)
        tau:       temperature
        threshold: binarization threshold (default 0.5)

    Returns:
        g_t:    (batch, 1) gate values — hard {0,1} in forward,
                               differentiable in backward
        g_soft: (batch, 1) soft gate (for logging / entropy computation)
    """
    log_p   = torch.log(p.clamp(1e-8, 1 - 1e-8))
    g_soft  = gumbel_sigmoid(log_p, tau=tau)              # soft ∈ (0,1)
    g_hard  = (g_soft > threshold).float()                 # hard ∈ {0,1}

    # Straight-Through: forward = hard, backward = soft
    # g_hard.detach() ensures gradient doesn't flow through g_hard
    g_t = g_hard + (g_soft - g_soft.detach())
    # Equivalent and numerically identical to:
    #   g_t = g_soft + (g_hard - g_hard.detach())

    return g_t, g_soft


# ─── Temperature annealing ────────────────────────────────────────────────────

class TemperatureScheduler:
    """
    Exponential temperature annealing schedule (Eq. 9 in paper):
        τ_t = max(τ_min, τ₀ · λ^⌊t / Δ⌋)

    where t = cumulative environment timesteps (not episodes).

    Args:
        tau_0:   initial temperature (default 1.0)
        lambda_: decay factor per interval (default 0.995)
        tau_min: minimum temperature floor (default 0.1)
        delta:   decay interval in timesteps (default 1000)
    """
    def __init__(
        self,
        tau_0:   float = 1.0,
        lambda_: float = 0.995,
        tau_min: float = 0.1,
        delta:   int   = 1000,
    ):
        self.tau_0   = tau_0
        self.lambda_ = lambda_
        self.tau_min = tau_min
        self.delta   = delta
        self._t      = 0        # cumulative timesteps

    def step(self, n_steps: int = 1) -> None:
        """Advance the schedule by n_steps environment timesteps."""
        self._t += n_steps

    def get_tau(self) -> float:
        """Return current temperature τ_t."""
        decay_steps = self._t // self.delta
        tau = self.tau_0 * (self.lambda_ ** decay_steps)
        return max(self.tau_min, tau)

    @property
    def current(self) -> float:
        return self.get_tau()

    def reset(self) -> None:
        """Reset timestep counter (call between weight vectors)."""
        self._t = 0


# ─── Gate entropy ─────────────────────────────────────────────────────────────

def gate_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    Binary entropy H(p) = -p log p - (1-p) log(1-p).
    Used as entropy bonus for the gating network.

    Args:
        p: (batch, 1) gate probabilities
    Returns:
        entropy: (batch, 1)
    """
    p = p.clamp(1e-8, 1 - 1e-8)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


# ─── Message aggregation ──────────────────────────────────────────────────────

def aggregate_messages(
    messages: torch.Tensor,
    agent_idx: int,
    n_agents: int,
) -> torch.Tensor:
    """
    Mean-pool messages from all agents except agent_idx.

    m̄^i_t = 1/(n-1) Σ_{j≠i} m^j_t

    Args:
        messages:  (n_agents, batch, msg_dim)
        agent_idx: index of the receiving agent
        n_agents:  total number of agents

    Returns:
        context: (batch, msg_dim)
    """
    # Mask out agent's own message
    others = [j for j in range(n_agents) if j != agent_idx]
    other_msgs = messages[others]                           # (n-1, batch, msg_dim)
    return other_msgs.mean(dim=0)                           # (batch, msg_dim)


def aggregate_messages_batch(
    messages: torch.Tensor,
    n_agents: int,
) -> torch.Tensor:
    """
    Compute aggregated context for ALL agents simultaneously.
    More efficient than calling aggregate_messages n times.

    Args:
        messages: (n_agents, batch, msg_dim)
    Returns:
        contexts: (n_agents, batch, msg_dim)
                  contexts[i] = mean of all messages except agent i
    """
    # Sum all messages
    total = messages.sum(dim=0, keepdim=True)             # (1, batch, msg_dim)
    # For agent i: subtract own message and divide by (n-1)
    contexts = (total - messages) / max(n_agents - 1, 1)  # (n_agents, batch, msg_dim)
    return contexts
