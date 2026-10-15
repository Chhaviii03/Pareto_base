"""
train.py — ParetoComm full training pipeline.

Trains one actor-critic policy per weight vector w ∈ W,
collecting a Pareto front approximation over the RWARE environment.

Usage:
    python train.py                          # default config (RWARE)
    python train.py --env rware --n_agents 4 --episodes 800
    python train.py --quick                  # fast test (50 eps, 3 weights)

Output:
    results/
      paretocomm_w{i}/  checkpoints per weight vector
      metrics.json       all logged metrics
      pareto_front.pdf   Pareto front figure
      training_curves.pdf training curves
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Tuple

# Local imports
from model import ParetoCommAgent
from gating import (
    straight_through_gumbel,
    TemperatureScheduler,
    gate_entropy,
    aggregate_messages_batch,
)
from environment_wrapper import make_env
from utils import (
    setup_logger,
    MetricsTracker,
    hypervolume_2d,
    non_dominated_set,
    compute_returns,
    compute_advantages,
    save_checkpoint,
    plot_pareto_front,
    plot_training_curves,
    plot_communication_tradeoff,
)


# ─── Default Hyperparameters ──────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Environment
    "env":          "rware",
    "n_agents":     4,
    "max_steps":    500,
    # Training
    "episodes":     800,
    "gamma":        0.99,
    "lr":           3e-4,
    "value_coef":   0.5,
    "ent_coef":     0.01,
    "gate_ent_coef": 0.005,
    # Architecture
    "hidden_dim":   64,
    "msg_dim":      16,
    # Gumbel-Sigmoid annealing
    "tau_0":        1.0,
    "tau_lambda":   0.995,
    "tau_min":      0.1,
    "tau_delta":    1000,
    # Weight grid
    "w_step":       0.1,         # step size; 0.1 → 11 weight vectors
    # Evaluation
    "eval_episodes": 50,
    # Output
    "results_dir":  "results",
    "seed":         42,
    "device":       "cpu",       # "cuda" if GPU available
}


# ─── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ParetoComm Training")
    p.add_argument("--env",          type=str,   default="rware")
    p.add_argument("--n_agents",     type=int,   default=4)
    p.add_argument("--episodes",     type=int,   default=800)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--hidden_dim",   type=int,   default=64)
    p.add_argument("--msg_dim",      type=int,   default=16)
    p.add_argument("--w_step",       type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--results_dir",  type=str,   default="results")
    p.add_argument("--quick",        action="store_true",
                   help="Quick test: 50 episodes, 3 weight vectors")
    return p.parse_args()


# ─── Trajectory buffer ────────────────────────────────────────────────────────

class EpisodeBuffer:
    """Stores one episode of transitions for all agents."""
    def __init__(self):
        self.obs:          List[List[np.ndarray]] = []  # [t][agent]
        self.actions:      List[List[int]]        = []
        self.gate_vals:    List[np.ndarray]       = []  # [t] shape (n_agents,)
        self.gate_logprobs:List[torch.Tensor]     = []  # [t] shape (n_agents,)
        self.act_logprobs: List[torch.Tensor]     = []  # [t] shape (n_agents,)
        self.values:       List[torch.Tensor]     = []  # [t] shape (n_agents,)
        self.scalar_rewards: List[float]          = []  # [t] scalar r_w
        self.vector_rewards: List[np.ndarray]     = []  # [t] (2,)
        self.gate_probs:   List[torch.Tensor]     = []  # [t] (n_agents,) for entropy

    def add(self, obs, actions, gate_vals, gate_logprobs,
            act_logprobs, values, scalar_r, vector_r, gate_probs):
        self.obs.append(obs)
        self.actions.append(actions)
        self.gate_vals.append(gate_vals)
        self.gate_logprobs.append(gate_logprobs)
        self.act_logprobs.append(act_logprobs)
        self.values.append(values)
        self.scalar_rewards.append(scalar_r)
        self.vector_rewards.append(vector_r)
        self.gate_probs.append(gate_probs)

    def __len__(self):
        return len(self.scalar_rewards)


# ─── One episode rollout ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_episode(
    env,
    agent:    ParetoCommAgent,
    tau_sched: TemperatureScheduler,
    w_vec:    np.ndarray,          # (2,) weight vector
    n_agents: int,
    device:   torch.device,
    deterministic: bool = False,
) -> EpisodeBuffer:
    """
    Run one episode and collect a trajectory buffer.
    Uses no_grad for efficiency; gradients computed during update.

    Args:
        deterministic: if True, use argmax actions (for evaluation)
    """
    buf  = EpisodeBuffer()
    obs  = env.reset()           # list of (obs_dim,) arrays
    done = False
    tau  = tau_sched.current

    while not done:
        # Convert observations to tensor
        obs_t = torch.FloatTensor(np.stack(obs)).to(device)  # (n_agents, obs_dim)

        # 1. Encode observations
        h_t = agent.encode(obs_t)                             # (n_agents, hidden_dim)

        # 2. Compute gate probabilities
        p_t = agent.gate_prob(h_t)                            # (n_agents, 1)

        # 3. Compute gated messages using Straight-Through Gumbel
        # We need grad here for the update step — but collect_episode
        # is @no_grad, so we just get the hard gate value for rollout.
        # Gradients are recomputed in the update step.
        gate_hard = (torch.sigmoid(agent.gate.net(h_t)) > 0.5).float()  # (n_agents, 1)
        gate_np   = gate_hard.squeeze(-1).cpu().numpy()                  # (n_agents,)

        # 4. Encode messages
        raw_msgs = agent.encode_message(h_t)                  # (n_agents, msg_dim)
        msgs = gate_hard * raw_msgs                           # gated messages

        # 5. Aggregate messages for each agent
        msgs_stack = msgs.unsqueeze(1)                        # (n_agents, 1, msg_dim)
        # Manually aggregate: for each agent i, mean of others
        msg_ctx = aggregate_messages_batch(
            msgs.unsqueeze(0).transpose(0, 1),                # (n_agents, 1, msg_dim)
            n_agents
        ).squeeze(1)                                          # (n_agents, msg_dim)
        # Simpler approach for clarity:
        msg_ctx = _mean_other_messages(msgs, n_agents)        # (n_agents, msg_dim)

        # 6. Sample actions
        dists   = agent.act(h_t, msg_ctx)                     # Categorical
        if deterministic:
            actions = dists.probs.argmax(dim=-1).tolist()
        else:
            actions = dists.sample().tolist()

        act_lp  = dists.log_prob(
            torch.tensor(actions, device=device)
        )                                                     # (n_agents,)

        # 7. Gate log-prob (Bernoulli)
        p_clamp = p_t.squeeze(-1).clamp(1e-8, 1-1e-8)
        gate_t  = torch.tensor(gate_np, device=device, dtype=torch.float32)
        gate_lp = gate_t * torch.log(p_clamp) + \
                  (1 - gate_t) * torch.log(1 - p_clamp)      # (n_agents,)

        # 8. Value estimate
        val = agent.get_value(h_t, msg_ctx).squeeze(-1)       # (n_agents,)

        # 9. Step environment
        next_obs, r_vec, done, info = env.step(actions, gate_np)

        # 10. Scalarize reward
        r_scalar = float(np.dot(w_vec, r_vec))

        buf.add(
            obs=list(obs),
            actions=actions,
            gate_vals=gate_np,
            gate_logprobs=gate_lp.cpu(),
            act_logprobs=act_lp.cpu(),
            values=val.cpu(),
            scalar_r=r_scalar,
            vector_r=r_vec,
            gate_probs=p_clamp.cpu(),
        )

        obs = next_obs
        tau_sched.step(1)
        tau = tau_sched.current

    return buf


def _mean_other_messages(msgs: torch.Tensor, n_agents: int) -> torch.Tensor:
    """
    Compute mean of other agents' messages for each agent.
    msgs: (n_agents, msg_dim) → out: (n_agents, msg_dim)
    """
    total = msgs.sum(dim=0, keepdim=True)          # (1, msg_dim)
    other_mean = (total - msgs) / max(n_agents - 1, 1)  # (n_agents, msg_dim)
    return other_mean


# ─── Actor-Critic Update ──────────────────────────────────────────────────────

def update_actor_critic(
    agent:      ParetoCommAgent,
    optimizer:  torch.optim.Optimizer,
    buf:        EpisodeBuffer,
    tau_sched:  TemperatureScheduler,
    w_vec:      np.ndarray,
    n_agents:   int,
    gamma:      float,
    value_coef: float,
    ent_coef:   float,
    gate_ent_coef: float,
    device:     torch.device,
) -> Dict[str, float]:
    """
    One actor-critic update from a complete episode trajectory.

    Returns dict of loss values for logging.
    """
    T   = len(buf)
    tau = tau_sched.current

    # ── 1. Compute MC returns ───────────────────────────────────────────────
    returns_np  = []
    R = 0.0
    for r in reversed(buf.scalar_rewards):
        R = r + gamma * R
        returns_np.insert(0, R)
    returns_t = torch.tensor(returns_np, dtype=torch.float32, device=device)

    # ── 2. Recompute values and log-probs WITH gradients ───────────────────
    all_act_lp   = []  # (T, n_agents)
    all_gate_lp  = []  # (T, n_agents)
    all_values   = []  # (T, n_agents)
    all_gate_ent = []  # (T, n_agents)

    for t in range(T):
        obs_t  = torch.FloatTensor(np.stack(buf.obs[t])).to(device)
        h_t    = agent.encode(obs_t)

        # Recompute gate with Straight-Through Gumbel (WITH grad)
        p_t    = agent.gate_prob(h_t)                         # (n_agents, 1)

        # Straight-Through gate
        g_t, g_soft = straight_through_gumbel(p_t, tau=tau)  # both (n_agents, 1)

        raw_msgs = agent.encode_message(h_t)
        msgs     = g_t * raw_msgs                             # gated messages (with grad)

        msg_ctx  = _mean_other_messages(msgs.squeeze(-1) if msgs.dim() == 3
                                        else msgs, n_agents)

        dists    = agent.act(h_t, msg_ctx)
        actions_t = torch.tensor(buf.actions[t], device=device)
        act_lp    = dists.log_prob(actions_t)                 # (n_agents,)

        # Gate log-prob (using hard gate for correct likelihood)
        p_clamp  = p_t.squeeze(-1).clamp(1e-8, 1-1e-8)
        gate_t   = torch.tensor(buf.gate_vals[t], dtype=torch.float32, device=device)
        gate_lp  = gate_t * torch.log(p_clamp) + \
                   (1 - gate_t) * torch.log(1 - p_clamp)

        val      = agent.get_value(h_t, msg_ctx).squeeze(-1)  # (n_agents,)

        # Binary entropy of gate
        g_ent    = gate_entropy(p_clamp.unsqueeze(-1)).squeeze(-1)

        all_act_lp.append(act_lp)
        all_gate_lp.append(gate_lp)
        all_values.append(val)
        all_gate_ent.append(g_ent)

    # Stack: (T, n_agents)
    act_lp_all  = torch.stack(all_act_lp,   dim=0)
    gate_lp_all = torch.stack(all_gate_lp,  dim=0)
    values_all  = torch.stack(all_values,   dim=0)
    gate_ent_all = torch.stack(all_gate_ent, dim=0)

    # Mean across agents → (T,)
    act_lp_mean  = act_lp_all.mean(dim=1)
    gate_lp_mean = gate_lp_all.mean(dim=1)
    values_mean  = values_all.mean(dim=1)
    gate_ent_mean = gate_ent_all.mean(dim=1)

    # ── 3. Compute advantages ──────────────────────────────────────────────
    advantages = (returns_t - values_mean.detach())
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── 4. Losses ──────────────────────────────────────────────────────────
    # Policy loss: -E[A_t * (log π + log p(g))]
    policy_loss = -(advantages * (act_lp_mean + gate_lp_mean)).mean()

    # Value loss: MSE between returns and value estimates
    value_loss  = F.mse_loss(values_mean, returns_t)

    # Entropy bonuses (encourage exploration)
    action_entropy = -act_lp_mean.mean()               # proxy for action entropy
    gate_entropy_  = gate_ent_mean.mean()

    # Combined loss
    loss = (policy_loss
            + value_coef   * value_loss
            - ent_coef     * action_entropy
            - gate_ent_coef * gate_entropy_)

    # ── 5. Optimize ────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    optimizer.step()

    return {
        "loss":          loss.item(),
        "policy_loss":   policy_loss.item(),
        "value_loss":    value_loss.item(),
        "action_entropy": action_entropy.item(),
        "gate_entropy":   gate_entropy_.item(),
    }


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    env,
    agent:    ParetoCommAgent,
    n_agents: int,
    n_eps:    int,
    device:   torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate agent over n_eps episodes.
    Returns (mean_task_return, mean_comm_return, mean_comm_rate).
    """
    task_returns = []
    comm_returns = []
    comm_rates   = []

    dummy_tau = TemperatureScheduler(tau_0=0.1, tau_min=0.1, delta=10**9)
    w_dummy   = np.array([1.0, 0.0])  # only care about task for evaluation

    for _ in range(n_eps):
        buf = collect_episode(
            env, agent, dummy_tau, w_dummy, n_agents, device,
            deterministic=True
        )
        ep_task = sum(r[0] for r in buf.vector_rewards)
        ep_comm = sum(r[1] for r in buf.vector_rewards)
        ep_gates = np.stack(buf.gate_vals)             # (T, n_agents)
        comm_rate = float(ep_gates.mean())

        task_returns.append(ep_task)
        comm_returns.append(ep_comm)
        comm_rates.append(comm_rate)

    return (
        float(np.mean(task_returns)),
        float(np.mean(comm_returns)),
        float(np.mean(comm_rates)),
    )


# ─── Main training loop ───────────────────────────────────────────────────────

def train(config: dict):
    # ── Setup ──────────────────────────────────────────────────────────────
    os.makedirs(config["results_dir"], exist_ok=True)
    logger  = setup_logger(
        "paretocomm",
        log_file=os.path.join(config["results_dir"], "train.log")
    )
    device  = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    logger.info("=" * 60)
    logger.info("ParetoComm Training")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)

    # ── Build weight grid ──────────────────────────────────────────────────
    step = config["w_step"]
    w1_values = np.arange(0.0, 1.0 + step / 2, step)
    weight_grid = [(round(w1, 2), round(1 - w1, 2)) for w1 in w1_values]
    logger.info(f"Weight grid: {len(weight_grid)} vectors")

    # ── Environment (used for obs_dim and n_actions) ────────────────────────
    env = make_env(config["env"], n_agents=config["n_agents"],
                   max_steps=config["max_steps"])
    obs_dim   = env.obs_dim
    n_actions = env.n_actions
    n_agents  = config["n_agents"]
    logger.info(f"Env: {config['env']}  obs_dim={obs_dim}  n_actions={n_actions}")

    # ── Metrics tracker ────────────────────────────────────────────────────
    tracker    = MetricsTracker()
    pf_points  = []      # (task_return, comm_return) per weight vector
    comm_rates = []

    # Reference point for Hypervolume:
    # r_task ∈ [0, ∞),  r_comm ∈ [-n*T, 0]
    # Use (0, -n_agents) as a lower bound that all reasonable policies dominate
    ref_point = np.array([0.0, -float(n_agents)])

    # ── Main loop over weight vectors ──────────────────────────────────────
    for w_idx, (w1, w2) in enumerate(weight_grid):
        w_vec     = np.array([w1, w2])
        w_str     = f"w=({w1:.1f},{w2:.1f})"
        ckpt_dir  = os.path.join(config["results_dir"], f"paretocomm_w{w_idx:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        logger.info(f"\n{'='*50}")
        logger.info(f"Weight vector {w_idx+1}/{len(weight_grid)}: {w_str}")
        logger.info(f"{'='*50}")

        # ── Initialize model and optimizer ──────────────────────────────
        agent = ParetoCommAgent(
            obs_dim=obs_dim, hidden_dim=config["hidden_dim"],
            msg_dim=config["msg_dim"], n_actions=n_actions,
        ).to(device)

        optimizer = Adam(agent.parameters(), lr=config["lr"])
        tau_sched = TemperatureScheduler(
            tau_0   = config["tau_0"],
            lambda_ = config["tau_lambda"],
            tau_min = config["tau_min"],
            delta   = config["tau_delta"],
        )

        best_scalar_return = -float("inf")

        # ── Episode loop ────────────────────────────────────────────────
        for episode in range(1, config["episodes"] + 1):
            t_start = time.time()

            # Collect trajectory
            buf = collect_episode(
                env, agent, tau_sched, w_vec, n_agents, device
            )

            # Compute episode metrics
            ep_task   = sum(r[0] for r in buf.vector_rewards)
            ep_comm   = sum(r[1] for r in buf.vector_rewards)
            ep_scalar = float(np.dot(w_vec, [ep_task, ep_comm]))
            ep_gates  = np.stack(buf.gate_vals)
            ep_comm_rate = float(ep_gates.mean())
            ep_tau    = tau_sched.current

            # Update parameters
            loss_info = update_actor_critic(
                agent, optimizer, buf, tau_sched, w_vec,
                n_agents, config["gamma"], config["value_coef"],
                config["ent_coef"], config["gate_ent_coef"], device,
            )

            # Log
            tracker.log(
                w_idx=w_idx,
                episode=episode,
                weight_str=w_str,
                task_return=ep_task,
                comm_return=ep_comm,
                scalar_return=ep_scalar,
                comm_rate=ep_comm_rate,
                tau=ep_tau,
                **loss_info,
            )

            # Console logging every 100 episodes
            if episode % 100 == 0 or episode == 1:
                hv = hypervolume_2d(
                    np.array([[ep_task, ep_comm]]), ref_point
                )
                ep_time = time.time() - t_start
                logger.info(
                    f"  [{w_str}] ep={episode:4d}/{config['episodes']}"
                    f"  task={ep_task:6.3f}  comm={ep_comm:6.1f}"
                    f"  gate={ep_comm_rate:.2f}  τ={ep_tau:.3f}"
                    f"  loss={loss_info['loss']:.4f}  t={ep_time:.2f}s"
                )

            # Save best checkpoint
            if ep_scalar > best_scalar_return:
                best_scalar_return = ep_scalar
                save_checkpoint(
                    agent, optimizer, w_idx, episode,
                    {"scalar_return": ep_scalar},
                    os.path.join(ckpt_dir, "best.pt")
                )

        # ── Evaluate this weight vector's policy ─────────────────────────
        logger.info(f"  Evaluating {w_str} over {config['eval_episodes']} episodes...")
        eval_task, eval_comm, eval_comm_rate = evaluate(
            env, agent, n_agents, config["eval_episodes"], device
        )
        logger.info(
            f"  EVAL {w_str}: task={eval_task:.4f}  "
            f"comm={eval_comm:.4f}  comm_rate={eval_comm_rate:.3f}"
        )

        # Log eval metrics
        tracker.log(
            w_idx=w_idx,
            episode=config["episodes"],  # mark as final
            eval_task_return=eval_task,
            eval_comm_return=eval_comm,
            eval_comm_rate=eval_comm_rate,
        )

        pf_points.append([eval_task, eval_comm])
        comm_rates.append(eval_comm_rate)

    env.close()

    # ── Post-training: compute and report Pareto front ────────────────────
    pf_array   = np.array(pf_points)
    comm_array = np.array(comm_rates)

    hv_total = hypervolume_2d(pf_array, ref_point)
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"  Total policies:      {len(pf_array)}")
    logger.info(f"  Hypervolume:         {hv_total:.4f}")
    logger.info(f"  Reference point:     {ref_point}")
    logger.info(f"  Mean comm rate:      {comm_array.mean():.3f}")
    logger.info(f"{'='*60}")

    # Print Pareto front table
    logger.info("\nPareto Front Points:")
    logger.info(f"  {'w1':>4} {'w2':>4} | {'task_ret':>10} {'comm_ret':>10} {'comm_rate':>10}")
    logger.info(f"  {'-'*50}")
    for (w1, w2), (t_r, c_r), c_rate in zip(weight_grid, pf_points, comm_rates):
        logger.info(f"  {w1:4.1f} {w2:4.1f} | {t_r:10.4f} {c_r:10.4f} {c_rate:10.3f}")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_path = os.path.join(config["results_dir"], "metrics.json")
    tracker.save(metrics_path)

    # ── Generate figures ──────────────────────────────────────────────────
    logger.info("\nGenerating figures...")

    fig_dir = os.path.join(config["results_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    try:
        plot_pareto_front(
            pf_points   = pf_array,
            ref_point   = ref_point,
            save_path   = os.path.join(fig_dir, "pareto_front"),
        )
        plot_training_curves(
            metrics    = tracker,
            ref_point  = ref_point,
            save_path  = os.path.join(fig_dir, "training_curves"),
        )
        plot_communication_tradeoff(
            pf_points  = pf_array,
            comm_rates = comm_array,
            save_path  = os.path.join(fig_dir, "comm_tradeoff"),
        )
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    logger.info(f"\nAll outputs saved to: {config['results_dir']}/")
    logger.info("Training complete.")

    return pf_array, hv_total, tracker


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    # Override with CLI args
    config.update({
        "env":         args.env,
        "n_agents":    args.n_agents,
        "episodes":    args.episodes,
        "lr":          args.lr,
        "gamma":       args.gamma,
        "hidden_dim":  args.hidden_dim,
        "msg_dim":     args.msg_dim,
        "w_step":      args.w_step,
        "seed":        args.seed,
        "device":      args.device,
        "results_dir": args.results_dir,
    })

    # Quick test mode
    if args.quick:
        config["episodes"]    = 50
        config["w_step"]      = 0.5   # 3 weight vectors: (0.0,1.0), (0.5,0.5), (1.0,0.0)
        config["eval_episodes"] = 5
        config["max_steps"]   = 100
        print("[QUICK MODE] Running minimal test: 50 episodes, 3 weight vectors")

    # Auto-detect CUDA
    if torch.cuda.is_available() and config["device"] == "cpu":
        config["device"] = "cuda"
        print(f"CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")

    train(config)
