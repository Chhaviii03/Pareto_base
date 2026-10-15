"""
baselines.py — Baseline implementations for ParetoComm comparison.

Implements:
  B1: Always-communicate MAPPO (no gating, task reward only)
  B2: Scalar-penalized MARL (lambda sweep: 0.1, 0.5, 1.0)
  B3: Never communicate (lower bound)

Run: python baselines.py --env rware --episodes 800
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Tuple

from model import ParetoCommAgent
from environment_wrapper import make_env
from utils import (
    setup_logger, MetricsTracker, hypervolume_2d,
    compute_returns, compute_advantages,
    save_checkpoint, plot_pareto_front,
)


DEFAULT_CONFIG = {
    "env":          "rware",
    "n_agents":     4,
    "max_steps":    500,
    "episodes":     800,
    "gamma":        0.99,
    "lr":           3e-4,
    "value_coef":   0.5,
    "ent_coef":     0.01,
    "hidden_dim":   64,
    "msg_dim":      16,
    "seed":         42,
    "device":       "cpu",
    "results_dir":  "results_baselines",
    "eval_episodes": 50,
    "lambda_values": [0.1, 0.5, 1.0],  # for SP-MARL
}


# ─── Baseline: Never communicate ─────────────────────────────────────────────

def never_communicate_baseline(env, n_agents, n_eval_eps):
    """
    Evaluate a never-communicate policy (random actions, no messages).
    Returns (mean_task_return, mean_comm_return=0, comm_rate=0).
    """
    task_returns = []
    for _ in range(n_eval_eps):
        obs = env.reset()
        done = False
        ep_task = 0.0
        while not done:
            actions = [env._env.action_space[i].sample() for i in range(n_agents)]
            gate_vals = np.zeros(n_agents)
            _, r_vec, done, _ = env.step(actions, gate_vals)
            ep_task += r_vec[0]
        task_returns.append(ep_task)
    return float(np.mean(task_returns)), 0.0, 0.0


# ─── Baseline: Always communicate (MAPPO) ────────────────────────────────────

def always_communicate_baseline(env, agent, optimizer, config, logger):
    """
    Train a policy that always communicates (gate ≡ 1).
    Task reward only (no communication cost in reward).
    """
    n_agents = config["n_agents"]
    device   = torch.device(config["device"])

    # Import training utilities
    from train import collect_episode, update_actor_critic, evaluate
    from gating import TemperatureScheduler
    import time

    w_vec    = np.array([1.0, 0.0])   # task reward only
    tau_sched = TemperatureScheduler(tau_0=0.001, tau_min=0.001, delta=10**9)
    # tau ≈ 0 → gate ≈ hard always-1 (we will force gate=1 below)

    best_return = -float("inf")

    for episode in range(1, config["episodes"] + 1):
        # Force always-communicate: override gate network
        # We do this by setting gate network weights to output large logit
        with torch.no_grad():
            for param in agent.gate.parameters():
                param.fill_(10.0)   # large logit → p ≈ 1 → gate ≡ 1

        buf = collect_episode(env, agent, tau_sched, w_vec,
                              n_agents, device, deterministic=False)

        if episode % 100 == 0:
            ep_task = sum(r[0] for r in buf.vector_rewards)
            logger.info(f"  [Always-Comm] ep={episode}  task={ep_task:.3f}")

    eval_task, eval_comm, eval_comm_rate = evaluate(
        env, agent, n_agents, config["eval_episodes"], device
    )
    # Comm return = -n_agents * T (always communicating)
    eval_comm_return = -float(n_agents)  # per-step cost summed over episode

    return eval_task, eval_comm_return, 1.0  # comm_rate = 1.0


# ─── Baseline: Scalar-penalized MARL ─────────────────────────────────────────

def scalar_penalized_baseline(env, agent, optimizer, config, lam, logger):
    """
    Train a policy with scalar reward: r = r_task - λ * r_comm.
    Returns (eval_task_return, eval_comm_return, eval_comm_rate).
    """
    from train import collect_episode, update_actor_critic, evaluate
    from gating import TemperatureScheduler

    n_agents = config["n_agents"]
    device   = torch.device(config["device"])

    # Weight vector: w = [1/(1+λ), λ/(1+λ)] to keep same scale
    # More simply: treat as w = (1.0, lam) unnormalized, normalize to simplex
    w_sum  = 1.0 + lam
    w_vec  = np.array([1.0 / w_sum, lam / w_sum])

    tau_sched = TemperatureScheduler(
        tau_0=1.0, lambda_=0.995, tau_min=0.1, delta=1000
    )

    for episode in range(1, config["episodes"] + 1):
        buf = collect_episode(env, agent, tau_sched, w_vec,
                              n_agents, device)
        update_actor_critic(
            agent, optimizer, buf, tau_sched, w_vec,
            n_agents, config["gamma"], config["value_coef"],
            config["ent_coef"], 0.005, device,
        )

        if episode % 200 == 0:
            ep_task = sum(r[0] for r in buf.vector_rewards)
            ep_gates = np.stack(buf.gate_vals).mean()
            logger.info(f"  [SP λ={lam}] ep={episode}  task={ep_task:.3f}  gate={ep_gates:.2f}")

    eval_task, eval_comm, eval_comm_rate = evaluate(
        env, agent, n_agents, config["eval_episodes"], device
    )
    return eval_task, eval_comm, eval_comm_rate


# ─── Main baseline runner ─────────────────────────────────────────────────────

def run_baselines(config: dict):
    os.makedirs(config["results_dir"], exist_ok=True)
    logger = setup_logger("baselines",
                          log_file=os.path.join(config["results_dir"], "baselines.log"))
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    env = make_env(config["env"], n_agents=config["n_agents"],
                   max_steps=config["max_steps"])

    baseline_points = {}   # method → (task, comm) point

    # ── B3: Never communicate ────────────────────────────────────────────
    logger.info("B3: Never communicate baseline")
    task_nc, comm_nc, rate_nc = never_communicate_baseline(
        env, config["n_agents"], config["eval_episodes"]
    )
    baseline_points["Never communicate"] = np.array([[task_nc, comm_nc]])
    logger.info(f"  task={task_nc:.4f}  comm={comm_nc:.4f}  rate={rate_nc:.3f}")

    # ── B2: Scalar-penalized MARL (lambda sweep) ─────────────────────────
    for lam in config["lambda_values"]:
        logger.info(f"\nB2: Scalar-penalized MARL λ={lam}")
        agent = ParetoCommAgent(
            obs_dim=env.obs_dim, hidden_dim=config["hidden_dim"],
            msg_dim=config["msg_dim"], n_actions=env.n_actions,
        ).to(device)
        optimizer = Adam(agent.parameters(), lr=config["lr"])
        task_sp, comm_sp, rate_sp = scalar_penalized_baseline(
            env, agent, optimizer, config, lam, logger
        )
        baseline_points[f"SP-MARL λ={lam}"] = np.array([[task_sp, comm_sp]])
        logger.info(f"  task={task_sp:.4f}  comm={comm_sp:.4f}  rate={rate_sp:.3f}")
        env.close()
        env = make_env(config["env"], n_agents=config["n_agents"],
                       max_steps=config["max_steps"])

    env.close()

    # Save and print summary
    logger.info("\n=== Baseline Summary ===")
    ref_point = np.array([0.0, -float(config["n_agents"])])
    for name, pts in baseline_points.items():
        hv = hypervolume_2d(pts, ref_point)
        logger.info(f"  {name:30s}: task={pts[0,0]:.4f}  comm={pts[0,1]:.4f}  HV={hv:.4f}")

    # Save for plotting alongside ParetoComm
    import json
    out = {k: v.tolist() for k, v in baseline_points.items()}
    with open(os.path.join(config["results_dir"], "baselines.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nBaseline results saved to {config['results_dir']}/baselines.json")

    return baseline_points


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env",      type=str,  default="rware")
    p.add_argument("--episodes", type=int,  default=800)
    p.add_argument("--seed",     type=int,  default=42)
    p.add_argument("--quick",    action="store_true")
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["env"]     = args.env
    cfg["episodes"] = args.episodes
    cfg["seed"]    = args.seed
    if args.quick:
        cfg["episodes"]     = 50
        cfg["eval_episodes"] = 5
        cfg["lambda_values"] = [0.5]

    run_baselines(cfg)
