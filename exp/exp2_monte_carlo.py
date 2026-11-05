"""Experiment 2: Monte Carlo simulation study of the RL dynamic strategy selection
algorithm (Section S.6 of the Electronic Supplement to "Dynamic Strategy Selection").

For each combination of training-sample size (240, 600, 1200 months) and number of
irrelevant AR(1) state variables (0, 2, 4, 8), simulates the four-state Markov-chain
data-generating process of Section S.6.1, estimates the RL policy (fitted Q-iteration,
Algorithm S.1) and the myopic Greedy policy from the training sample, and evaluates both
out of sample together with the analytic oracle policy and the fixed "always Strategy 1"
benchmark. Monte Carlo means and standard errors across replications are reported and
compared with the paper's reported oracle/RL/Greedy/fixed-strategy mean rewards.

Run: ``python -m exp.exp2_monte_carlo`` (reduced replication count) or
``python -m exp.exp2_monte_carlo --full`` (the paper's 500 replications per configuration).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.generate_synthetic import make_dataset
from src import config
from src.metrics import mean_reward_pct, switching_frequency
from src.oracle import oracle_mean_reward, simulate_oracle_policy
from src.policy_evaluation import run_estimated_policy, run_fixed_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _ret_vals_from(returns: np.ndarray) -> np.ndarray:
    """Pads ``returns`` (row k = Ret_{k+1}) with a leading zero row so that
    ``ret_vals[k] = Ret_k`` for k >= 1, the convention used by
    ``src.policy_evaluation.run_estimated_policy``/``run_fixed_policy``."""
    return np.vstack([np.zeros((1, returns.shape[1])), returns])


def run_one_replication(n_train: int, n_noise: int, seed: int, rho_grid: np.ndarray) -> dict:
    """RL, Greedy, oracle and fixed-Strategy-1 out-of-sample mean rewards and switching
    frequencies for one simulated path (Section S.6.1 and S.6.4 "Simulation design")."""
    K = np.array(config.EXP2_TRANSITION_MATRIX)
    dataset = make_dataset(
        K=K, n_train=n_train, n_test=config.EXP2_OOS_SIZE, n_noise=n_noise,
        beta=config.EXP2_BETA, sigma_eps=config.EXP2_SIGMA_EPS,
        strategy_corr=config.EXP2_STRATEGY_RETURN_CORR, noise_ar1_coef=config.EXP2_NOISE_AR1_COEF,
        noise_corr=config.EXP2_NOISE_CORR, seed=seed,
    )
    ret_vals = _ret_vals_from(dataset.returns)
    dates = pd.RangeIndex(ret_vals.shape[0])
    cost_bps = config.EXP2_SWITCHING_COST_BPS
    eta = config.EXP2_ETA
    # A refit frequency at or beyond the test horizon fits the Q-function once on the
    # n_train estimation sample and holds it fixed for every out-of-sample decision,
    # matching "training sample sizes ...; out-of-sample evaluation over 500 observations"
    # (Section S.6.1), rather than the expanding-window re-estimation of Experiment 1.
    refit_every_months = config.EXP2_OOS_SIZE + 1

    rl = run_estimated_policy(
        dataset.raw, ret_vals, dataset.binary_mask, dates, n_actions=config.EXP2_N_ACTIONS,
        initial_window=n_train, cost_bps=cost_bps, eta=eta, gamma=config.GAMMA,
        refit_every_months=refit_every_months, rho_grid=rho_grid,
    )
    greedy = run_estimated_policy(
        dataset.raw, ret_vals, dataset.binary_mask, dates, n_actions=config.EXP2_N_ACTIONS,
        initial_window=n_train, cost_bps=cost_bps, eta=eta, gamma=0.0,
        refit_every_months=refit_every_months, rho_grid=rho_grid,
    )
    oracle = simulate_oracle_policy(dataset.x, dataset.returns, cost_bps=cost_bps, eta=eta, start_index=n_train)
    fixed1 = run_fixed_policy(ret_vals, dates, action_idx=0, initial_window=n_train, eta=eta)

    out = {}
    for name, policy in (("RL", rl), ("Greedy", greedy), ("Oracle", oracle), ("Strategy1", fixed1)):
        out[f"{name}_mean_reward_pct"] = mean_reward_pct(policy.rewards)
        out[f"{name}_switching_freq"] = switching_frequency(policy.actions)
    return out


def run_experiment(n_replications: int, full: bool) -> pd.DataFrame:
    rho_min = config.GCV_RHO_OVER_N_MIN_SIMULATION
    rho_max = config.GCV_RHO_OVER_N_MAX_SIMULATION
    rho_grid = np.geomspace(rho_min, rho_max, config.GCV_GRID_SIZE)

    train_sizes = config.EXP2_TRAIN_SIZES_PAPER
    noise_counts = config.EXP2_NOISE_COUNTS
    rows = []
    K = np.array(config.EXP2_TRANSITION_MATRIX)
    cost = config.EXP2_SWITCHING_COST_BPS / 10000.0
    analytic_oracle = oracle_mean_reward(K, config.EXP2_BETA, config.EXP2_SIGMA_EPS, cost, config.EXP2_ETA) * 100.0
    logger.info("Analytic oracle long-run mean reward: %.4f%% (Section S.6.3, stationary-distribution benchmark)",
                analytic_oracle)

    for n_noise in noise_counts:
        for n_train in train_sizes:
            t0 = time.time()
            reps = []
            for rep in range(n_replications):
                seed = config.RANDOM_SEED + 1000 * n_noise + 10 * n_train + rep
                reps.append(run_one_replication(n_train, n_noise, seed, rho_grid))
            df = pd.DataFrame(reps)
            elapsed = time.time() - t0
            logger.info("noise=%d train=%d: %d replications in %.1fs (%.2fs/rep)",
                        n_noise, n_train, n_replications, elapsed, elapsed / n_replications)
            summary = {"n_noise": n_noise, "n_train": n_train, "n_replications": n_replications}
            for col in df.columns:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_mc_se"] = df[col].std(ddof=1) / np.sqrt(n_replications)
            summary["analytic_oracle_mean_reward_pct"] = analytic_oracle
            rows.append(summary)
    return pd.DataFrame(rows)


def make_plot(results: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_counts = sorted(results["n_noise"].unique())
    fig, axes = plt.subplots(1, len(noise_counts), figsize=(4 * len(noise_counts), 4), sharey=True)
    if len(noise_counts) == 1:
        axes = [axes]
    for ax, n_noise in zip(axes, noise_counts):
        sub = results[results["n_noise"] == n_noise].sort_values("n_train")
        for policy, marker in (("RL", "o"), ("Greedy", "s"), ("Oracle", "^"), ("Strategy1", "d")):
            ax.plot(sub["n_train"], sub[f"{policy}_mean_reward_pct_mean"], marker=marker, label=policy)
        ax.set_title(f"{n_noise} noise variables")
        ax.set_xlabel("Training sample size")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Out-of-sample mean reward (%)")
    axes[0].legend()
    fig.suptitle("Experiment 2: RL vs Greedy vs Oracle vs fixed Strategy 1 (Section S.6)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="run the paper's 500 Monte Carlo replications per configuration")
    args = parser.parse_args()

    n_replications = config.EXP2_N_REPLICATIONS_PAPER if args.full else config.EXP2_N_REPLICATIONS_RUN
    logger.info("Experiment 2: %d replications per configuration (paper: %d)",
                n_replications, config.EXP2_N_REPLICATIONS_PAPER)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiment(n_replications, args.full)
    results.to_csv(RESULTS_DIR / "exp2_monte_carlo_results.csv", index=False)
    make_plot(results, RESULTS_DIR / "exp2_monte_carlo_reward_vs_sample_size.png")

    log = {
        "n_replications_executed": n_replications,
        "n_replications_paper": config.EXP2_N_REPLICATIONS_PAPER,
        "train_sizes": list(config.EXP2_TRAIN_SIZES_PAPER),
        "noise_counts": list(config.EXP2_NOISE_COUNTS),
        "oos_size": config.EXP2_OOS_SIZE,
    }
    (RESULTS_DIR / "exp2_run_log.json").write_text(json.dumps(log, indent=2))
    logger.info("Experiment 2 complete: results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
