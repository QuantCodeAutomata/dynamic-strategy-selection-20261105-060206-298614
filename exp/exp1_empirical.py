"""Experiment 1: empirical factor rotation in the U.S. equity market (Section 4 of
"Dynamic Strategy Selection").

Monthly rotation among four U.S. equity factor strategies -- momentum (MOM), value
(HML), profitability (RMW) and a defensive low-volatility factor (DEF) -- using the
fitted Q-iteration RL policy (Algorithm S.1), the myopic Greedy policy (gamma=0), and
the four fixed-factor benchmarks. All policies are evaluated out of sample on an
expanding window across three state information sets (Trend, Trend+Cycle,
Trend+Cycle+Macro), two risk-aversion levels (eta in {0, 3}) and three switching costs
(20, 80, 120 bps).

Run: ``python -m exp.exp1_empirical`` (reduced re-estimation frequency) or
``python -m exp.exp1_empirical --full`` (the paper's monthly re-estimation).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import build_def_factor, default_def_universe, load_french_factors, load_fred_series
from src import config
from src.metrics import (
    action_shares_by_group,
    annualised_sharpe,
    mean_reward_pct,
    reward_diff_tstat,
    switching_frequency,
)
from src.policy_evaluation import run_estimated_policy, run_fixed_policy
from src.state_variables import assemble_state, classify_cycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ACTIONS = config.EXP1_ACTIONS
N_ACTIONS = len(ACTIONS)


def build_factor_returns() -> pd.DataFrame:
    """MOM, HML, RMW and MKT_RF from the Kenneth R. French Data Library, plus the DEF
    low-volatility proxy, aligned on a monthly PeriodIndex and trimmed to the sample
    period of Section 4.1 (July 1963 to December 2025)."""
    french = load_french_factors()
    def_tickers = default_def_universe()
    start = pd.Period(config.EXP1_SAMPLE_START, freq="M").strftime("%Y-%m-%d")
    end = pd.Period(config.EXP1_SAMPLE_END, freq="M").strftime("%Y-%m-%d")
    def_factor = build_def_factor(def_tickers, start, end)

    merged = french.join(def_factor, how="outer")
    sample_start = pd.Period(config.EXP1_SAMPLE_START, freq="M")
    sample_end = pd.Period(config.EXP1_SAMPLE_END, freq="M")
    merged = merged.loc[(merged.index >= sample_start) & (merged.index <= sample_end)]
    logger.info("Factor returns: %d months, %s to %s, columns=%s",
                len(merged), merged.index.min(), merged.index.max(), list(merged.columns))
    return merged


def common_valid_window(factor_returns: pd.DataFrame, fred: pd.DataFrame) -> pd.DatetimeIndex:
    """The first date at which every one of the three state specifications (in particular
    the richest, Trend+Cycle+Macro) and all four factor returns are simultaneously defined,
    so that the RL/Greedy policies are compared across specifications on an identical set
    of out-of-sample decisions (Section 4.1 reports a single decision count for all specs).
    """
    state, _ = assemble_state(factor_returns, ACTIONS, factor_returns["MKT_RF"], fred, "Trend+Cycle+Macro")
    valid = state.notna().all(axis=1) & factor_returns[list(ACTIONS)].notna().all(axis=1)
    valid_dates = state.index[valid]
    return valid_dates


def to_timestamp_index(period_index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return period_index.to_timestamp(how="end").normalize()


def run_grid(factor_returns: pd.DataFrame, fred: pd.DataFrame, refit_every_months: int,
             combos: tuple[tuple[str, int, float], ...], etas_for_fixed: tuple[float, ...]) -> dict:
    """Runs the RL and Greedy policies for every (spec, cost, eta) combination in ``combos``,
    and the four fixed-factor policies for every eta in ``etas_for_fixed``, on the common
    out-of-sample window (Section 4.1). Returns a dict with the headline comparison table,
    the rotation-analysis inputs and the raw policy results keyed by configuration, for later
    metric computation and plotting.
    """
    valid_dates = common_valid_window(factor_returns, fred)
    ret_vals_full = factor_returns.loc[valid_dates, list(ACTIONS)].values
    n_total = len(valid_dates)
    initial_window = config.EXP1_INITIAL_WINDOW_MONTHS
    assert initial_window < n_total - 1, "sample too short for the paper's 120-month initial window"
    dates_ts = to_timestamp_index(valid_dates)

    oos_start = dates_ts[initial_window + 1]
    n_decisions = n_total - 1 - initial_window
    logger.info("Common window: %d months (%s to %s); initial window %d months; "
                "out-of-sample decisions: %d, starting %s (paper: %d decisions from %s)",
                n_total, dates_ts[0].date(), dates_ts[-1].date(), initial_window, n_decisions,
                oos_start.date(), config.EXP1_OOS_N_DECISIONS_PAPER, config.EXP1_OOS_START)

    rho_grid = np.geomspace(config.GCV_RHO_OVER_N_MIN_EMPIRICAL, config.GCV_RHO_OVER_N_MAX_EMPIRICAL,
                             config.GCV_GRID_SIZE)

    cycle_labels = classify_cycle(factor_returns["MKT_RF"]).loc[valid_dates, "cycle_label"]

    policy_results: dict[str, object] = {}
    rows = []
    state_cache: dict[str, tuple] = {}
    for spec, cost_bps, eta in combos:
        if spec not in state_cache:
            state, binary_mask = assemble_state(factor_returns, ACTIONS, factor_returns["MKT_RF"], fred, spec)
            state_cache[spec] = (state.loc[valid_dates].values, binary_mask)
        state_vals, binary_mask = state_cache[spec]
        t0 = time.time()
        rl = run_estimated_policy(state_vals, ret_vals_full, binary_mask, dates_ts, N_ACTIONS,
                                   initial_window, cost_bps, eta, config.GAMMA,
                                   refit_every_months=refit_every_months, rho_grid=rho_grid)
        greedy = run_estimated_policy(state_vals, ret_vals_full, binary_mask, dates_ts, N_ACTIONS,
                                       initial_window, cost_bps, eta, 0.0,
                                       refit_every_months=refit_every_months, rho_grid=rho_grid)
        elapsed = time.time() - t0
        tstat, pval = reward_diff_tstat(rl.rewards, greedy.rewards)
        for name, policy in (("RL", rl), ("Greedy", greedy)):
            rows.append({
                "spec": spec, "cost_bps": cost_bps, "eta": eta, "policy": name,
                "mean_reward_pct": mean_reward_pct(policy.rewards),
                "annualised_sharpe": annualised_sharpe(policy.net_returns),
                "switching_frequency": switching_frequency(policy.actions),
                "n_decisions": len(policy.actions),
                "rl_vs_greedy_tstat": tstat if name == "RL" else np.nan,
                "rl_vs_greedy_pvalue": pval if name == "RL" else np.nan,
            })
        policy_results[f"{spec}|{cost_bps}|{eta}|RL"] = rl
        policy_results[f"{spec}|{cost_bps}|{eta}|Greedy"] = greedy
        logger.info("spec=%s cost=%dbps eta=%.0f: RL reward=%.4f%% Sharpe=%.3f switch=%.3f | "
                    "Greedy reward=%.4f%% Sharpe=%.3f switch=%.3f (%.1fs)",
                    spec, cost_bps, eta, mean_reward_pct(rl.rewards), annualised_sharpe(rl.net_returns),
                    switching_frequency(rl.actions), mean_reward_pct(greedy.rewards),
                    annualised_sharpe(greedy.net_returns), switching_frequency(greedy.actions), elapsed)

    for eta in etas_for_fixed:
        for idx, action in enumerate(ACTIONS):
            fixed = run_fixed_policy(ret_vals_full, dates_ts, idx, initial_window, eta)
            rows.append({
                "spec": "Fixed", "cost_bps": np.nan, "eta": eta, "policy": f"Fixed-{action}",
                "mean_reward_pct": mean_reward_pct(fixed.rewards),
                "annualised_sharpe": annualised_sharpe(fixed.net_returns),
                "switching_frequency": switching_frequency(fixed.actions),
                "n_decisions": len(fixed.actions), "rl_vs_greedy_tstat": np.nan, "rl_vs_greedy_pvalue": np.nan,
            })
            policy_results[f"Fixed|{action}|{eta}"] = fixed

    results_table = pd.DataFrame(rows)
    return {
        "results_table": results_table,
        "policy_results": policy_results,
        "cycle_labels": cycle_labels,
        "oos_start": oos_start,
        "n_decisions": n_decisions,
        "dates": dates_ts,
        "ret_vals": ret_vals_full,
    }


def rotation_analysis(policy_results: dict, cycle_labels: pd.Series, headline_key: str) -> pd.DataFrame:
    """Share of months each factor is held, broken down by market-cycle state (Section 4.2),
    for the headline RL policy configuration."""
    rl = policy_results[headline_key]
    cycle_labels = cycle_labels.copy()
    cycle_labels.index = to_timestamp_index(cycle_labels.index)
    groups = cycle_labels.reindex(rl.dates)
    actions = pd.Series([ACTIONS[a] for a in rl.actions], index=rl.dates)
    return action_shares_by_group(actions, groups, ACTIONS)


def make_plot(policy_results: dict, headline_rl_key: str, headline_greedy_key: str,
              fixed_mom_key: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rl = policy_results[headline_rl_key]
    greedy = policy_results[headline_greedy_key]
    fixed_mom = policy_results[fixed_mom_key]

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, policy in (("RL", rl), ("Greedy", greedy), ("Fixed MOM", fixed_mom)):
        cum = (1.0 + pd.Series(policy.net_returns, index=policy.dates)).cumprod()
        ax.plot(cum.index, cum.values, label=label)
    ax.set_yscale("log")
    ax.set_title("Experiment 1: cumulative net return, Trend+Cycle state, eta=3, cost=120bps (Section 4.2)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative growth of $1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="run the paper's monthly Q-function re-estimation (slow)")
    parser.add_argument("--combo-index", type=int, default=None,
                         help="internal: compute only this one combo (for running the independent "
                              "(spec, cost, eta) combinations of EXP1_RUN_COMBOS as separate parallel "
                              "processes); writes a partial result instead of the merged tables")
    parser.add_argument("--partial-dir", type=str, default=None,
                         help="internal: directory for partial per-combo outputs (used with --combo-index)")
    parser.add_argument("--merge-partial-dir", type=str, default=None,
                         help="internal: assemble the final tables from partial outputs already written "
                              "to this directory by separate --combo-index runs, instead of recomputing")
    args = parser.parse_args()

    refit_every_months = config.EXP1_REFIT_EVERY_MONTHS_PAPER
    if args.full:
        combos = tuple((spec, cost, eta) for spec in config.EXP1_STATE_SPECS
                        for cost in config.SWITCHING_COSTS_BPS for eta in config.RISK_AVERSION_LEVELS)
    else:
        combos = config.EXP1_RUN_COMBOS
    etas_for_fixed = config.RISK_AVERSION_LEVELS

    factor_returns = build_factor_returns()
    fred = load_fred_series(config.EXP1_SAMPLE_START, config.EXP1_SAMPLE_END)

    if args.combo_index is not None:
        # One combination of EXP1_RUN_COMBOS, computed in its own OS process so that the
        # independent (spec, cost, eta) points of the default grid can run concurrently on
        # separate cores; see the parallel driver invoked by `reproduce.sh`/README for exp1.
        one_combo = (combos[args.combo_index],)
        out = run_grid(factor_returns, fred, refit_every_months, one_combo, etas_for_fixed=())
        partial_dir = Path(args.partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        out["results_table"].to_csv(partial_dir / f"rows_{args.combo_index}.csv", index=False)
        import pickle
        with open(partial_dir / f"policy_{args.combo_index}.pkl", "wb") as fh:
            pickle.dump({"policy_results": out["policy_results"], "cycle_labels": out["cycle_labels"],
                         "oos_start": out["oos_start"], "n_decisions": out["n_decisions"],
                         "dates": out["dates"]}, fh)
        logger.info("combo %d (%s) done, partial results in %s", args.combo_index, one_combo[0], partial_dir)
        return

    logger.info("Experiment 1: re-estimation every %d month(s); %d (spec, cost, eta) combinations "
                "(paper's full grid: %d)", refit_every_months, len(combos),
                len(config.EXP1_STATE_SPECS) * len(config.SWITCHING_COSTS_BPS) * len(config.RISK_AVERSION_LEVELS))

    if args.merge_partial_dir is not None:
        import pickle
        partial_dir = Path(args.merge_partial_dir)
        row_frames, policy_results = [], {}
        cycle_labels = oos_start = n_decisions = dates = None
        for idx in range(len(combos)):
            row_frames.append(pd.read_csv(partial_dir / f"rows_{idx}.csv"))
            with open(partial_dir / f"policy_{idx}.pkl", "rb") as fh:
                partial = pickle.load(fh)
            policy_results.update(partial["policy_results"])
            cycle_labels, oos_start = partial["cycle_labels"], partial["oos_start"]
            n_decisions, dates = partial["n_decisions"], partial["dates"]
        valid_dates = common_valid_window(factor_returns, fred)
        ret_vals_full = factor_returns.loc[valid_dates, list(ACTIONS)].values
        for eta in etas_for_fixed:
            for action_idx, action in enumerate(ACTIONS):
                fixed = run_fixed_policy(ret_vals_full, dates, action_idx, config.EXP1_INITIAL_WINDOW_MONTHS, eta)
                row_frames.append(pd.DataFrame([{
                    "spec": "Fixed", "cost_bps": np.nan, "eta": eta, "policy": f"Fixed-{action}",
                    "mean_reward_pct": mean_reward_pct(fixed.rewards),
                    "annualised_sharpe": annualised_sharpe(fixed.net_returns),
                    "switching_frequency": switching_frequency(fixed.actions),
                    "n_decisions": len(fixed.actions), "rl_vs_greedy_tstat": np.nan, "rl_vs_greedy_pvalue": np.nan,
                }]))
                policy_results[f"Fixed|{action}|{eta}"] = fixed
        out = {"results_table": pd.concat(row_frames, ignore_index=True), "policy_results": policy_results,
               "cycle_labels": cycle_labels, "oos_start": oos_start, "n_decisions": n_decisions, "dates": dates,
               "ret_vals": ret_vals_full}
    else:
        out = run_grid(factor_returns, fred, refit_every_months, combos, etas_for_fixed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out["results_table"].to_csv(RESULTS_DIR / "exp1_performance_table.csv", index=False)

    headline_rl_key = "Trend+Cycle|120|3.0|RL"
    headline_greedy_key = "Trend+Cycle|120|3.0|Greedy"
    fixed_mom_key = "Fixed|MOM|3.0"
    rotation = rotation_analysis(out["policy_results"], out["cycle_labels"], headline_rl_key)
    rotation.to_csv(RESULTS_DIR / "exp1_rotation_by_cycle.csv")
    make_plot(out["policy_results"], headline_rl_key, headline_greedy_key, fixed_mom_key,
              RESULTS_DIR / "exp1_cumulative_returns.png")

    log = {
        "refit_every_months_executed": refit_every_months,
        "refit_every_months_paper": config.EXP1_REFIT_EVERY_MONTHS_PAPER,
        "n_decisions_executed": int(out["n_decisions"]),
        "n_decisions_paper": config.EXP1_OOS_N_DECISIONS_PAPER,
        "oos_start_executed": str(out["oos_start"].date()),
        "oos_start_paper": config.EXP1_OOS_START,
        "sample_start": str(out["dates"][0].date()),
        "sample_end": str(out["dates"][-1].date()),
        "def_universe_size": config.EXP1_UNIVERSE_SIZE_RUN,
    }
    (RESULTS_DIR / "exp1_run_log.json").write_text(json.dumps(log, indent=2))
    logger.info("Experiment 1 complete: results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
