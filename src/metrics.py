"""Performance metrics shared by both experiments (Section 4.1 "Performance measures").

Annualised Sharpe ratios of net returns, switching frequency, and the t-statistic
for the difference in mean rewards between two policies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def annualised_sharpe(monthly_net_returns: np.ndarray | pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio of a monthly net-return series: mean / std * sqrt(12)
    (Section 4.1: "annualised Sharpe ratios of net returns"). Net returns already include the
    switching cost deducted at the month of a strategy change.
    """
    x = np.asarray(monthly_net_returns, dtype=float)
    x = x[~np.isnan(x)]
    assert x.size > 1, "need at least two observations for a Sharpe ratio"
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(periods_per_year))


def switching_frequency(actions: np.ndarray | pd.Series) -> float:
    """Share of out-of-sample decisions at which the held strategy changes from the previous
    month (Section 4.1: "switching frequencies")."""
    a = np.asarray(actions)
    assert a.size > 1
    return float(np.mean(a[1:] != a[:-1]))


def mean_reward_pct(rewards: np.ndarray | pd.Series) -> float:
    """Monthly mean reward, in percent (Section 4.1: "monthly mean rewards (%)")."""
    r = np.asarray(rewards, dtype=float)
    r = r[~np.isnan(r)]
    assert r.size > 0
    return float(r.mean() * 100.0)


def reward_diff_tstat(rewards_a: np.ndarray, rewards_b: np.ndarray) -> tuple[float, float]:
    """t-statistic (and two-sided p-value) for the null hypothesis that the paired mean
    reward of policy A equals that of policy B (Section 4.1: "test statistical significance
    of RL vs Greedy reward differences using t-statistics"). A paired t-test on the reward
    difference series is the natural statistic the paper's wording describes; the paper does
    not specify a heteroskedasticity-and-autocorrelation (Newey-West) correction, so the
    plain paired t-test (``scipy.stats.ttest_rel``) is used here.
    """
    a = np.asarray(rewards_a, dtype=float)
    b = np.asarray(rewards_b, dtype=float)
    assert a.shape == b.shape
    mask = ~(np.isnan(a) | np.isnan(b))
    result = stats.ttest_rel(a[mask], b[mask])
    return float(result.statistic), float(result.pvalue)


def action_shares_by_group(actions: pd.Series, groups: pd.Series, action_labels: tuple[str, ...]) -> pd.DataFrame:
    """Share of months each action is held, broken down by a grouping variable (for example
    the market-cycle label), used for the state-dependent rotation analysis of Section 4.2."""
    df = pd.DataFrame({"action": actions.values, "group": groups.values})
    table = pd.crosstab(df["group"], df["action"], normalize="index")
    for label in action_labels:
        if label not in table.columns:
            table[label] = 0.0
    return table[list(action_labels)]
