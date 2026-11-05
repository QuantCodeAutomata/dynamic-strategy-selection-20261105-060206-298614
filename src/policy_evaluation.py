"""Expanding-window out-of-sample evaluation shared by the RL, Greedy and fixed-factor
policies of the empirical factor-rotation experiment (Section 4.1).

At each out-of-sample decision date the Q-function is estimated (or re-estimated, every
``refit_every_months`` months) on all data available up to that date -- an expanding
window, so no information from after the decision date ever enters the estimate -- and
the policy selects the strategy that maximises the estimated action-value function given
the currently held strategy. The realised reward and net return use the factor return of
the following month, which is unknown at decision time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.rl_core import FittedQIterationResult, fit_q_function, utility


@dataclass
class PolicyResult:
    dates: pd.DatetimeIndex  # dates at which the reward/net return is realised (decision date + 1 month)
    actions: np.ndarray  # held action index for the month ending at ``dates``
    net_returns: np.ndarray  # Ret_{t+1}^{a'} - Cost_t^{(a,a')}, decimal
    rewards: np.ndarray  # U(net_return), decimal
    n_refits: int  # number of times the Q-function was (re-)estimated

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"action": self.actions, "net_return": self.net_returns, "reward": self.rewards},
                             index=self.dates)


def run_estimated_policy(state_vals: np.ndarray, ret_vals: np.ndarray, binary_mask: np.ndarray,
                          dates: pd.DatetimeIndex, n_actions: int, initial_window: int, cost_bps: float,
                          eta: float, gamma: float, refit_every_months: int = 1,
                          rho_grid: np.ndarray | None = None) -> PolicyResult:
    """Expanding-window backtest of the estimated policy (RL if ``gamma`` > 0, Greedy if
    ``gamma`` = 0, Section 4.1 and Section 3 "Estimation Algorithm").

    ``state_vals`` is the (N, L) array of already scaled/clipped raw state variables
    (``src.state_variables.assemble_state``), ``ret_vals`` the (N, n_actions) array of
    factor returns aligned to the same ``dates`` index, and ``binary_mask`` the matching
    boolean mask. The first decision uses ``state_vals[initial_window]``; the realised
    reward uses ``ret_vals[initial_window + 1]``, and so on.
    """
    n_total = state_vals.shape[0]
    assert ret_vals.shape[0] == n_total and len(dates) == n_total
    assert initial_window < n_total - 1, "initial window leaves no out-of-sample months"
    max_iter = 1 if gamma == 0.0 else None  # Section 4.1: Greedy = one fitted Q-iteration (gamma=0)

    cost = cost_bps / 10000.0
    held_action: int | None = None
    out_dates, out_actions, out_net, out_reward = [], [], [], []
    fit: FittedQIterationResult | None = None
    n_refits = 0

    for i in range(initial_window, n_total - 1):
        if fit is None or (i - initial_window) % refit_every_months == 0:
            raw_t = state_vals[0:i]
            raw_tp1 = state_vals[1:i + 1]
            ret_tp1 = ret_vals[1:i + 1]
            kwargs = {}
            if max_iter is not None:
                kwargs["max_iter"] = max_iter
            fit = fit_q_function(raw_t, raw_tp1, ret_tp1, binary_mask, n_actions, cost_bps, eta, gamma,
                                  rho_grid=rho_grid, **kwargs)
            n_refits += 1

        raw_now = state_vals[i:i + 1]
        prev_for_q = np.array([held_action if held_action is not None else 0])
        q_vals = fit.predict_all(raw_now, prev_for_q)[0]
        new_action = int(np.argmax(q_vals))

        switch_cost = cost if (held_action is not None and new_action != held_action) else 0.0
        ret_realized = ret_vals[i + 1, new_action]
        net_return = ret_realized - switch_cost
        reward = float(utility(np.array([net_return]), eta)[0])

        out_dates.append(dates[i + 1])
        out_actions.append(new_action)
        out_net.append(net_return)
        out_reward.append(reward)
        held_action = new_action

    return PolicyResult(dates=pd.DatetimeIndex(out_dates), actions=np.array(out_actions),
                         net_returns=np.array(out_net), rewards=np.array(out_reward), n_refits=n_refits)


def run_fixed_policy(ret_vals: np.ndarray, dates: pd.DatetimeIndex, action_idx: int, initial_window: int,
                      eta: float) -> PolicyResult:
    """Always-hold-strategy-``action_idx`` benchmark: no switching, so no switching cost is ever
    incurred (Section 4.1: "fixed policies maintain the same strategy throughout")."""
    n_total = ret_vals.shape[0]
    out_dates = dates[initial_window + 1:n_total]
    ret_realized = ret_vals[initial_window + 1:n_total, action_idx]
    rewards = utility(ret_realized, eta)
    actions = np.full(len(out_dates), action_idx)
    return PolicyResult(dates=pd.DatetimeIndex(out_dates), actions=actions, net_returns=ret_realized,
                         rewards=rewards, n_refits=0)
