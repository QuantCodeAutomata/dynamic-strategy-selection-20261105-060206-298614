"""Analytic oracle policy for the Monte Carlo simulation (Section S.6.3 of the
Electronic Supplement).

The simulated data-generating process is deliberately tractable: the raw state
is a four-state Markov chain :math:`X_t` with known transition matrix **K**
(equation S.27), and strategy returns depend only on :math:`X_{t+1}`. This
lets the optimal ("oracle") policy be derived analytically rather than
estimated, and used as a benchmark for the estimated RL and Greedy policies.

State encoding: the 16-state system :math:`S_t^a = (\\tilde S_t, a)` of
Section S.1 is encoded as ``s = 4 * x + a_prev`` with ``x, a_prev in {0,1,2,3}``
(0-indexed; ``x`` corresponds to the paper's :math:`X_t \\in \\{1,2,3,4\\}` and
``a_prev`` to the previously held strategy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_STATES = 4
N_ACTIONS = 4


def expected_reward(K: np.ndarray, x: int, a_next: int, a_prev: int, beta: float, sigma_eps: float,
                     cost_val: float, eta: float) -> float:
    """E[R_{t+1}^{(a_prev,a_next)} | X_t = x], equation of Section S.6.3.

    Derivation (paper Section S.6.3, Example 1 reward with Example 2 fixed cost):
    Ret_{t+1}^{a'} = beta * 1{X_{t+1}=a'} + eps, eps ~ (0, sigma_eps^2), independent of X.
    R = U(Ret - Cost) = (Ret - Cost) - (eta/2)(Ret - Cost)^2, Cost = cost_val * 1{a_prev != a_next}.
    Taking expectations given X_t = x (so P(X_{t+1}=a_next | X_t=x) = K[x, a_next]):

        E[R] = [beta - (eta/2)(beta^2 - 2*beta*cost)] * P - cost - (eta/2)(sigma_eps^2 + cost^2)

    with P = K[x, a_next] and cost = cost_val * 1{a_prev != a_next}.
    """
    p = K[x, a_next]
    cost = cost_val * float(a_prev != a_next)
    return (beta - 0.5 * eta * (beta**2 - 2.0 * beta * cost)) * p - cost - 0.5 * eta * (sigma_eps**2 + cost**2)


def _state_index(x: int, a_prev: int) -> int:
    return N_STATES * x + a_prev


def policy_transition_and_reward(K: np.ndarray, policy_action: np.ndarray, beta: float, sigma_eps: float,
                                  cost_val: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """K^pi (16x16) and r_eta^pi (16,) for a deterministic policy, Section S.6.3.

    ``policy_action`` is a length-16 array mapping each encoded state ``s`` to the
    action a_next in {0,1,2,3} chosen by the policy at that state.
    """
    n = N_STATES * N_ACTIONS
    K_pi = np.zeros((n, n))
    r_pi = np.zeros(n)
    for x in range(N_STATES):
        for a_prev in range(N_ACTIONS):
            s = _state_index(x, a_prev)
            a_next = int(policy_action[s])
            r_pi[s] = expected_reward(K, x, a_next, a_prev, beta, sigma_eps, cost_val, eta)
            for x_next in range(N_STATES):
                s_next = _state_index(x_next, a_next)
                K_pi[s, s_next] = K[x, x_next]
    return K_pi, r_pi


def policy_value_function(K: np.ndarray, policy_action: np.ndarray, beta: float, sigma_eps: float,
                           cost_val: float, eta: float, gamma: float) -> np.ndarray:
    """V_eta^pi = (I - gamma K^pi)^{-1} r_eta^pi, matrix form of equation (S.1)."""
    K_pi, r_pi = policy_transition_and_reward(K, policy_action, beta, sigma_eps, cost_val, eta)
    n = K_pi.shape[0]
    V = np.linalg.solve(np.eye(n) - gamma * K_pi, r_pi)
    return V


def oracle_policy_action(x: np.ndarray | int) -> np.ndarray | int:
    """Oracle policy (Section S.6.1): Strategy 1 (index 0) if X_t in {1,3} (0-indexed {0,2}),
    Strategy 2 (index 1) if X_t in {2,4} (0-indexed {1,3})."""
    x_arr = np.asarray(x)
    action = np.where((x_arr == 0) | (x_arr == 2), 0, 1)
    return action.item() if np.ndim(x) == 0 else action


def oracle_policy_action_vector() -> np.ndarray:
    """Length-16 policy_action vector, independent of a_prev (equation in Section S.6.3)."""
    policy_action = np.zeros(N_STATES * N_ACTIONS, dtype=int)
    for x in range(N_STATES):
        for a_prev in range(N_ACTIONS):
            policy_action[_state_index(x, a_prev)] = int(oracle_policy_action(x))
    return policy_action


@dataclass
class BellmanCheck:
    is_optimal: bool
    min_margin: float  # smallest V_eta^pi(s) - RHS(s, a') over all states and actions


def verify_bellman_optimality(K: np.ndarray, policy_action: np.ndarray, beta: float, sigma_eps: float,
                               cost_val: float, eta: float, gamma: float, tol: float = 1e-10) -> BellmanCheck:
    """Checks V_eta^pi(S_t^a) >= K^{a'} R_eta(S_t^a) + gamma K^{a'} V_eta^pi(S_t^a) for every
    state and every deviation a' (Section S.6.3, "Bellman optimality inequalities").
    """
    V = policy_value_function(K, policy_action, beta, sigma_eps, cost_val, eta, gamma)
    min_margin = np.inf
    for x in range(N_STATES):
        for a_prev in range(N_ACTIONS):
            s = _state_index(x, a_prev)
            for a_next in range(N_ACTIONS):
                reward = expected_reward(K, x, a_next, a_prev, beta, sigma_eps, cost_val, eta)
                continuation = sum(K[x, x_next] * V[_state_index(x_next, a_next)] for x_next in range(N_STATES))
                rhs = reward + gamma * continuation
                margin = V[s] - rhs
                min_margin = min(min_margin, margin)
    return BellmanCheck(is_optimal=min_margin >= -tol, min_margin=float(min_margin))


def oracle_mean_reward(K: np.ndarray, beta: float, sigma_eps: float, cost_val: float, eta: float,
                        stationary_dist: np.ndarray | None = None) -> float:
    """Long-run average one-period reward of the oracle policy under the chain's stationary
    distribution over (x, a_prev): used as an analytic cross-check for the simulated oracle
    out-of-sample mean reward."""
    policy_action = oracle_policy_action_vector()
    K_pi, r_pi = policy_transition_and_reward(K, policy_action, beta, sigma_eps, cost_val, eta)
    if stationary_dist is None:
        # stationary distribution of the 16-state chain K_pi via the dominant left eigenvector
        eigvals, eigvecs = np.linalg.eig(K_pi.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi_vec = np.real(eigvecs[:, idx])
        pi_vec = pi_vec / pi_vec.sum()
    else:
        pi_vec = stationary_dist
    return float(pi_vec @ r_pi)


def simulate_oracle_policy(x: np.ndarray, returns: np.ndarray, cost_bps: float, eta: float, start_index: int):
    """Simulates the oracle policy (Section S.6.1) along one realised path.

    At each decision date ``t`` the oracle observes ``X_t`` (not ``X_{t+1}``) and selects the
    state-dependent oracle action of :func:`oracle_policy_action`; the realised reward uses
    ``Ret_{t+1}`` and the fixed switching cost for the transition into the strategy held over
    month ``t+1``. ``returns`` follows the convention of ``data.generate_synthetic.make_dataset``:
    row ``k`` gives ``Ret_{k+1}``. Returns a ``src.policy_evaluation.PolicyResult``.
    """
    from src.policy_evaluation import PolicyResult
    from src.rl_core import utility

    cost = cost_bps / 10000.0
    n_total = len(x)
    held_action: int | None = None
    out_actions, out_net, out_reward = [], [], []
    for t in range(start_index, n_total - 1):
        new_action = int(oracle_policy_action(x[t]))
        switch_cost = cost if (held_action is not None and new_action != held_action) else 0.0
        ret_realized = returns[t, new_action]  # Ret_{t+1}
        net_return = ret_realized - switch_cost
        reward = float(utility(np.array([net_return]), eta)[0])
        out_actions.append(new_action)
        out_net.append(net_return)
        out_reward.append(reward)
        held_action = new_action
    dates = pd.RangeIndex(start_index + 1, n_total)
    return PolicyResult(dates=dates, actions=np.array(out_actions), net_returns=np.array(out_net),
                         rewards=np.array(out_reward), n_refits=0)
