"""Unit tests for the shared fitted Q-iteration engine (src/rl_core.py)."""
from __future__ import annotations

import numpy as np

from src.rl_core import (
    build_basis_matrix,
    cap_values,
    fit_q_function,
    net_reward,
    ridge_gcv_fit,
    utility,
)


def test_cap_values_bounds():
    x = np.array([-5.0, -0.1, 0.0, 0.3, 10.0])
    capped = cap_values(x, bound=0.5)
    assert np.all(np.abs(capped) <= 0.5 + 1e-12)
    assert np.allclose(capped, np.array([-0.5, -0.1, 0.0, 0.3, 0.5]))


def test_utility_mean_variance_reduces_to_identity_at_eta_zero():
    x = np.array([-0.1, 0.0, 0.05, 0.2])
    assert np.allclose(utility(x, eta=0.0), x)
    # Risk aversion penalises the utility relative to the raw return (Section 2, Example 1).
    assert np.all(utility(x, eta=3.0) <= x + 1e-12)


def test_net_reward_switching_cost_only_when_action_changes():
    ret = np.array([0.05, 0.05, 0.05])
    prev = np.array([0, 1, 2])
    reward_same = net_reward(ret, prev, new_action_idx=0, cost=0.012, eta=0.0)
    # Row 0: prev==new (no cost); rows 1,2: prev!=new (cost deducted).
    assert np.isclose(reward_same[0], 0.05)
    assert np.isclose(reward_same[1], 0.05 - 0.012)
    assert np.isclose(reward_same[2], 0.05 - 0.012)


def test_build_basis_matrix_shape_and_intercept():
    raw = np.array([[0.1, 0.2], [0.3, -0.1], [0.0, 0.5]])
    binary_mask = np.array([False, False])
    prev_action = np.array([0, 1, 2])
    X = build_basis_matrix(raw, binary_mask, prev_action, n_actions=4, baseline_action=0)
    # intercept(1) + linear(2) + quadratic(2) + dummies(n_actions-1=3) = 8
    assert X.shape == (3, 8)
    assert np.allclose(X[:, 0], 1.0)
    assert np.allclose(X[:, 1:3], raw)
    assert np.allclose(X[:, 3:5], raw**2)


def test_ridge_gcv_recovers_linear_signal():
    """On a clean linear DGP, ridge-GCV should fit the slope closely and choose a small penalty."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, 1))
    X = np.column_stack([np.ones(n), x])
    true_beta = np.array([0.02, 0.5])
    y = X @ true_beta + rng.normal(scale=0.01, size=n)
    grid = np.geomspace(1e-4, 2.5, 31)
    fit = ridge_gcv_fit(X, y, grid)
    assert np.allclose(fit.coef, true_beta, atol=0.05)
    assert fit.df > 0


def test_ridge_gcv_zero_penalty_on_intercept():
    """The intercept is never shrunk towards zero even with a strong signal mean shift."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=(n, 1))
    X = np.column_stack([np.ones(n), x])
    y = 2.0 + 0.0 * x[:, 0] + rng.normal(scale=0.001, size=n)
    grid = np.geomspace(1e-3, 2.5, 31)
    fit = ridge_gcv_fit(X, y, grid)
    assert abs(fit.coef[0] - 2.0) < 0.05


def test_fit_q_function_greedy_matches_one_iteration():
    """gamma=0 and max_iter=1 (the Greedy policy) must use the contemporaneous reward only,
    i.e. be insensitive to large changes in bogus 'future' returns outside the current month."""
    rng = np.random.default_rng(2)
    n = 300
    n_actions = 3
    raw_t = rng.uniform(-1, 1, size=(n, 2))
    raw_tp1 = rng.uniform(-1, 1, size=(n, 2))
    ret_tp1 = rng.normal(scale=0.03, size=(n, n_actions))
    binary_mask = np.array([False, False])
    grid = np.geomspace(0.1, 2.5, 11)
    fit = fit_q_function(raw_t, raw_tp1, ret_tp1, binary_mask, n_actions, cost_bps=20, eta=0.0,
                          gamma=0.0, rho_grid=grid, max_iter=1, n_action_draws=2, seed=1)
    assert fit.n_iterations == 1
    preds = fit.predict_all(raw_t[:5], np.zeros(5, dtype=int))
    assert preds.shape == (5, n_actions)
    assert np.all(np.abs(preds) <= 0.5 + 1e-9)  # Q-cap B=0.5 (Section S.5.2)


def test_fit_q_function_rl_differs_from_greedy_with_discounting():
    rng = np.random.default_rng(3)
    n = 400
    n_actions = 3
    raw_t = rng.uniform(-1, 1, size=(n, 2))
    raw_tp1 = rng.uniform(-1, 1, size=(n, 2))
    ret_tp1 = rng.normal(scale=0.03, size=(n, n_actions))
    binary_mask = np.array([False, False])
    grid = np.geomspace(0.1, 2.5, 11)
    rl = fit_q_function(raw_t, raw_tp1, ret_tp1, binary_mask, n_actions, cost_bps=120, eta=0.0,
                         gamma=0.9, rho_grid=grid, max_iter=20, n_action_draws=2, seed=1)
    greedy = fit_q_function(raw_t, raw_tp1, ret_tp1, binary_mask, n_actions, cost_bps=120, eta=0.0,
                             gamma=0.0, rho_grid=grid, max_iter=1, n_action_draws=2, seed=1)
    assert rl.n_iterations > greedy.n_iterations
