"""Unit tests for the analytic oracle policy (src/oracle.py, Section S.6.3)."""
from __future__ import annotations

import numpy as np

from src.config import EXP2_SIGMA_EPS, EXP2_STRATEGY_RETURN_CORR, EXP2_SWITCHING_COST_BPS, EXP2_TRANSITION_MATRIX
from src.oracle import (
    expected_reward,
    oracle_mean_reward,
    oracle_policy_action,
    oracle_policy_action_vector,
    verify_bellman_optimality,
)

K = np.array(EXP2_TRANSITION_MATRIX)
BETA = 0.01
COST = EXP2_SWITCHING_COST_BPS / 10000.0


def test_transition_matrix_rows_sum_to_one():
    assert np.allclose(K.sum(axis=1), 1.0)


def test_oracle_policy_action_matches_paper_rule():
    # Section S.6.1: Strategy 1 (index 0) when X_t in {1,3} (0-indexed {0,2}); Strategy 2
    # (index 1) when X_t in {2,4} (0-indexed {1,3}).
    assert oracle_policy_action(0) == 0
    assert oracle_policy_action(2) == 0
    assert oracle_policy_action(1) == 1
    assert oracle_policy_action(3) == 1


def test_oracle_policy_vector_independent_of_prev_action():
    vec = oracle_policy_action_vector()
    assert vec.shape == (16,)
    for x in range(4):
        same = {vec[4 * x + a_prev] for a_prev in range(4)}
        assert len(same) == 1


def test_bellman_optimality_holds_at_eta_zero():
    policy = oracle_policy_action_vector()
    check = verify_bellman_optimality(K, policy, BETA, EXP2_SIGMA_EPS, COST, eta=0.0, gamma=0.9)
    assert check.is_optimal, f"oracle policy violates Bellman optimality, margin={check.min_margin}"


def test_bellman_optimality_holds_at_eta_three():
    # The paper states the oracle policy is optimal for eta in [0, 164] (Section S.6.1).
    policy = oracle_policy_action_vector()
    check = verify_bellman_optimality(K, policy, BETA, EXP2_SIGMA_EPS, COST, eta=3.0, gamma=0.9)
    assert check.is_optimal, f"oracle policy violates Bellman optimality at eta=3, margin={check.min_margin}"


def test_expected_reward_decreases_with_switching_cost():
    r_no_switch = expected_reward(K, x=0, a_next=0, a_prev=0, beta=BETA, sigma_eps=EXP2_SIGMA_EPS,
                                   cost_val=COST, eta=0.0)
    r_switch = expected_reward(K, x=0, a_next=0, a_prev=1, beta=BETA, sigma_eps=EXP2_SIGMA_EPS,
                                cost_val=COST, eta=0.0)
    assert r_switch < r_no_switch


def test_oracle_mean_reward_is_finite_and_positive():
    reward = oracle_mean_reward(K, BETA, EXP2_SIGMA_EPS, COST, eta=0.0)
    assert np.isfinite(reward)
    assert reward > 0  # the oracle targets the high-mean strategy more often than chance
