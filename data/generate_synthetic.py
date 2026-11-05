"""Synthetic data-generating process for the Monte Carlo simulation study
(Section S.6.1 of the Electronic Supplement).

A four-state Markov chain X_t drives strategy returns and the (known) oracle
policy; independent AR(1) "noise" variables are appended to the state to study
the effect of irrelevant information on the estimated policy. Everything here
is simulated -- no external data is used, matching the paper's simulation
design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_STATES = 4
N_ACTIONS = 4


def simulate_markov_chain(K: np.ndarray, n_total: int, rng: np.random.Generator,
                           x0: int | None = None) -> np.ndarray:
    """Simulates X_t in {0,1,2,3} for t = 0..n_total-1 from the transition matrix K
    (equation S.27). The chain is started from its stationary distribution (``x0=None``)
    so that the early part of the path is not atypical of the chain's long-run behaviour.
    """
    assert K.shape == (N_STATES, N_STATES)
    assert np.allclose(K.sum(axis=1), 1.0), "rows of the transition matrix must sum to one"
    if x0 is None:
        eigvals, eigvecs = np.linalg.eig(K.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = stat / stat.sum()
        x0 = rng.choice(N_STATES, p=stat)
    x = np.empty(n_total, dtype=int)
    x[0] = x0
    for t in range(1, n_total):
        x[t] = rng.choice(N_STATES, p=K[x[t - 1]])
    return x


def simulate_strategy_returns(x: np.ndarray, beta: float, sigma_eps: float, corr: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Ret_{t+1}^a = beta * 1{X_{t+1}=a} + eps_{t+1}^a (Section S.6.1), for t = 0..n_total-2.

    ``eps`` has a compound-symmetric covariance sigma_eps^2 * [(1-corr) I + corr * 11'] across
    the four strategies (Section S.6.1: "pairwise correlation of innovations ... = 0.3").
    Returns a (n_total - 1, 4) array, row k giving Ret_{k+1} (realised using x[k+1]).
    """
    n_total = len(x)
    cov = sigma_eps**2 * ((1 - corr) * np.eye(N_ACTIONS) + corr * np.ones((N_ACTIONS, N_ACTIONS)))
    eps = rng.multivariate_normal(np.zeros(N_ACTIONS), cov, size=n_total - 1)
    hit = (x[1:, None] == np.arange(N_ACTIONS)[None, :]).astype(float)  # 1{X_{t+1}=a}
    return beta * hit + eps


def simulate_noise_variables(n_total: int, n_noise: int, ar1_coef: float, corr: float,
                              rng: np.random.Generator) -> np.ndarray:
    """``n_noise`` independent-of-X AR(1) state variables (Section S.6.1): x_t = ar1_coef *
    x_{t-1} + u_t, u_t ~ N(0, 1 - ar1_coef^2) so that each series is marginally standardised,
    with pairwise correlation ``corr`` across the innovations. Returns (n_total, n_noise)."""
    if n_noise == 0:
        return np.zeros((n_total, 0))
    innov_var = 1.0 - ar1_coef**2
    cov = innov_var * ((1 - corr) * np.eye(n_noise) + corr * np.ones((n_noise, n_noise)))
    u = rng.multivariate_normal(np.zeros(n_noise), cov, size=n_total)
    out = np.empty((n_total, n_noise))
    out[0] = u[0] / np.sqrt(innov_var)  # start each series at unit variance, its stationary level
    for t in range(1, n_total):
        out[t] = ar1_coef * out[t - 1] + u[t]
    return out


def build_dummy_state(x: np.ndarray) -> np.ndarray:
    """Dummy encoding of X_t with category 0 omitted (Section S.6.1: "the first dummy omitted
    and an intercept included"): columns are 1{X_t=1}, 1{X_t=2}, 1{X_t=3}."""
    return (x[:, None] == np.array([1, 2, 3])[None, :]).astype(float)


@dataclass
class SimulatedDataset:
    x: np.ndarray  # (n_total,) Markov-chain state, 0-indexed
    returns: np.ndarray  # (n_total - 1, 4) strategy returns, row k = Ret_{k+1}
    raw: np.ndarray  # (n_total, 3 + n_noise) raw state: X dummies + noise variables
    binary_mask: np.ndarray  # (3 + n_noise,) boolean, True for the three X dummies
    n_train: int
    n_test: int


def make_dataset(K: np.ndarray, n_train: int, n_test: int, n_noise: int, beta: float, sigma_eps: float,
                  strategy_corr: float, noise_ar1_coef: float, noise_corr: float,
                  seed: int) -> SimulatedDataset:
    """One Monte Carlo replication's simulated path, long enough to fit the Q-function on the
    first ``n_train`` transitions and evaluate ``n_test`` further out-of-sample decisions
    without discarding or restarting the chain (Section S.6.1: "training sample sizes ...;
    out-of-sample evaluation over 500 observations")."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test + 1
    x = simulate_markov_chain(K, n_total, rng)
    returns = simulate_strategy_returns(x, beta, sigma_eps, strategy_corr, rng)
    noise = simulate_noise_variables(n_total, n_noise, noise_ar1_coef, noise_corr, rng)
    dummies = build_dummy_state(x)
    raw = np.concatenate([dummies, noise], axis=1)
    binary_mask = np.array([True, True, True] + [False] * n_noise)
    return SimulatedDataset(x=x, returns=returns, raw=raw, binary_mask=binary_mask, n_train=n_train, n_test=n_test)
