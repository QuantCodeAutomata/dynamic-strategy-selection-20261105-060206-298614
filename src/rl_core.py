"""Shared fitted Q-iteration engine for dynamic strategy selection.

Implements Algorithm S.1 of the Electronic Supplement ("Q-Function Estimation
for Model Selection"), the additive action-value model of equation (A.2)/(A.3)
with quadratic basis functions (Section S.5.2), ridge regression with the
penalty selected by generalised cross-validation -- GCV, Golub, Heath and Wahba
(1979) -- and the averaging across N_A simulated action sequences of equation
(S.12). This module is shared, unchanged, by both experiments.

Custom implementation: scikit-learn's ridge estimators (Ridge / RidgeCV) apply
one global penalty to every coefficient, including, implicitly, the intercept
via centering, and expose no per-grid generalised-cross-validation score with
a zero-penalty intercept constraint compatible with the paper's additive
model; the estimator below implements the paper's ridge-GCV procedure
directly (Section S.5.2) using a singular value decomposition so that the
whole penalty grid is evaluated without a Python-level loop, which is
required for the repeated monthly re-estimation of Section 4.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import FQI_CONVERGENCE_TOL, FQI_MAX_ITER, GAMMA, N_SIMULATED_ACTION_SEQUENCES, Q_CAP_B, RANDOM_SEED


def cap_values(x: np.ndarray, bound: float = Q_CAP_B) -> np.ndarray:
    """Capping operator [x]_B = sign(x) * min(|x|, B) (Section S.2, equation before (S.9))."""
    return np.sign(x) * np.minimum(np.abs(x), bound)


def utility(x: np.ndarray, eta: float) -> np.ndarray:
    """Mean-variance utility U(x) = x - (eta/2) x^2 (Section 2, Example 1)."""
    return x - 0.5 * eta * x**2


def net_reward(ret_next: np.ndarray, prev_action_idx: np.ndarray, new_action_idx: int,
               cost: float, eta: float) -> np.ndarray:
    """Reward R_{t+1}^{(a,a')} = U(Ret_{t+1}^{(a')} - Cost_t^{(a,a')}) (Section 4.1, "Reward and estimation").

    ``prev_action_idx`` is the previously held strategy a (integer-coded, one value per row);
    ``new_action_idx`` is the candidate new strategy a'; ``cost`` is the fixed switching cost
    (decimal, e.g. 0.0020 for 20 bps).
    """
    switch_cost = cost * (prev_action_idx != new_action_idx).astype(float)
    return utility(ret_next - switch_cost, eta)


def build_basis_matrix(raw_features: np.ndarray, binary_mask: np.ndarray,
                        prev_action_idx: np.ndarray, n_actions: int, baseline_action: int) -> np.ndarray:
    """Additive quadratic basis phi(S_t) for the Q-function of action ``baseline_action``.

    Equation (A.2)/(A.3) and Section S.5.2: f_a((s_tilde, a')) = sum_l s_tilde_l b_l^a +
    sum_j b_{L+j}^a 1{a' != a_j}, implemented as one-hot dummies of the previous action with the
    category a' = baseline_action dropped (used as the reference level, avoiding multicollinearity
    with the intercept). Non-binary raw state variables enter as a linear term and its square;
    binary (Cycle dummy) variables enter linearly only. An intercept column is prepended.

    Parameters
    ----------
    raw_features : (n, L) array of already scaled/clipped raw state variables.
    binary_mask : (L,) boolean array, True where the column is a 0/1 dummy (no quadratic term).
    prev_action_idx : (n,) integer array with the previous action held at each row.
    n_actions : number of candidate actions |A|.
    baseline_action : the action index a whose Q-function this basis serves.

    Returns
    -------
    (n, K) design matrix, K = 1 (intercept) + L + sum(~binary_mask) (quadratic terms) + (n_actions - 1).
    """
    n, L = raw_features.shape
    cols = [np.ones(n)]
    cols.append(raw_features)  # linear terms, all L raw variables
    if (~binary_mask).any():
        quad = raw_features[:, ~binary_mask] ** 2
        cols.append(quad)
    other_actions = [a for a in range(n_actions) if a != baseline_action]
    if other_actions:
        dummies = np.column_stack([(prev_action_idx == a).astype(float) for a in other_actions])
        cols.append(dummies)
    return np.column_stack(cols)


def _penalty_mask(L: int, n_quad: int, n_dummies: int) -> np.ndarray:
    """1 for penalised (non-intercept) columns, 0 for the intercept (Section S.5.2: 'penalty for the
    intercept is set to 0')."""
    n_cols = 1 + L + n_quad + n_dummies
    mask = np.ones(n_cols)
    mask[0] = 0.0
    return mask


@dataclass
class RidgeGCVResult:
    coef: np.ndarray
    chosen_rho_over_n: float
    gcv_score: float
    df: float


def ridge_gcv_fit(X: np.ndarray, y: np.ndarray, rho_over_n_grid: np.ndarray) -> RidgeGCVResult:
    """Ridge regression with the penalty rho/n chosen by generalised cross-validation.

    Standardises every non-intercept column to a z-score computed on the supplied sample
    (Section S.5.2: "we first standardise the variables into z-scores using only observations
    available at the time of the decision"), fits ridge with zero penalty on the intercept for
    every candidate rho/n using a single singular value decomposition (so the whole grid is
    evaluated without a per-candidate linear solve), and transforms the coefficients back to the
    original scale. GCV score = n * RSS / (n - df)^2 with df = trace of the ridge hat matrix,
    following Golub, Heath and Wahba (1979), as referenced in Section 4.1.
    """
    n, p = X.shape
    assert n == y.shape[0]
    intercept_col = X[:, 0]
    assert np.allclose(intercept_col, 1.0), "first column of X must be the intercept"
    raw = X[:, 1:]
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    Xz = (raw - mean) / std_safe
    ybar = y.mean()
    yc = y - ybar

    # Economy SVD of the standardised, penalised block: Xz = U diag(s) V'
    U, s, _Vt = np.linalg.svd(Xz, full_matrices=False)
    d = U.T @ yc  # rotated target, length = rank(Xz)
    s2 = s**2

    lam_grid = rho_over_n_grid * n  # lambda = (rho/n) * n = rho
    # shrinkage[i, k] = s_i^2 / (s_i^2 + lambda_k): fraction of variance explained along
    # singular direction i, used for fitted values (U @ shrinkage * d) and for the GCV
    # degrees of freedom (Golub, Heath & Wahba, 1979; Section S.5.2).
    shrinkage = s2[:, None] / (s2[:, None] + lam_grid[None, :])
    yc_hat_all = U @ (shrinkage * d[:, None])  # (n, n_grid)
    rss_all = np.sum((yc[:, None] - yc_hat_all) ** 2, axis=0)
    df_all = shrinkage.sum(axis=0) + 1.0  # +1 for the unpenalised intercept
    denom = np.maximum(n - df_all, 1e-6)
    gcv_all = n * rss_all / denom**2

    best_k = int(np.argmin(gcv_all))
    lam = lam_grid[best_k]
    # Rotated ridge coefficient alpha_hat = s / (s^2 + lambda) * d (one power of s fewer
    # than the shrinkage factor above, since beta = V @ alpha_hat while the fitted values
    # use U @ diag(s) @ alpha_hat = U @ shrinkage @ d).
    alpha_hat = (s / (s2 + lam)) * d
    beta_std = _Vt.T @ alpha_hat

    beta_orig = beta_std / std_safe
    intercept_orig = ybar - np.sum(beta_std * mean / std_safe)
    coef = np.concatenate([[intercept_orig], beta_orig])
    return RidgeGCVResult(coef=coef, chosen_rho_over_n=float(rho_over_n_grid[best_k]),
                           gcv_score=float(gcv_all[best_k]), df=float(df_all[best_k]))


@dataclass
class FittedQIterationResult:
    """Averaged (equation S.12), capped Q-function coefficients for every action."""

    coefs: dict  # action_index -> (n_actions,) list giving one coefficient vector per draw u
    n_iterations: int
    n_actions: int
    binary_mask: np.ndarray

    def predict_q(self, raw_features: np.ndarray, prev_action_idx: np.ndarray, action: int) -> np.ndarray:
        """Average, over the N_A simulated action sequences, of the capped Q(.,action) prediction."""
        preds = []
        for coef in self.coefs[action]:
            X = build_basis_matrix(raw_features, self.binary_mask, prev_action_idx, self.n_actions, action)
            preds.append(cap_values(X @ coef))
        return np.mean(preds, axis=0)

    def predict_all(self, raw_features: np.ndarray, prev_action_idx: np.ndarray) -> np.ndarray:
        """(n, n_actions) matrix of averaged capped Q-values for every action."""
        return np.column_stack([self.predict_q(raw_features, prev_action_idx, a) for a in range(self.n_actions)])

    def greedy_action(self, raw_features: np.ndarray, prev_action_idx: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_all(raw_features, prev_action_idx), axis=1)


def _fit_one_draw(raw_t: np.ndarray, raw_tp1: np.ndarray, ret_tp1: np.ndarray, binary_mask: np.ndarray,
                   n_actions: int, cost: float, eta: float, gamma: float, rho_grid: np.ndarray,
                   max_iter: int, tol: float, rng: np.random.Generator) -> tuple[dict, int]:
    """One realisation u of Algorithm S.1: i.i.d. uniform previous actions augment the sample,
    then the Q-function is iterated to (approximate) convergence."""
    n = raw_t.shape[0]
    prev_action_idx = rng.integers(0, n_actions, size=n)  # A_{t-1}, i.i.d. uniform (Section S.2.1)

    v_prev = np.zeros((n, n_actions))  # V_hat^{(0)} = 0, one column per candidate next-period previous action
    coefs_by_action: dict = {a: None for a in range(n_actions)}
    n_iter_used = max_iter
    for j in range(1, max_iter + 1):
        for a in range(n_actions):
            # Regress R_{t+1}^{(prev, a)} + gamma * V_hat^{(j-1)}(S_{t+1}^a) on phi(S_t) (Section S.2.4)
            y = net_reward(ret_tp1[:, a], prev_action_idx, a, cost, eta) + gamma * v_prev[:, a]
            X = build_basis_matrix(raw_t, binary_mask, prev_action_idx, n_actions, a)
            fit = ridge_gcv_fit(X, y, rho_grid)
            coefs_by_action[a] = fit.coef

        # V_hat^{(j)}(raw_tp1, c) = max_{a'} Q_hat^{(j)}(raw_tp1, c; a') for every candidate previous
        # action c at t+1, needed as the continuation value for the next iteration's regression target.
        v_next = np.zeros((n, n_actions))
        for c in range(n_actions):
            qs = np.column_stack([
                cap_values(build_basis_matrix(raw_tp1, binary_mask, np.full(n, c), n_actions, ap)
                           @ coefs_by_action[ap])
                for ap in range(n_actions)
            ])
            v_next[:, c] = np.max(qs, axis=1)
        max_delta = float(np.max(np.abs(v_next - v_prev)))
        v_prev = v_next
        n_iter_used = j
        if j > 1 and max_delta < tol:
            break
    return coefs_by_action, n_iter_used


def fit_q_function(raw_t: np.ndarray, raw_tp1: np.ndarray, ret_tp1: np.ndarray, binary_mask: np.ndarray,
                    n_actions: int, cost_bps: float, eta: float, gamma: float = GAMMA,
                    rho_grid: np.ndarray | None = None, max_iter: int = FQI_MAX_ITER,
                    tol: float = FQI_CONVERGENCE_TOL, n_action_draws: int = N_SIMULATED_ACTION_SEQUENCES,
                    seed: int = RANDOM_SEED) -> FittedQIterationResult:
    """Algorithm S.1 averaged over ``n_action_draws`` i.i.d. simulated action sequences (equation S.12).

    Set ``gamma=0`` and ``max_iter=1`` to obtain the myopic Greedy policy (Section 4.1: "This is
    achieved using one fitted Q-iteration ... equivalent to setting gamma = 0").
    """
    if rho_grid is None:
        from src.config import GCV_GRID_SIZE, GCV_RHO_OVER_N_MAX_EMPIRICAL, GCV_RHO_OVER_N_MIN_EMPIRICAL
        rho_grid = np.geomspace(GCV_RHO_OVER_N_MIN_EMPIRICAL, GCV_RHO_OVER_N_MAX_EMPIRICAL, GCV_GRID_SIZE)
    cost = cost_bps / 10000.0

    coefs_by_action_draws: dict = {a: [] for a in range(n_actions)}
    iters_used = []
    for u in range(n_action_draws):
        rng = np.random.default_rng(seed + u)
        coefs, n_iter = _fit_one_draw(raw_t, raw_tp1, ret_tp1, binary_mask, n_actions, cost, eta, gamma,
                                       rho_grid, max_iter, tol, rng)
        for a in range(n_actions):
            coefs_by_action_draws[a].append(coefs[a])
        iters_used.append(n_iter)
    return FittedQIterationResult(coefs=coefs_by_action_draws, n_iterations=int(np.mean(iters_used)),
                                   n_actions=n_actions, binary_mask=binary_mask)
