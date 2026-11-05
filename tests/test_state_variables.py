"""Unit tests for state-variable construction (src/state_variables.py, Section 4.1 / S.5.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.state_variables import assemble_state, build_trend, classify_cycle, scale_and_clip


def test_scale_and_clip_bounds():
    x = pd.Series([-100.0, -0.05, 0.0, 0.05, 100.0])
    out = scale_and_clip(x, scale=0.1)
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert np.isclose(out.iloc[2], 0.0)


def test_build_trend_uses_only_trailing_information():
    idx = pd.period_range("2000-01", periods=24, freq="M")
    # A return series that jumps at month 13: the trend value at month 12 (index 11) must not
    # reflect data from month 13 onward (no look-ahead).
    returns = pd.Series([0.01] * 12 + [1.0] * 12, index=idx)
    df = pd.DataFrame({"MOM": returns})
    trend = build_trend(df, ("MOM",))
    assert np.isnan(trend["Trend_MOM"].iloc[10])  # fewer than 12 trailing obs
    val_at_12 = trend["Trend_MOM"].iloc[11]
    assert val_at_12 <= scale_and_clip(pd.Series([0.01]), 0.1).iloc[0] + 1e-9


def test_classify_cycle_four_categories_are_exhaustive_and_mutually_exclusive():
    idx = pd.period_range("2000-01", periods=36, freq="M")
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(scale=0.03, size=36), index=idx)
    out = classify_cycle(mkt)
    valid = out.dropna()
    dummy_sum = valid[["Bear", "Correction", "Rebound"]].sum(axis=1)
    # Bull is encoded as all three dummies equal to zero (the omitted baseline category).
    assert set(dummy_sum.unique()).issubset({0.0, 1.0})
    assert set(valid["cycle_label"].unique()).issubset({"Bull", "Bear", "Correction", "Rebound"})


def test_classify_cycle_rule_matches_sign_definition():
    idx = pd.period_range("2000-01", periods=13, freq="M")
    mkt = pd.Series([0.05] * 12 + [0.05], index=idx)  # positive 12m average, positive current -> Bull
    out = classify_cycle(mkt)
    assert out["cycle_label"].iloc[-1] == "Bull"

    mkt2 = pd.Series([0.05] * 12 + [-0.05], index=idx)  # positive average, negative current -> Correction
    out2 = classify_cycle(mkt2)
    assert out2["cycle_label"].iloc[-1] == "Correction"

    mkt3 = pd.Series([-0.05] * 12 + [-0.05], index=idx)  # negative average, negative current -> Bear
    out3 = classify_cycle(mkt3)
    assert out3["cycle_label"].iloc[-1] == "Bear"

    mkt4 = pd.Series([-0.05] * 12 + [0.05], index=idx)  # negative average, positive current -> Rebound
    out4 = classify_cycle(mkt4)
    assert out4["cycle_label"].iloc[-1] == "Rebound"


def test_assemble_state_binary_mask_lengths_by_spec():
    idx = pd.period_range("1990-01", periods=60, freq="M")
    rng = np.random.default_rng(1)
    actions = ("MOM", "HML", "RMW", "DEF")
    factor_returns = pd.DataFrame({a: rng.normal(scale=0.03, size=60) for a in actions}, index=idx)
    mkt = pd.Series(rng.normal(scale=0.03, size=60), index=idx)
    fred = pd.DataFrame({
        "FEDFUNDS": np.full(60, 2.0), "GS10": np.full(60, 3.0), "GS1": np.full(60, 1.0),
        "BAA": np.full(60, 5.0), "AAA": np.full(60, 4.0),
        "CPILFENS": 100 * (1.002 ** np.arange(60)), "CPIAUCNS": 100 * (1.003 ** np.arange(60)),
    }, index=idx)

    state_trend, mask_trend = assemble_state(factor_returns, actions, mkt, fred, "Trend")
    assert state_trend.shape[1] == 4 and mask_trend.sum() == 0

    state_tc, mask_tc = assemble_state(factor_returns, actions, mkt, fred, "Trend+Cycle")
    assert state_tc.shape[1] == 7 and mask_tc.sum() == 3

    state_tcm, mask_tcm = assemble_state(factor_returns, actions, mkt, fred, "Trend+Cycle+Macro")
    assert state_tcm.shape[1] == 12 and mask_tcm.sum() == 3
