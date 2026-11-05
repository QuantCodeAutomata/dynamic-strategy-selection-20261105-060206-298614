"""State-variable construction for the empirical factor-rotation experiment.

Builds the Trend, Cycle and Macro state blocks described in Section 4.1 of
"Dynamic Strategy Selection" and applies the scaling/clipping rule of
Section S.5.1: every non-binary raw variable is divided by a prespecified
constant and clipped to [-1, 1] before entering the ridge-GCV basis of
``src.rl_core``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EXP1_INFLATION_LAG_MONTHS,
    EXP1_SCALE_CORE_INFLATION,
    EXP1_SCALE_CREDIT_SPREAD,
    EXP1_SCALE_FEDFUNDS,
    EXP1_SCALE_FOOD_ENERGY,
    EXP1_SCALE_TERM_SPREAD,
    EXP1_SCALE_TREND,
    EXP1_TREND_WINDOW_MONTHS,
)


def scale_and_clip(x: pd.Series, scale: float) -> pd.Series:
    """x / scale, clipped to [-1, 1] (Section S.5.1)."""
    return (x / scale).clip(-1.0, 1.0)


def build_trend(factor_returns: pd.DataFrame, actions: tuple[str, ...]) -> pd.DataFrame:
    """Trend_t: trailing 12-month average monthly return of each factor (Section 4.1 "State"),
    scaled by ``EXP1_SCALE_TREND`` and clipped to [-1, 1]. Uses only information available at
    the end of month t (a trailing, backward-looking rolling mean), so no look-ahead is
    introduced when the state at t is used to choose the action held over month t+1."""
    trend = factor_returns[list(actions)].rolling(EXP1_TREND_WINDOW_MONTHS, min_periods=EXP1_TREND_WINDOW_MONTHS).mean()
    trend.columns = [f"Trend_{a}" for a in actions]
    return trend.apply(lambda col: scale_and_clip(col, EXP1_SCALE_TREND))


def classify_cycle(market_excess_return: pd.Series) -> pd.DataFrame:
    """Bull/Bear/Correction/Rebound classification (Goulding, Harvey and Mazzoleni, 2023,
    as referenced in Section 4.1), from the signs of the trailing 12-month average market
    excess return and the current one-month market excess return:

        Bull       : 12-month average > 0 and current month > 0
        Correction : 12-month average > 0 and current month <= 0
        Rebound    : 12-month average <= 0 and current month > 0
        Bear       : 12-month average <= 0 and current month <= 0

    Returns dummy columns for Bear, Correction and Rebound; Bull is the omitted baseline
    category (Section 4.1: "we omit the Bull dummy to avoid multicollinearity").
    """
    avg_12m = market_excess_return.rolling(12, min_periods=12).mean()
    cur = market_excess_return
    bull = (avg_12m > 0) & (cur > 0)
    correction = (avg_12m > 0) & (cur <= 0)
    rebound = (avg_12m <= 0) & (cur > 0)
    bear = (avg_12m <= 0) & (cur <= 0)
    out = pd.DataFrame({
        "Bear": bear.astype(float),
        "Correction": correction.astype(float),
        "Rebound": rebound.astype(float),
    }, index=market_excess_return.index)
    out.loc[avg_12m.isna(), :] = np.nan
    labels = pd.Series(np.select(
        [bull, correction, rebound, bear], ["Bull", "Correction", "Rebound", "Bear"], default=np.nan
    ), index=market_excess_return.index)
    labels[avg_12m.isna()] = np.nan
    out["cycle_label"] = labels
    return out


def build_macro(fred: pd.DataFrame) -> pd.DataFrame:
    """FEDFUNDS, TermSpread, CreditSpread, CoreInflation, FoodEnergy (Section 4.1 "State").

    ``fred`` must contain monthly columns FEDFUNDS, GS10, GS1, BAA, AAA, CPILFENS (core,
    not seasonally adjusted) and CPIAUCNS (headline, not seasonally adjusted). CoreInflation
    is the 12-month log difference of CPILFENS; FoodEnergy is the difference between the
    12-month headline and core inflation rates. Both inflation variables are lagged by one
    month to reflect the CPI release delay (Section 4.1, footnote on "Inflation variables
    lagged by one month due to release delay").
    """
    term_spread = fred["GS10"] - fred["GS1"]
    credit_spread = fred["BAA"] - fred["AAA"]
    core_inflation = np.log(fred["CPILFENS"]).diff(12)
    headline_inflation = np.log(fred["CPIAUCNS"]).diff(12)
    food_energy = headline_inflation - core_inflation

    out = pd.DataFrame({
        "FEDFUNDS": scale_and_clip(fred["FEDFUNDS"], EXP1_SCALE_FEDFUNDS),
        "TermSpread": scale_and_clip(term_spread, EXP1_SCALE_TERM_SPREAD),
        "CreditSpread": scale_and_clip(credit_spread, EXP1_SCALE_CREDIT_SPREAD),
        "CoreInflation": scale_and_clip(core_inflation, EXP1_SCALE_CORE_INFLATION),
        "FoodEnergy": scale_and_clip(food_energy, EXP1_SCALE_FOOD_ENERGY),
    }, index=fred.index)
    lag_cols = ["CoreInflation", "FoodEnergy"]
    out[lag_cols] = out[lag_cols].shift(EXP1_INFLATION_LAG_MONTHS)
    return out


def assemble_state(factor_returns: pd.DataFrame, actions: tuple[str, ...], market_excess_return: pd.Series,
                    fred: pd.DataFrame, spec: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Builds the raw-state DataFrame for one of the three information sets and the matching
    binary mask (True = 0/1 dummy, no quadratic term; False = continuous, gets a square term).

    ``spec`` in {"Trend", "Trend+Cycle", "Trend+Cycle+Macro"} (Section 4).
    """
    trend = build_trend(factor_returns, actions)
    blocks = [trend]
    binary_mask = [False] * trend.shape[1]
    if spec in ("Trend+Cycle", "Trend+Cycle+Macro"):
        cycle = classify_cycle(market_excess_return)[["Bear", "Correction", "Rebound"]]
        blocks.append(cycle)
        binary_mask += [True] * cycle.shape[1]
    if spec == "Trend+Cycle+Macro":
        macro = build_macro(fred)
        blocks.append(macro)
        binary_mask += [False] * macro.shape[1]
    state = pd.concat(blocks, axis=1)
    return state, np.array(binary_mask)
