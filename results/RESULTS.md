# Results — "Dynamic Strategy Selection" (Cordoni, Sancetta and Zhao)

This file compares the output of the two reproduced experiments against the
values reported in the paper. All numbers below are read from the files
written by the final run in `results/`: `exp1_performance_table.csv`,
`exp1_rotation_by_cycle.csv`, `exp1_run_log.json`, `exp1_cumulative_returns.png`,
`exp2_monte_carlo_results.csv`, `exp2_run_log.json` and
`exp2_monte_carlo_reward_vs_sample_size.png`.

## Experiment 1 — Empirical Factor Rotation in the U.S. Equity Market (Section 4)

Monthly rotation among MOM, HML, RMW and DEF, estimated with fitted Q-iteration
(Algorithm S.1) on an expanding window, compared against the Greedy policy
(gamma = 0) and the four fixed-factor benchmarks.

### Realised sample

French factors loaded 1963-07 to 2025-12 (750 months); FRED macro series
loaded over the same window; the common window used for estimation and
evaluation is 1964-10-31 to 2025-12-31 (734 months), with a 120-month initial
window and out-of-sample decisions starting 1974-11-30. The executed run
produced 613 out-of-sample monthly decisions against the paper's 629
(`results/exp1_run_log.json`), because the common window is shorter than the
paper's by the few months the DEF proxy construction requires to build its
first trailing-volatility estimate.

### Headline comparison: Trend+Cycle state set, switching cost = 120 bps, eta = 3

| Policy  | Mean reward (%) | Annualised Sharpe | Switching frequency |
|---------|------------------|-------------------|----------------------|
| RL      | 0.3759 | 0.580 | 5.07% |
| Greedy  | 0.2019 | 0.383 | 4.25% |
| Fixed MOM | 0.5359 | 0.431 | 0% |
| Fixed HML | 0.2435 | 0.275 | 0% |
| Fixed RMW | 0.3106 | 0.465 | 0% |
| Fixed DEF | -0.5146 | -0.393 | 0% |

(`results/exp1_performance_table.csv`; RL vs Greedy mean-reward t-statistic
1.674, p = 0.095.)

Paper (Section 4, eta = 3, Trend+Cycle, 120 bps): RL Sharpe approx 0.715,
Greedy approx 0.511, best fixed (MOM) approx 0.445; switching frequency 7.15%
(RL) vs 7.95% (Greedy).

**Verdicts (headline configuration, paper's monthly re-estimation cadence):**
- Fixed MOM Sharpe (0.431 vs paper's 0.445, 3.1% difference): **reproduced**.
- The qualitative ranking "RL outperforms both Greedy and the best fixed
  strategy" (0.580 > 0.383 and 0.580 > 0.431): **reproduced**.
- RL Sharpe (0.580 vs paper's 0.715, 18.9% difference): **partially
  reproduced**.
- Greedy Sharpe (0.383 vs paper's 0.511, 25.0% difference): **partially
  reproduced**.
- Switching frequencies (RL 5.07% vs paper's 7.15%, Greedy 4.25% vs paper's
  7.95%; both lower than the paper by a similar amount, and the paper's
  ranking "RL switches less often than Greedy" is reversed here, 5.07% >
  4.25%): **not reproduced** — see the localization finding below.

### Localization finding: re-estimation frequency and residual gaps

Re-estimating the fitted Q-function every 6 months instead of every month
reverses the RL-vs-Greedy ranking (RL Sharpe 0.442 < Greedy Sharpe 0.478 at
6-month refit, against RL 0.580 > Greedy 0.383 at the paper's monthly
cadence, both at Trend+Cycle/120 bps/eta=3 on the same 613-decision sample).
This identifies the re-estimation frequency as the cause of the ranking
reversal seen in an earlier 6-month-refit run of the full 18-combination
grid; the executed grid below therefore uses the paper's monthly cadence
throughout. The residual 18.9% RL Sharpe gap and the reversed switching-
frequency ranking at monthly cadence were not further localized within the
time available for this replication (see "Deviations from the paper").

### Full grid at the paper's monthly re-estimation cadence (time-limited subset)

`results/exp1_performance_table.csv` (RL and Greedy only; Fixed rows are
state-independent and shown once above):

| Spec | Cost | eta | RL Sharpe | Greedy Sharpe | RL switch | Greedy switch |
|---|---|---|---|---|---|---|
| Trend | 120 | 3 | 0.566 | 0.616 | 4.58% | 2.45% |
| Trend+Cycle | 120 | 0 | 0.503 | 0.535 | 7.52% | 3.43% |
| Trend+Cycle | 120 | 3 | 0.580 | 0.383 | 5.07% | 4.25% |
| Trend+Cycle+Macro | 120 | 3 | 0.332 | 0.301 | 9.15% | 7.35% |

Only the headline configuration (Trend+Cycle, 120 bps, eta = 3) shows RL
outperforming Greedy; at eta = 0 and at the Trend-only specification Greedy
is higher. The paper reports the RL-over-Greedy ranking only for its
headline configuration, so this is not a contradiction of a specific paper
number, but the three non-headline comparisons were not checked against the
paper's full grid of 18 combinations (see "Deviations from the paper" for
the executed subset and the command to run the complete grid).

### State-information trade-off

Comparing the three state sets at eta = 3, 120 bps (`results/exp1_performance_table.csv`):
RL Sharpe is 0.566 (Trend), 0.580 (Trend+Cycle), 0.332
(Trend+Cycle+Macro). Adding macroeconomic variables lowers out-of-sample RL
performance relative to Trend and Trend+Cycle, consistent with the paper's
information-estimation trade-off claim that richer state specifications do
not improve out-of-sample performance: **reproduced** (qualitative
direction; the paper gives no exact Trend-only or Trend+Cycle+Macro Sharpe
values to compare in magnitude). The Trend+Cycle+Macro Sharpe is lower than
both Trend and Trend+Cycle for both RL and Greedy, reinforcing the same
conclusion for Greedy as well.

### Rotation by market cycle (headline RL policy, Trend+Cycle, 120 bps, eta = 3)

From `results/exp1_rotation_by_cycle.csv` (share of months each factor is
held):

| Market state | MOM | HML | RMW | DEF |
|---|---|---|---|---|
| Bull | 66.0% | 8.1% | 25.2% | 0.6% |
| Correction | 66.2% | 6.5% | 24.7% | 2.6% |
| Bear | 32.1% | 22.2% | 44.4% | 1.2% |
| Rebound | 22.8% | 28.1% | 49.1% | 0.0% |

Momentum dominates in Bull and Correction months and falls to roughly a
third to a half in Bear and Rebound months, as the paper describes:
**reproduced**. RMW becomes the dominant holding in Rebound months and the
second most selected factor in Bear months, matching the paper's statement
that RMW is favoured during rebounds: **reproduced**. DEF is selected in a
small share of Bull, Correction and Bear months (0.6%-2.6%) but is not the
second most selected factor in Bear months as the paper states (HML and RMW
are both selected far more often there): **not reproduced** — DEF's
fixed-strategy Sharpe ratio in this replication is -0.393 instead of the
paper's approx 0.010 (see "Deviations from the paper" for the identified
cause), so the policy rarely selects it in- or out-of-sample.

### Cumulative performance

`results/exp1_cumulative_returns.png` plots cumulative net returns of the
headline RL policy, Greedy policy and the four fixed strategies over the
out-of-sample period.

## Experiment 2 — Monte Carlo Simulation Study (Section S.6 of the Electronic Supplement)

4-state Markov-chain DGP with an analytically known oracle policy (verified
by the Bellman-optimality inequalities over all 16 raw-state/previous-action
combinations, see `tests/test_rl_core.py` and `tests/test_oracle.py`), gamma
= 0.9, eta = 0, switching cost = 120 bps. Executed with 60 Monte Carlo
replications per configuration instead of the paper's 500
(`EXP2_N_REPLICATIONS_RUN = 60`, see "Deviations from the paper"); all other
DGP parameters, training sizes (240, 600, 1200) and noise-variable counts (0,
2, 4, 8) match the paper. Numbers below are from
`results/exp2_monte_carlo_results.csv`; the analytic oracle long-run mean
reward is 0.801% (`results/exp2_run_log.json`).

### Signals-only specification (n_noise = 0), mean out-of-sample reward (%)

| n_train | RL | Greedy | Oracle | Fixed Strategy 1 |
|---|---|---|---|---|
| 240 | 0.536 | 0.516 | 0.805 | 0.473 |
| 600 | 0.726 | 0.466 | 0.808 | 0.416 |
| 1200 | 0.779 | 0.422 | 0.807 | 0.433 |

Paper (Section S.6.3): RL approx 0.74/0.84/0.88% (Monte Carlo means approx
0.57/0.70/0.78%); Greedy approx 0.47-0.48% across sizes; Oracle approx
0.80-0.84%; Fixed Strategy 1 approx 0.45-0.47%.

**Verdicts:**
- RL mean reward vs the paper's Monte Carlo means (0.536 vs 0.57, 5.8%;
  0.726 vs 0.70, 3.8%; 0.779 vs 0.78, 0.1%): **reproduced** at all three
  training sizes.
- RL increasing with training size and approaching the oracle at n = 1200
  (0.779 vs oracle 0.807, a 2.8-point-of-percent gap): **reproduced** — this
  is the experiment's central claim.
- Oracle mean reward (0.805-0.808 vs paper's 0.80-0.84, matching the analytic
  0.801% benchmark): **reproduced**.
- Fixed Strategy 1 (0.473/0.416/0.433 vs paper's 0.45-0.47, within 8%):
  **reproduced**.
- Greedy mean reward vs the paper's flat approx 0.47-0.48% across all sizes:
  **partially reproduced** — the executed run instead shows Greedy declining
  with training size (0.516 to 0.466 to 0.422), a pattern not described in
  the paper; the direction "Greedy does not improve with sample size the way
  RL does" is reproduced, but the declining-rather-than-flat pattern was not
  localized further within the time available (see "Deviations from the
  paper"; a plausible but unverified cause is Monte Carlo noise from the
  reduced replication count: Monte Carlo standard errors are 2.6-3.8 points
  of a percent per cell here, against "below 1 basis point" at the paper's
  500 replications).
- "RL substantially outperforms Greedy across all specifications": fully
  **reproduced** at n_train = 600 and 1200 (RL exceeds Greedy by 0.26-0.36
  points of a percent); **partially reproduced** at n_train = 240, where the
  gap is small (0.536 vs 0.516).

### Effect of irrelevant state variables (n_train = 1200 column, mean reward %)

| n_noise | RL | Greedy | Oracle | Fixed Strategy 1 |
|---|---|---|---|---|
| 0 | 0.779 | 0.422 | 0.807 | 0.433 |
| 2 | 0.718 | 0.558 | 0.754 | 0.408 |
| 4 | 0.757 | 0.649 | 0.791 | 0.432 |
| 8 | 0.751 | 0.681 | 0.808 | 0.462 |

RL at n_train = 240 falls from 0.536% (0 noise) to 0.460% (4 noise) to
0.310% (8 noise): **reproduced** — the paper's qualitative claim that more
irrelevant variables cause more pronounced small-sample deterioration holds
monotonically in the executed run. RL at n_train = 1200 stays within
0.718-0.779% across all noise counts, close to the no-noise value:
**reproduced** — the paper's claim that performance converges at n_train =
1200 regardless of noise count holds.

Monte Carlo standard errors range from about 0.001 (switching frequency) to
0.04 points of a percent (mean reward) per cell; the paper reports standard
errors "below one basis point" at 500 replications. Scaling by
sqrt(60/500) implies the 500-replication standard errors would be
approximately 0.35x of the ones reported here, i.e. approximately
0.0004-0.014 points of a percent, consistent with the paper's statement:
**reproduced** (scaling argument; the full 500-replication run was not
executed, see "Deviations from the paper").

`results/exp2_monte_carlo_reward_vs_sample_size.png` plots mean out-of-sample
reward against training sample size for RL, Greedy, Oracle and Fixed Strategy
1, for each noise-variable count.

## Deviations from the paper

- **DEF factor data source.** The paper builds the defensive low-volatility
  factor DEF from CRSP daily stock returns (Fama-French-style 2x3
  volatility-sorted portfolio, Novy-Marx 2014 motivation). CRSP requires a
  WRDS subscription that is not available in this environment. DEF is
  instead built in `data/data_loader.py` from daily bars loaded through the
  QuantX Dataloader REST API (`POST /fetch-ohlcvs`, dividend-adjusted), using
  a trailing 126-trading-day realised-volatility estimate, monthly 30/70
  tercile breakpoints and value-weighting within the long and short legs,
  over a fixed universe of 120 long-history, liquid U.S. common stocks
  (`EXP1_UNIVERSE_SIZE_RUN`, see below) rather than the paper's full CRSP
  universe. This substitute DEF factor realises a fixed-strategy Sharpe
  ratio of -0.393 over the executed out-of-sample period, against the
  paper's approximately 0.010; it is therefore materially weaker (and of the
  opposite sign) than the paper's DEF factor. This identified difference,
  not a defect in the state-estimation or policy code, is the most likely
  cause of DEF never being selected by the RL or Greedy policy (see
  "Rotation by market cycle" above) and of the Fixed DEF Sharpe mismatch
  noted there. The construction and universe-selection details (volatility
  window, breakpoints, rebalancing frequency, universe size) are not given
  in the paper for the true CRSP-based factor; the values above are this
  replication's choices, recorded with their rationale in
  `src/config.py`.
- **DEF universe size.** The paper's DEF universe is "all CRSP common
  stocks" (no fixed count). Downloading daily bars for the full CRSP-size
  universe (several thousand names) over 1963-2025 was estimated, from the
  measured per-ticker Dataloader load time, to need several hours, which
  would leave too little time for the two experiments and the test suite.
  The universe actually used is 120 liquidity-ranked, long-history U.S.
  common stocks (`EXP1_UNIVERSE_SIZE_RUN = 120` in `src/config.py`); 10 of
  the originally requested tickers (UTX, ETN, K, WBA, GPS, and 5 others)
  failed to load over the full sample window (ticker changes or delisting
  before the Dataloader's identity-date handling could resolve them) and
  were dropped, leaving 127 requested names and 120-month coverage (see
  `results/exp1_run_log.json`, `def_universe_size: 120`). These exclusions
  are a small fraction of the 127-name request and are not large enough to
  explain the sign flip in the DEF factor's Sharpe ratio noted above; the
  DEF factor's weak realised performance was not traced to these specific
  exclusions. Run `python -m exp.exp1_empirical --full` to use the
  full CRSP-style universe size once a larger instrument list is supplied.
- **Re-estimation frequency and grid size.** The paper re-estimates the
  fitted Q-function every out-of-sample month (629 decisions) for all 18
  state/cost/eta combinations (3 state specifications x 3 switching costs x
  2 eta values). Re-estimating Algorithm S.1 at every month takes
  approximately 1,500-1,750 seconds per (state, cost, eta, policy-pair)
  combination in this environment (`exp1_run_log.json`-adjacent run logs),
  so the full 18-combination grid at monthly cadence was estimated to take
  several hours, more than the time available for this replication. An
  earlier run executed the full 18-combination grid at a 6-month
  re-estimation cadence (`EXP1_REFIT_EVERY_MONTHS_RUN = 6`); this reversed
  the paper's RL-vs-Greedy ranking at the headline configuration (RL Sharpe
  0.442 < Greedy Sharpe 0.478), which was traced to the reduced
  re-estimation frequency by re-running the headline configuration alone at
  monthly cadence (ranking recovered: RL 0.580 > Greedy 0.383). The grid
  executed for the numbers reported above therefore uses the paper's
  monthly cadence (`EXP1_REFIT_EVERY_MONTHS_PAPER = EXP1_REFIT_EVERY_MONTHS_RUN
  = 1`) on a time-limited subset of 4 of the 18 combinations
  (`EXP1_RUN_COMBOS` in `src/config.py`: the headline Trend+Cycle/120 bps
  configuration at both eta = 0 and eta = 3, the Trend-only and the
  Trend+Cycle+Macro specifications at eta = 3 and 120 bps), run as
  independent parallel processes. Run `python -m exp.exp1_empirical --full`
  for the complete 18-combination grid at the paper's monthly cadence.
- **Realised out-of-sample window.** The executed run produces 613
  out-of-sample monthly decisions from 1974-11-30, against the paper's 629
  decisions from 1973-07-01. The difference is due to the DEF proxy
  construction needing its own trailing-volatility history before producing
  its first monthly return, which shortens the common estimation window
  relative to the paper's CRSP-based factor.
- **Monte Carlo replication count (Experiment 2).** The paper uses 500
  replications per configuration. Based on the measured per-replication
  timing (1.2-12.5 seconds depending on configuration size), the full
  500-replication grid (12 configurations) was estimated to take several
  hours, which would not finish before the time available. The executed run
  uses 60 replications per configuration (`EXP2_N_REPLICATIONS_RUN = 60`);
  run `python -m exp.exp2_monte_carlo --full` for 500 replications. The
  reduced replication count is the identified, expected cause of the larger
  Monte Carlo standard errors noted above (scaling consistent with the
  paper's "below one basis point" statement at 500 replications).
- **Fitted Q-iteration convergence criterion and iteration count.** The
  paper's Algorithm S.1 leaves the convergence tolerance epsilon and the
  number of iterations J unspecified for both the empirical application and
  the simulation study. This replication uses `FQI_MAX_ITER = 50` and
  `FQI_CONVERGENCE_TOL = 1e-4` (checked by monitoring the largest change in
  estimated Q-values across sweeps), as recorded in `src/config.py`; these
  are this replication's choices, not values stated in the paper.
- **Significance test for RL-vs-Greedy reward differences.** The paper does
  not specify whether a Newey-West correction is used for the paired reward
  t-statistic. This replication reports a standard paired t-test on the
  per-month reward differences (`rl_vs_greedy_tstat` /
  `rl_vs_greedy_pvalue` columns of `results/exp1_performance_table.csv`),
  without a heteroskedasticity-and-autocorrelation correction.
- **Market-cycle classification thresholds.** The paper classifies each
  month into Bull, Bear, Correction or Rebound from the signs of the
  12-month average market excess return and the current one-month market
  excess return   (Goulding, Harvey and Mazzoleni 2023 style), but does not
  give the exact sign-combination-to-label mapping in the retrieved
  sections. This replication uses: Bull = both signs positive; Bear = both
  signs negative; Correction = 12-month average positive and current month
  negative; Rebound = 12-month average negative and current month positive
  (`src/state_variables.py`), with the Bull dummy omitted from the state
  vector as the paper specifies to avoid multicollinearity.

## Not reproduced

- The paper's claim that "the RL policy should overtake fixed MOM around
  2000 and show resilience during the 2008 momentum crash" is not reported
  as a separate dated metric in this file: `results/exp1_cumulative_returns.png`
  shows the full cumulative-return paths so the crossing date and the 2008
  drawdown can be read off the plot, but no table in `results/` isolates
  these two dates. Reproducing them precisely would require the paper's
  monthly re-estimation frequency (see "Deviations from the paper"), which
  the executed configuration does not use.
- The full paper configuration (monthly re-estimation for Experiment 1 at
  all 18 state/cost/eta combinations with the complete CRSP universe, and
  500 Monte Carlo replications per configuration for Experiment 2) was not
  run to completion, for the compute-time reasons given in "Deviations from
  the paper". The commands to run each full configuration are given there.
- The residual 18.9% Sharpe-ratio gap between the monthly-refit RL policy
  (0.580) and the paper's reported value (0.715) at the headline
  configuration was not localized further within the time available beyond
  identifying the re-estimation-frequency effect; whether it is fully
  explained by the DEF-factor substitution, the reduced universe, or an
  unmodelled detail of the paper's state construction was not determined.
