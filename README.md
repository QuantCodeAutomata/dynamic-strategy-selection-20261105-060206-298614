# Dynamic Strategy Selection — Replication

Replication of selected results from:

**Dynamic Strategy Selection** — Cordoni, F., Sancetta, A., Zhao, H.

This repository reproduces two experiments from the paper's reinforcement-learning
(fitted Q-iteration) framework for dynamic rotation among a finite set of trading
strategies with switching costs:

1. **Experiment 1 — Empirical Factor Rotation in the U.S. Equity Market** (Section 4):
   monthly rotation among four U.S. equity factor strategies (momentum MOM, value HML,
   profitability RMW, and a defensive low-volatility factor DEF), comparing the RL
   policy, the myopic Greedy policy, and the four fixed-factor benchmarks across state
   information sets (Trend, Trend+Cycle, Trend+Cycle+Macro), risk-aversion levels
   (η ∈ {0, 3}) and switching costs (20, 80, 120 bps).
2. **Experiment 2 — Monte Carlo Simulation Study** (Section S.6 of the Electronic
   Supplement): a controlled four-state Markov-chain data-generating process with a
   known oracle policy, used to check that the estimated RL policy approaches the
   oracle as the training sample grows, and to study the effect of irrelevant state
   variables.

## Repository layout

```
src/    shared library: RL core (fitted Q-iteration, ridge-GCV), state variables,
        oracle, policy evaluation, performance metrics, config.py
exp/    experiment scripts (exp1_empirical.py, exp2_monte_carlo.py)
data/   data_loader.py (French factors, FRED, QuantX Dataloader) and
        generate_synthetic.py (Monte Carlo data-generating process)
tests/  pytest suite
results/ tables, plots, logs and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for Experiment 1
```

## Tests

```bash
python -m pytest -m "not market_data"      # fast unit tests, no network access
python -m pytest                            # full suite, requires API keys for market_data tests
```

## Running the experiments

```bash
python -m exp.exp2_monte_carlo              # reduced Monte Carlo replication count
python -m exp.exp2_monte_carlo --full       # the paper's 500 replications per configuration

python -m exp.exp1_empirical                # executed configuration (see results/RESULTS.md)
python -m exp.exp1_empirical --full         # the paper's full monthly re-estimation
```

`reproduce.sh` runs the test suite and both experiments with these commands.

Every run overwrites the tables, logs and plots in `results/`; the numbers and
verdicts in `results/RESULTS.md` are written from those files and must be checked
and updated by hand after a rerun.

See `results/RESULTS.md` for the comparison against the paper's reported values,
the deviations from the paper, and the results not reproduced.
