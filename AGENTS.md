# Repository guide

This repository reproduces two experiments from **Dynamic Strategy Selection**
(Cordoni, F., Sancetta, A., Zhao, H.): a fitted-Q-iteration reinforcement-learning
framework for dynamic rotation among a finite set of trading strategies with
switching costs.

1. **Experiment 1** (Section 4): monthly rotation among four U.S. equity factor
   strategies (MOM, HML, RMW, DEF), comparing the RL policy against a Greedy
   policy and four fixed-factor benchmarks across state information sets,
   risk-aversion levels and switching costs.
2. **Experiment 2** (Section S.6 of the Electronic Supplement): a Monte Carlo
   study on a four-state Markov-chain data-generating process with a known
   oracle policy, checking convergence of the estimated RL policy to the
   oracle and the effect of irrelevant state variables.

## Layout

```
src/     shared library, flat modules only:
         config.py            all parameters, grouped by experiment, as UPPER_CASE
                               module-level variables with a source comment
         rl_core.py           fitted Q-iteration (Algorithm S.1), additive basis
                               functions (Eq. A.2), ridge regression with GCV,
                               Q-value capping and N_A-action averaging (Eq. S.12)
         state_variables.py   Trend / Cycle / Macro state construction and scaling
         oracle.py            analytic oracle policy and value function for the
                               Monte Carlo DGP (Section S.6)
         policy_evaluation.py out-of-sample policy simulation, rewards, switching
                               frequency, Sharpe ratios, significance tests
         metrics.py           shared performance-metric helpers
exp/     experiment entry points only (exp1_empirical.py, exp2_monte_carlo.py);
         call src/ functions with their documented signatures, no private logic
data/    data_loader.py       French factor library, FRED, and the QuantX
                               Dataloader REST API (DEF proxy construction)
         generate_synthetic.py Monte Carlo data-generating process (Experiment 2)
tests/   pytest suite (unit tests on fixtures/synthetic inputs; market_data
         tests use a cached, reduced real-data slice)
results/ tables, logs and plots written by the experiment scripts, plus
         RESULTS.md (the only place deviations from the paper are described)
```

`src/`, `exp/` and `data/` each contain an empty `__init__.py`. No market data,
downloaded datasets, or API responses are committed; `data/cache/` is
git-ignored.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY (Experiment 1 only)
```

API keys are read from `.env` via `python-dotenv` inside `data/data_loader.py`
and must never be committed, logged, or placed in `src/config.py`.

## Tests

```bash
python -m pytest -m "not market_data"   # fast unit tests, no network access
python -m pytest                        # full suite; market_data tests need API keys
                                         # and download a small cached real-data slice
```

`market_data`-marked tests are registered and skipped automatically in
`tests/conftest.py` when `DATALOADER_API_KEY` is not set.

## Running the experiments

```bash
python -m exp.exp2_monte_carlo          # executed Monte Carlo replication count
python -m exp.exp2_monte_carlo --full   # the paper's 500 replications per configuration

python -m exp.exp1_empirical            # executed state/cost/eta subset (see results/RESULTS.md)
python -m exp.exp1_empirical --full     # the paper's full 18-combination grid at monthly re-estimation
```

`reproduce.sh` runs the non-market-data tests and both experiments with the
commands above.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  UPPER_CASE variable with a one-line source comment (a paper section or
  equation, or "not specified in the paper"); no classes or parameter dicts.
  Where a parameter was reduced to fit the time available, both the paper
  value and the executed value are defined side by side
  (`*_PAPER` / `*_RUN`), and an experiment's `--full` flag selects the paper
  values.
- All data loading goes through `data/data_loader.py` (market data) or
  `data/generate_synthetic.py` (the Monte Carlo DGP); `src/` and `exp/`
  never fetch or generate data themselves.
- `exp/` scripts call `src/` functions using their documented argument names
  and return keys; they do not add undocumented keyword arguments.
- Every run of an experiment script overwrites its tables, logs and plots in
  `results/`. The numbers and verdicts in `results/RESULTS.md` are not
  regenerated automatically: after a rerun, check them against the new files
  in `results/` and update `results/RESULTS.md` by hand.
- Deviations from the paper (data substitutions, reduced grids or replication
  counts, unresolved implementation choices) are recorded only in
  `results/RESULTS.md`, never silently in code.
