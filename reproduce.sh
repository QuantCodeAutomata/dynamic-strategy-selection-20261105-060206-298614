#!/usr/bin/env bash
# Runs the test suite and both experiments with the commands given in README.md.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp2_monte_carlo
python -m exp.exp1_empirical
