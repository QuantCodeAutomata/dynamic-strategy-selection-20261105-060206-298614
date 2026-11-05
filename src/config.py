"""Module-level configuration constants for the dynamic strategy selection replication.

Every parameter used by the shared pipeline (src/), the data-preparation code
(data/) and the experiment scripts (exp/) is declared here as an UPPER_CASE
module-level variable. Each variable carries a comment stating its meaning,
its unit and the paper section/equation that specifies it (or states that the
paper does not specify it).
"""

# ---------------------------------------------------------------------------
# Shared reinforcement-learning (fitted Q-iteration) parameters
# Algorithm S.1 and Section 4.1 of "Dynamic Strategy Selection"
# ---------------------------------------------------------------------------

GAMMA = 0.9  # discount factor, dimensionless (Section 4.1: implies a ~6.6-month discount half-life)

RISK_AVERSION_LEVELS = (0.0, 3.0)  # eta in U(x) = x - (eta/2) x^2, dimensionless (Section 4.1)

SWITCHING_COSTS_BPS = (20, 80, 120)  # fixed switching cost, basis points (Section 4.1)

Q_CAP_B = 0.5  # capping bound for the Q-function, [-B, B] in decimals (Section S.5.2: B = 0.5)

GCV_GRID_SIZE = 31  # number of logarithmically spaced ridge-penalty candidates (Section S.5.2)
GCV_RHO_OVER_N_MIN_EMPIRICAL = 0.1   # lower bound of the rho/n grid, empirical application (Section S.5.2)
GCV_RHO_OVER_N_MAX_EMPIRICAL = 2.5   # upper bound of the rho/n grid, empirical application (Section S.5.2)
GCV_RHO_OVER_N_MIN_SIMULATION = 1e-6  # lower bound of the rho/n grid, Monte Carlo study (Section S.6.1)
GCV_RHO_OVER_N_MAX_SIMULATION = 2.5   # upper bound of the rho/n grid, Monte Carlo study (Section S.6.1)

N_SIMULATED_ACTION_SEQUENCES = 10  # N_A in equation (S.12): number of i.i.d. simulated action paths averaged

FQI_MAX_ITER = 50  # maximum number of fitted Q-iteration sweeps J; not specified in the paper, chosen
# generously so that the value function stabilises well before the cap is reached (checked by convergence below)
FQI_CONVERGENCE_TOL = 1e-4  # convergence tolerance epsilon for Algorithm S.1; not specified in the paper

RANDOM_SEED = 12345  # base seed for the i.i.d. uniform simulated actions and all Monte Carlo draws;
# not specified in the paper, fixed here for reproducibility

# ---------------------------------------------------------------------------
# Experiment 1: Empirical factor rotation in the U.S. equity market
# Section 4 and Section S.5 of the paper
# ---------------------------------------------------------------------------

EXP1_ACTIONS = ("MOM", "HML", "RMW", "DEF")  # the four candidate factor strategies (Section 4)

EXP1_SAMPLE_START = "1963-07-01"  # start of the factor-return sample (Section 4.1: "July 1963")
EXP1_SAMPLE_END = "2025-12-31"  # end of the factor-return sample (Section 4.1: "to December 2025")

EXP1_INITIAL_WINDOW_MONTHS = 120  # expanding-window initial estimation sample, months (Section 4.1)
EXP1_OOS_START = "1973-07-01"  # first out-of-sample decision month (Section 4.1: "begins in July 1973")
EXP1_OOS_N_DECISIONS_PAPER = 629  # number of out-of-sample monthly decisions reported in the paper (Section 4.1)

EXP1_TREND_WINDOW_MONTHS = 12  # trailing window for the factor-trend state variable, months (Section 4.1 "State")

# Scaling constants: each raw variable is divided by its constant, then clipped to [-1, 1] (Section S.5.1)
EXP1_SCALE_TREND = 0.1           # factor trend scaling constant, decimal return units (Section S.5.1)
EXP1_SCALE_FEDFUNDS = 20.0       # FEDFUNDS scaling constant, percentage points (Section S.5.1)
EXP1_SCALE_TERM_SPREAD = 10.0    # TermSpread scaling constant, percentage points (Section S.5.1)
EXP1_SCALE_CREDIT_SPREAD = 10.0  # CreditSpread scaling constant, percentage points (Section S.5.1)
EXP1_SCALE_CORE_INFLATION = 15.0  # CoreInflation scaling constant, percentage points (Section S.5.1)
EXP1_SCALE_FOOD_ENERGY = 5.0      # FoodEnergy scaling constant, percentage points (Section S.5.1)

EXP1_INFLATION_LAG_MONTHS = 1  # CPI release-delay lag applied to inflation state variables (Section 4.1, footnote 1)

EXP1_STATE_SPECS = ("Trend", "Trend+Cycle", "Trend+Cycle+Macro")  # the three state information sets (Section 4)

# DEF factor construction: CRSP daily stock returns are not accessible in this environment (no WRDS
# subscription); the defensive low-volatility factor is instead built from QuantX Dataloader daily bars
# (see data/data_loader.py and "Deviations from the paper" in results/RESULTS.md).
EXP1_DEF_VOL_WINDOW_DAYS = 126  # trailing daily-volatility estimation window, trading days (not specified in
# the paper for the CRSP construction; chosen as a standard 6-month volatility window, Novy-Marx 2014 style)
EXP1_DEF_REBALANCE = "M"  # monthly rebalancing frequency for the volatility sort (Section 4.1: factors are monthly)
EXP1_DEF_LOW_QUANTILE = 0.3   # bottom 30% of trailing volatility bought long (Fama-French 2x3 style tercile cut)
EXP1_DEF_HIGH_QUANTILE = 0.7  # top 30% of trailing volatility sold short (Fama-French 2x3 style tercile cut)

EXP1_UNIVERSE_SIZE_PAPER = "all CRSP common stocks"  # paper's DEF universe (Section 4.1): not a fixed count
EXP1_UNIVERSE_SIZE_RUN = 120  # number of long-history, liquid U.S. common stocks used to build the DEF proxy
# (reduced from the full CRSP universe; selection rule: liquidity-ranked large caps with long Dataloader
# history, see data/data_loader.py and "Deviations from the paper" in results/RESULTS.md)

EXP1_REFIT_EVERY_MONTHS_PAPER = 1  # the paper re-estimates the policy every out-of-sample month
# every (state spec, switching cost, eta) combination actually computed by the default,
# time-limited run, at the paper's own monthly re-estimation frequency: the headline
# Trend+Cycle/120bps/eta=3 point (Section 4.1), its eta=0 counterpart to gauge risk
# aversion, and the two other state specifications at the headline cost/eta to gauge the
# information-estimation trade-off (Section 4.2). `--full` instead runs the complete
# 3 specs x 3 costs x 2 etas = 18-combination grid of Section 4.1.
EXP1_RUN_COMBOS = (
    ("Trend+Cycle", 120, 3.0),
    ("Trend+Cycle", 120, 0.0),
    ("Trend", 120, 3.0),
    ("Trend+Cycle+Macro", 120, 3.0),
)
# (Section 4.1: "recursive" expanding-window evaluation; 629 monthly decisions)
EXP1_REFIT_EVERY_MONTHS_RUN = 1  # executed re-estimation frequency: every out-of-sample month, matching
# the paper (Section 4.1). An earlier draft re-estimated every six months to save computation time; this
# changed the sign of the RL-versus-Greedy Sharpe-ratio ranking relative to the paper (Section 4, Table of
# Sharpe ratios), so the reduction is not used; the time saved instead comes from the smaller combination
# grid of EXP1_RUN_COMBOS above. Run with --full for the paper's complete combination grid.

# ---------------------------------------------------------------------------
# Experiment 2: Monte Carlo simulation study (Section S.6 of the Electronic Supplement)
# ---------------------------------------------------------------------------

EXP2_N_STATES = 4  # number of Markov-chain states for X_t (Section S.6.1)
EXP2_N_ACTIONS = 4  # number of candidate strategies (Section S.6.1)

# Transition matrix K, K_ij = Pr(X_{t+1}=j | X_t=i) -- equation (S.27)
EXP2_TRANSITION_MATRIX = (
    (0.85, 0.05, 0.10, 0.00),
    (0.05, 0.85, 0.00, 0.10),
    (0.90, 0.00, 0.10, 0.00),
    (0.00, 0.90, 0.00, 0.10),
)

EXP2_BETA = 0.01  # conditional mean return of strategy a when X_{t+1}=a, decimal (Section S.6.1)
EXP2_SIGMA_EPS = 0.04  # idiosyncratic monthly return volatility, decimal (Section S.6.1: "4%")
EXP2_STRATEGY_RETURN_CORR = 0.3  # pairwise correlation of strategy-return innovations (Section S.6.1)
EXP2_NOISE_AR1_COEF = 0.5  # AR(1) coefficient of the irrelevant state variables (Section S.6.1)
EXP2_NOISE_CORR = 0.3  # pairwise correlation of the noise-variable innovations (Section S.6.1)
EXP2_NOISE_COUNTS = (0, 2, 4, 8)  # number of irrelevant/noisy state variables considered (Section S.6.1)

EXP2_ETA = 0.0  # risk aversion used in the simulation; the oracle policy is optimal for eta in [0, 164] (Section S.6.1)
EXP2_SWITCHING_COST_BPS = 120  # fixed switching cost used throughout the simulation, bps (Section S.6.1)

EXP2_TRAIN_SIZES_PAPER = (240, 600, 1200)  # training-sample sizes n_est, months (Section S.6.1)
EXP2_OOS_SIZE = 500  # out-of-sample evaluation horizon, months (Section S.6.1)

EXP2_N_REPLICATIONS_PAPER = 500  # full number of Monte Carlo replications per configuration (Section S.6.1)
# Executed replication count: reduced from the paper's 500 to fit the available computation time while
# preserving every DGP parameter, training size and noise specification. Run with --full for 500 replications.
EXP2_N_REPLICATIONS_RUN = 60

# ---------------------------------------------------------------------------
# Data sources (Section 4.1 and Section 8 "Data")
# ---------------------------------------------------------------------------

FRENCH_3FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
)
FRENCH_5FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
FRENCH_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
)

FRED_SERIES = ("FEDFUNDS", "GS10", "GS1", "BAA", "AAA", "CPILFENS", "CPIAUCNS")  # FRED mnemonics used (Section S.5.1)
