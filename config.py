"""
Configuration for Queue Digital Twin simulation.
"""

# ============================================================================
# PLANT MODEL (Ground Truth)
# ============================================================================

# Arrival process: Poisson with exponential interarrival times
PLANT_ARRIVAL_RATE = 0.4  # jobs/min (λ = 0.4, mean interarrival = 2.5 min)

# Service time: Lognormal distribution
PLANT_SERVICE_MEAN = 5.0  # minutes
PLANT_SERVICE_STDEV = 2.0  # minutes

# Number of servers (operators)
NUM_SERVERS = 2

# Operator fatigue: linear drift in service mean
# service_mean(t) = base_mean * (1 + FATIGUE_RATE * t / SHIFT_DURATION)
FATIGUE_RATE = 0.2  # 20% increase over shift
SHIFT_DURATION = 480  # 8 hours in minutes

# ============================================================================
# DIGITAL TWIN (Initial State - Wrong!)
# ============================================================================

TWIN_INITIAL_ARRIVAL_RATE = 0.30  # jobs/min (incorrect: true is 0.40)
TWIN_INITIAL_SERVICE_MEAN = 4.0  # minutes (incorrect: true is 5.0)
TWIN_INITIAL_SERVICE_STDEV = 1.5  # minutes (incorrect: true is 2.0)
TWIN_ASSUMES_FATIGUE = False  # Twin doesn't know about fatigue initially

# ============================================================================
# DIGITAL TWIN LEARNING
# ============================================================================

# Rolling window size for parameter estimation
ROLLING_WINDOW_SIZE = 200  # last N arrivals/services

# Exponential moving average smoothing factor
EMA_ALPHA = 0.1  # lower = more smoothing, higher = more responsive

# Event-driven updates: after every completion/arrival
# State reconciliation sync frequency
SYNC_INTERVAL = 5.0  # minutes (fixed interval for metrics snapshot)

# ============================================================================
# SIMULATION
# ============================================================================

SIM_DURATION = 480  # 8 hours in minutes
NUM_SEEDS = 10  # number of independent runs
RANDOM_SEED_BASE = 42

# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = "outputs"
LOG_DIR = f"{OUTPUT_DIR}/logs"
PLOT_DIR = f"{OUTPUT_DIR}/plots"
REPORT_DIR = f"{OUTPUT_DIR}/reports"

# CSV event log columns
EVENT_LOG_COLUMNS = [
    "timestamp",
    "event_type",  # "arrival", "service_start", "service_end"
    "job_id",
    "queue_length_before",
    "service_time",  # None for arrival events
    "operator_id",  # None for arrival events
]

# ============================================================================
# METRICS & CONVERGENCE
# ============================================================================

# Convergence definition: within X% for Y minutes
CONVERGENCE_THRESHOLD_PCT = 5.0  # 5% error
CONVERGENCE_DURATION = 60.0  # 60 minutes stable
