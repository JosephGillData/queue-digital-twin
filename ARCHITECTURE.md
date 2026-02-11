# Architecture & Design Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION ORCHESTRATOR                  │
│                   (scripts/run_simulation.py)               │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼──────────┐        ┌──────▼────────────┐
        │  PLANT (Ground   │        │  DIGITAL TWIN    │
        │  Truth Model)    │        │  (Learner)       │
        │                  │        │                  │
        │ • Arrivals       │        │ • Same structure │
        │ • 2 Operators    │        │ • Wrong params   │
        │ • Service times  │        │ • Updates from   │
        │ • Operator       │        │   plant logs     │
        │   fatigue        │        │ • MLE + EMA      │
        │                  │        │   learning       │
        └────────┬─────────┘        └──────┬───────────┘
                 │                         │
                 │  Events flow            │  Sync & Learn
                 │                         │
                 └────────────┬────────────┘
                              │
                    ┌─────────▼────────────┐
                    │   EVENT LOG          │
                    │  (event_log.py)      │
                    │                      │
                    │ • CSV storage        │
                    │ • In-memory list     │
                    │ • Filter methods     │
                    └─────────┬────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
        ┌───────▼──────────┐        ┌───────▼────────┐
        │  ESTIMATORS      │        │  METRICS &     │
        │  (estimator.py)  │        │  REPORTING     │
        │                  │        │ (metrics.py)   │
        │ • Rolling MLE    │        │                │
        │ • EMA smoothing  │        │ • Errors       │
        │ • Arrival rate   │        │ • Convergence  │
        │ • Service time   │        │ • RMSE, MAE    │
        └──────────────────┘        │ • Plots        │
                                    │ • JSON reports │
                                    └────────────────┘
```

## Component Details

### 1. Plant Model (`src/plant_model.py`)
**Role:** Simulates the real-world queue system

**Key Features:**
- M/M/2 queue (Poisson arrivals, lognormal service)
- Arrival process: exponential interarrival times (λ=0.4 jobs/min)
- Service: lognormal(mean=5, stdev=2) per operator
- **Operator fatigue**: service_mean(t) = base_mean × (1 + 0.2 × t/480)
- Event logging: every arrival, start, completion

**Equation:**
```
Service time ~ LogNormal(μ, σ)
where μ, σ are adjusted for fatigue factor
```

### 2. Digital Twin (`src/twin_model.py`)
**Role:** Learns system dynamics from plant observations

**Initial State (Wrong):**
- λ_init = 0.30 (true: 0.40)
- mean_init = 4.0 (true: 5.0)
- stdev_init = 1.5 (true: 2.0)
- Fatigue = OFF (learns later)

**Learning Mechanism:**
- Event-driven updates on every arrival/completion
- Periodic sync (every 5 min) for metrics reconciliation
- Automatic fatigue detection via service time comparison

### 3. Estimators (`src/estimator.py`)

#### ArrivalRateEstimator
```
λ̂(t) = EMA[1/E[interarrival], α=0.1]

- Maintains rolling window of last N=200 interarrival times
- Computes MLE: λ = 1 / mean(window)
- Applies EMA smoothing: λ̂ = α×λ + (1-α)×λ̂_prev
```

#### RollingEstimator (Service Time)
```
μ̂(t) = EMA[mean(last 200 services), α=0.1]
σ̂(t) = EMA[std(last 200 services), α=0.1]

- Independent for mean and stdev
- Rolling window avoids "forgetting" old data while adapting to drift
- EMA provides smooth transitions
```

### 4. Event Log (`src/event_log.py`)
**Role:** Records and stores all simulation events

**CSV Schema:**
```
timestamp | event_type    | job_id | queue_length | service_time | operator_id
----------|---------------|--------|--------------|--------------|----------
0.5       | arrival       | 0      | 0            | null         | null
0.8       | service_start | 0      | 0            | null         | 0
5.7       | service_end   | 0      | 1            | 4.9          | 0
```

### 5. Metrics & Reporting (`src/metrics.py`)
**Computed Metrics:**
```
Parameter Errors:
  - |λ̂ - λ_true| / λ_true × 100 (%)
  - |μ̂ - μ_true| / μ_true × 100 (%)
  - |σ̂ - σ_true| / σ_true × 100 (%)

Convergence:
  - Time until error < threshold (5%) for duration (60 min)
  - Percentage of runs that converge

Queue Metrics:
  - RMSE of predicted queue vs observed
  - Mean/stdev of wait times
  - Throughput error vs arrival rate

Performance:
  - Total jobs completed
  - System utilization
  - Server efficiency
```

## Learning Algorithm: Rolling MLE + EMA

### Why This Approach?

1. **Rolling Window (MLE)**
   - ✅ Captures recent trend without full recalculation
   - ✅ Robust: uses only actual observed values
   - ❌ Slow to detect step changes (window lag)

2. **EMA Smoothing**
   - ✅ Fast adaptation to small changes
   - ✅ Reduces noise from stochastic variations
   - ❌ Can overshoot if α too high

**Combined:** Rolling MLE (robust) + EMA (responsive) = Best of both worlds

### Example: Service Time Learning
```
t=0:    μ̂ = 4.0 (initial wrong guess)
t=5:    μ̂ = EMA[4.8, α=0.1] = 0.1×4.8 + 0.9×4.0 = 4.08
t=10:   μ̂ = EMA[5.1, α=0.1] = 0.1×5.1 + 0.9×4.08 = 4.17
t=20:   μ̂ = EMA[5.2, α=0.1] = 0.1×5.2 + 0.9×4.33 = 4.40
...
t=100:  μ̂ → 5.0 (true value)
```

## Data Flow

### Per Simulation Run:

1. **Initialization**
   - Create plant with TRUE parameters
   - Create twin with WRONG parameters
   - Create event log

2. **Per Time Step (SimPy Event)**
   - Plant generates event (arrival/completion)
   - Event logged to CSV
   - Twin reads event
   - Twin updates estimates (if arrival/completion)
   - Metrics recorded

3. **Every SYNC_INTERVAL (5 min)**
   - Twin reconciles state
   - Check for fatigue drift
   - Record twin estimates to history

4. **After Simulation (t=480 min)**
   - Compute final parameter errors
   - Determine convergence
   - Generate plots
   - Save JSON report

## Configuration Parameters Explained

| Parameter | Default | Tuning Effect |
|-----------|---------|---|
| `ROLLING_WINDOW_SIZE` | 200 | ↑ Slower adaptation, more robust; ↓ Faster but noisier |
| `EMA_ALPHA` | 0.1 | ↑ More responsive, more overshoot; ↓ Smoother, lags more |
| `SYNC_INTERVAL` | 5.0 | ↑ Less overhead, coarser tracking; ↓ More overhead |
| `FATIGUE_RATE` | 0.2 | ↑ Stronger drift, easier to detect; ↓ Subtle, harder to learn |
| `SIM_DURATION` | 480 | ↑ More time to converge, more cost; ↓ Faster runs |

## Convergence Definition

Twin is "converged" when:
1. Arrival rate error < 5% AND service mean error < 5%
2. Both conditions hold for at least 60 consecutive minutes
3. After that point, estimates remain stable

## Expected Performance (from config)

**Plant Parameters:**
- λ = 0.4 jobs/min
- μ = 5.0 min
- σ = 2.0 min
- Fatigue: +20% over 480 min

**Expected Convergence:**
- **Arrival rate:** 5-10% error → converge by ~100 min
- **Service mean:** 5-15% error → converge by ~150 min
- **Fatigue detection:** By ~200 min (requires comparing enough windows)

## Future Enhancements

1. **Bayesian Parameter Estimation**
   - Prior distributions on λ, μ
   - Posterior updates with each observation
   - Faster convergence for high-confidence priors

2. **Kalman Filter**
   - Optimal state estimation under noise
   - Better handling of fatigue as state variable

3. **Queue Length Prediction**
   - Currently uses Erlang formula approximation
   - Could use actual twin queue dynamics

4. **Multi-shift Simulation**
   - Reset fatigue each day
   - Learn shift-specific patterns

5. **Visualization**
   - Interactive Plotly dashboards
   - Real-time twin vs plant comparison
   - Parameter trace animations

6. **Advanced Learning**
   - Deep learning to forecast arrivals/service
   - Detect distributional changes (non-stationary)
   - Multi-variate analysis (time-of-day effects)
