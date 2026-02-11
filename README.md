# Queue Digital Twin

A SimPy-based simulation project demonstrating a **digital twin** that learns from a **physical plant** simulation.

## Overview

### The Physical Plant (Ground Truth)
A discrete-event simulation representing a real-world queue system:
- **1 FIFO queue** with **2 parallel servers (operators)**
- **Poisson arrivals** (λ = 0.4 jobs/min)
- **Lognormal service times** (mean = 5 min, stdev = 2 min)
- **Operator fatigue** (20% increase in service time over 8-hour shift)
- **Event logs** tracking arrivals, service start/end, queue lengths

### The Digital Twin
A parallel SimPy model that:
- Starts with **incorrect parameter estimates** (intentionally wrong)
- **Observes events** from the plant in near-real-time
- **Learns and adapts** using rolling maximum likelihood estimation (MLE) + exponential moving average (EMA)
- **Predicts** queue lengths and wait times based on current estimates
- Detects **fatigue drift** automatically

## Key Features

✅ **Event-driven synchronization** – Twin updates on every arrival/completion  
✅ **Rolling window MLE** – Uses last N=200 observations for robust estimation  
✅ **EMA smoothing** – Exponential moving average reduces noise (α=0.1)  
✅ **Fatigue detection** – Automatically learns service time drift  
✅ **Comprehensive metrics** – RMSE, MAE, convergence time, parameter errors  
✅ **Visualizations** – Parameter convergence plots, queue length traces  
✅ **Reproducible** – Configurable random seeds, batch runs (10 seeds by default)  

## Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLANT_ARRIVAL_RATE` | 0.4 | Jobs per minute (Poisson) |
| `PLANT_SERVICE_MEAN` | 5.0 | Mean service time (minutes) |
| `PLANT_SERVICE_STDEV` | 2.0 | Service time std dev |
| `FATIGUE_RATE` | 0.2 | 20% increase over 480 min shift |
| `NUM_SERVERS` | 2 | Number of parallel operators |
| `TWIN_INITIAL_ARRIVAL_RATE` | 0.30 | Twin's wrong initial guess (true: 0.40) |
| `TWIN_INITIAL_SERVICE_MEAN` | 4.0 | Twin's wrong initial guess (true: 5.0) |
| `SIM_DURATION` | 480 | Simulation length (minutes) |
| `NUM_SEEDS` | 10 | Number of independent runs |

## Installation

```bash
# Clone the repo
cd queue-digital-twin

# Create virtual environment (optional)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Simulation

```bash
# Run batch simulation (10 independent runs)
python scripts/run_simulation.py

# Or run a single simulation
python -c "from scripts.run_simulation import run_single_simulation; run_single_simulation(0)"
```

### Output Structure

```
outputs/
├── logs/
│   ├── events_plant_run0.csv    # Plant event log (run 0)
│   ├── events_twin_run0.csv     # Twin event log (run 0)
│   └── ...
├── plots/
│   ├── convergence_run0.png     # Parameter convergence (3 subplots)
│   ├── queue_run0.png           # Queue length trace
│   └── ...
└── reports/
    ├── report_run0.json         # Detailed metrics (JSON)
    └── ...
```

## Example Output

```
==================================================
SUMMARY
==================================================

Arrival Rate Error (%):
  Mean: 2.34%
  Std:  1.56%
  Min:  0.18%
  Max:  5.47%

Service Mean Error (%):
  Mean: 3.12%
  Std:  2.01%
  Min:  0.91%
  Max:  7.23%

Convergence Times (runs that converged):
  Count: 9 / 10
  Mean: 142.3 min
  Min:  87.5 min
  Max:  203.1 min
```

## Project Structure

```
queue-digital-twin/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # All configuration
├── src/
│   ├── __init__.py
│   ├── plant_model.py           # Ground truth plant
│   ├── twin_model.py            # Digital twin
│   ├── event_log.py             # Event logging
│   ├── estimator.py             # MLE + EMA estimators
│   └── metrics.py               # Metrics & reporting
├── scripts/
│   └── run_simulation.py         # Main orchestration
├── outputs/                     # Simulation outputs (auto-created)
│   ├── logs/
│   ├── plots/
│   └── reports/
└── tests/
    └── test_basic.py            # Unit tests
```

## Running Tests

```bash
pytest tests/test_basic.py -v
```

## Key Algorithms

### Arrival Rate Estimation
```
λ̂(t) = EMA( 1 / mean_interarrival, α=0.1 )
```
Updates on every arrival event.

### Service Time Estimation
```
μ̂(t) = EMA( mean(last 200 services), α=0.1 )
σ̂(t) = EMA( stdev(last 200 services), α=0.1 )
```
Updates on every completion event.

### Fatigue Detection
Compares early vs recent service times to detect linear drift:
```
fatigue_rate = (mean_recent - mean_early) / mean_early / elapsed_time
```

### Queue Length Prediction (M/M/c approximation)
```
ρ = λ̂ / (c * μ̂)
L_q ≈ (ρ^c / (c!(1-ρ))) / (sum of Erlang B formula terms)
```

## Dependencies

- **simpy** – Discrete event simulation
- **numpy** – Numerical computations
- **pandas** – Data manipulation
- **matplotlib** – Plotting
- **scipy** – Statistical functions

## References

- Ross, S. M. (2014). *Introduction to Probability Models*
- Hopp, W. J., & Spearman, M. L. (2011). *Factory Physics*
- Kleijnen, J. P. (1995). *Verification and validation of simulation models*

## License

MIT License

## Contact

Created as a demonstration of digital twin technology using SimPy.
