# Quick Start Guide

## 1. Install Dependencies

```bash
# Navigate to project directory
cd queue-digital-twin

# Install all required packages
pip install -r requirements.txt
```

## 2. Run the Full Simulation (10 independent runs)

```bash
python scripts/run_simulation.py
```

This will:
- Simulate 10 independent runs (configurable in `config.py`)
- Each run simulates 480 minutes (8 hours)
- Generate event logs, plots, and JSON reports
- Print summary statistics to console

**Expected runtime:** ~2-5 minutes (depending on your machine)

## 3. Check the Results

After running, outputs are organized in `outputs/`:

```
outputs/
├── logs/           # CSV event logs (plant and twin)
├── plots/          # Convergence and queue length plots
└── reports/        # JSON reports with detailed metrics
```

### View a specific report:
```bash
cat outputs/reports/report_run0.json
```

### View convergence plot:
Open `outputs/plots/convergence_run0.png` in your image viewer

## 4. Run Tests

```bash
pytest tests/test_basic.py -v
```

## 5. Customize Parameters

Edit `config.py` to change:
- **Plant parameters** (arrival rate, service times, fatigue)
- **Twin's initial wrong estimates**
- **Learning rates** (window size, EMA factor)
- **Simulation duration**
- **Number of runs**

Example: To run just 1 shorter simulation (100 minutes):
```python
# config.py
SIM_DURATION = 100  # Change from 480
NUM_SEEDS = 1       # Change from 10
```

Then run:
```bash
python scripts/run_simulation.py
```

## 6. Understand the Output

### Console Output Example:
```
==================================================
SUMMARY
==================================================

Arrival Rate Error (%):
  Mean: 2.34%    ← Twin's estimate vs true
  
Service Mean Error (%):
  Mean: 3.12%    ← How well twin learned service time
  
Convergence Times (runs that converged):
  Count: 9 / 10   ← How many runs converged
  Mean: 142.3 min ← Average time to converge
```

### What the plots show:

1. **convergence_run0.png**
   - Top: Arrival rate learning (red dashed = true, blue = twin's estimate)
   - Middle: Service mean learning
   - Bottom: Service stdev learning
   - Shows how quickly twin learns true parameters

2. **queue_run0.png**
   - Queue length over time
   - Blue = arrivals, Orange = completions

3. **report_run0.json**
   - Detailed metrics in JSON format
   - Parameter errors, convergence time, throughput, wait times

## 7. Advanced: Run a Single Simulation Programmatically

```python
from scripts.run_simulation import run_single_simulation

# Run run 0
report = run_single_simulation(run_id=0)

# Print results
print(f"Arrival rate error: {report['parameter_errors']['arrival_rate_error_pct']:.2f}%")
print(f"Converged: {report['convergence']['converged']}")
if report['convergence']['converged']:
    print(f"Convergence time: {report['convergence']['convergence_time']:.1f} min")
```

## 8. Troubleshooting

**"ModuleNotFoundError: No module named simpy"**
- Run: `pip install -r requirements.txt`

**"Outputs folder exists but no files generated"**
- Check that simulation actually ran (should print progress)
- Verify `config.SIM_DURATION > 0`

**Plots look empty**
- This can happen if not enough events occurred
- Try increasing `PLANT_ARRIVAL_RATE` in config.py

**Want to understand the code?**
- Read [src/plant_model.py](src/plant_model.py) first (the "real world")
- Then [src/twin_model.py](src/twin_model.py) (the learner)
- Then [src/estimator.py](src/estimator.py) (learning algorithm)

## Next Steps

1. **Modify the learning algorithm** in `src/estimator.py` (try different EMA factors, window sizes)
2. **Add new metrics** to `src/metrics.py` (e.g., Erlang C queue predictions)
3. **Extend fatigue model** in `src/plant_model.py` (non-linear fatigue, operator breaks)
4. **Add visualization** (plotly for interactive plots, matplotlib animations)
5. **Implement Kalman filter** or Bayesian parameter estimation for faster convergence

Happy simulating! 🚀
