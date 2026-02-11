"""
Main orchestration script for queue digital twin simulation.
Runs plant and twin in parallel and generates reports.
"""

from pathlib import Path
import sys

# Ensure root directory is in path for config import
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import simpy
import config
from src.plant_model import Plant
from src.twin_model import DigitalTwin
from src.event_log import EventLog
from src.metrics import MetricsComputer


def run_single_simulation(run_id: int = 0) -> dict:
    """Run a single plant + twin simulation.

    Args:
        run_id: Identifier for this run (seed)

    Returns:
        Dictionary with results
    """
    # Set random seed
    seed = config.RANDOM_SEED_BASE + run_id
    np.random.seed(seed)
    
    # Create SimPy environments
    env = simpy.Environment()
    
    # Create output directories
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create event logs
    plant_log = EventLog(output_dir=config.LOG_DIR, run_id=f"plant_run{run_id}")
    twin_log = EventLog(output_dir=config.LOG_DIR, run_id=f"twin_run{run_id}")
    
    # Create and start plant
    plant = Plant(
        env=env,
        event_log=plant_log,
        arrival_rate=config.PLANT_ARRIVAL_RATE,
        service_mean=config.PLANT_SERVICE_MEAN,
        service_stdev=config.PLANT_SERVICE_STDEV,
        num_servers=config.NUM_SERVERS,
        fatigue_rate=config.FATIGUE_RATE,
        shift_duration=config.SHIFT_DURATION,
        random_seed=seed,
    )
    plant.run()
    
    # Create and start digital twin
    twin = DigitalTwin(
        env=env,
        plant_event_log=plant_log,
        twin_event_log=twin_log,
        initial_arrival_rate=config.TWIN_INITIAL_ARRIVAL_RATE,
        initial_service_mean=config.TWIN_INITIAL_SERVICE_MEAN,
        initial_service_stdev=config.TWIN_INITIAL_SERVICE_STDEV,
        num_servers=config.NUM_SERVERS,
        random_seed=seed + 1000,
    )
    twin.run()
    
    # Track twin estimates over time
    twin_history = []
    
    def record_twin_state():
        """Periodic recording of twin state."""
        while True:
            yield env.timeout(config.SYNC_INTERVAL)
            twin_history.append({
                "timestamp": env.now,
                "est_lambda": twin.estimated_arrival_rate,
                "est_mean": twin.estimated_service_mean,
                "est_stdev": twin.estimated_service_stdev,
                "est_fatigue": twin.estimated_fatigue_rate,
            })
    
    env.process(record_twin_state())
    
    # Run simulation
    print(f"[Run {run_id}] Starting simulation...")
    env.run(until=config.SIM_DURATION)
    print(f"[Run {run_id}] Simulation complete.")
    
    # Compute metrics
    plant_params = {
        "arrival_rate": config.PLANT_ARRIVAL_RATE,
        "service_mean": config.PLANT_SERVICE_MEAN,
        "service_stdev": config.PLANT_SERVICE_STDEV,
    }
    
    metrics_computer = MetricsComputer(
        plant_log=plant_log,
        twin_log=twin_log,
        plant_params=plant_params,
        twin_history=twin_history,
        output_dir=config.REPORT_DIR,
        run_id=f"run{run_id}",
    )
    
    report = metrics_computer.generate_report(config.SIM_DURATION)
    metrics_computer.save_report_json(report)
    metrics_computer.plot_parameter_convergence()
    metrics_computer.plot_queue_length()
    
    print(f"[Run {run_id}] Report saved.")
    
    return report


def run_batch_simulation(num_seeds: int = config.NUM_SEEDS):
    """Run multiple simulations with different seeds.

    Args:
        num_seeds: Number of independent runs
    """
    print(f"Queue Digital Twin Simulation")
    print(f"=" * 50)
    print(f"Configuration:")
    print(f"  Plant arrival rate: {config.PLANT_ARRIVAL_RATE} jobs/min")
    print(f"  Plant service mean: {config.PLANT_SERVICE_MEAN} min")
    print(f"  Plant service stdev: {config.PLANT_SERVICE_STDEV} min")
    print(f"  Fatigue rate: {config.FATIGUE_RATE * 100}% over {config.SHIFT_DURATION} min")
    print(f"  Simulation duration: {config.SIM_DURATION} min")
    print(f"  Number of runs: {num_seeds}")
    print(f"=" * 50)
    
    all_reports = []
    
    for run_id in range(num_seeds):
        print(f"\n--- Run {run_id + 1}/{num_seeds} ---")
        report = run_single_simulation(run_id)
        all_reports.append(report)
    
    # Summary statistics
    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    
    arrival_errors = [r["parameter_errors"]["arrival_rate_error_pct"] for r in all_reports]
    mean_errors = [r["parameter_errors"]["service_mean_error_pct"] for r in all_reports]
    stdev_errors = [r["parameter_errors"]["service_stdev_error_pct"] for r in all_reports]
    
    print(f"\nArrival Rate Error (%):")
    print(f"  Mean: {np.mean(arrival_errors):.2f}%")
    print(f"  Std:  {np.std(arrival_errors):.2f}%")
    print(f"  Min:  {np.min(arrival_errors):.2f}%")
    print(f"  Max:  {np.max(arrival_errors):.2f}%")
    
    print(f"\nService Mean Error (%):")
    print(f"  Mean: {np.mean(mean_errors):.2f}%")
    print(f"  Std:  {np.std(mean_errors):.2f}%")
    print(f"  Min:  {np.min(mean_errors):.2f}%")
    print(f"  Max:  {np.max(mean_errors):.2f}%")
    
    print(f"\nService Stdev Error (%):")
    print(f"  Mean: {np.mean(stdev_errors):.2f}%")
    print(f"  Std:  {np.std(stdev_errors):.2f}%")
    print(f"  Min:  {np.min(stdev_errors):.2f}%")
    print(f"  Max:  {np.max(stdev_errors):.2f}%")
    
    convergence_times = [
        r["convergence"]["convergence_time"] 
        for r in all_reports 
        if r["convergence"]["converged"]
    ]
    
    if convergence_times:
        print(f"\nConvergence Times (runs that converged):")
        print(f"  Count: {len(convergence_times)} / {num_seeds}")
        print(f"  Mean: {np.mean(convergence_times):.1f} min")
        print(f"  Min:  {np.min(convergence_times):.1f} min")
        print(f"  Max:  {np.max(convergence_times):.1f} min")
    else:
        print(f"\nConvergence: No runs converged to threshold")
    
    print(f"\n{'=' * 50}")
    print(f"Outputs:")
    print(f"  Event logs: {config.LOG_DIR}")
    print(f"  Reports:   {config.REPORT_DIR}")
    print(f"  Plots:     {config.OUTPUT_DIR}/plots")
    print(f"{'=' * 50}")


def main():
    """Entry point for the simulation."""
    run_batch_simulation(num_seeds=config.NUM_SEEDS)


if __name__ == "__main__":
    main()
