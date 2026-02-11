"""
Metrics computation and reporting for the digital twin project.
Compares plant vs twin, measures convergence, and generates reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from src.event_log import EventLog


class MetricsComputer:
    """Compute performance metrics and generate reports."""

    def __init__(
        self,
        plant_log: EventLog,
        twin_log: EventLog,
        plant_params: Dict,
        twin_history: List[Dict],
        output_dir: str = config.REPORT_DIR,
        run_id: str = "default",
    ):
        """Initialize metrics computer.

        Args:
            plant_log: EventLog from plant
            twin_log: EventLog from twin
            plant_params: Plant parameters (true values)
            twin_history: List of (timestamp, est_lambda, est_mean, est_stdev) dicts
            output_dir: Output directory for reports
            run_id: Identifier for this run
        """
        self.plant_log = plant_log
        self.twin_log = twin_log
        self.plant_params = plant_params
        self.twin_history = twin_history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def compute_queue_lengths(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute queue lengths over time from event logs.

        Returns:
            (timestamps, plant_queue_lengths, twin_queue_lengths)
        """
        # Plant queue lengths
        plant_df = self.plant_log.get_dataframe()
        if plant_df.empty:
            return np.array([]), np.array([]), np.array([])
        
        plant_arrivals = plant_df[plant_df["event_type"] == "arrival"].copy()
        plant_completions = plant_df[plant_df["event_type"] == "service_end"].copy()
        
        # Compute queue at each timestamp
        timestamps = sorted(set(plant_arrivals["timestamp"]) | set(plant_completions["timestamp"]))
        
        plant_queues = []
        for t in timestamps:
            arrivals_by_t = len(plant_arrivals[plant_arrivals["timestamp"] <= t])
            completions_by_t = len(plant_completions[plant_completions["timestamp"] <= t])
            queue_len = arrivals_by_t - min(completions_by_t, config.NUM_SERVERS)
            plant_queues.append(queue_len)
        
        return np.array(timestamps), np.array(plant_queues), np.array([])

    def compute_wait_times(self) -> Tuple[List[float], float, float]:
        """Compute wait times for plant jobs.

        Returns:
            (wait_times, mean_wait, stdev_wait)
        """
        plant_df = self.plant_log.get_dataframe()
        if plant_df.empty:
            return [], 0.0, 0.0
        
        # Match arrivals with completions
        arrivals = plant_df[plant_df["event_type"] == "arrival"].set_index("job_id")
        completions = plant_df[plant_df["event_type"] == "service_end"].set_index("job_id")
        
        wait_times = []
        for job_id in arrivals.index:
            if job_id in completions.index:
                arrival_time = arrivals.loc[job_id, "timestamp"]
                completion_time = completions.loc[job_id, "timestamp"]
                wait = completion_time - arrival_time
                wait_times.append(wait)
        
        if not wait_times:
            return [], 0.0, 0.0
        
        return wait_times, np.mean(wait_times), np.std(wait_times)

    def compute_throughput(self, duration: float) -> Tuple[float, float]:
        """Compute throughput (jobs per time unit).

        Args:
            duration: Simulation duration

        Returns:
            (plant_throughput, theoretical_arrival_rate)
        """
        completions = len(self.plant_log.get_completions())
        plant_throughput = completions / duration if duration > 0 else 0.0
        
        return plant_throughput, self.plant_params["arrival_rate"]

    def compute_parameter_errors(self) -> Dict[str, float]:
        """Compute final parameter estimation errors.

        Returns:
            Dict with arrival_rate_error, service_mean_error, service_stdev_error (in %)
        """
        if not self.twin_history:
            return {
                "arrival_rate_error_pct": np.nan,
                "service_mean_error_pct": np.nan,
                "service_stdev_error_pct": np.nan,
            }
        
        final = self.twin_history[-1]
        
        true_lambda = self.plant_params["arrival_rate"]
        true_mean = self.plant_params["service_mean"]
        true_stdev = self.plant_params["service_stdev"]
        
        errors = {
            "arrival_rate_error_pct": abs(final["est_lambda"] - true_lambda) / true_lambda * 100,
            "service_mean_error_pct": abs(final["est_mean"] - true_mean) / true_mean * 100,
            "service_stdev_error_pct": abs(final["est_stdev"] - true_stdev) / true_stdev * 100,
        }
        
        return errors

    def compute_rmse_queue_length(self) -> float:
        """Compute RMSE of queue length predictions.

        Note: Currently limited since we don't have perfect plant queue tracking.

        Returns:
            RMSE value
        """
        # Placeholder: would require detailed queue tracking
        return 0.0

    def compute_convergence_metrics(self) -> Dict[str, Optional[float]]:
        """Compute convergence statistics.

        Returns:
            Dict with convergence_time, stable_at_time, etc.
        """
        if not self.twin_history or len(self.twin_history) < 2:
            return {"convergence_time": None, "converged": False}
        
        true_lambda = self.plant_params["arrival_rate"]
        true_mean = self.plant_params["service_mean"]
        
        # Check when error falls below threshold
        threshold = config.CONVERGENCE_THRESHOLD_PCT
        stable_duration = config.CONVERGENCE_DURATION
        
        convergence_time = None
        for i, record in enumerate(self.twin_history):
            lambda_error = abs(record["est_lambda"] - true_lambda) / true_lambda * 100
            mean_error = abs(record["est_mean"] - true_mean) / true_mean * 100
            
            if lambda_error < threshold and mean_error < threshold:
                # Check if it stays stable
                future_records = self.twin_history[i:]
                future_times = [r["timestamp"] for r in future_records]
                
                if len(future_times) > 1:
                    elapsed = future_times[-1] - future_times[0]
                    if elapsed >= stable_duration:
                        convergence_time = record["timestamp"]
                        break
        
        return {
            "convergence_time": convergence_time,
            "converged": convergence_time is not None,
            "threshold_pct": threshold,
            "stable_duration": stable_duration,
        }

    def generate_report(self, sim_duration: float) -> Dict:
        """Generate comprehensive metrics report.

        Args:
            sim_duration: Total simulation duration

        Returns:
            Report dictionary
        """
        param_errors = self.compute_parameter_errors()
        convergence = self.compute_convergence_metrics()
        plant_throughput, true_lambda = self.compute_throughput(sim_duration)
        wait_times, mean_wait, stdev_wait = self.compute_wait_times()
        
        report = {
            "run_id": self.run_id,
            "simulation_duration": sim_duration,
            "plant_parameters": self.plant_params,
            "final_estimates": (
                self.twin_history[-1] if self.twin_history else {}
            ),
            "parameter_errors": param_errors,
            "convergence": convergence,
            "throughput": {
                "actual_jobs_completed": len(self.plant_log.get_completions()),
                "plant_throughput": plant_throughput,
                "expected_arrival_rate": true_lambda,
                "throughput_error_pct": (
                    abs(plant_throughput - true_lambda) / true_lambda * 100
                    if true_lambda > 0 else 0
                ),
            },
            "wait_times": {
                "mean_wait": mean_wait,
                "stdev_wait": stdev_wait,
                "samples": len(wait_times),
            },
            "queue": {
                "rmse": self.compute_rmse_queue_length(),
            },
        }
        
        return report

    def save_report_json(self, report: Dict) -> str:
        """Save report as JSON file.

        Args:
            report: Report dictionary

        Returns:
            Path to saved file
        """
        path = self.output_dir / f"report_{self.run_id}.json"
        
        # Handle non-serializable values
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if obj is None or isinstance(obj, float):
                return obj
            return str(obj)
        
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=default_serializer)
        
        return str(path)

    def plot_parameter_convergence(self) -> str:
        """Plot parameter convergence over time.

        Returns:
            Path to saved figure
        """
        if not self.twin_history:
            return ""
        
        df = pd.DataFrame(self.twin_history)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Arrival rate
        true_lambda = self.plant_params["arrival_rate"]
        axes[0].plot(df["timestamp"], df["est_lambda"], label="Twin estimate", linewidth=2)
        axes[0].axhline(true_lambda, color="r", linestyle="--", label="True value", linewidth=2)
        axes[0].set_ylabel("Arrival Rate (jobs/min)")
        axes[0].set_title("Arrival Rate Convergence")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Service mean
        true_mean = self.plant_params["service_mean"]
        axes[1].plot(df["timestamp"], df["est_mean"], label="Twin estimate", linewidth=2)
        axes[1].axhline(true_mean, color="r", linestyle="--", label="True value", linewidth=2)
        axes[1].set_ylabel("Service Mean (min)")
        axes[1].set_title("Service Mean Convergence")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Service stdev
        true_stdev = self.plant_params["service_stdev"]
        axes[2].plot(df["timestamp"], df["est_stdev"], label="Twin estimate", linewidth=2)
        axes[2].axhline(true_stdev, color="r", linestyle="--", label="True value", linewidth=2)
        axes[2].set_ylabel("Service Stdev (min)")
        axes[2].set_xlabel("Simulation Time (min)")
        axes[2].set_title("Service Stdev Convergence")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        path = self.output_dir.parent / "plots" / f"convergence_{self.run_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        
        return str(path)

    def plot_queue_length(self) -> str:
        """Plot queue length over time.

        Returns:
            Path to saved figure
        """
        plant_df = self.plant_log.get_dataframe()
        
        if plant_df.empty:
            return ""
        
        arrivals = plant_df[plant_df["event_type"] == "arrival"]
        completions = plant_df[plant_df["event_type"] == "service_end"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if not arrivals.empty and not completions.empty:
            # Simple visualization: plot arrival and completion times
            ax.scatter(arrivals["timestamp"], arrivals["queue_length_before"], 
                      label="Arrivals", alpha=0.5, s=20)
            ax.scatter(completions["timestamp"], completions["queue_length_before"],
                      label="Completions", alpha=0.5, s=20)
        
        ax.set_xlabel("Simulation Time (min)")
        ax.set_ylabel("Queue Length")
        ax.set_title("Queue Length Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = self.output_dir.parent / "plots" / f"queue_{self.run_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        
        return str(path)
