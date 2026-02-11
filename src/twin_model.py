"""
Digital Twin model: learns from plant observations.
Starts with incorrect parameters and updates via rolling MLE + EMA.
"""

import simpy
import numpy as np
from typing import Optional, List
import config
from src.estimator import RollingEstimator, ArrivalRateEstimator
from src.event_log import Event, EventLog


class DigitalTwin:
    """Digital twin that learns plant dynamics from event logs."""

    def __init__(
        self,
        env: simpy.Environment,
        plant_event_log: EventLog,
        twin_event_log: EventLog,
        initial_arrival_rate: float = config.TWIN_INITIAL_ARRIVAL_RATE,
        initial_service_mean: float = config.TWIN_INITIAL_SERVICE_MEAN,
        initial_service_stdev: float = config.TWIN_INITIAL_SERVICE_STDEV,
        num_servers: int = config.NUM_SERVERS,
        random_seed: Optional[int] = None,
    ):
        """Initialize digital twin.

        Args:
            env: SimPy environment
            plant_event_log: EventLog from the plant
            twin_event_log: EventLog for twin's own events
            initial_arrival_rate: Twin's initial (wrong) arrival rate estimate
            initial_service_mean: Twin's initial (wrong) service mean
            initial_service_stdev: Twin's initial (wrong) service stdev
            num_servers: Number of servers (same as plant)
            random_seed: Random seed
        """
        self.env = env
        self.plant_event_log = plant_event_log
        self.twin_event_log = twin_event_log
        self.num_servers = num_servers
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Estimated parameters (initialized incorrectly)
        self.estimated_arrival_rate = initial_arrival_rate
        self.estimated_service_mean = initial_service_mean
        self.estimated_service_stdev = initial_service_stdev
        self.estimated_fatigue_rate = 0.0  # Twin assumes no fatigue initially
        
        # Estimators (learn from observations)
        self.arrival_estimator = ArrivalRateEstimator()
        self.service_estimator = RollingEstimator()
        
        # SimPy resources (mirror of plant)
        self.servers = simpy.Resource(env, capacity=num_servers)
        
        # Job tracking
        self.job_counter = 0
        self.arrival_times = {}
        self.completed_jobs = 0
        
        # Last sync time (to avoid processing same events twice)
        self.last_sync_time = 0.0
        self.last_processed_event_index = 0

    def sync_with_plant(self):
        """Synchronize with plant observations.
        
        Updates estimated parameters from plant event log.
        """
        plant_events = self.plant_event_log.get_events_since(self.last_sync_time)
        
        # Process arrivals and completions to update estimates
        for event in plant_events:
            if event.event_type == "arrival":
                self.arrival_estimator.observe_arrival(event.timestamp)
                
            elif event.event_type == "service_end" and event.service_time is not None:
                self.service_estimator.add_observation(event.service_time)
        
        # Update estimates
        if self.plant_event_log.get_arrivals():
            self.estimated_arrival_rate = self.arrival_estimator.get_estimate()
        
        if len(self.service_estimator.data_window) > 0:
            mean, stdev = self.service_estimator.get_estimate()
            self.estimated_service_mean = mean
            self.estimated_service_stdev = stdev
        
        # Detect fatigue (simple: check if service mean drifts upward)
        # Compare recent vs early service times
        all_completions = self.plant_event_log.get_completions()
        if len(all_completions) > 100:
            early_services = [
                e.service_time for e in all_completions[:50]
                if e.service_time is not None
            ]
            recent_services = [
                e.service_time for e in all_completions[-50:]
                if e.service_time is not None
            ]
            
            early_mean = np.mean(early_services) if early_services else 0
            recent_mean = np.mean(recent_services) if recent_services else 0
            
            if recent_mean > early_mean * 1.05:  # 5% increase suggests fatigue
                # Estimate fatigue rate
                elapsed = self.env.now
                if elapsed > 0:
                    self.estimated_fatigue_rate = (recent_mean - early_mean) / early_mean / (elapsed / 480)
        
        self.last_sync_time = self.env.now

    def get_service_mean(self) -> float:
        """Get current estimated service mean (with learned fatigue).

        Returns:
            Current estimated service mean
        """
        t = self.env.now
        # Shift duration assumed to be 480
        fatigue_factor = 1.0 + self.estimated_fatigue_rate * (t / 480.0)
        return self.estimated_service_mean * fatigue_factor

    def arrival_process(self):
        """SimPy process: generate job arrivals based on estimates."""
        while True:
            # Exponential interarrival based on estimated rate
            if self.estimated_arrival_rate > 0:
                interarrival = np.random.exponential(
                    1.0 / self.estimated_arrival_rate
                )
            else:
                interarrival = 1.0  # Fallback
            
            yield self.env.timeout(interarrival)

            job_id = self.job_counter
            self.job_counter += 1
            self.arrival_times[job_id] = self.env.now

            # Start service
            self.env.process(self.service_process(job_id))

    def service_process(self, job_id: int):
        """SimPy process: service a job based on estimates.

        Args:
            job_id: Job identifier
        """
        with self.servers.request() as req:
            yield req
            
            # Sample service time from estimated distribution
            if self.estimated_service_mean > 0:
                service_time = np.random.lognormal(
                    mean=np.log(self.get_service_mean()),
                    sigma=self.estimated_service_stdev
                )
            else:
                service_time = 1.0  # Fallback
            
            yield self.env.timeout(service_time)
            self.completed_jobs += 1

    def run(self):
        """Start the digital twin."""
        self.env.process(self.arrival_process())
        # Sync periodically
        self.env.process(self.sync_process())

    def sync_process(self):
        """Periodically sync with plant data."""
        while True:
            yield self.env.timeout(config.SYNC_INTERVAL)
            self.sync_with_plant()

    def predict_queue_length(self) -> float:
        """Predict current queue length based on estimates.

        Returns:
            Estimated queue length
        """
        # Rough estimate: (lambda - 2*mu) proportional to queue
        mu = 1.0 / self.get_service_mean() if self.get_service_mean() > 0 else 0.0
        
        if self.estimated_arrival_rate >= 2 * mu:
            # System is overloaded; queue grows
            rho = self.estimated_arrival_rate / (2 * mu)
            queue_est = (rho - 1) / (1 - rho) if rho < 1 else np.inf
            return max(0, queue_est)
        return 0.0

    def predict_wait_time(self) -> float:
        """Predict expected wait time.

        Returns:
            Estimated mean wait time
        """
        queue = self.predict_queue_length()
        mu = 1.0 / self.get_service_mean() if self.get_service_mean() > 0 else 1.0
        return queue / (2 * mu) if mu > 0 else 0.0
