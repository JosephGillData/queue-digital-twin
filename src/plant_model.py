"""
Plant model: the ground truth simulation.
Represents the real physical queue system with arrivals, service, and fatigue.
"""

import simpy
import numpy as np
from typing import Optional
import config
from src.event_log import EventLog, Event


class Plant:
    """Ground truth queue system with 2 servers and operator fatigue."""

    def __init__(
        self,
        env: simpy.Environment,
        event_log: EventLog,
        arrival_rate: float = config.PLANT_ARRIVAL_RATE,
        service_mean: float = config.PLANT_SERVICE_MEAN,
        service_stdev: float = config.PLANT_SERVICE_STDEV,
        num_servers: int = config.NUM_SERVERS,
        fatigue_rate: float = config.FATIGUE_RATE,
        shift_duration: float = config.SHIFT_DURATION,
        random_seed: Optional[int] = None,
    ):
        """Initialize plant model.

        Args:
            env: SimPy environment
            event_log: EventLog instance for recording events
            arrival_rate: Poisson arrival rate (jobs per time unit)
            service_mean: Mean service time
            service_stdev: Service time std dev
            num_servers: Number of parallel servers
            fatigue_rate: Fatigue rate (fractional increase over shift)
            shift_duration: Shift duration in time units
            random_seed: Random seed for reproducibility
        """
        self.env = env
        self.event_log = event_log
        self.arrival_rate = arrival_rate
        self.service_mean = service_mean
        self.service_stdev = service_stdev
        self.num_servers = num_servers
        self.fatigue_rate = fatigue_rate
        self.shift_duration = shift_duration
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # SimPy resources
        self.servers = simpy.Resource(env, capacity=num_servers)
        self.queue = []
        
        # Job tracking
        self.job_counter = 0
        self.arrival_times = {}  # job_id -> arrival time
        
        # Metrics
        self.completed_jobs = 0

    def get_service_mean(self) -> float:
        """Get current service mean (with fatigue).

        Returns:
            Service mean at current simulation time
        """
        t = self.env.now
        fatigue_factor = 1.0 + self.fatigue_rate * (t / self.shift_duration)
        return self.service_mean * fatigue_factor

    def arrival_process(self):
        """SimPy process: generate job arrivals."""
        while True:
            # Exponential interarrival time
            interarrival = np.random.exponential(1.0 / self.arrival_rate)
            yield self.env.timeout(interarrival)

            job_id = self.job_counter
            self.job_counter += 1
            self.arrival_times[job_id] = self.env.now

            # Log arrival
            queue_len = len(self.queue) + (self.num_servers - len(self.servers.users))
            event = Event(
                timestamp=self.env.now,
                event_type="arrival",
                job_id=job_id,
                queue_length_before=queue_len,
            )
            self.event_log.log_event(event)

            # Start service process for this job
            self.env.process(self.service_process(job_id))

    def service_process(self, job_id: int):
        """SimPy process: service a single job.

        Args:
            job_id: Unique job identifier
        """
        # Request a server
        with self.servers.request() as req:
            yield req
            
            # Log service start
            queue_len = len(self.queue)
            operator_id = len(self.servers.users) - 1  # Rough assignment
            event = Event(
                timestamp=self.env.now,
                event_type="service_start",
                job_id=job_id,
                queue_length_before=queue_len,
                operator_id=operator_id,
            )
            self.event_log.log_event(event)

            # Sample service time (lognormal)
            service_time = np.random.lognormal(
                mean=np.log(self.get_service_mean()),
                sigma=self.service_stdev
            )
            
            # Service duration
            yield self.env.timeout(service_time)

            # Log service completion
            queue_len = len(self.servers.users)
            event = Event(
                timestamp=self.env.now,
                event_type="service_end",
                job_id=job_id,
                queue_length_before=queue_len,
                service_time=service_time,
                operator_id=operator_id,
            )
            self.event_log.log_event(event)
            
            self.completed_jobs += 1

    def run(self):
        """Start the plant simulation."""
        self.env.process(self.arrival_process())
