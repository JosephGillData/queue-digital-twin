"""
Basic tests for the digital twin simulation.
"""

import pytest
import simpy
import numpy as np
from pathlib import Path
import config
from src.plant_model import Plant
from src.event_log import EventLog, Event
from src.estimator import ArrivalRateEstimator, RollingEstimator


def test_event_log_creation(tmp_path):
    """Test event log creation and CSV export."""
    log_dir = str(tmp_path)
    event_log = EventLog(output_dir=log_dir, run_id="test")
    
    # Add some events
    event1 = Event(
        timestamp=0.0,
        event_type="arrival",
        job_id=1,
        queue_length_before=0,
    )
    event_log.log_event(event1)
    
    assert len(event_log.events) == 1
    assert event_log.get_arrivals()[0].job_id == 1


def test_arrival_rate_estimator():
    """Test arrival rate estimator."""
    estimator = ArrivalRateEstimator(window_size=10, ema_alpha=0.1)
    
    # Simulate regular arrivals (1 per time unit)
    for i in range(20):
        estimator.observe_arrival(float(i))
    
    est_lambda = estimator.get_estimate()
    # Expected: ~1.0 job per time unit
    assert 0.8 < est_lambda < 1.2, f"Expected ~1.0, got {est_lambda}"


def test_rolling_estimator():
    """Test rolling window estimator."""
    estimator = RollingEstimator(window_size=50, ema_alpha=0.1)
    
    # Add observations from normal distribution
    np.random.seed(42)
    true_mean = 5.0
    true_stdev = 1.0
    
    for _ in range(100):
        value = np.random.normal(true_mean, true_stdev)
        estimator.add_observation(value)
    
    est_mean, est_stdev = estimator.get_estimate()
    
    assert abs(est_mean - true_mean) < 0.5, f"Expected mean ~{true_mean}, got {est_mean}"
    assert abs(est_stdev - true_stdev) < 0.5, f"Expected stdev ~{true_stdev}, got {est_stdev}"


def test_plant_basic_run(tmp_path):
    """Test plant model runs without errors."""
    log_dir = str(tmp_path)
    event_log = EventLog(output_dir=log_dir, run_id="test")
    
    env = simpy.Environment()
    plant = Plant(
        env=env,
        event_log=event_log,
        arrival_rate=0.5,
        service_mean=5.0,
        service_stdev=2.0,
        num_servers=2,
        random_seed=42,
    )
    plant.run()
    
    # Run for 100 time units
    env.run(until=100)
    
    # Check that events were logged
    arrivals = event_log.get_arrivals()
    completions = event_log.get_completions()
    
    assert len(arrivals) > 0, "Should have arrivals"
    assert len(completions) > 0, "Should have completions"


def test_config_parameters():
    """Test that config parameters are reasonable."""
    assert config.PLANT_ARRIVAL_RATE > 0
    assert config.PLANT_SERVICE_MEAN > 0
    assert config.NUM_SERVERS > 0
    assert config.SIM_DURATION > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
