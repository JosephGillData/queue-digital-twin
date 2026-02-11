"""
Estimator for the digital twin.
Uses rolling window MLE and exponential moving average (EMA) smoothing.
"""

from collections import deque
from typing import Tuple, Optional
import numpy as np
import config


class RollingEstimator:
    """Rolling window estimator with EMA smoothing."""

    def __init__(
        self,
        window_size: int = config.ROLLING_WINDOW_SIZE,
        ema_alpha: float = config.EMA_ALPHA,
    ):
        """Initialize rolling estimator.

        Args:
            window_size: Size of rolling window for MLE
            ema_alpha: EMA smoothing factor (0-1), lower = smoother
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.data_window = deque(maxlen=window_size)
        
        # EMA state
        self.ema_mean: Optional[float] = None
        self.ema_stdev: Optional[float] = None

    def add_observation(self, value: float):
        """Add a new observation.

        Args:
            value: Observation value
        """
        self.data_window.append(value)
        self._update_ema()

    def _update_ema(self):
        """Update EMA with current window statistics."""
        if len(self.data_window) < 2:
            return

        current_mean = np.mean(self.data_window)
        current_stdev = np.std(self.data_window)

        if self.ema_mean is None:
            self.ema_mean = current_mean
            self.ema_stdev = current_stdev
        else:
            self.ema_mean = (
                self.ema_alpha * current_mean +
                (1 - self.ema_alpha) * self.ema_mean
            )
            self.ema_stdev = (
                self.ema_alpha * current_stdev +
                (1 - self.ema_alpha) * self.ema_stdev
            )

    def get_estimate(self) -> Tuple[float, float]:
        """Get current MLE + EMA estimate.

        Returns:
            (estimated_mean, estimated_stdev)
        """
        if self.ema_mean is None:
            return 0.0, 0.0

        return self.ema_mean, self.ema_stdev

    def get_mle_only(self) -> Tuple[float, float]:
        """Get raw MLE without EMA (current window only).

        Returns:
            (mean, stdev)
        """
        if len(self.data_window) < 1:
            return 0.0, 0.0

        mean = np.mean(self.data_window)
        stdev = np.std(self.data_window)
        return mean, stdev


class ArrivalRateEstimator:
    """Estimate arrival rate from interarrival times."""

    def __init__(
        self,
        window_size: int = config.ROLLING_WINDOW_SIZE,
        ema_alpha: float = config.EMA_ALPHA,
    ):
        """Initialize arrival rate estimator.

        Args:
            window_size: Size of rolling window
            ema_alpha: EMA smoothing factor
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.interarrival_window = deque(maxlen=window_size)
        
        self.ema_lambda: Optional[float] = None
        self.last_arrival_time: Optional[float] = None

    def observe_arrival(self, timestamp: float) -> bool:
        """Observe an arrival event.

        Args:
            timestamp: Arrival timestamp

        Returns:
            True if interarrival time was recorded
        """
        if self.last_arrival_time is not None:
            interarrival = timestamp - self.last_arrival_time
            self.interarrival_window.append(interarrival)
            self._update_ema()

        self.last_arrival_time = timestamp
        return len(self.interarrival_window) > 0

    def _update_ema(self):
        """Update EMA estimate of lambda."""
        if len(self.interarrival_window) == 0:
            return

        mean_interarrival = np.mean(self.interarrival_window)
        # lambda = 1 / mean_interarrival (jobs per unit time)
        current_lambda = 1.0 / mean_interarrival if mean_interarrival > 0 else 0.0

        if self.ema_lambda is None:
            self.ema_lambda = current_lambda
        else:
            self.ema_lambda = (
                self.ema_alpha * current_lambda +
                (1 - self.ema_alpha) * self.ema_lambda
            )

    def get_estimate(self) -> float:
        """Get current estimated arrival rate (jobs per time unit).

        Returns:
            Estimated lambda
        """
        if self.ema_lambda is None:
            return 0.0

        return self.ema_lambda

    def get_mle_only(self) -> float:
        """Get raw MLE without EMA.

        Returns:
            Estimated lambda from current window
        """
        if len(self.interarrival_window) == 0:
            return 0.0

        mean_interarrival = np.mean(self.interarrival_window)
        return 1.0 / mean_interarrival if mean_interarrival > 0 else 0.0
