"""
Event logging utilities for both plant and twin models.
Stores events to CSV and maintains in-memory log.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd
import config


@dataclass
class Event:
    """Represents a single event in the queue system."""
    timestamp: float
    event_type: str  # "arrival", "service_start", "service_end"
    job_id: int
    queue_length_before: int
    service_time: Optional[float] = None
    operator_id: Optional[int] = None


class EventLog:
    """Manages event logging to CSV and in-memory storage."""

    def __init__(self, output_dir: str = config.LOG_DIR, run_id: str = "default"):
        """Initialize event logger.

        Args:
            output_dir: Directory to store CSV logs
            run_id: Identifier for this run (used in filename)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = run_id
        self.events: List[Event] = []
        
        # CSV file path
        self.csv_path = self.output_dir / f"events_{run_id}.csv"
        
        # Write header
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with header."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=config.EVENT_LOG_COLUMNS)
            writer.writeheader()

    def log_event(self, event: Event):
        """Log a single event to memory and CSV.

        Args:
            event: Event object to log
        """
        self.events.append(event)
        
        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=config.EVENT_LOG_COLUMNS)
            writer.writerow(asdict(event))

    def get_dataframe(self) -> pd.DataFrame:
        """Return in-memory events as pandas DataFrame."""
        if not self.events:
            return pd.DataFrame(columns=config.EVENT_LOG_COLUMNS)
        
        return pd.DataFrame([asdict(e) for e in self.events])

    def get_events_since(self, timestamp: float) -> List[Event]:
        """Get all events after a given timestamp.

        Args:
            timestamp: Filter events after this time

        Returns:
            List of events
        """
        return [e for e in self.events if e.timestamp > timestamp]

    def get_arrivals(self) -> List[Event]:
        """Get all arrival events."""
        return [e for e in self.events if e.event_type == "arrival"]

    def get_completions(self) -> List[Event]:
        """Get all service completion events."""
        return [e for e in self.events if e.event_type == "service_end"]

    def save_csv(self) -> str:
        """Return path to CSV file."""
        return str(self.csv_path)
