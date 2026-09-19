import csv
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path


class CSVDataLogger:
    """Episode-scoped CSV logger for force-like vector signals."""

    def __init__(
        self,
        task_name: str,
        log_dir: str | os.PathLike,
        signal_name: str = "contact_force",
        signal_dim: int = 3,
        signal_labels: Iterable[str] | None = None,
    ):
        self.task_name = task_name
        self.log_dir = Path(log_dir)
        self.signal_name = signal_name
        self.signal_dim = int(signal_dim)
        self.signal_labels = (
            list(signal_labels)
            if signal_labels is not None
            else [f"{self.signal_name}_{axis}" for axis in ("x", "y", "z")[: self.signal_dim]]
        )
        if len(self.signal_labels) != self.signal_dim:
            raise ValueError("signal_labels length must match signal_dim")

        self.csv_file = None
        self.csv_writer = None
        self.episode_count = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CSVDataLogger] Logging CSV data to: {self.log_dir}")

    def start_new_episode(self):
        """Close the active file and open a new per-episode CSV."""
        self.close()
        self.episode_count += 1

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"{self.task_name}_ep{self.episode_count:03d}_{timestamp}.csv"
        filepath = self.log_dir / filename

        self.csv_file = filepath.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "is_engaged", "is_engaged_half", "is_success", *self.signal_labels])
        print(f"[CSVDataLogger] Started episode {self.episode_count}: {filename}")

    def log_step(
        self,
        step: int,
        is_engaged: bool,
        is_engaged_half: bool,
        is_success: bool,
        signal=None,
        contact_force=None,
    ):
        """Append one simulation step to the active CSV."""
        if self.csv_writer is None:
            return

        if signal is None:
            signal = contact_force
        if signal is None:
            raise ValueError("Either signal or contact_force must be provided")

        values = self._as_float_list(signal)
        if len(values) != self.signal_dim:
            raise ValueError(f"Expected signal_dim={self.signal_dim}, got {len(values)}")

        self.csv_writer.writerow([
            int(step),
            int(bool(is_engaged)),
            int(bool(is_engaged_half)),
            int(bool(is_success)),
            *values,
        ])

    def close(self):
        """Close the active CSV file if one is open."""
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"[CSVDataLogger] Closed episode {self.episode_count}.")
            self.csv_file = None
            self.csv_writer = None

    @staticmethod
    def _as_float_list(value) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "flatten"):
            value = value.flatten()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (int, float)):
            value = [value]
        return [float(v) for v in value]
