from collections import deque
from typing import Any

import numpy as np


class CustomLogger:
    def __init__(self) -> None:
        self.logs = ""
        self.end = "\n"
        self.sep = " "

    def print(self, *objects: Any) -> None:
        self.logs += self.sep.join(map(str, objects)) + self.end

    def print_numeric(self, label, value, end="\n") -> None:
        """Print a labeled numeric value with consistent formatting."""
        if isinstance(value, float):
            self.logs += f"{label} {value:.5f}"
        else:
            self.logs += f"{label} {value}"

        self.logs += end

    def flush(self):
        print(self.logs)
        self.logs = ""
