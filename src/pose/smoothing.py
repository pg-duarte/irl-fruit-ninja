# src/pose/smoothing.py
#
# Generic smoothing utilities (no OpenPose dependency).
# - EMA filter for (x,y) points
# - Confidence-weighted update
# - Simple helpers used by HandTracker

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EMA2D:
    """
    Exponential moving average for 2D points.
    alpha in (0,1): higher = less smoothing (more responsive)
    """
    alpha: float = 0.35
    x: Optional[float] = None
    y: Optional[float] = None
    initialized: bool = False

    def reset(self):
        self.x, self.y = None, None
        self.initialized = False

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if not self.initialized or self.x is None or self.y is None:
            self.x, self.y = float(x), float(y)
            self.initialized = True
            return self.x, self.y

        a = float(self.alpha)
        self.x = a * float(x) + (1.0 - a) * self.x
        self.y = a * float(y) + (1.0 - a) * self.y
        return self.x, self.y


def lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b
