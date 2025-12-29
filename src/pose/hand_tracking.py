# src/pose/hand_tracking.py
#
# Converts raw wrists (Hands) into stable tracked hands + trails.
# Responsibilities:
# - gating by confidence
# - hold-last for brief dropouts
# - smoothing (EMA)
# - estimate velocity (optional but useful for slice logic)
# - maintain trails buffers
#
# No OpenPose inside: you feed it raw Hands from OpenPoseWrapper.

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import math
import time

from .openpose_wrapper import Hands, HandPoint
from .smoothing import EMA2D


@dataclass
class TrailPoint:
    x: float
    y: float
    t: float
    conf: float


@dataclass
class Trails:
    left: List[TrailPoint] = field(default_factory=list)
    right: List[TrailPoint] = field(default_factory=list)


@dataclass
class HandKinematics:
    x: Optional[float] = None
    y: Optional[float] = None
    conf: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class HandTrackingConfig:
    # Confidence gating
    min_conf: float = 0.07

    # Hold-last behaviour
    hold_frames: int = 6             # how many consecutive frames we keep last position after dropout
    conf_decay: float = 0.85         # applied each dropout frame while holding

    # Smoothing
    ema_alpha: float = 0.35

    # Trails
    trail_max_points: int = 24
    trail_min_conf: float = 0.05     # only add points to trail above this conf
    trail_min_dist_px: float = 6.0   # avoid adding almost-identical points


class HandTracker:
    def __init__(self, cfg: HandTrackingConfig = HandTrackingConfig()):
        self.cfg = cfg

        self._ema_L = EMA2D(alpha=cfg.ema_alpha)
        self._ema_R = EMA2D(alpha=cfg.ema_alpha)

        self._last_L = HandKinematics()
        self._last_R = HandKinematics()

        self._drop_L = 0
        self._drop_R = 0

        self.trails = Trails()

        self._last_t: Optional[float] = None

    def reset(self):
        self._ema_L.reset()
        self._ema_R.reset()
        self._last_L = HandKinematics()
        self._last_R = HandKinematics()
        self._drop_L = 0
        self._drop_R = 0
        self.trails = Trails()
        self._last_t = None

    def _dist2(self, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x1 - x2
        dy = y1 - y2
        return dx * dx + dy * dy

    def _update_one(
        self,
        raw: HandPoint,
        ema: EMA2D,
        last: HandKinematics,
        drop_count: int,
        dt: float,
    ) -> Tuple[HandKinematics, int]:
        cfg = self.cfg

        valid = raw.conf >= cfg.min_conf and raw.x is not None and raw.y is not None

        if valid:
            # reset dropout
            drop_count = 0

            # smooth
            sx, sy = ema.update(raw.x, raw.y)

            # velocity (use last smoothed position)
            if last.x is not None and last.y is not None and dt > 1e-6:
                vx = (sx - last.x) / dt
                vy = (sy - last.y) / dt
            else:
                vx = 0.0
                vy = 0.0

            return HandKinematics(x=sx, y=sy, conf=float(raw.conf), vx=float(vx), vy=float(vy)), drop_count

        # invalid -> hold last for a bit
        if last.x is not None and last.y is not None and drop_count < cfg.hold_frames:
            drop_count += 1
            held_conf = last.conf * (cfg.conf_decay ** drop_count)
            return HandKinematics(x=last.x, y=last.y, conf=float(held_conf), vx=0.0, vy=0.0), drop_count

        # fully lost
        drop_count = min(drop_count + 1, cfg.hold_frames + 1)
        return HandKinematics(x=None, y=None, conf=0.0, vx=0.0, vy=0.0), drop_count

    def update(self, raw_hands: Hands, t: Optional[float] = None) -> Tuple[Hands, Trails, dict]:
        """
        Input:
          raw_hands: Hands from OpenPoseWrapper (wrists or elbows fallback)
          t: optional timestamp (seconds). If None, uses time.time()

        Output:
          (smoothed_hands, trails, metrics)
        """
        now = float(time.time() if t is None else t)

        if self._last_t is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-6, now - self._last_t)
        self._last_t = now

        self._last_L, self._drop_L = self._update_one(raw_hands.left, self._ema_L, self._last_L, self._drop_L, dt)
        self._last_R, self._drop_R = self._update_one(raw_hands.right, self._ema_R, self._last_R, self._drop_R, dt)

        # update trails
        self._update_trail_side("L", now, self._last_L)
        self._update_trail_side("R", now, self._last_R)

        smoothed = Hands(
            left=HandPoint(self._last_L.x, self._last_L.y, float(self._last_L.conf)),
            right=HandPoint(self._last_R.x, self._last_R.y, float(self._last_R.conf)),
        )

        metrics = {
            "dt": dt,
            "left_v": math.hypot(self._last_L.vx, self._last_L.vy),
            "right_v": math.hypot(self._last_R.vx, self._last_R.vy),
            "left_conf": self._last_L.conf,
            "right_conf": self._last_R.conf,
        }
        return smoothed, self.trails, metrics

    def _update_trail_side(self, side: str, t: float, hk: HandKinematics):
        cfg = self.cfg
        if hk.x is None or hk.y is None or hk.conf < cfg.trail_min_conf:
            return

        trail = self.trails.left if side == "L" else self.trails.right

        if trail:
            lastp = trail[-1]
            if self._dist2(hk.x, hk.y, lastp.x, lastp.y) < (cfg.trail_min_dist_px ** 2):
                # too close -> skip
                return

        trail.append(TrailPoint(x=float(hk.x), y=float(hk.y), t=float(t), conf=float(hk.conf)))

        if len(trail) > cfg.trail_max_points:
            del trail[: len(trail) - cfg.trail_max_points]
