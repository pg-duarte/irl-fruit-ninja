# src/pose/hand_tracking.py
# Tracker simples: smoothing (EMA) + hold-last + trails
# + dynamic jump gating:
#   rejects detections that are too far AND too fast to be plausible

import time
import math


class HandTracker:
    def __init__(
        self,
        alpha=0.35,
        hold_frames=6,
        conf_decay=0.85,
        trail_len=24,
        min_conf=0.07,
        max_jump_px=160.0,        # soft distance limit (px). <=0 disables.
        max_speed_px_s=2800.0,    # NEW: allow fast moves if speed is plausible (px/s). <=0 disables.
    ):
        self.alpha = float(alpha)
        self.hold_frames = int(hold_frames)
        self.conf_decay = float(conf_decay)
        self.trail_len = int(trail_len)
        self.min_conf = float(min_conf)
        self.max_jump_px = float(max_jump_px)
        self.max_speed_px_s = float(max_speed_px_s)
        self.reset()

    def reset(self):
        self.L = {"x": None, "y": None, "c": 0.0, "vx": 0.0, "vy": 0.0, "drop": 0}
        self.R = {"x": None, "y": None, "c": 0.0, "vx": 0.0, "vy": 0.0, "drop": 0}
        self.trailL = []
        self.trailR = []
        self._t = None

    def _ema(self, prev, new):
        if prev is None:
            return float(new)
        a = self.alpha
        return a * float(new) + (1.0 - a) * float(prev)

    def _is_implausible_jump(self, state, x, y, dt):
        """
        Return True if (x,y) is a likely "wrist swap / hallucination".
        Rule: reject only if it's far AND (if enabled) too fast.
        This keeps fast real slices while blocking teleports across screen.
        """
        if state["x"] is None or state["y"] is None:
            return False

        dx = float(x) - float(state["x"])
        dy = float(y) - float(state["y"])
        d2 = dx * dx + dy * dy

        # Distance gate (optional)
        dist_gate = False
        if self.max_jump_px > 0:
            dist_gate = d2 > (self.max_jump_px * self.max_jump_px)

        # Speed gate (optional)
        speed_gate = False
        if self.max_speed_px_s > 0 and dt > 1e-6:
            v2 = d2 / (dt * dt)
            speed_gate = v2 > (self.max_speed_px_s * self.max_speed_px_s)

        # If speed gate is disabled, fall back to distance only.
        if self.max_speed_px_s <= 0:
            return dist_gate

        # Reject only if BOTH far and too fast.
        return dist_gate and speed_gate

    def _update_one(self, state, raw, dt):
        x, y, c = raw
        ok = (c >= self.min_conf) and (x is not None) and (y is not None)

        # Dynamic jump gating (NEW behavior)
        if ok and self._is_implausible_jump(state, x, y, dt):
            ok = False

        if ok:
            # smoothing
            sx = self._ema(state["x"], x)
            sy = self._ema(state["y"], y)

            # velocity
            if state["x"] is not None and dt > 1e-6:
                state["vx"] = (sx - state["x"]) / dt
                state["vy"] = (sy - state["y"]) / dt
            else:
                state["vx"] = 0.0
                state["vy"] = 0.0

            state["x"], state["y"], state["c"] = sx, sy, float(c)
            state["drop"] = 0
            return

        # dropout: hold-last
        if state["x"] is not None and state["drop"] < self.hold_frames:
            state["drop"] += 1
            state["c"] = state["c"] * (self.conf_decay ** state["drop"])
            state["vx"] = 0.0
            state["vy"] = 0.0
            return

        # lost
        state["x"], state["y"], state["c"] = None, None, 0.0
        state["vx"], state["vy"] = 0.0, 0.0
        state["drop"] = min(state["drop"] + 1, self.hold_frames + 1)

    def _push_trail(self, trail, state, now):
        if state["x"] is None or state["c"] <= 0:
            return
        trail.append((float(state["x"]), float(state["y"]), now, float(state["c"])))
        if len(trail) > self.trail_len:
            del trail[0 : len(trail) - self.trail_len]

    def update(self, rawL, rawR, t=None):
        now = time.time() if t is None else float(t)
        dt = (now - self._t) if self._t is not None else 1.0 / 30.0
        self._t = now
        if dt <= 0:
            dt = 1.0 / 30.0

        self._update_one(self.L, rawL, dt)
        self._update_one(self.R, rawR, dt)

        self._push_trail(self.trailL, self.L, now)
        self._push_trail(self.trailR, self.R, now)

        met = {
            "dt": dt,
            "Lv": math.hypot(self.L["vx"], self.L["vy"]),
            "Rv": math.hypot(self.R["vx"], self.R["vy"]),
            "Lc": self.L["c"],
            "Rc": self.R["c"],
        }

        L = (self.L["x"], self.L["y"], self.L["c"])
        R = (self.R["x"], self.R["y"], self.R["c"])
        return L, R, self.trailL, self.trailR, met
