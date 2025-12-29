# src/pose/hand_tracking.py
# Tracker simples: smoothing (EMA) + hold-last + trails

import time
import math


class HandTracker:
    def __init__(self, alpha=0.35, hold_frames=6, conf_decay=0.85, trail_len=24, min_conf=0.07):
        self.alpha = float(alpha)
        self.hold_frames = int(hold_frames)
        self.conf_decay = float(conf_decay)
        self.trail_len = int(trail_len)
        self.min_conf = float(min_conf)
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

    def _update_one(self, state, raw, dt):
        x, y, c = raw
        ok = (c >= self.min_conf) and (x is not None) and (y is not None)

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
            del trail[0:len(trail) - self.trail_len]

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
        # devolve (x,y,conf) também
        L = (self.L["x"], self.L["y"], self.L["c"])
        R = (self.R["x"], self.R["y"], self.R["c"])
        return L, R, self.trailL, self.trailR, met
