# src/pose/hand_tracking.py
# Tracker melhorado: prediction (const-velocity) + anti-swap L/R + alpha adaptativo
# Mantém a mesma classe, mesma assinatura e o mesmo output do teu HandTracker atual.

import time
import math


class HandTracker:
    def __init__(
        self,
        alpha=0.35,               # base smoothing (ainda existe, mas agora é adaptativo)
        hold_frames=6,
        conf_decay=0.85,
        trail_len=24,
        min_conf=0.07,
        max_jump_px=160.0,        # gate distância (vs predição)
        max_speed_px_s=2800.0,    # gate velocidade
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

    # ---------- core helpers ----------

    def _ema(self, prev, new, a):
        if prev is None:
            return float(new)
        return a * float(new) + (1.0 - a) * float(prev)

    def _predict_xy(self, state, dt):
        """Constant-velocity prediction."""
        if state["x"] is None or state["y"] is None:
            return None, None
        return (float(state["x"]) + float(state["vx"]) * dt,
                float(state["y"]) + float(state["vy"]) * dt)

    def _adaptive_alpha(self, c, speed_px_s):
        """
        Alpha adaptativo:
          - conf alta + velocidade alta => mais responsivo (alpha maior)
          - conf baixa + parado => mais suave (alpha menor)
        Mantém intervalo estável.
        """
        c01 = max(0.0, min(1.0, float(c)))
        v = float(speed_px_s)

        # normaliza v para ~0..1 em torno de 0..(max_speed)
        v01 = 0.0
        if self.max_speed_px_s > 1e-6:
            v01 = max(0.0, min(1.0, v / self.max_speed_px_s))

        # base alpha + boost com velocidade e confiança
        a = self.alpha
        a = a + 0.35 * v01 + 0.10 * (c01 - 0.5)

        # clamp: evita jitter por alpha demasiado alto e evita lag por alpha demasiado baixo
        return max(0.12, min(0.85, a))

    def _is_implausible(self, state, meas_x, meas_y, meas_c, dt):
        """
        Gate contra PREVISÃO (não contra last sample).
        Rejeita se for longe demais e rápido demais.
        """
        if meas_x is None or meas_y is None:
            return True
        if float(meas_c) < self.min_conf:
            return True
        if state["x"] is None or state["y"] is None:
            return False

        px, py = self._predict_xy(state, dt)
        if px is None:
            return False

        dx = float(meas_x) - float(px)
        dy = float(meas_y) - float(py)
        d2 = dx * dx + dy * dy

        dist_gate = False
        if self.max_jump_px > 0:
            dist_gate = d2 > (self.max_jump_px * self.max_jump_px)

        speed_gate = False
        if self.max_speed_px_s > 0 and dt > 1e-6:
            v2 = d2 / (dt * dt)
            speed_gate = v2 > (self.max_speed_px_s * self.max_speed_px_s)

        # Se speed gate desligado: distancia basta
        if self.max_speed_px_s <= 0:
            return dist_gate

        return dist_gate and speed_gate

    def _assoc_cost2(self, state, meas, dt):
        """
        Custo de associar medida -> track, usando distância à PREVISÃO.
        Retorna dist^2 (menor = melhor). Se track vazio, custo baixo.
        """
        mx, my, mc = meas
        if mx is None or my is None or float(mc) < self.min_conf:
            return 1e30

        if state["x"] is None or state["y"] is None:
            # se track ainda não existe, custo baseado só na confiança (preferir conf alta)
            return 1e6 - 1e6 * max(0.0, min(1.0, float(mc)))

        px, py = self._predict_xy(state, dt)
        dx = float(mx) - float(px)
        dy = float(my) - float(py)
        return dx * dx + dy * dy

    def _update_one(self, state, raw, dt):
        mx, my, mc = raw
        ok = not self._is_implausible(state, mx, my, mc, dt)

        if ok:
            # predição para estimar velocidade atual (para alpha adaptativo)
            px, py = self._predict_xy(state, dt)
            if px is None:
                px, py = float(mx), float(my)

            # velocidade "medida" (vs predição)
            dxp = float(mx) - float(px)
            dyp = float(my) - float(py)
            speed = 0.0 if dt <= 1e-6 else math.hypot(dxp, dyp) / dt

            a = self._adaptive_alpha(mc, speed)

            # smoothing
            sx = self._ema(state["x"], mx, a)
            sy = self._ema(state["y"], my, a)

            # velocidade filtrada
            if state["x"] is not None and dt > 1e-6:
                vx_m = (sx - float(state["x"])) / dt
                vy_m = (sy - float(state["y"])) / dt

                # suaviza a velocidade para não explodir com jitter
                vfa = 0.45
                state["vx"] = vfa * vx_m + (1.0 - vfa) * float(state["vx"])
                state["vy"] = vfa * vy_m + (1.0 - vfa) * float(state["vy"])
            else:
                state["vx"] = 0.0
                state["vy"] = 0.0

            state["x"], state["y"], state["c"] = float(sx), float(sy), float(mc)
            state["drop"] = 0
            return

        # dropout: hold-last com decaimento
        if state["x"] is not None and state["drop"] < self.hold_frames:
            state["drop"] += 1
            state["c"] = float(state["c"]) * (self.conf_decay ** state["drop"])
            # continua a "andar" pela predição, mas reduz a velocidade para não derivar
            state["vx"] *= 0.75
            state["vy"] *= 0.75
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

    # ---------- public API (unchanged) ----------

    def update(self, rawL, rawR, t=None):
        now = time.time() if t is None else float(t)
        dt = (now - self._t) if self._t is not None else 1.0 / 30.0
        self._t = now
        if dt <= 0:
            dt = 1.0 / 30.0

        # --- Anti-swap association (NEW) ---
        # Se ambos os raw têm conf, decide se é melhor manter ou trocar com base no custo à previsão.
        # Isto resolve a maioria dos swaps L/R do OpenPose sem mudar API.
        cLL = self._assoc_cost2(self.L, rawL, dt)
        cRR = self._assoc_cost2(self.R, rawR, dt)
        cLR = self._assoc_cost2(self.L, rawR, dt)
        cRL = self._assoc_cost2(self.R, rawL, dt)

        keep_cost = cLL + cRR
        swap_cost = cLR + cRL

        useL = rawL
        useR = rawR

        # Só troca se for claramente melhor (histerese) e ambas forem "válidas"
        def _valid(raw):
            x, y, c = raw
            return (x is not None) and (y is not None) and (float(c) >= self.min_conf)

        if _valid(rawL) and _valid(rawR):
            if swap_cost + 2500.0 < keep_cost:  # margem ~50px^2 para evitar flip-flop
                useL, useR = rawR, rawL

        # update tracks
        self._update_one(self.L, useL, dt)
        self._update_one(self.R, useR, dt)

        # trails
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
