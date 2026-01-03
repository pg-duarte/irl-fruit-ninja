# src/ui/menu.py
# UI super simples: botões retangulares + hover + dwell + swipe-to-click.

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import cv2


@dataclass
class Button:
    key: str
    label: str
    rect: Tuple[int, int, int, int]  # x, y, w, h

    def contains(self, x: int, y: int) -> bool:
        rx, ry, rw, rh = self.rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


class MenuUI:
    def __init__(
        self,
        buttons: List[Button],
        dwell_s: float = 0.8,
        swipe_speed_px_s: float = 1800.0,
        click_cooldown_s: float = 0.4,
    ):
        self.buttons = buttons
        self.dwell_s = float(dwell_s)
        self.swipe_speed_px_s = float(swipe_speed_px_s)
        self.click_cooldown_s = float(click_cooldown_s)

        self._hover_key: Optional[str] = None
        self._hover_t0: float = 0.0
        self._last_click_t: float = 0.0

    def _cooldown_ok(self, now: float) -> bool:
        return (now - self._last_click_t) >= self.click_cooldown_s

    def update(self, cursor_xy: Optional[Tuple[int, int]], cursor_speed: float, now: float) -> Optional[str]:
        """
        Returns button.key if a selection was triggered (dwell or swipe).
        """
        if cursor_xy is None:
            self._hover_key = None
            return None

        cx, cy = cursor_xy

        hovered: Optional[Button] = None
        for b in self.buttons:
            if b.contains(cx, cy):
                hovered = b
                break

        if hovered is None:
            self._hover_key = None
            return None

        # update hover timer
        if hovered.key != self._hover_key:
            self._hover_key = hovered.key
            self._hover_t0 = now

        # trigger by swipe
        if cursor_speed >= self.swipe_speed_px_s and self._cooldown_ok(now):
            self._last_click_t = now
            return hovered.key

        # trigger by dwell
        if (now - self._hover_t0) >= self.dwell_s and self._cooldown_ok(now):
            self._last_click_t = now
            return hovered.key

        return None

    def draw(self, frame, cursor_xy: Optional[Tuple[int, int]]):
        """
        Draw buttons + simple hover highlight.
        """
        for b in self.buttons:
            x, y, w, h = b.rect
            is_hover = (b.key == self._hover_key)
            # simple style: white border; fill if hovered
            if is_hover:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), thickness=-1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

            # label
            cv2.putText(
                frame,
                b.label,
                (x + 16, y + int(h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        # cursor
        if cursor_xy is not None:
            cx, cy = cursor_xy
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), thickness=2)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), thickness=-1)
