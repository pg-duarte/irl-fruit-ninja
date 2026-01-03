# src/game/fruit.py
import os
import cv2 as cv
import numpy as np

import random

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "images"



class Fruit:
    _img_cache = {}

    def __init__(self, x, y, size=64):
        """
        size: rendered diameter in pixels
        """
        self.x = float(x)
        self.y = float(y)
        self.size = int(size)
        self.radius = self.size // 2
        self.alive = True

        # physics
        # physics (spawner sets vx/vy)
        self.vx = 0.0
        self.vy = 0.0
        self.g = 900.0


        # load image once
        self.img = self._load_image("watermelon.png")

    @classmethod
    def _load_image(cls, name):
        if name in cls._img_cache:
            return cls._img_cache[name]

        path = os.path.join(ASSETS_DIR, name)
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Asset not found: {path}")

        cls._img_cache[name] = img
        return img

    def update(self, dt):
        self.vy += self.g * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, frame):
        h, w = frame.shape[:2]

        img = cv.resize(self.img, (self.size, self.size), interpolation=cv.INTER_AREA)

        x0 = int(self.x - self.radius)
        y0 = int(self.y - self.radius)
        x1 = x0 + self.size
        y1 = y0 + self.size

        # clip to screen
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            return

        fx0 = max(0, x0)
        fy0 = max(0, y0)
        fx1 = min(w, x1)
        fy1 = min(h, y1)

        ix0 = fx0 - x0
        iy0 = fy0 - y0
        ix1 = ix0 + (fx1 - fx0)
        iy1 = iy0 + (fy1 - fy0)

        roi = frame[fy0:fy1, fx0:fx1]
        sprite = img[iy0:iy1, ix0:ix1]

        if sprite.shape[2] == 4:
            alpha = sprite[:, :, 3:4] / 255.0
            roi[:] = (1 - alpha) * roi + alpha * sprite[:, :, :3]
        else:
            roi[:] = sprite[:, :, :3]

    def is_offscreen(self, width, height):
        return (
            self.y - self.radius > height + 50 or
            self.x + self.radius < -50 or
            self.x - self.radius > width + 50
        )
