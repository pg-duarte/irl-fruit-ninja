# src/game/fruit.py
import os
import random

from pathlib import Path

import cv2 as cv
import numpy as np

# assets/images
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

        # physics (spawner typically sets vx/vy; these are safe defaults)
        self.vx = 0.0
        self.vy = 0.0
        self.g = 900.0

        # rotation
        self.angle = random.uniform(0.0, 360.0)        # degrees
        self.omega = random.uniform(-180.0, 180.0)     # deg/s

        # load image once
        self.img = self._load_image("watermelon.png")

    @classmethod
    def _load_image(cls, name: str):
        if name in cls._img_cache:
            return cls._img_cache[name]

        path = ASSETS_DIR / name
        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Asset not found: {path}")

        cls._img_cache[name] = img
        return img

    def update(self, dt: float):
        self.vy += self.g * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # update rotation
        self.angle = (self.angle + self.omega * dt) % 360.0

    def draw(self, frame):
        h, w = frame.shape[:2]

        # resize sprite to desired on-screen size
        img0 = cv.resize(self.img, (self.size, self.size), interpolation=cv.INTER_AREA)

        # rotate around center (keep alpha)
        M = cv.getRotationMatrix2D((self.radius, self.radius), float(self.angle), 1.0)
        img = cv.warpAffine(
            img0,
            M,
            (self.size, self.size),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_TRANSPARENT,
        )

        x0 = int(self.x - self.radius)
        y0 = int(self.y - self.radius)
        x1 = x0 + self.size
        y1 = y0 + self.size

        # offscreen quick reject
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            return

        # clip to frame
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

        # alpha blend if RGBA
        if sprite.ndim == 3 and sprite.shape[2] == 4:
            alpha = (sprite[:, :, 3:4].astype(np.float32)) / 255.0
            roi[:] = (1.0 - alpha) * roi + alpha * sprite[:, :, :3]
        else:
            roi[:] = sprite[:, :, :3]

    def is_offscreen(self, width, height):
        return (
            self.y - self.radius > height + 50 or
            self.x + self.radius < -50 or
            self.x - self.radius > width + 50
        )
