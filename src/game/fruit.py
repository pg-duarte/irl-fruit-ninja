# src/game/fruit.py
import random
from pathlib import Path

import cv2 as cv
import numpy as np

# assets/images
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "images"


def _alpha_blit(dst_bgr: np.ndarray, sprite_rgba: np.ndarray):
    """Alpha blend RGBA sprite onto BGR dst ROI (same size)."""
    if sprite_rgba.ndim == 3 and sprite_rgba.shape[2] == 4:
        alpha = (sprite_rgba[:, :, 3:4].astype(np.float32)) / 255.0
        dst_bgr[:] = (1.0 - alpha) * dst_bgr + alpha * sprite_rgba[:, :, :3]
    else:
        dst_bgr[:] = sprite_rgba[:, :, :3]


def _rotate_rgba(img_rgba: np.ndarray, angle_deg: float):
    """Rotate square RGBA image around center, keep alpha."""
    h, w = img_rgba.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    return cv.warpAffine(
        img_rgba, M, (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_TRANSPARENT
    )


def _pad_to_square(img_rgba: np.ndarray):
    """Pad RGBA image to square (centered) with transparent background."""
    h, w = img_rgba.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 4), dtype=img_rgba.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img_rgba
    return out


class FruitPiece:
    """A sliced half piece (still circular collision is OK, we just draw sprite)."""

    def __init__(self, x, y, img_rgba: np.ndarray, size: int):
        self.x = float(x)
        self.y = float(y)

        self.size = int(size)
        self.radius = self.size // 2
        self.alive = True

        # physics
        self.vx = 0.0
        self.vy = 0.0
        self.g = 900.0

        # rotation
        self.angle = random.uniform(0.0, 360.0)
        self.omega = random.uniform(-360.0, 360.0)

        # sprite for this piece (RGBA)
        self.img = img_rgba

    def update(self, dt: float):
        self.vy += self.g * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle = (self.angle + self.omega * dt) % 360.0

    def draw(self, frame):
        h, w = frame.shape[:2]

        img0 = cv.resize(self.img, (self.size, self.size), interpolation=cv.INTER_AREA)
        img = _rotate_rgba(img0, self.angle)

        x0 = int(self.x - self.radius)
        y0 = int(self.y - self.radius)
        x1 = x0 + self.size
        y1 = y0 + self.size

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
        _alpha_blit(roi, sprite)

    def is_offscreen(self, width, height):
        return (
            self.y - self.radius > height + 50 or
            self.x + self.radius < -50 or
            self.x - self.radius > width + 50
        )


class Fruit:
    _img_cache = {}

    def __init__(self, x, y, size=64):
        self.x = float(x)
        self.y = float(y)
        self.size = int(size)
        self.radius = self.size // 2
        self.alive = True

        # physics (spawner sets vx/vy usually)
        self.vx = 0.0
        self.vy = 0.0
        self.g = 900.0

        # rotation
        self.angle = random.uniform(0.0, 360.0)
        self.omega = random.uniform(-180.0, 180.0)

        # load sprite
        self.img = self._load_image("watermelon.png")

    @classmethod
    def _load_image(cls, name: str):
        if name in cls._img_cache:
            return cls._img_cache[name]

        path = ASSETS_DIR / name
        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Asset not found: {path}")

        # ensure RGBA
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        cls._img_cache[name] = img
        return img

    def update(self, dt: float):
        self.vy += self.g * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle = (self.angle + self.omega * dt) % 360.0

    def draw(self, frame):
        h, w = frame.shape[:2]

        img0 = cv.resize(self.img, (self.size, self.size), interpolation=cv.INTER_AREA)
        img = _rotate_rgba(img0, self.angle)

        x0 = int(self.x - self.radius)
        y0 = int(self.y - self.radius)
        x1 = x0 + self.size
        y1 = y0 + self.size

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
        _alpha_blit(roi, sprite)

    def is_offscreen(self, width, height):
        return (
            self.y - self.radius > height + 50 or
            self.x + self.radius < -50 or
            self.x - self.radius > width + 50
        )

    def slice_into_halves(self):
        """
        Returns 2 FruitPiece objects (left half, right half).
        Simple vertical split of the sprite -> looks like it's cut.
        """
        # make sure we split an RGBA image
        src = self.img
        if src.ndim == 3 and src.shape[2] == 3:
            src = cv.cvtColor(src, cv.COLOR_BGR2BGRA)

        h, w = src.shape[:2]
        mid = w // 2

        left = src[:, :mid, :].copy()
        right = src[:, mid:, :].copy()

        # pad halves to square so rotation works nicely about center
        left = _pad_to_square(left)
        right = _pad_to_square(right)

        # pieces inherit current position
        pL = FruitPiece(self.x, self.y, left, self.size)
        pR = FruitPiece(self.x, self.y, right, self.size)

        # inherit gravity
        pL.g = self.g
        pR.g = self.g

        # inherit base velocities + add outward impulse
        kick = 160.0
        pL.vx = self.vx - kick
        pR.vx = self.vx + kick
        pL.vy = self.vy - 60.0
        pR.vy = self.vy - 60.0

        # make them spin more
        pL.omega = random.uniform(-540.0, -240.0)
        pR.omega = random.uniform(240.0, 540.0)

        return pL, pR
