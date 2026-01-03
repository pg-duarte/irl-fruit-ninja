# src/main/main.py

import sys
import os
import time
import threading
from typing import Optional, Tuple

import cv2

# Treat "src" as project root for imports: game.*, pose.*, ui.*, common.*
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from game.game_core import Game
from pose.openpose_wrapper import OpenPose
from pose.hand_tracking import HandTracker

WIDTH, HEIGHT = 640, 480

# =========================
# Performance knobs
# =========================

# 1) Run pose only every N frames (decimation).
#    2 = pose at ~15 Hz if camera is ~30 FPS.
POSE_EVERY_N_FRAMES = 3

# 2) Smaller DNN input => faster inference, less accuracy.
POSE_IN_SIZE = 256

# Trail fade settings
TRAIL_TTL_SEC = 4.0

# Camera index (change if needed)
CAM_INDEX = 1


# =========================
# Capture thread
# =========================
class FrameGrabber:
    """
    Simple camera reader thread that always keeps the latest frame.
    This reduces stutter and prevents the main loop from blocking on cap.read().
    """
    def __init__(self, cam_index: int):
        self.cap = cv2.VideoCapture(cam_index)
        # Try to reduce internal buffering (not supported on all backends)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest: Optional[Tuple[float, any]] = None  # (timestamp, frame)
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            t = time.time()
            with self.lock:
                self.latest = (t, frame)

    def read_latest(self) -> Optional[Tuple[float, any]]:
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        self.cap.release()


# =========================
# Trail drawing utils
# =========================
def _fade_color(base_bgr, fade01: float):
    """fade01: 1.0=full, 0.0=gone"""
    f = max(0.0, min(1.0, fade01))
    return (int(base_bgr[0] * f), int(base_bgr[1] * f), int(base_bgr[2] * f))


def _trail_to_xy_list(trail_points, now_t: float):
    """
    trail_points: [(x,y,t,conf), ...]  (from your HandTracker)
    -> [(x,y), ...] filtered by TTL.
    """
    out = []
    for x, y, t, c in trail_points:
        if (now_t - t) <= TRAIL_TTL_SEC:
            out.append((int(x), int(y)))
    return out


def _draw_trail(frame, trail_points, now_t: float, base_color_bgr, thickness=2, step=1):
    """
    Draw trail segments with fade based on age (TTL).
    step>1 draws fewer segments (faster).
    """
    pts = [(x, y, t, c) for (x, y, t, c) in trail_points if (now_t - t) <= TRAIL_TTL_SEC]
    if len(pts) < 2:
        return

    # optional downsample for speed
    if step > 1:
        pts = pts[::step]
        if len(pts) < 2:
            return

    for i in range(len(pts) - 1):
        x1, y1, t1, c1 = pts[i]
        x2, y2, t2, c2 = pts[i + 1]

        age = now_t - t2
        fade = 1.0 - (age / TRAIL_TTL_SEC)  # 1 -> 0
        color = _fade_color(base_color_bgr, fade)

        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


# =========================
# Main
# =========================
def main():
    game = Game(WIDTH, HEIGHT)

    # OpenPose model path (graph_opt.pb inside src/pose/)
    model_path = os.path.abspath(os.path.join(SRC_DIR, "pose", "graph_opt.pb"))

    # 2) Smaller in_size = faster
    pose = OpenPose(
        model_path=model_path,
        in_size=POSE_IN_SIZE,
        thr=0.07,
        elbow_fallback=True,
        swap_rb=False,
    )

    tracker = HandTracker(
        alpha=0.35,
        hold_frames=6,
        conf_decay=0.85,
        trail_len=120,
        min_conf=0.07
    )

    grabber = FrameGrabber(CAM_INDEX).start()
    cv2.namedWindow("Fruit Ninja Test")

    last_loop_t = time.time()
    frame_idx = 0

    # pose cache (to reuse on frames where we skip pose)
    last_rawL = (None, None, 0.0)
    last_rawR = (None, None, 0.0)

    # FPS counter
    fps_last = time.time()
    fps_count = 0
    fps = 0.0

    try:
        while True:
            latest = grabber.read_latest()
            if latest is None:
                time.sleep(0.005)
                continue

            cam_t, frame = latest

            # Resize/flip in main thread (keeps capture thread minimal)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - last_loop_t
            last_loop_t = now
            if dt <= 0:
                dt = 1.0 / 60.0

            # 1) Pose decimation: only run every N frames
            if (frame_idx % POSE_EVERY_N_FRAMES) == 0:
                try:
                    last_rawL, last_rawR = pose.wrists(frame)
                except Exception:
                    # If inference fails, keep last known (tracker will decay/hold)
                    pass

            frame_idx += 1

            # Tracking every frame (cheap)
            L, R, trailL, trailR, met = tracker.update(last_rawL, last_rawR, t=now)

            trails_xy = {
                "left":  _trail_to_xy_list(trailL, now),
                "right": _trail_to_xy_list(trailR, now),
            }

            # Game update/draw
            game.update(dt, trails_xy)
            frame = game.draw(frame)

            # Trails draw (optional downsample step=2 to draw fewer segments faster)
            _draw_trail(frame, trailL, now, base_color_bgr=(255, 0, 0), thickness=2, step=2)
            _draw_trail(frame, trailR, now, base_color_bgr=(0, 255, 255), thickness=2, step=2)

            # HUD
            cv2.putText(frame, f"Score: {game.score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # FPS update (~1s)
            fps_count += 1
            if now - fps_last >= 1.0:
                fps = fps_count / (now - fps_last)
                fps_count = 0
                fps_last = now

            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Fruit Ninja Test", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
