# src/main/main.py

import sys
import os
import time
import threading
from typing import Optional, Tuple, List

import cv2

# Treat "src" as project root for imports: game.*, pose.*, ui.*
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from game.game_core import Game
from pose.openpose_wrapper import OpenPose
from pose.hand_tracking import HandTracker
from ui.state import AppState
from ui.menu import Button, MenuUI

WIDTH, HEIGHT = 640, 480

# =========================
# DEV controls
# =========================
DEV_MOUSE = True  # True => cursor/click comes from mouse instead of hand

# Mouse slicing trail behavior (DEV)
MOUSE_TRAIL_MAX_POINTS = 40
MOUSE_TRAIL_MIN_DIST_PX = 4  # only add a point if mouse moved enough

# Performance knobs
POSE_EVERY_N_FRAMES = 2
POSE_IN_SIZE = 256
TRAIL_TTL_SEC = 0.6
CAM_INDEX = 0

# UI behavior
DWELL_S = 0.8
SWIPE_SPEED_PX_S = 1800.0


class FrameGrabber:
    def __init__(self, cam_index: int):
        self.cap = cv2.VideoCapture(cam_index)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest: Optional[Tuple[float, any]] = None
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


def _fade_color(base_bgr, fade01: float):
    f = max(0.0, min(1.0, fade01))
    return (int(base_bgr[0] * f), int(base_bgr[1] * f), int(base_bgr[2] * f))


def _trail_to_xy_list(trail_points, now_t: float):
    out = []
    for x, y, t, c in trail_points:
        if (now_t - t) <= TRAIL_TTL_SEC:
            out.append((int(x), int(y)))
    return out


def _draw_trail(frame, trail_points, now_t: float, base_color_bgr, thickness=2, step=2):
    pts = [(x, y, t, c) for (x, y, t, c) in trail_points if (now_t - t) <= TRAIL_TTL_SEC]
    if len(pts) < 2:
        return
    if step > 1:
        pts = pts[::step]
        if len(pts) < 2:
            return

    for i in range(len(pts) - 1):
        x1, y1, t1, c1 = pts[i]
        x2, y2, t2, c2 = pts[i + 1]
        age = now_t - t2
        fade = 1.0 - (age / TRAIL_TTL_SEC)
        color = _fade_color(base_color_bgr, fade)
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def _clamp_xy(x: float, y: float):
    xi = int(max(0, min(WIDTH - 1, x)))
    yi = int(max(0, min(HEIGHT - 1, y)))
    return xi, yi


def _choose_cursor(L, R, met):
    lx, ly, lc = L
    rx, ry, rc = R

    if (lc <= 0 and rc <= 0) or (lx is None and rx is None):
        return None, 0.0

    if rc > lc and rx is not None and ry is not None:
        return _clamp_xy(rx, ry), float(met.get("Rv", 0.0))
    if lx is not None and ly is not None:
        return _clamp_xy(lx, ly), float(met.get("Lv", 0.0))

    return None, 0.0


def _load_highscore(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip() or "0")
    except Exception:
        return 0


def _save_highscore(path: str, value: int):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(int(value)))
    except Exception:
        pass


# =========================
# Mouse input (DEV)
# =========================
_MOUSE_X = 0
_MOUSE_Y = 0
_MOUSE_CLICK = False


def _mouse_cb(event, x, y, flags, userdata):
    global _MOUSE_X, _MOUSE_Y, _MOUSE_CLICK
    _MOUSE_X, _MOUSE_Y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        _MOUSE_CLICK = True


def _maybe_push_mouse_point(trail_xy: List[Tuple[int, int]], x: int, y: int):
    if not trail_xy:
        trail_xy.append((x, y))
        return

    px, py = trail_xy[-1]
    dx = x - px
    dy = y - py
    if (dx * dx + dy * dy) >= (MOUSE_TRAIL_MIN_DIST_PX * MOUSE_TRAIL_MIN_DIST_PX):
        trail_xy.append((x, y))
        if len(trail_xy) > MOUSE_TRAIL_MAX_POINTS:
            del trail_xy[0 : len(trail_xy) - MOUSE_TRAIL_MAX_POINTS]


def _draw_mouse_trail(frame, trail_xy: List[Tuple[int, int]]):
    if len(trail_xy) < 2:
        return
    for i in range(len(trail_xy) - 1):
        cv2.line(frame, trail_xy[i], trail_xy[i + 1], (255, 255, 255), 2)


def main():
    repo_root = os.path.abspath(os.path.join(SRC_DIR, ".."))
    highscore_path = os.path.join(repo_root, "highscore.txt")
    highscore = _load_highscore(highscore_path)

    # Pose (still runs, but in DEV_MOUSE you mainly use it for overlay/testing)
    model_path = os.path.abspath(os.path.join(SRC_DIR, "pose", "graph_opt.pb"))
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
        min_conf=0.07,
        max_jump_px=160.0,
        max_speed_px_s=2800.0,
    )

    # Game
    game = Game(WIDTH, HEIGHT)

    # Menus
    menu = MenuUI(
        buttons=[
            Button("start", "Start Game", (220, 150, 200, 55)),
            Button("high", "Highscore", (220, 220, 200, 55)),
            Button("credits", "Credits", (220, 290, 200, 55)),
            Button("exit", "Exit", (220, 360, 200, 55)),
        ],
        dwell_s=DWELL_S,
        swipe_speed_px_s=SWIPE_SPEED_PX_S,
    )

    pause_menu = MenuUI(
        buttons=[
            Button("resume", "Resume", (220, 200, 200, 55)),
            Button("menu", "Main Menu", (220, 270, 200, 55)),
        ],
        dwell_s=DWELL_S,
        swipe_speed_px_s=SWIPE_SPEED_PX_S,
    )

    back_menu = MenuUI(
        buttons=[Button("back", "Back", (20, 20, 120, 45))],
        dwell_s=DWELL_S,
        swipe_speed_px_s=SWIPE_SPEED_PX_S,
    )

    pause_btn = Button("pause", "Pause", (20, 20, 120, 45))
    pause_ui = MenuUI([pause_btn], dwell_s=0.7, swipe_speed_px_s=SWIPE_SPEED_PX_S)

    state = AppState.MENU

    grabber = FrameGrabber(CAM_INDEX).start()
    win_name = "Fruit Ninja"
    cv2.namedWindow(win_name)

    if DEV_MOUSE:
        cv2.setMouseCallback(win_name, _mouse_cb)

    last_loop_t = time.time()
    frame_idx = 0

    last_rawL = (None, None, 0.0)
    last_rawR = (None, None, 0.0)

    # Mouse slicing trail (DEV)
    mouse_trail_xy: List[Tuple[int, int]] = []

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
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - last_loop_t
            last_loop_t = now
            if dt <= 0:
                dt = 1.0 / 60.0

            # Pose decimation
            if (frame_idx % POSE_EVERY_N_FRAMES) == 0:
                try:
                    last_rawL, last_rawR = pose.wrists(frame)
                except Exception:
                    pass
            frame_idx += 1

            # Tracking (hands)
            L, R, trailL, trailR, met = tracker.update(last_rawL, last_rawR, t=now)

            # Cursor source
            if DEV_MOUSE:
                cursor_xy = (_MOUSE_X, _MOUSE_Y)
                cursor_speed = 0.0
            else:
                cursor_xy, cursor_speed = _choose_cursor(L, R, met)

            # Trails for gameplay
            if DEV_MOUSE:
                # Build a simple trail from mouse movement and feed it to the game.
                if cursor_xy is not None:
                    _maybe_push_mouse_point(mouse_trail_xy, cursor_xy[0], cursor_xy[1])

                trails_xy = {"left": [], "right": mouse_trail_xy}
            else:
                mouse_trail_xy.clear()
                trails_xy = {
                    "left": _trail_to_xy_list(trailL, now),
                    "right": _trail_to_xy_list(trailR, now),
                }

            # One-shot mouse click -> instant menu selection (treat as huge swipe)
            global _MOUSE_CLICK
            if DEV_MOUSE and _MOUSE_CLICK:
                cursor_speed = 999999.0
                _MOUSE_CLICK = False

            # -----------------
            # State machine
            # -----------------
            if state == AppState.MENU:
                picked = menu.update(cursor_xy, cursor_speed, now)
                menu.draw(frame, cursor_xy)

                cv2.putText(frame, "Fruit Ninja", (190, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
                cv2.putText(frame, "Select: swipe fast or hold", (170, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if picked == "start":
                    game = Game(WIDTH, HEIGHT)
                    mouse_trail_xy.clear()
                    state = AppState.PLAYING
                elif picked == "high":
                    state = AppState.HIGHSCORES
                elif picked == "credits":
                    state = AppState.CREDITS
                elif picked == "exit":
                    break

            elif state == AppState.PLAYING:
                game.update(dt, trails_xy)
                frame = game.draw(frame)

                if DEV_MOUSE:
                    _draw_mouse_trail(frame, mouse_trail_xy)
                else:
                    _draw_trail(frame, trailL, now, base_color_bgr=(255, 0, 0), thickness=2, step=2)
                    _draw_trail(frame, trailR, now, base_color_bgr=(0, 255, 255), thickness=2, step=2)

                cv2.putText(frame, f"Score: {game.score}", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                picked = pause_ui.update(cursor_xy, cursor_speed, now)
                pause_ui.draw(frame, cursor_xy)
                if picked == "pause":
                    if game.score > highscore:
                        highscore = game.score
                        _save_highscore(highscore_path, highscore)
                    state = AppState.PAUSED

            elif state == AppState.PAUSED:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), thickness=-1)
                frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

                cv2.putText(frame, "Paused", (250, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                picked = pause_menu.update(cursor_xy, cursor_speed, now)
                pause_menu.draw(frame, cursor_xy)

                if picked == "resume":
                    state = AppState.PLAYING
                elif picked == "menu":
                    if game.score > highscore:
                        highscore = game.score
                        _save_highscore(highscore_path, highscore)
                    state = AppState.MENU

            elif state == AppState.HIGHSCORES:
                picked = back_menu.update(cursor_xy, cursor_speed, now)
                back_menu.draw(frame, cursor_xy)

                cv2.putText(frame, "Highscore", (215, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, f"{highscore}", (290, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)

                if picked == "back":
                    state = AppState.MENU

            elif state == AppState.CREDITS:
                picked = back_menu.update(cursor_xy, cursor_speed, now)
                back_menu.draw(frame, cursor_xy)

                cv2.putText(frame, "Credits", (250, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Fruit Ninja (IRL) - TAPDI", (160, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Controls: swipe fast or hold", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if picked == "back":
                    state = AppState.MENU

            # FPS
            fps_count += 1
            if now - fps_last >= 1.0:
                fps = fps_count / (now - fps_last)
                fps_count = 0
                fps_last = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if DEV_MOUSE:
                cv2.putText(frame, "DEV_MOUSE: move to slice, click to select", (20, HEIGHT - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
