import os
import sys
import time
import cv2
import numpy as np

# ========= OpenCL (ANTES de importar stabilization) =========
os.environ.setdefault("PYOPENCL_CTX", "0")
os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "1")

# ========= Imports do projeto =========
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pose.openpose_wrapper import OpenPose
from pose.hand_tracking import HandTracker
from game.game_core import Game
from ui.state import AppState
from ui.menu import Button, MenuUI

from stabilization.face_detection import detectar_e_preparar_rosto
from stabilization.stabilizer import GPUImageStabilizer


# ========= Config =========
WIDTH, HEIGHT = 640, 480
CAM_INDEX = 1

USE_STABILIZATION = True

POSE_EVERY_N_FRAMES = 2
POSE_IN_SIZE = 256

TRAIL_TTL_SEC = 0.6

DWELL_S = 0.8
SWIPE_SPEED_PX_S = 1800.0


# ========= Helpers simples =========
def gray01(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def gray01_to_bgr(gray):
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def trail_xy(trail, now):
    return [(int(x), int(y)) for x, y, t, c in trail if now - t <= TRAIL_TTL_SEC]


def draw_trail(frame, trail, now, color):
    pts = [(x, y, t) for x, y, t, c in trail if now - t <= TRAIL_TTL_SEC]
    if len(pts) < 2:
        return

    for i in range(len(pts) - 1):
        x1, y1, t1 = pts[i]
        x2, y2, t2 = pts[i + 1]
        age = now - t2
        fade = max(0.0, 1.0 - age / TRAIL_TTL_SEC)
        col = (int(color[0] * fade), int(color[1] * fade), int(color[2] * fade))
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)


def choose_cursor(L, R, met):
    lx, ly, lc = L
    rx, ry, rc = R

    if rc > lc and rx is not None:
        return (int(rx), int(ry)), met.get("Rv", 0.0)
    if lx is not None:
        return (int(lx), int(ly)), met.get("Lv", 0.0)

    return None, 0.0


# ========= Main =========
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Erro ao abrir webcam")
        return

    # ---------- Estabilização (init simples) ----------
    stabilizer = None
    if USE_STABILIZATION:
        print("A detetar face para estabilização...")
        time.sleep(2)

        ret, frame0 = cap.read()
        if ret:
            frame0 = cv2.resize(frame0, (WIDTH, HEIGHT))
            frame0 = cv2.flip(frame0, 1)

            template, x0, y0, dbg = detectar_e_preparar_rosto(frame0)
            if template is not None:
                template01 = template.astype(np.float32) / 255.0
                stabilizer = GPUImageStabilizer(
                    template01,
                    int(x0),
                    int(y0),
                    (HEIGHT, WIDTH),
                    faceMargin=100,
                    numAngles=4,
                )
                print("Estabilização ON")
            else:
                print("Face não detetada → estabilização OFF")

    # ---------- OpenPose + Tracker ----------
    model_path = os.path.join(SRC_DIR, "pose", "graph_opt.pb")
    pose = OpenPose(model_path, in_size=POSE_IN_SIZE, thr=0.07)
    tracker = HandTracker(trail_len=120)

    # ---------- Jogo + UI ----------
    game = Game(WIDTH, HEIGHT)
    state = AppState.MENU

    menu = MenuUI(
        [
            Button("start", "Start Game", (220, 160, 200, 55)),
            Button("exit", "Exit", (220, 240, 200, 55)),
        ],
        dwell_s=DWELL_S,
        swipe_speed_px_s=SWIPE_SPEED_PX_S,
    )

    frame_idx = 0
    last_rawL = (None, None, 0.0)
    last_rawR = (None, None, 0.0)
    last_t = time.time()

    cv2.namedWindow("Fruit Ninja")

    # ---------- Loop principal ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_t
        last_t = now

        # ---------- Estabilização ----------
        base_frame = frame
        if stabilizer is not None:
            try:
                g = gray01(frame)
                g_stab, _, _, _ = stabilizer.process(g)
                base_frame = gray01_to_bgr(g_stab)
            except Exception:
                base_frame = frame

        # ---------- OpenPose (decimado) ----------
        if frame_idx % POSE_EVERY_N_FRAMES == 0:
            try:
                last_rawL, last_rawR = pose.wrists(base_frame)
            except Exception:
                pass
        frame_idx += 1

        # ---------- Tracking ----------
        L, R, trailL, trailR, met = tracker.update(last_rawL, last_rawR, t=now)
        cursor, speed = choose_cursor(L, R, met)

        trails = {
            "left": trail_xy(trailL, now),
            "right": trail_xy(trailR, now),
        }

        # ---------- State machine ----------
        if state == AppState.MENU:
            picked = menu.update(cursor, speed, now)
            menu.draw(base_frame, cursor)

            cv2.putText(base_frame, "Fruit Ninja", (200, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

            if picked == "start":
                game = Game(WIDTH, HEIGHT)
                state = AppState.PLAYING
            elif picked == "exit":
                break

        elif state == AppState.PLAYING:
            game.update(dt, trails)
            base_frame = game.draw(base_frame)

            # trails visuais
            draw_trail(base_frame, trailL, now, (255, 0, 0))
            draw_trail(base_frame, trailR, now, (0, 255, 255))

        cv2.imshow("Fruit Ninja", base_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
