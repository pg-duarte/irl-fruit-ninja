# main.py
import os
import sys
import time
import cv2
import numpy as np

import config  # <- ficheiro config.py (mesma pasta deste main.py)

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


# ========= Helpers =========
def gray01(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def trail_xy(trail, now):
    return [(int(x), int(y)) for x, y, t, c in trail if now - t <= config.TRAIL_TTL_SEC]


def trail_length_px(trail):
    if len(trail) < 2:
        return 0.0

    length = 0.0
    for i in range(len(trail) - 1):
        x1, y1 = trail[i]
        x2, y2 = trail[i + 1]
        length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return length


def draw_trail(frame, trail, now, color):
    pts = [(x, y, t) for x, y, t, c in trail if now - t <= config.TRAIL_TTL_SEC]
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        x1, y1, t1 = pts[i]
        x2, y2, t2 = pts[i + 1]
        age = now - t2
        fade = max(0.0, 1.0 - age / config.TRAIL_TTL_SEC)
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


# ========= Mouse =========
clicked = False


def mouse_callback(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True


def main():
    global clicked

    W, H = config.WIDTH, config.HEIGHT

    cap = cv2.VideoCapture(config.CAM_INDEX)
    if not cap.isOpened():
        print("Erro ao abrir webcam")
        return

    stabilizer = None
    template = None
    x0 = y0 = None

    # ---------- Pré-view para escolher frame e inicializar estabilização ----------
    if config.USE_STABILIZATION:
        print("A detetar face para estabilização...")
        time.sleep(1)

        win_pick = "Pick frame (click esquerdo) / ESC sai"
        cv2.namedWindow(win_pick, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_pick, mouse_callback)

        clicked = False
        chosen_frame = None

        while True:
            ret, frame0 = cap.read()
            if not ret:
                break

            cv2.imshow(win_pick, frame0)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if clicked:
                chosen_frame = frame0.copy()
                break

        cv2.destroyWindow(win_pick)

        if chosen_frame is None:
            cap.release()
            cv2.destroyAllWindows()
            return

        # prepara frame escolhido
        frame0 = cv2.resize(chosen_frame, (W, H))
        frame0 = cv2.flip(frame0, 1)

        template, x0, y0, processed_frame = detectar_e_preparar_rosto(frame0)
        cv2.imshow("Face Detection", processed_frame)

        if template is not None:
            cv2.imshow("Face", template)

            template01 = template.astype(np.float32) / 255.0
            stabilizer = GPUImageStabilizer(
                template01,
                int(x0),
                int(y0),
                (H, W),
                faceMargin=int(config.FACE_MARGIN),
                numAngles=int(config.NUM_ANGLES),
            )
            print("Estabilização ON")
        else:
            print("Face não detetada → estabilização OFF")
            stabilizer = None

    # ---------- OpenPose + Tracker ----------
    model_path = os.path.join(SRC_DIR, "pose", "graph_opt.pb")
    pose = OpenPose(model_path, in_size=int(config.POSE_IN_SIZE), thr=float(config.POSE_THRESHOLD))
    tracker = HandTracker(trail_len=120)

    # ---------- Jogo + UI ----------
    game = Game(W, H)
    state = AppState.MENU

    menu = MenuUI(
        [
            Button("start", "Start Game", (220, 160, 200, 55)),
            Button("exit", "Exit", (220, 240, 200, 55)),
        ],
        dwell_s=float(config.DWELL_S),
        swipe_speed_px_s=float(config.SWIPE_SPEED_PX_S),
    )

    frame_idx = 0
    last_rawL = (None, None, 0.0)
    last_rawR = (None, None, 0.0)
    last_t = time.time()

    cv2.namedWindow("Fruit Ninja")
    cv2.setMouseCallback("Fruit Ninja", mouse_callback)  # 1x, fora do loop
    clicked = False

    # ---------- Loop principal ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # normaliza frame SEMPRE antes de usar
        frame = cv2.resize(frame, (W, H))
        frame = cv2.flip(frame, 1)

        base_frame = frame  # GARANTIDO sempre

        now = time.time()
        dt = now - last_t
        last_t = now

        # ---------- Re-detect template por clique (só se estabilização ativa) ----------
        if clicked and stabilizer is not None and config.USE_STABILIZATION:
            clicked = False
            print("Re-detect template...")

            template_backup = template.copy() if template is not None else None

            new_template, new_x0, new_y0, processed_frame = detectar_e_preparar_rosto(frame)
            cv2.imshow("Face Detection", processed_frame)

            if new_template is not None:
                template = new_template
                x0, y0 = new_x0, new_y0
                cv2.imshow("Face", template)

                stabilizer.updtate_template(template.astype(np.float32) / 255.0)

                if hasattr(stabilizer, "reset"):
                    stabilizer.reset(int(x0), int(y0))
            else:
                if template_backup is not None:
                    template = template_backup
                    cv2.imshow("Face", template)

        # ---------- Estabilização ----------
        if stabilizer is not None and config.USE_STABILIZATION:
            try:
                g = gray01(frame)
                stabilized_bgr, _, _, _ = stabilizer.process(g, frame, zoom=float(config.STAB_ZOOM))
                base_frame = stabilized_bgr

                # Cost map (best effort)
                try:
                    heat_map = stabilizer.get_cost_map_visual()
                    cv2.imshow("Cost Map", heat_map)
                except Exception:
                    pass

            except Exception:
                base_frame = frame

        # ---------- OpenPose (decimado) ----------
        if frame_idx % int(config.POSE_EVERY_N_FRAMES) == 0:
            try:
                last_rawL, last_rawR = pose.wrists(base_frame)

                # NEW: desenhar esqueleto completo (opcional via config.py)
                if config.SHOW_OPENPOSE_SKELETON:
                    base_frame = pose.draw_skeleton(base_frame)

            except Exception:
                pass
        frame_idx += 1

        # ---------- Tracking ----------
        L, R, trailL, trailR, met = tracker.update(last_rawL, last_rawR, t=now)
        cursor, speed = choose_cursor(L, R, met)

        # ---------- State machine ----------
        if state == AppState.MENU:
            picked = menu.update(cursor, speed, now)
            menu.draw(base_frame, cursor)

            cv2.putText(
                base_frame, "Fruit Ninja", (200, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3
            )

            if picked == "start":
                game = Game(W, H)
                state = AppState.PLAYING
            elif picked == "exit":
                break

        elif state == AppState.PLAYING:
            trails = {
                "left": trail_xy(trailL, now),
                "right": trail_xy(trailR, now),
            }

            len_L = trail_length_px(trails["left"])
            len_R = trail_length_px(trails["right"])
            trail_length = len_L + len_R

            if trail_length < config.TRAIL_LEN_THRS:
                state = AppState.MENU

            game.update(dt, trails)
            base_frame = game.draw(base_frame)

            draw_trail(base_frame, trailL, now, (255, 0, 0))
            draw_trail(base_frame, trailR, now, (0, 255, 255))

        # ---------- Render ----------
        cv2.imshow("Fruit Ninja", base_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
