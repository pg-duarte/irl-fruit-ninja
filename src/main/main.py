# src/main/main.py
# Teste webcam: OpenPose wrists -> tracker -> desenhar pontos + trails
# Funciona mesmo correndo diretamente (corrige sys.path)

import os
import sys
import time
import cv2 as cv

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pose.openpose_wrapper import OpenPose
from src.pose.hand_tracking import HandTracker


def draw_point(img, p, label):
    x, y, c = p
    if x is None or y is None or c <= 0:
        return
    x, y = int(x), int(y)
    cv.circle(img, (x, y), 8, (255, 255, 255), -1)
    cv.putText(img, f"{label} {c:.2f}", (x + 10, y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)


def draw_trail(img, trail, max_age=0.6):
    now = time.time()
    pts = [p for p in trail if now - p[2] <= max_age]
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        x0, y0 = int(pts[i-1][0]), int(pts[i-1][1])
        x1, y1 = int(pts[i][0]), int(pts[i][1])
        cv.line(img, (x0, y0), (x1, y1), (255, 255, 255), 2)


def main():
    model_path = os.path.join(_ROOT, "src", "pose", "graph_opt.pb")

    pose = OpenPose(model_path=model_path, in_size=368, thr=0.07, elbow_fallback=True, swap_rb=False)
    tracker = HandTracker(alpha=0.35, hold_frames=6, conf_decay=0.85, trail_len=24, min_conf=0.07)

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam.")

    show_trails = True
    last = time.time()
    fps = 0.0

    print("q sair | t trails | r reset | +/- threshold")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1)

        rawL, rawR = pose.wrists(frame)
        L, R, trailL, trailR, met = tracker.update(rawL, rawR)

        out = frame.copy()
        draw_point(out, L, "L")
        draw_point(out, R, "R")

        if show_trails:
            draw_trail(out, trailL)
            draw_trail(out, trailR)

        # fps
        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        cv.putText(out, f"FPS {fps:.1f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(out, f"thr {pose.thr:.2f}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(out, f"Lc {met['Lc']:.2f} Lv {met['Lv']:.0f}px/s", (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(out, f"Rc {met['Rc']:.2f} Rv {met['Rv']:.0f}px/s", (10, 97), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow("Pessoa B - wrists + tracking", out)

        k = cv.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("t"):
            show_trails = not show_trails
        if k == ord("r"):
            tracker.reset()
        if k in (ord("+"), ord("=")):
            pose.thr = min(0.5, pose.thr + 0.01)
        if k in (ord("-"), ord("_")):
            pose.thr = max(0.01, pose.thr - 0.01)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
