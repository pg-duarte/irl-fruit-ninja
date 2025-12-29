# src/main/main.py
#
# Webcam test for Pessoa B pipeline:
#   OpenPoseWrapper (wrists) -> HandTracker (smoothing + hold-last + trails) -> overlay
#
# This file is "arranged" to work BOTH ways:
#   1) Recommended: from project root
#        python -m src.main.main
#   2) Also works if you run the file directly:
#        python src/main/main.py
#
# Keys:
#   q  - quit
#   t  - toggle trails
#   s  - toggle skeleton debug (from wrapper)
#   a  - toggle all keypoints debug (from wrapper)
#   r  - reset tracker
#   + / -  - increase/decrease wrist threshold

from __future__ import annotations

import os
import sys
import time
import math

# -------------------------
# Make imports work even when running this file directly
# -------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2 as cv  # noqa: E402

from src.pose.openpose_wrapper import OpenPoseWrapper, OpenPoseConfig, Hands  # noqa: E402
from src.pose.hand_tracking import HandTracker, HandTrackingConfig  # noqa: E402


def draw_trails(frame, trails, max_age_s: float = 0.6) -> None:
    """Draw trails as fading polylines (keeps it simple for debugging)."""
    now = time.time()

    for side_points in (trails.left, trails.right):
        if len(side_points) < 2:
            continue

        pts = [p for p in side_points if (now - p.t) <= max_age_s]
        if len(pts) < 2:
            continue

        for i in range(1, len(pts)):
            p0 = pts[i - 1]
            p1 = pts[i]
            cv.line(frame, (int(p0.x), int(p0.y)), (int(p1.x), int(p1.y)), (255, 255, 255), 2)


def draw_hands(frame, hands: Hands) -> None:
    """Draw hand points + confidence."""
    def draw_one(label: str, hp) -> None:
        if hp.x is None or hp.y is None or hp.conf <= 0:
            return
        x, y = int(hp.x), int(hp.y)
        cv.circle(frame, (x, y), 8, (255, 255, 255), -1)
        cv.putText(
            frame, f"{label} {hp.conf:.2f}",
            (x + 10, y - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA
        )

    draw_one("L", hands.left)
    draw_one("R", hands.right)


def resolve_model_path() -> str:
    """
    Prefer graph_opt.pb next to openpose_wrapper.py (src/pose/graph_opt.pb).
    If you store it elsewhere, hardcode or read from env.
    """
    pose_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, "src", "pose"))
    model_path = os.path.join(pose_dir, "graph_opt.pb")
    return model_path


def main() -> None:
    model_path = resolve_model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Não encontrei o modelo em: {model_path}\n"
            f"Coloca o graph_opt.pb em src/pose/ ou altera resolve_model_path()."
        )

    pose_cfg = OpenPoseConfig(
        model_path=model_path,
        in_width=368,
        in_height=368,
        thr_wrist=0.07,
        thr_skeleton=0.15,
        elbow_fallback=True,
        swap_rb=False,  # se os resultados forem maus, experimenta True
    )
    pose = OpenPoseWrapper(pose_cfg)

    tracker = HandTracker(HandTrackingConfig(
        min_conf=0.07,
        hold_frames=6,
        conf_decay=0.85,
        ema_alpha=0.35,
        trail_max_points=24,
        trail_min_conf=0.05,
        trail_min_dist_px=6.0,
    ))

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam (index 0).")

    show_trails = True
    show_skeleton = False
    show_all_keypoints = False

    last_time = time.time()
    fps = 0.0

    print("Webcam test started.")
    print("Keys: q quit | t trails | s skeleton | a all keypoints | r reset | +/- wrist thr")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for more intuitive interaction
        frame = cv.flip(frame, 1)

        # OpenPose -> raw wrists
        raw_hands = pose.estimate_hands(frame)

        # Tracking -> smoothed hands + trails
        hands, trails, met = tracker.update(raw_hands)

        out = frame.copy()

        # Wrapper debug draw
        if show_skeleton or show_all_keypoints:
            out, _ = pose.draw_debug(out, draw_skeleton=show_skeleton, draw_all_keypoints=show_all_keypoints)

        # Hands + trails overlays
        draw_hands(out, hands)
        if show_trails:
            draw_trails(out, trails)

        # FPS calc
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        # HUD
        cv.putText(out, f"FPS: {fps:.1f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(out, f"thr_wrist: {pose.cfg.thr_wrist:.2f}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(out, f"L v: {met['left_v']:.0f}px/s  conf:{met['left_conf']:.2f}", (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(out, f"R v: {met['right_v']:.0f}px/s conf:{met['right_conf']:.2f}", (10, 97), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow("Pose + Tracking (Pessoa B)", out)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            show_trails = not show_trails
        elif key == ord("s"):
            show_skeleton = not show_skeleton
        elif key == ord("a"):
            show_all_keypoints = not show_all_keypoints
        elif key == ord("r"):
            tracker.reset()
        elif key in (ord("+"), ord("=")):
            pose.cfg.thr_wrist = min(0.50, float(pose.cfg.thr_wrist) + 0.01)
        elif key in (ord("-"), ord("_")):
            pose.cfg.thr_wrist = max(0.01, float(pose.cfg.thr_wrist) - 0.01)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
