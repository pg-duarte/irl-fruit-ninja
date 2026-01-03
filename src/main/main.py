import sys
import os
import cv2
import time

# --- Fix Python path so "src" is treated as project root ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.game_core import Game

from pose.openpose_wrapper import OpenPoseWrapper, OpenPoseConfig
from pose.hand_tracking import HandTracker, HandTrackingConfig

WIDTH, HEIGHT = 640, 480

# === Trail fade settings (EDIT THIS) ===
TRAIL_TTL_SEC = 4.0          # how long the trail stays visible
DRAW_SKELETON_DEBUG = False  # set True if you want skeleton lines (slower)

def _fade_color(base_bgr, fade01: float):
    """fade01: 1.0=full, 0.0=gone"""
    f = max(0.0, min(1.0, fade01))
    return (int(base_bgr[0] * f), int(base_bgr[1] * f), int(base_bgr[2] * f))

def _trail_to_xy_list(trail_points, now_t: float):
    """Convert TrailPoint list -> [(x,y), ...] filtered by TTL."""
    out = []
    for p in trail_points:
        if (now_t - p.t) <= TRAIL_TTL_SEC:
            out.append((int(p.x), int(p.y)))
    return out

def _draw_trail(frame, trail_points, now_t: float, base_color_bgr, thickness=2):
    """Draw segments with fade based on age (TTL)."""
    # keep only points within TTL
    pts = [p for p in trail_points if (now_t - p.t) <= TRAIL_TTL_SEC]
    if len(pts) < 2:
        return

    # draw older->newer segments with increasing intensity
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        age = now_t - p2.t
        fade = 1.0 - (age / TRAIL_TTL_SEC)  # 1 -> 0
        color = _fade_color(base_color_bgr, fade)

        cv2.line(frame, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color, thickness)

def main():
    game = Game(WIDTH, HEIGHT)

    # --- OpenPose model path ---
    model_path = os.path.join(os.path.dirname(__file__), "..", "pose", "graph_opt.pb")
    model_path = os.path.abspath(model_path)

    pose_cfg = OpenPoseConfig(
        model_path=model_path,
        in_width=368,
        in_height=368,
        thr_wrist=0.07,
        thr_skeleton=0.15,
        elbow_fallback=True,
        swap_rb=False,
    )
    pose = OpenPoseWrapper(pose_cfg)

    # Hand tracker (smoothing + hold-last + trails + velocity)
    ht_cfg = HandTrackingConfig(
        min_conf=0.07,
        ema_alpha=0.35,
        trail_max_points=120,      # can be bigger because TTL will filter
        trail_min_conf=0.05,
        trail_min_dist_px=6.0
    )
    tracker = HandTracker(ht_cfg)

    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Fruit Ninja Test")

    last_time = time.time()

    # --- FPS counter state ---
    fps_last = time.time()
    fps_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_time
        last_time = now

        # --- Pose inference ---
        if DRAW_SKELETON_DEBUG:
            frame, raw_hands = pose.draw_debug(frame, draw_skeleton=True, draw_all_keypoints=False)
        else:
            raw_hands = pose.estimate_hands(frame)

        # --- Tracking (smoothing + trails with timestamps) ---
        smoothed_hands, pose_trails, metrics = tracker.update(raw_hands, t=now)

        # Convert pose_trails -> format that Game.update expects
        trails_xy = {
            "left":  _trail_to_xy_list(pose_trails.left, now),
            "right": _trail_to_xy_list(pose_trails.right, now),
        }

        # --- Game update ---
        game.update(dt, trails_xy)

        # --- Draw fruits (sprites) ---
        frame = game.draw(frame)

        # --- Draw trails with fade (TTL) ---
        _draw_trail(frame, pose_trails.left,  now, base_color_bgr=(255, 0, 0), thickness=2)    # left = blue
        _draw_trail(frame, pose_trails.right, now, base_color_bgr=(0, 255, 255), thickness=2)  # right = yellow

        # --- HUD ---
        cv2.putText(frame, f"Score: {game.score}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- FPS update (once per ~1s) ---
        fps_count += 1
        if now - fps_last >= 1.0:
            fps = fps_count / (now - fps_last)
            fps_count = 0
            fps_last = now

        # --- FPS draw (top-left corner) ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Fruit Ninja Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
