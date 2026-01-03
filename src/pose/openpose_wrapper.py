# src/pose/openpose_wrapper.py
#
# OpenPose (COCO-style body keypoints) via OpenCV DNN (TensorFlow .pb).
# Intended use (Pessoa B):
#   - Initialize once
#   - Per-frame: estimate_hands(frame_bgr) -> Hands (wrists, with elbow fallback)
#
# Notes:
# - This model is BODY-only (no 21-point hand keypoints). We use wrists (and elbow fallback).
# - Wrapper stays "pure": no smoothing, no trails, no game logic. Those live in hand_tracking.py.

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


# =========================
# Types (idealmente vêm de src/common/types.py)
# =========================
@dataclass
class HandPoint:
    x: Optional[float]
    y: Optional[float]
    conf: float


@dataclass
class Hands:
    left: HandPoint
    right: HandPoint


# =========================
# Config
# =========================
@dataclass
class OpenPoseConfig:
    # Path to TensorFlow graph (.pb)
    model_path: str = "graph_opt.pb"

    # Inference input resolution (bigger = better wrists, slower)
    in_width: int = 368
    in_height: int = 368

    # Thresholds (separados)
    thr_wrist: float = 0.07        # low to keep wrists when arms go up
    thr_skeleton: float = 0.15     # higher to avoid messy skeleton debug

    # Preprocessing
    swap_rb: bool = False          # model-dependent; keep configurable
    mean: Tuple[float, float, float] = (127.5, 127.5, 127.5)

    # If True, uses OpenCV DNN backend/target if available
    prefer_backend: bool = True

    # If True, allow elbow fallback when wrist confidence is low
    elbow_fallback: bool = True


# =========================
# COCO BODY_PARTS mapping (18 + background)
# =========================
BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18,
}

POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
]

# Indices we care about
R_WRIST = BODY_PARTS["RWrist"]   # 4
L_WRIST = BODY_PARTS["LWrist"]   # 7
R_ELBOW = BODY_PARTS["RElbow"]   # 3
L_ELBOW = BODY_PARTS["LElbow"]   # 6


class OpenPoseWrapper:
    """
    OpenPose model via OpenCV DNN (TensorFlow .pb).

    - init loads the network once.
    - per-frame inference returns wrists (x, y, conf), with elbow fallback.
    """

    def __init__(self, cfg: OpenPoseConfig):
        self.cfg = cfg

        if not os.path.isfile(cfg.model_path):
            raise FileNotFoundError(
                f"OpenPose model not found: {cfg.model_path}\n"
                f"Tip: set OpenPoseConfig(model_path=...) or put graph_opt.pb next to this file."
            )
        import cv2 as cv
        self.cv = cv
        self.net = self.cv.dnn.readNetFromTensorflow(cfg.model_path)

        # Default safe: CPU
        def _set_cpu():
            try:
                self.net.setPreferableBackend(self.cv.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(self.cv.dnn.DNN_TARGET_CPU)
            except Exception:
                pass
            print("[OpenPose] Using CPU")

        if not cfg.prefer_backend:
            _set_cpu()
            return

        # Try CUDA (FP16 first), but verify by doing a tiny forward later
        try:
            self.net.setPreferableBackend(self.cv.dnn.DNN_BACKEND_CUDA)
            try:
                self.net.setPreferableTarget(self.cv.dnn.DNN_TARGET_CUDA_FP16)
                print("[OpenPose] Requested CUDA FP16")
            except Exception:
                self.net.setPreferableTarget(self.cv.dnn.DNN_TARGET_CUDA)
                print("[OpenPose] Requested CUDA FP32")

            # IMPORTANT: some builds accept setPreferable* but fail on forward.
            # We'll mark it and validate on first forward call.
            self._cuda_requested = True

        except Exception:
            self._cuda_requested = False
            _set_cpu()


    # -------------------------
    # Internal: run network + decode all maxima
    # -------------------------
    def _infer_points(self, frame_bgr: np.ndarray) -> Tuple[List[Optional[Tuple[int, int]]], List[float]]:
        """
        Returns:
          - points[i] = (x,y) maxima location (in frame pixels) for each body part
          - confs[i]  = maxima confidence value
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return [None] * len(BODY_PARTS), [0.0] * len(BODY_PARTS)

        
                # one-time validation: if CUDA was requested but is invalid, fallback to CPU
        if getattr(self, "_cuda_requested", False) and not getattr(self, "_cuda_validated", False):
            try:
                # run a tiny dummy forward to force backend/target validation
                dummy = np.zeros((self.cfg.in_height, self.cfg.in_width, 3), dtype=np.uint8)
                blob0 = self.cv.dnn.blobFromImage(
                    dummy, 1.0, (self.cfg.in_width, self.cfg.in_height),
                    self.cfg.mean, swapRB=bool(self.cfg.swap_rb), crop=False
                )
                self.net.setInput(blob0)
                _ = self.net.forward()
                print("[OpenPose] CUDA validated OK")
            except Exception as e:
                # fallback to CPU
                try:
                    self.net.setPreferableBackend(self.cv.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(self.cv.dnn.DNN_TARGET_CPU)
                except Exception:
                    pass
                print("[OpenPose] CUDA failed at runtime -> fallback to CPU:", str(e))
            finally:
                self._cuda_validated = True

        
        frame_h, frame_w = frame_bgr.shape[:2]

        blob = self.cv.dnn.blobFromImage(
            frame_bgr,
            1.0,
            (self.cfg.in_width, self.cfg.in_height),
            self.cfg.mean,
            swapRB=bool(self.cfg.swap_rb),
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()

        if out.ndim != 4 or out.shape[1] < 19:
            raise RuntimeError(f"Unexpected network output shape: {out.shape} (expected N x >=19 x H x W)")

        # Keep only keypoint heatmaps (18 + background)
        out = out[:, :19, :, :]

        H = out.shape[2]
        W = out.shape[3]

        points: List[Optional[Tuple[int, int]]] = []
        confs: List[float] = []

        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = self.cv.minMaxLoc(heatMap)

            x = (frame_w * point[0]) / float(W)
            y = (frame_h * point[1]) / float(H)

            points.append((int(x), int(y)))
            confs.append(float(conf))

        return points, confs

    # -------------------------
    # Internal: pick wrists with threshold + fallback
    # -------------------------
    def _pick_wrists(self, points: List[Optional[Tuple[int, int]]], confs: List[float]) -> Hands:
        thr = float(self.cfg.thr_wrist)

        lp = points[L_WRIST] if confs[L_WRIST] >= thr else None
        lc = confs[L_WRIST] if lp is not None else 0.0

        rp = points[R_WRIST] if confs[R_WRIST] >= thr else None
        rc = confs[R_WRIST] if rp is not None else 0.0

        # elbow fallback
        if self.cfg.elbow_fallback:
            if rp is None and confs[R_ELBOW] >= thr:
                rp = points[R_ELBOW]
                rc = confs[R_ELBOW]
            if lp is None and confs[L_ELBOW] >= thr:
                lp = points[L_ELBOW]
                lc = confs[L_ELBOW]

        left = HandPoint(float(lp[0]), float(lp[1]), float(lc)) if lp is not None else HandPoint(None, None, 0.0)
        right = HandPoint(float(rp[0]), float(rp[1]), float(rc)) if rp is not None else HandPoint(None, None, 0.0)
        return Hands(left=left, right=right)

    # -------------------------
    # Public: wrists only
    # -------------------------
    def estimate_hands(self, frame_bgr: np.ndarray) -> Hands:
        points, confs = self._infer_points(frame_bgr)
        return self._pick_wrists(points, confs)

    # -------------------------
    # Public: optional debug draw (for dev builds)
    # -------------------------
    def draw_debug(
        self,
        frame_bgr: np.ndarray,
        draw_skeleton: bool = True,
        draw_all_keypoints: bool = False,
    ):
        """
        Returns (frame_out, hands).
        - Skeleton threshold uses cfg.thr_skeleton
        - Wrist selection uses cfg.thr_wrist
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, Hands(HandPoint(None, None, 0.0), HandPoint(None, None, 0.0))

        points, confs = self._infer_points(frame_bgr)
        hands = self._pick_wrists(points, confs)

        out = frame_bgr.copy()
        thr_skel = float(self.cfg.thr_skeleton)

        if draw_skeleton:
            for a, b in POSE_PAIRS:
                ia = BODY_PARTS[a]
                ib = BODY_PARTS[b]
                if confs[ia] >= thr_skel and confs[ib] >= thr_skel:
                    pa = points[ia]
                    pb = points[ib]
                    if pa is not None and pb is not None:
                        self.cv.line(out, pa, pb, (0, 255, 0), 2)

        if draw_all_keypoints:
            for i in range(len(BODY_PARTS)):
                p = points[i]
                if p is None:
                    continue
                c = confs[i]
                color = (0, 255, 0) if c >= thr_skel else (0, 165, 255)
                self.cv.circle(out, p, 2, color, self.cv.FILLED)

        # draw chosen hands
        if hands.left.conf > 0 and hands.left.x is not None and hands.left.y is not None:
            self.cv.circle(out, (int(hands.left.x), int(hands.left.y)), 7, (0, 255, 0), -1)
        if hands.right.conf > 0 and hands.right.x is not None and hands.right.y is not None:
            self.cv.circle(out, (int(hands.right.x), int(hands.right.y)), 7, (0, 255, 0), -1)

        return out, hands
