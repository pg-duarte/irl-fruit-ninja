import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class HandPoint:
    x: Optional[float]
    y: Optional[float]
    conf: float


@dataclass
class Hands:
    left: HandPoint
    right: HandPoint


@dataclass
class OpenPoseConfig:
    # Path to TensorFlow graph (.pb)
    model_path: str = "graph_opt.pb"

    # Inference input resolution (smaller = faster)
    in_width: int = 320
    in_height: int = 240

    # Confidence threshold for keypoints
    thr: float = 0.2

    # If True, uses OpenCV DNN backend/target if available
    prefer_backend: bool = True


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

R_WRIST = BODY_PARTS["RWrist"]  # 4
L_WRIST = BODY_PARTS["LWrist"]  # 7


class OpenPoseWrapper:
    """
    OpenPose model via OpenCV DNN (TensorFlow .pb).

    Single-person heuristic: global maxima per heatmap.
    Returns wrists (x, y, conf).
    """

    def __init__(self, cfg: OpenPoseConfig):
        self.cfg = cfg

        if not os.path.isfile(cfg.model_path):
            raise FileNotFoundError(
                f"OpenPose model not found: {cfg.model_path}\n"
                f"Tip: set OpenPoseConfig(model_path=...) or place graph_opt.pb accordingly."
            )

        import cv2 as cv
        self.cv = cv
        self.net = self.cv.dnn.readNetFromTensorflow(cfg.model_path)

        if cfg.prefer_backend:
            try:
                self.net.setPreferableBackend(self.cv.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(self.cv.dnn.DNN_TARGET_CPU)
            except Exception:
                pass

    def _infer_points(self, frame_bgr: np.ndarray) -> Tuple[List[Optional[Tuple[int, int]]], List[float]]:
        if frame_bgr is None or frame_bgr.size == 0:
            points = [None] * len(BODY_PARTS)
            confs = [0.0] * len(BODY_PARTS)
            return points, confs

        frame_h, frame_w = frame_bgr.shape[:2]

        blob = self.cv.dnn.blobFromImage(
            frame_bgr,
            1.0,
            (self.cfg.in_width, self.cfg.in_height),
            (127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()

        # Keep only the first 19 heatmaps (18 keypoints + background)
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

            if conf > self.cfg.thr:
                points.append((int(x), int(y)))
                confs.append(float(conf))
            else:
                points.append(None)
                confs.append(float(conf))

        return points, confs

    def estimate_hands(self, frame_bgr: np.ndarray) -> Hands:
        points, confs = self._infer_points(frame_bgr)

        lp = points[L_WRIST]
        lc = confs[L_WRIST]
        left = HandPoint(float(lp[0]), float(lp[1]), lc) if lp is not None else HandPoint(None, None, 0.0)

        rp = points[R_WRIST]
        rc = confs[R_WRIST]
        right = HandPoint(float(rp[0]), float(rp[1]), rc) if rp is not None else HandPoint(None, None, 0.0)

        return Hands(left=left, right=right)

    def process_frame(self, frame_bgr: np.ndarray, draw_skeleton: bool = False):
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, Hands(HandPoint(None, None, 0.0), HandPoint(None, None, 0.0))

        points, _confs = self._infer_points(frame_bgr)
        hands = self.estimate_hands(frame_bgr)

        out = frame_bgr.copy()

        if draw_skeleton:
            for pair in POSE_PAIRS:
                id_from = BODY_PARTS[pair[0]]
                id_to = BODY_PARTS[pair[1]]
                if points[id_from] and points[id_to]:
                    self.cv.line(out, points[id_from], points[id_to], (0, 255, 0), 2)
                    self.cv.circle(out, points[id_from], 3, (0, 0, 255), self.cv.FILLED)
                    self.cv.circle(out, points[id_to], 3, (0, 0, 255), self.cv.FILLED)

        return out, hands


if __name__ == "__main__":
    import cv2 as cv

    cfg = OpenPoseConfig(
        model_path=os.path.join(os.path.dirname(__file__), "graph_opt.pb"),
        in_width=368,
        in_height=368,
        thr=0.1,
    )

    pose = OpenPoseWrapper(cfg)

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam (index 0).")

    print("A correr. Carrega 'q' para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out, _hands = pose.process_frame(frame, draw_skeleton=True)
        cv.imshow("OpenPose (OpenCV DNN) webcam", out)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
