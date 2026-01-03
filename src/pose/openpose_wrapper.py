# src/pose/openpose_wrapper.py
# Wrapper simples: OpenPose (COCO body) via OpenCV DNN (.pb)
# Devolve só punhos (com fallback para cotovelos)

import os
import cv2 as cv


BODY = {
    "RElbow": 3, "RWrist": 4,
    "LElbow": 6, "LWrist": 7,
}

class OpenPose:
    def __init__(self, model_path, in_size=368, thr=0.07, elbow_fallback=True, swap_rb=False):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        self.net = cv.dnn.readNetFromTensorflow(model_path)
        self.in_size = int(in_size)
        self.thr = float(thr)
        self.elbow_fallback = bool(elbow_fallback)
        self.swap_rb = bool(swap_rb)

    def _infer(self, frame):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(
            frame, 1.0, (self.in_size, self.in_size),
            (127.5, 127.5, 127.5), swapRB=self.swap_rb, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()

        if out.ndim != 4 or out.shape[1] < 19:
            raise RuntimeError(f"Saída inesperada da rede: {out.shape}")

        out = out[:, :19, :, :]  # 18 + background
        H, W = out.shape[2], out.shape[3]

        pts = []
        confs = []
        for i in range(19):
            hm = out[0, i, :, :]
            _, c, _, p = cv.minMaxLoc(hm)
            x = int(w * p[0] / W)
            y = int(h * p[1] / H)
            pts.append((x, y))
            confs.append(float(c))
        return pts, confs

    def wrists(self, frame):
        pts, conf = self._infer(frame)

        def pick(wrist_i, elbow_i):
            if conf[wrist_i] >= self.thr:
                return pts[wrist_i], conf[wrist_i]
            if self.elbow_fallback and conf[elbow_i] >= self.thr:
                return pts[elbow_i], conf[elbow_i]
            return None, 0.0

        r, rc = pick(BODY["RWrist"], BODY["RElbow"])
        l, lc = pick(BODY["LWrist"], BODY["LElbow"])

        # devolve (x,y,conf) ou (None,None,0)
        L = (l[0], l[1], lc) if l else (None, None, 0.0)
        R = (r[0], r[1], rc) if r else (None, None, 0.0)
        return L, R
