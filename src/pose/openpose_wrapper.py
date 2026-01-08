# src/pose/openpose_wrapper.py
# Wrapper simples: OpenPose (COCO body) via OpenCV DNN (.pb)
# Devolve só punhos (com fallback para cotovelos)
#
# + NEW: guarda os keypoints do último forward e permite desenhar o esqueleto completo

import os
import cv2 as cv


BODY = {
    "RElbow": 3, "RWrist": 4,
    "LElbow": 6, "LWrist": 7,
}

# Ligações COCO (pares de índices) para desenhar esqueleto
POSE_PAIRS = [
    (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
]


class OpenPose:
    def __init__(
        self,
        model_path,
        in_size=512,
        thr=0.02,
        elbow_fallback=True,
        swap_rb=False,
        heatmap_blur_ksize=5,   # 0 desliga blur
        refine_window=3,        # 0 desliga refinamento; recomendado 3 ou 5
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        self.net = cv.dnn.readNetFromTensorflow(model_path)
        self.in_size = int(in_size)
        self.thr = float(thr)
        self.elbow_fallback = bool(elbow_fallback)
        self.swap_rb = bool(swap_rb)

        self.heatmap_blur_ksize = int(heatmap_blur_ksize)
        self.refine_window = int(refine_window)

        # NEW: cache do último forward (para desenhar esqueleto)
        self._last_pts = None
        self._last_confs = None

    def _infer(self, frame):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(
            frame,
            1.0,
            (self.in_size, self.in_size),
            (127.5, 127.5, 127.5),
            swapRB=self.swap_rb,
            crop=False,
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

            # blur para reduzir picos espúrios
            if self.heatmap_blur_ksize and self.heatmap_blur_ksize >= 3:
                k = self.heatmap_blur_ksize
                if k % 2 == 0:
                    k += 1
                hm_s = cv.GaussianBlur(hm, (k, k), 0)
            else:
                hm_s = hm

            # pico inicial
            _, c, _, p = cv.minMaxLoc(hm_s)
            px, py = int(p[0]), int(p[1])

            # refinamento por centro-de-massa numa janela pequena
            if self.refine_window and self.refine_window >= 3:
                rw = self.refine_window
                if rw % 2 == 0:
                    rw += 1
                r = rw // 2

                x0 = max(0, px - r)
                y0 = max(0, py - r)
                x1 = min(W - 1, px + r)
                y1 = min(H - 1, py + r)

                patch = hm_s[y0:y1 + 1, x0:x1 + 1]
                if patch.size > 0:
                    sumv = float(patch.sum())
                    if sumv > 1e-9:
                        xs = 0.0
                        ys = 0.0
                        for yy in range(patch.shape[0]):
                            row = patch[yy, :]
                            y_abs = y0 + yy
                            for xx in range(patch.shape[1]):
                                v = float(row[xx])
                                xs += v * float(x0 + xx)
                                ys += v * float(y_abs)
                        px_f = xs / sumv
                        py_f = ys / sumv
                    else:
                        px_f, py_f = float(px), float(py)
                else:
                    px_f, py_f = float(px), float(py)
            else:
                px_f, py_f = float(px), float(py)

            # map heatmap coords -> frame coords
            x = int(round(w * px_f / float(W)))
            y = int(round(h * py_f / float(H)))

            # clamp
            if x < 0:
                x = 0
            elif x >= w:
                x = w - 1
            if y < 0:
                y = 0
            elif y >= h:
                y = h - 1

            pts.append((x, y))
            confs.append(float(c))

        # NEW: guardar para draw_skeleton()
        self._last_pts = pts
        self._last_confs = confs

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

        L = (l[0], l[1], lc) if l else (None, None, 0.0)
        R = (r[0], r[1], rc) if r else (None, None, 0.0)
        return L, R

    # NEW: desenhar esqueleto completo (COCO)
    def draw_skeleton(self, frame):
        if self._last_pts is None or self._last_confs is None:
            return frame

        # joints
        for (x, y), c in zip(self._last_pts, self._last_confs):
            if c >= self.thr:
                cv.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # bones
        for i, j in POSE_PAIRS:
            if self._last_confs[i] >= self.thr and self._last_confs[j] >= self.thr:
                x1, y1 = self._last_pts[i]
                x2, y2 = self._last_pts[j]
                cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        return frame
