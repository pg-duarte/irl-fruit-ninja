# src/pose/openpose_wrapper.py
# Wrapper simples: OpenPose (COCO body) via OpenCV DNN (.pb)
# Devolve só punhos (com fallback para cotovelos)
#
# Fix de estabilidade/precisão:
# - Gaussian blur no heatmap antes de procurar o pico
# - Refinamento do pico por centro-de-massa numa janela pequena (sub-pixel em coords do heatmap)
# Isto reduz jitter e picos espúrios sem ficar pesado.

import os
import cv2 as cv


BODY = {
    "RElbow": 3, "RWrist": 4,
    "LElbow": 6, "LWrist": 7,
}


class OpenPose:
    def __init__(
        self,
        model_path,
        in_size=368,
        thr=0.07,
        elbow_fallback=True,
        swap_rb=False,
        heatmap_blur_ksize=5,   # NEW: 0 desliga blur
        refine_window=3,        # NEW: 0 desliga refinamento; recomendado 3 ou 5
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

            # --- NEW: blur para reduzir picos espúrios ---
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

            # --- NEW: refinamento por centro-de-massa numa janela pequena ---
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
                # evita problemas se patch for vazio
                if patch.size > 0:
                    # pesos: valores do heatmap (>=0 normalmente)
                    sumv = float(patch.sum())
                    if sumv > 1e-9:
                        # coordenadas locais
                        xs = 0.0
                        ys = 0.0
                        for yy in range(patch.shape[0]):
                            row = patch[yy, :]
                            y_abs = y0 + yy
                            # soma ponderada em x
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
