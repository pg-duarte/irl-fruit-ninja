import os

class HandsProvider:
    """
    Provides hands in the project standard format:
      {"left": (x,y,conf), "right": (x,y,conf)}

    Strategy:
      - If OpenPose model exists -> use OpenPoseWrapper
      - Else -> fallback to mouse mock (so the game always runs)
    """

    def __init__(self, width, height, model_path=None):
        self.width = width
        self.height = height
        self.mouse_pos = (0, 0)

        # Default model location: src/pose/graph_opt.pb
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "graph_opt.pb")

        self.use_openpose = False
        self.pose = None

        # Try to init OpenPose only if model exists
        if os.path.isfile(model_path):
            try:
                from .openpose_wrapper import OpenPoseWrapper, OpenPoseConfig
                cfg = OpenPoseConfig(model_path=model_path, in_width=368, in_height=368, thr=0.2)
                self.pose = OpenPoseWrapper(cfg)
                self.use_openpose = True
            except Exception:
                self.use_openpose = False
                self.pose = None

    def set_mouse_pos(self, x, y):
        self.mouse_pos = (x, y)

    def get_hands(self, frame=None):
        """
        Returns (hands_dict, frame_out)
        - hands_dict: {"left": (x,y,conf), "right": (x,y,conf)}
        - frame_out: frame with optional skeleton drawn (if OpenPose active), else original frame
        """
        # --- OpenPose path ---
        if self.use_openpose and self.pose is not None and frame is not None:
            frame_out, h = self.pose.process_frame(frame, draw_skeleton=True)

            lx = float(h.left.x) if h.left.x is not None else 0.0
            ly = float(h.left.y) if h.left.y is not None else 0.0
            lc = float(h.left.conf)

            rx = float(h.right.x) if h.right.x is not None else 0.0
            ry = float(h.right.y) if h.right.y is not None else 0.0
            rc = float(h.right.conf)

            return {"left": (lx, ly, lc), "right": (rx, ry, rc)}, frame_out

        # --- Fallback mouse mock ---
        x, y = self.mouse_pos
        left = (float(x), float(y), 1.0)

        rx = min(self.width - 1, x + 120)
        right = (float(rx), float(y), 1.0)

        return {"left": left, "right": right}, frame

