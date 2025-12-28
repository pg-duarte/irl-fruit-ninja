# Temporary provider: returns hands from mouse (mock).
# Later, replace get_hands() to call OpenPose wrapper and return the same dict format:
# {"left": (x,y,conf), "right": (x,y,conf)}

class HandsProvider:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mouse_pos = (0, 0)

    def set_mouse_pos(self, x, y):
        self.mouse_pos = (x, y)

    def get_hands(self, frame=None):
        x, y = self.mouse_pos

        left = (float(x), float(y), 1.0)

        # Mock right hand: offset from the mouse
        rx = min(self.width - 1, x + 120)
        right = (float(rx), float(y), 1.0)

        return {"left": left, "right": right}
