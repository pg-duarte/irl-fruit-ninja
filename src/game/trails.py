class Trails:
    def __init__(self, max_len=10, min_conf=0.2):
        self.max_len = max_len
        self.min_conf = min_conf
        self.data = {"left": [], "right": []}

    def update(self, hands):
        """
        hands: dict like {"left": (x,y,conf), "right": (x,y,conf)}
        """
        for k in ("left", "right"):
            x, y, conf = hands.get(k, (0.0, 0.0, 0.0))
            if conf >= self.min_conf:
                self.data[k].append((int(x), int(y)))
                if len(self.data[k]) > self.max_len:
                    self.data[k].pop(0)

    def get(self):
        return self.data
