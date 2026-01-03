from .spawner import Spawner
from .collision import segment_circle_collision

def _speed(p1, p2, dt):
    if dt <= 0:
        return 0.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx*dx + dy*dy) ** 0.5 / dt

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.spawner = Spawner(width, height)
        self.fruits = []
        self.score = 0
        self.min_slice_speed = 250.0  # px/s

    def update(self, dt, trails):
        """
        trails: dict like {"left": [(x,y),...], "right": [(x,y),...]}
        """
        fruit = self.spawner.update(dt)
        if fruit:
            self.fruits.append(fruit)

        for fruit in self.fruits:
            fruit.update(dt)

            if not fruit.alive:
                continue

            for hand in ("left", "right"):
                tr = trails.get(hand, [])
                if len(tr) < 2:
                    continue

                p1, p2 = tr[-2], tr[-1]
                if _speed(p1, p2, dt) < self.min_slice_speed:
                    continue

                if segment_circle_collision(p1, p2, fruit.x, fruit.y, fruit.radius):
                    fruit.alive = False
                    self.score += 10
                    break  # fruit already sliced

        self.fruits = [
            f for f in self.fruits
            if f.alive and not f.is_offscreen(self.width, self.height)
        ]
    def draw(self, frame):
        # draw all fruits (sprites)
        for fruit in self.fruits:
            fruit.draw(frame)

        # simple score overlay (optional)
        import cv2 as cv
        cv.putText(frame, f"Score: {self.score}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        return frame

