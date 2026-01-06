import random
from .fruit import Fruit

class Spawner:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.timer = 0.0
        self.spawn_interval = 1.0  # seconds

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.spawn_interval:
            self.timer = 0.0

            x = random.randint(80, self.width - 80)
            y = self.height + 40

            fruit = Fruit(x, y)

            # --- physics variation (Fruit Ninja–like) ---
            SLOW = 0.5  # 0.7 mild, 0.5 slow-mo

            # set gravity scaled (keeps height if vy is scaled too)
            fruit.g = 900.0 * SLOW

            # launch velocities scaled (same factor => same height)
            fruit.vx = random.uniform(-200.0, 200.0) * SLOW
            fruit.vy = -random.uniform(1000.0, 1350.0) * SLOW
            fruit.omega = random.uniform(-720.0, 720.0)
            return fruit
        return None

