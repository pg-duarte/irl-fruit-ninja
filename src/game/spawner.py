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
            x = random.randint(50, self.width - 50)
            y = self.height + 30
            return Fruit(x, y)
        return None
