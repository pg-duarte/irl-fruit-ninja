import random
from .fruit import Fruit

class Spawner:
    def __init__(self, width):
        self.width = width
        self.timer = 0
        self.spawn_interval = 1.0  # segundos

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.spawn_interval:
            self.timer = 0
            x = random.randint(50, self.width - 50)
            return Fruit(x, 480)
        return None
