from .spawner import Spawner
from .collision import segment_circle_collision

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.spawner = Spawner(width)
        self.fruits = []
        self.score = 0

    def update(self, dt, trail):
        fruit = self.spawner.update(dt)
        if fruit:
            self.fruits.append(fruit)

        for fruit in self.fruits:
            fruit.update(dt)

            if len(trail) >= 2:
                if segment_circle_collision(
                    trail[-2], trail[-1],
                    fruit.x, fruit.y, fruit.radius
                ):
                    if fruit.alive:
                        fruit.alive = False
                        self.score += 10

        self.fruits = [
            f for f in self.fruits
            if f.alive and not f.is_offscreen(self.height)
        ]
