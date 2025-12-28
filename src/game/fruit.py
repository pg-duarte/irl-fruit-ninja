class Fruit:
    def __init__(self, x, y, radius=25):
        self.x = x
        self.y = y
        self.radius = radius
        self.alive = True

    def update(self, dt):
        self.y -= 200 * dt  # sobe no ecrã

    def is_offscreen(self, height):
        return self.y + self.radius < 0
