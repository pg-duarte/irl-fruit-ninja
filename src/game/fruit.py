class Fruit:
    def __init__(self, x, y, radius=25):
        self.x = float(x)
        self.y = float(y)
        self.radius = int(radius)
        self.alive = True

        # simple physics
        self.vx = 0.0
        self.vy = -920.0  # initial upward velocity (px/s)
        self.g  = 900.0   # gravity (px/s^2)

    def update(self, dt):
        self.vy += self.g * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt

    def is_offscreen(self, width, height):
        return (self.y - self.radius > height + 50 or
                self.x + self.radius < -50 or
                self.x - self.radius > width + 50)
