import sys
import os
import cv2
import time

# --- Fix Python path so "src" is treated as project root ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.game_core import Game
from game.trails import Trails
from pose.hands_provider import HandsProvider

WIDTH, HEIGHT = 640, 480

def main():
    game = Game(WIDTH, HEIGHT)
    trails_tracker = Trails(max_len=10, min_conf=0.2)
    provider = HandsProvider(WIDTH, HEIGHT)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Fruit Ninja Test")

    # Mouse callback updates the provider (mock hands)
    def mouse_callback(event, x, y, flags, param):
        provider.set_mouse_pos(x, y)

    cv2.setMouseCallback("Fruit Ninja Test", mouse_callback)

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_time
        last_time = now

        # Get hands (mock for now; later OpenPose will replace provider internals)
        hands, frame = provider.get_hands(frame)


        # Update trails from hands
        trails_tracker.update(hands)
        trails = trails_tracker.get()

        # Update game using trails
        game.update(dt, trails)

        # Draw fruits
        for fruit in game.fruits:
            cv2.circle(
                frame,
                (int(fruit.x), int(fruit.y)),
                fruit.radius,
                (0, 255, 0),
                -1
            )

        # Draw trails: left (blue), right (yellow)
        left_tr = trails.get("left", [])
        right_tr = trails.get("right", [])

        for i in range(len(left_tr) - 1):
            cv2.line(frame, left_tr[i], left_tr[i + 1], (255, 0, 0), 2)

        for i in range(len(right_tr) - 1):
            cv2.line(frame, right_tr[i], right_tr[i + 1], (0, 255, 255), 2)

        # Draw score
        cv2.putText(
            frame,
            f"Score: {game.score}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Fruit Ninja Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
