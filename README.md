# irl-fruit-ninja
This project consists of the development of an interactive Fruit Ninja–style game controlled through hand movements captured by a camera. The player interacts with the game by slicing virtual fruits using their hands in real time, without any physical controller.

The system processes a video stream from a webcam and applies human pose estimation using OpenPose to extract keypoints corresponding to the player’s hands. To improve robustness and visual stability, the video is first stabilized by detecting the player’s face and performing template matching based on the correlation coefficient, followed by image translation implemented on the GPU using OpenCL.

Game logic is built on top of the stabilized video and pose data. Fruits are spawned dynamically on the screen and are considered “cut” when the trajectory of a hand intersects their position. The game includes a scoring system and is designed to be modular and extensible, allowing future additions such as power-ups, combos, and difficulty scaling.

The project emphasizes real-time performance, modular software architecture, and the efficient use of GPU acceleration, avoiding unnecessary data transfers between CPU and GPU where possible.
