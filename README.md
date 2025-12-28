# irl-fruit-ninja
This project consists of the development of an interactive Fruit Ninja–style game controlled through hand movements captured by a camera. The player interacts with the game by slicing virtual fruits using their hands in real time, without any physical controller.

The system processes a video stream from a webcam and applies human pose estimation using OpenPose to extract keypoints corresponding to the player’s hands. To improve robustness and visual stability, the video is first stabilized by detecting the player’s face and performing template matching based on the correlation coefficient, followed by image translation implemented on the GPU using OpenCL.

Game logic is built on top of the stabilized video and pose data. Fruits are spawned dynamically on the screen and are considered “cut” when the trajectory of a hand intersects their position. The game includes a scoring system and is designed to be modular and extensible, allowing future additions such as power-ups, combos, and difficulty scaling.

The project emphasizes real-time performance, modular software architecture, and the efficient use of GPU acceleration, avoiding unnecessary data transfers between CPU and GPU where possible.

## Team Workflow Guide

### 1. Clone the repository
Each team member must clone the repository (do NOT create folders manually):

cd Desktop
git clone https://github.com/pg-duarte/irl-fruit-ninja.git
cd irl-fruit-ninja

---

### 2. Create your own branch
Each person works on their own feature branch.

Stabilization (Person A):
git checkout -b feat/stabilization
git push -u origin feat/stabilization

Pose Estimation (Person B):
git checkout -b feat/pose
git push -u origin feat/pose

Game & UI (Person C):
git checkout -b feat/game

---

### 3. Folder responsibility
Each member must work only inside their assigned folder:

src/stabilization/ -> Stabilization & OpenCL
src/pose/ -> OpenPose & hand tracking
src/game/ and src/ui/ -> Game logic & rendering

Do not modify other folders without coordination.

---

### 4. Commit and push changes

git add .
git commit -m "Short description of the change"
git push

---

### 5. Keep your branch up to date
Before starting a new work session:

git checkout main
git pull
git checkout <your-branch>
git merge main

---

### 6. Pull Requests
When a feature is ready:

1. Open GitHub
2. Create a Pull Request to merge your branch into main
3. Wait for review before merging

Direct commits to main are not allowed.

---

### 7. General rules
- Never work directly on main
- Use clear commit messages
- Keep code modular and clean
- Communicate before changing shared files
