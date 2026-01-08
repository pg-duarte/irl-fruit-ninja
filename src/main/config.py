# config.py
# =========================
# Configuração global da aplicação
# =========================

# -------- Vídeo --------
WIDTH = 640
HEIGHT = 480
CAM_INDEX = 1

# -------- Estabilização --------
USE_STABILIZATION = False

FACE_MARGIN = 110     
NUM_ANGLES = 4         
STAB_ZOOM = 1.0        

# -------- OpenPose --------
POSE_EVERY_N_FRAMES = 1
POSE_IN_SIZE = 256
POSE_THRESHOLD = 0.07
SHOW_OPENPOSE_SKELETON = True


# -------- Trails --------
TRAIL_TTL_SEC = 0.6
TRAIL_LEN_THRS = 10

# -------- Menu / UI --------
DWELL_S = 0.8
SWIPE_SPEED_PX_S = 1800.0
