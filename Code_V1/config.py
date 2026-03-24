# ============================================================
# CAMERA
# ============================================================
W = 640
H = 480

CAM_ID = 0
USE_PICAMERA2 = True

# ============================================================
# VISION - LINE FOLLOWING
# ============================================================
ROI_START_FRAC = 0.50
GAUSS_KSIZE = 5

USE_BINARY_INV = True

KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2
MORPH_OPEN_IT = 1

BAND_THICKNESS = 6

# slices
SLICE_TOP_FRAC = 0.05
SLICE_MID_FRAC = 0.50
SLICE_BOT_FRAC = 0.95

# steering weights
W_TOP = 0.05
W_MID = 0.05
W_BOT = 0.85

# bottom gating
BOTTOM_SLICE_MAX_DIST = 200
CENTER_EXCEPTION_DIST = 60

# ============================================================
# T DETECTION (LOCAL STEM FRAME)
# ============================================================
# ============================================================
# T detection based on tracked trunk direction
# ============================================================
T_ENABLE = True

T_SCAN_OFFSET_PX = -35       # negatif = remonte le long du tronc vers le haut

T_BRANCH_HALF_LEN = 90
T_BRANCH_MIN_LEN = 22
T_BRANCH_THICKNESS = 2

T_TRUNK_CHECK_LEN = 28
T_TRUNK_MIN_HITS = 16

T_BAR_MIN_HITS = 28
T_CENTER_TOL_PX = 28

T_MIN_DIST_FROM_BOTTOM = 25
T_ANGLE_MIN_DEG = 65
# ============================================================
# CONTROL - LINE FOLLOWING
# ============================================================
STEER_SMOOTH_ALPHA = 0.35
STEER_DEADZONE = 5.0

HARD_TURN_THRESH = 90.0

T_CONFIRM_FRAMES = 3
T_MEMORY_FRAMES = 5
STOP_ON_T_END = True

# ============================================================
# LOST LINE / STATES
# ============================================================
BLIND_TURN_FRAMES = 100
SEARCH_FRAMES = 200

BLIND_TURN_STEER = 0.85
SEARCH_TURN_STEER = 0.60

# ============================================================
# SPEED
# ============================================================
BASE_SPEED = 0.40
TURN_SPEED = 0.30
BLIND_SPEED = 0.25
SEARCH_SPEED = 0.15

STEER_GAIN = 0.1

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True

# ============================================================
# AUTO FIRE / BLUE TARGET DETECTION
# ============================================================
import numpy as np

BLUE_LOW = np.array([95, 60, 25], dtype=np.uint8)
BLUE_HIGH = np.array([135, 255, 255], dtype=np.uint8)

MIN_TARGET_AREA = 1200

# aiming
CENTER_TOL = 15
LOCK_FRAMES_REQUIRED = 5
TURN_SHOT_SPEED = 0.25

# same target check after shot
TARGET_MATCH_DIST_PX = 80
POST_SHOT_CONFIRM_FRAMES = 5
POST_SHOT_SETTLE_TIME = 0.20

# mission
NOMBRE_DE_CIBLES = 1

# ============================================================
# SHOOTER
# ============================================================
MOTEUR_PIN = 19
SERVO_PIN = 18

PWM_FREQ = 50
ANGLE_LOADED = 21
ANGLE_FIRE_START = 10
ANGLE_INCREMENT = 23

MAG_CAPACITY = 7
MIN_TIME_BETWEEN_SHOTS = 0.5