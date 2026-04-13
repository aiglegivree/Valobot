# config.py

# ============================================================
# CAMERA
# ============================================================
W = 640
H = 480

CAM_ID = 0
USE_PICAMERA2 = True

# ============================================================
# VISION
# ============================================================
ROI_START_FRAC = 0.50
GAUSS_KSIZE = 15

KERNEL_SIZE = 15
MORPH_CLOSE_IT = 1

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

# black threshold in HSV
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 100

# ============================================================
# CORNER SHAPE DETECTION
# ============================================================
CORNER_PADDING = 1
CORNER_APPROX_EPSILON_FRAC = 0.04
CORNER_MIN_CONTOUR_AREA = 1500
SHARP_TURN_CORNERS = 6
T_CORNERS = 8

# ============================================================
# CONTROL
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
BASE_SPEED = 0.20
TURN_SPEED = 0.15
BLIND_SPEED = 0.25
SEARCH_SPEED = 0.15

STEER_GAIN = 0.1

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True

# ============================================================
# SHOOTING
# ============================================================
BLUE_LOW = (100, 140, 60)
BLUE_HIGH = (135, 255, 255)

MIN_TARGET_AREA = 1200
CENTER_TOL = 35
LOCK_FRAMES_REQUIRED = 5
TARGET_MATCH_DIST_PX = 80

MOTEUR_PIN = 19
SERVO_PIN = 18
PWM_FREQ = 50

ANGLE_LOADED = 21
ANGLE_FIRE_START = 10
ANGLE_INCREMENT = 23

MAG_CAPACITY = 4
MIN_TIME_BETWEEN_SHOTS = 0.8
POST_SHOT_SETTLE_TIME = 0.2
POST_SHOT_CONFIRM_FRAMES = 4
NOMBRE_DE_CIBLES = 4
