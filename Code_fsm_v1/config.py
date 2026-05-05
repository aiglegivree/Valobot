# ============================================================
# CAMERA
# ============================================================
W = 320
H = 240

CAM_ID = 0
USE_PICAMERA2 = True

# ============================================================
# VISION
# ============================================================
ROI_START_FRAC = 0.50
GAUSS_KSIZE = 7

KERNEL_SIZE = 15
MORPH_CLOSE_IT = 3
SHAPE_MORPH_MODE = 'remove_holes'

BAND_THICKNESS = 3
SLICE_POINT_HISTORY_FRAMES = 5

# slices
SLICE_TOP_FRAC = 0.05
SLICE_MID_FRAC = 0.50
SLICE_BOT_FRAC = 0.95

# steering weights
W_TOP = 0.15
W_MID = 0.25
W_BOT = 0.60

# bottom gating
BOTTOM_SLICE_MAX_DIST = 100
CENTER_EXCEPTION_DIST = 30

# black threshold in HSV
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 255
BLACK_V_HIGH = 160

# ============================================================
# CORNER SHAPE DETECTION
# ============================================================
CORNER_PADDING = 5
CORNER_APPROX_EPSILON_FRAC = 0.01
CORNER_MIN_CONTOUR_AREA = 1500
CORNER_MAX_ANGLE_DEG = 160.0
T_CORNERS = 8

# ============================================================
# CONTROL
# ============================================================
STEER_SMOOTH_ALPHA = 0.45
STEER_DEADZONE = 2.5

HARD_TURN_THRESH = 90.0

T_CONFIRM_FRAMES = 2
T_LINE_FRAMES = 30
T_BOTTOM_SLICE_WHITE_RATIO = 0.60
T_SIDE_MIN_X_OFFSET = 12

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
BASE_SPEED = 0.50
LOW_SPEED = 0.25
LOW_SPEED_HOLD_SEC = 4.0
TURN_SPEED = 0.30
BLIND_SPEED = 0.30
SEARCH_SPEED = 0.30

STEER_GAIN = 0.14
TOP_SLICE_LOW_WIDTH_THRESHOLD = 30
TOP_SLICE_LOW_WIDTH_FRAMES = 4

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True
RECORD_RUN_VIDEO = True
RECORD_RUN_FPS = 20.0
RECORDINGS_DIR = "run_recordings"

# ============================================================
# BLUE TARGETS
# ============================================================
BLUE_LOW_CORRIDOR = (81, 130, 22)
BLUE_HIGH_CORRIDOR = (114, 255, 99)
MIN_TARGET_AREA_CORRIDOR = 450
MIN_TARGET_WIDTH_CORRIDOR = 7

BLUE_LOW_SHOOTING = (101, 208, 22)
BLUE_HIGH_SHOOTING = (135, 255, 255)
MIN_TARGET_AREA_SHOOTING = 150
MIN_TARGET_WIDTH_SHOOTING = 25

# Backward-compatible aliases. Keep these mapped to the shooting profile.
BLUE_LOW = BLUE_LOW_SHOOTING
BLUE_HIGH = BLUE_HIGH_SHOOTING
MIN_TARGET_AREA = MIN_TARGET_AREA_SHOOTING
MIN_TARGET_WIDTH = MIN_TARGET_WIDTH_SHOOTING

# ============================================================
# SHOOTING
# ============================================================
CENTER_TOL = 8
MIN_AIM_TURN_SPEED = 0.25
LOCK_FRAMES_REQUIRED = 5
TARGET_MATCH_DIST_PX = 40
TARGET_LOCK_MISS_FRAMES = 10
SHOOTING_TARGET_MIN_RATIO = 0.30
SHOOTING_TARGET_MAX_RATIO = 0.70

MOTEUR_PIN = 19
SERVO_PIN = 18
PWM_FREQ = 50

ANGLE_LOADED = 21
ANGLE_FIRE_START = 10
ANGLE_INCREMENT = 23

MAG_CAPACITY = 7
MIN_TIME_BETWEEN_SHOTS = 1.5
POST_SHOT_SETTLE_TIME = 0.5
POST_SHOT_CONFIRM_FRAMES = 30
NOMBRE_DE_CIBLES = 3
