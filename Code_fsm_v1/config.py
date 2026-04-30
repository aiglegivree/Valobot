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
STEER_SMOOTH_ALPHA = 0.35
STEER_DEADZONE = 2.5

HARD_TURN_THRESH = 90.0

T_CONFIRM_FRAMES = 1
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
LOW_SPEED = 0.15
LOW_SPEED_HOLD_SEC = 3.0
TURN_SPEED = 0.30
BLIND_SPEED = 0.25
SEARCH_SPEED = 0.20

STEER_GAIN = 0.13
TOP_SLICE_LOW_WIDTH_THRESHOLD = 30
TOP_SLICE_LOW_WIDTH_FRAMES = 2

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True
RECORD_RUN_VIDEO = True
RECORD_RUN_FPS = 20.0
RECORDINGS_DIR = "run_recordings"

# ============================================================
# SHOOTING
# ============================================================
BLUE_LOW = (101, 198, 22)
BLUE_HIGH = (135, 255, 255)

MIN_TARGET_AREA = 150
MIN_TARGET_WIDTH = 12
CENTER_TOL = 18
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
MIN_TIME_BETWEEN_SHOTS = 0.8
POST_SHOT_SETTLE_TIME = 0.2
POST_SHOT_CONFIRM_FRAMES = 4
NOMBRE_DE_CIBLES = 3
