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

# black threshold in HSV
BLACK_H_LOW = 0
BLACK_S_LOW = 0
BLACK_V_LOW = 0

BLACK_H_HIGH = 180
BLACK_S_HIGH = 170
BLACK_V_HIGH = 110

# ============================================================
# T DETECTION
# ============================================================
T_ENABLE = True

# kept for compatibility, even if not used directly
T_SIDE_MARGIN = 20

# zone where we look for the top bar
T_SCAN_Y_START_FRAC = 0.12
T_SCAN_Y_END_FRAC   = 0.72

# the upper zone must spread far enough on both sides of the stem
T_MIN_SIDE_EXT = 32
T_MAX_SIDE_RATIO = 2.4

# candidate bar must be clearly wider than the bottom stem
T_WIDTH_RATIO = 1.9
T_MIN_ABS_WIDTH = 95
T_MIN_WIDTH_GAIN = 45

# tilted T support:
# require the spread condition to hold on several rows, not necessarily consecutive
T_MIN_BAR_ROWS = 6

# stem continuity below the bar zone
T_STEM_MIN_ROWS = 22
T_STEM_X_TOL = 26
T_STEM_MIN_HITS = 14

# optional plausibility check:
# do not reject slightly off-center T, only absurdly far ones
T_CENTER_MAX_ERR = 120

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
BASE_SPEED = 0.40
TURN_SPEED = 0.30
BLIND_SPEED = 0.25
SEARCH_SPEED = 0.15

STEER_GAIN = 0.1

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True