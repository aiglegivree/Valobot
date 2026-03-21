# ============================================================
# CAMERA
# ============================================================
W = 640
H = 480

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

# ============================================================
# T DETECTION (ROBUST VERSION)
# ============================================================

# horizontal bar detection
T_WIDTH_RATIO = 2.2
T_MIN_ABS_WIDTH = 110
T_MIN_SIDE_EXT = 35
T_MAX_SIDE_RATIO = 1.8

# vertical scan zone
T_SCAN_Y_START_FRAC = 0.18
T_SCAN_Y_END_FRAC   = 0.72
T_MIN_BAR_ROWS = 6

# NEW robustness constraints
T_CENTER_MAX_ERR = 100        # reject T if bottom stem too far from robot center
T_MIN_WIDTH_GAIN = 60        # bar must be significantly wider than bottom
T_STEM_MIN_ROWS = 18         # require vertical stem continuity
T_STEM_X_TOL = 28            # tolerance around stem x when checking continuity

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
CAM_ID = 0
USE_PICAMERA2 = True
