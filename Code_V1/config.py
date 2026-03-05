# config.py

# ---------------- Camera ----------------
W, H = 640, 480
FPS = 60

# ---------------- Vision ----------------
ROI_START_FRAC = 0.30              # was 0.60; bigger ROI => see more
LOOKAHEAD_Y_FRAC = 0.35            # inside ROI (0=top, 1=bottom)
BAND_THICKNESS = 10
MIN_AREA = 600
INVERT_AFTER_OTSU = True

# Morphology
KERNEL_SIZE = 5
MORPH_CLOSE_IT = 1
MORPH_OPEN_IT = 1

# ---------------- Control ----------------
# Speeds 0..100
BASE_SPEED = 25
MAX_SPEED  = 40
MIN_SPEED  = 20

# Blend errors (normalized [-1, 1])
DEADBAND = 0.05
W_LOOK   = 0.3
W_BOTTOM = 0.7
KP = 5.0

# Steering smoothing
STEER_LPF_ALPHA = 0.35

# ---------------- Line lost recovery ----------------
LINE_LOST_TIMEOUT_S = 0.08   # after this, enter search
SEARCH_FORWARD = 0.0         # 0 = turn in place (recommended for 90°)
SEARCH_STEER_START = 10.0    # initial turning strength
SEARCH_STEER_MAX = 30.0      # max turning strength
SEARCH_RAMP_TIME = 0.40      # time to ramp to max

# ---------------- Motors GPIO (BCM) ----------------
ENA, IN1, IN2 = 13, 27, 22
ENB, IN3, IN4 = 18, 23, 24
PWM_FREQ = 1000

INVERT_LEFT_MOTOR  = False
INVERT_RIGHT_MOTOR = False

# Debug video
SHOW_MASK_WINDOW = True