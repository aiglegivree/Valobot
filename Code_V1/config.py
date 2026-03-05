# config.py

# ---------------- Camera ----------------
CAM_ID = 0  # pas utilisé si PiCam (Picamera2), mais ok de le garder
W, H = 640, 480

# ROI (fractions of image height)
ROI_TOP_FRAC = 0.65
ROI_BOT_FRAC = 1.00

# Thresholding / detection
GAUSS_K = 5
MIN_CONTOUR_AREA = 150

# ---------------- Control ----------------
# speed scale 0..100 (% duty)
BASE_SPEED = 25
MIN_SPEED = 0
MAX_SPEED = 60

# Steering output expected in [-1, +1]
KP = 0.004
KD = 0.000

# EMA smoothing on error
EMA_ALPHA = 0.35

# Anti-jump gating (start large to avoid constant LOST)
GATE_PX = 160

# Line lost behavior
LOST_TIMEOUT_S = 0.35
SEARCH_STEER = 0.55
SEARCH_SPEED_FACTOR = 0.55

# ---------------- Motors GPIO (BCM) ----------------
ENA, IN1, IN2 = 13, 27, 22
ENB, IN3, IN4 = 18, 23, 24

PWM_FREQ = 1000

# Motor trim (compensate mismatch). 1.0 = no change.
LEFT_TRIM  = 1.00
RIGHT_TRIM = 1.00

# Direction fixes
INVERT_STEER = False
SWAP_MOTORS = False