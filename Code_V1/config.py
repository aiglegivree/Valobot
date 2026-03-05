# config.py

# ---------------- Camera ----------------
CAM_ID = 0
W, H = 640, 480

# ROI (fractions of image height)
ROI_TOP_FRAC = 0.60   # start of ROI (0..1)
ROI_BOT_FRAC = 1.00   # end of ROI

# Thresholding / detection
GAUSS_K = 5
INV_BINARY = True          # True if line is darker than floor (common)
THRESH = 120               # static threshold (tune)
MIN_CONTOUR_AREA = 400     # reject tiny blobs

# ---------------- Control ----------------
# Your speed scale is 0..100 (% duty)
BASE_SPEED = 25            # cruising speed (0..100)
MIN_SPEED = 0
MAX_SPEED = 60

# Steering output expected in [-1, +1]
KP = 0.004                 # steer gain in 1/px (tune)
KD = 0.000                 # optional (set >0 if needed)

# EMA filter on error
EMA_ALPHA = 0.35           # 0..1 (higher = less smoothing)

# Anti-jump gating: reject measurements too far from prediction
GATE_PX = 90               # px threshold (tune)

# Line lost behavior
LOST_TIMEOUT_S = 0.35      # after this, increase search aggressiveness
SEARCH_STEER = 0.55        # base steer in LOST (0..1)
SEARCH_SPEED_FACTOR = 0.55 # speed multiplier in LOST

# ---------------- Motors GPIO (BCM) ----------------
# IN4 -> GPIO24 (pin 18)
ENA, IN1, IN2 = 13, 27, 22
ENB, IN3, IN4 = 18, 23, 24

PWM_FREQ = 1000

# Motor trim (compensate mismatch). 1.0 = no change.
LEFT_TRIM  = 1.00
RIGHT_TRIM = 1.00

# If your wiring makes the robot steer the wrong way, flip one of these:
INVERT_STEER = False       # True -> steer sign inverted
SWAP_MOTORS = False        # True -> swap left/right commands