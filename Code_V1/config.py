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
# Le T est détecté dans le repère local du tronc, pas dans le repère image.

# le bas de la ligne suivie doit rester raisonnablement proche du centre
T_CENTER_MAX_ERR = 160

# zone verticale de scan de la ligne connectée
T_SCAN_Y_START_FRAC = 0.12
T_SCAN_Y_END_FRAC = 0.95

# tracking du tronc vers le haut
T_TRACK_MAX_STEP = 26
T_MIN_TRACK_POINTS = 18

# préfiltre candidat : la ligne doit s'élargir localement
T_CANDIDATE_MIN_WIDTH = 55
T_CANDIDATE_MIN_WIDTH_GAIN = 16
T_CANDIDATE_MIN_WIDTH_RATIO = 1.35

# sondes locales dans le repère du tronc
T_SIDE_PROBE_LEN = 38
T_SIDE_PROBE_HALF_WIDTH = 4

T_BACK_PROBE_LEN = 22
T_BACK_PROBE_HALF_WIDTH = 4

# nombre minimal de hits dans les sondes
T_BACK_MIN_HITS = 12
T_SIDE_MIN_HITS = 14

# évite les faux T très déséquilibrés
T_MIN_BALANCE_RATIO = 0.35

# contrainte d'orientation :
# si |dot(branch, trunk)| est trop grand -> branche trop parallèle au tronc
T_DIRECTION_DOT_MAX = 0.55

# gauche et droite doivent être vraiment opposées
# avec dot(left, right) proche de -1
T_MIN_SEPARATION = 0.30

# nombre de lignes candidates consécutives pour valider
T_CONFIRM_ROWS = 2

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