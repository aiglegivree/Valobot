# config.py

# ============================================================
# CAMERA
# ============================================================
W = 640
H = 480

# ============================================================
# VISION
# ============================================================
ROI_START_FRAC = 0.50          # on traite seulement la moitié basse
GAUSS_KSIZE = 5

# ligne noire sur fond blanc
USE_BINARY_INV = True

KERNEL_SIZE = 5
MORPH_CLOSE_IT = 2
MORPH_OPEN_IT = 1

BAND_THICKNESS = 6

# positions des 3 tranches dans la ROI
SLICE_TOP_FRAC = 0.05
SLICE_MID_FRAC = 0.50
SLICE_BOT_FRAC = 0.95

# poids pour l'erreur de steering
W_TOP = 0.05
W_MID = 0.05
W_BOT = 0.85

BOTTOM_SLICE_MAX_DIST = 200   # pixels from image center on bottom slice
CENTER_EXCEPTION_DIST = 60   # pixels

T_SIDE_MARGIN = 20

T_SCAN_Y_START_FRAC = 0.15
T_SCAN_Y_END_FRAC   = 0.75
T_MIN_BAR_ROWS = 3

T_WIDTH_RATIO = 2.2
T_MIN_ABS_WIDTH = 110
T_MIN_SIDE_EXT = 35
T_MAX_SIDE_RATIO = 1.8

# ============================================================
# CONTROL
# ============================================================
STEER_SMOOTH_ALPHA = 0.35      # lissage de la commande
STEER_DEADZONE = 5.0           # pour mettre à jour last_turn_sign

# seuil à partir duquel on considère qu'on prépare un gros virage
HARD_TURN_THRESH = 90.0

T_CONFIRM_FRAMES = 3
T_MEMORY_FRAMES = 5
STOP_ON_T_END = True
# ============================================================
# LOST LINE / STATES
# ============================================================
BLIND_TURN_FRAMES = 100          # nb de frames pendant lesquelles on continue à tourner
SEARCH_FRAMES = 200             # au-delà -> mode recherche

BLIND_TURN_STEER = 0.85        # steering forcé pendant blind turn
SEARCH_TURN_STEER = 0.60       # steering forcé pendant recherche

# ============================================================
# SPEED
# ============================================================
BASE_SPEED = 0.50              # 0..1
TURN_SPEED = 0.30              # gros virage
BLIND_SPEED = 0.25             # pendant perte ligne
SEARCH_SPEED = 0.15            # recherche



# mix steering -> roues
STEER_GAIN = 0.1             # influence du steering sur left/right

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True
CAM_ID = 0
USE_PICAMERA2 = True           # True si tu utilises la PiCam avec Picamera2
