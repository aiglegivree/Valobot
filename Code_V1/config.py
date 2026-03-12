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
W_TOP = 0.0
W_MID = 0.0
W_BOT = 1.0

# ============================================================
# CONTROL
# ============================================================
STEER_SMOOTH_ALPHA = 0.35      # lissage de la commande
STEER_DEADZONE = 5.0           # pour mettre à jour last_turn_sign

# seuil à partir duquel on considère qu'on prépare un gros virage
HARD_TURN_THRESH = 90.0

# ============================================================
# LOST LINE / STATES
# ============================================================
BLIND_TURN_FRAMES = 8          # nb de frames pendant lesquelles on continue à tourner
SEARCH_FRAMES = 18             # au-delà -> mode recherche

BLIND_TURN_STEER = 0.85        # steering forcé pendant blind turn
SEARCH_TURN_STEER = 0.60       # steering forcé pendant recherche

# ============================================================
# SPEED
# ============================================================
BASE_SPEED = 0.40              # 0..1
TURN_SPEED = 0.23              # gros virage
BLIND_SPEED = 0.25             # pendant perte ligne
SEARCH_SPEED = 0.15            # recherche



# mix steering -> roues
STEER_GAIN = 0.1              # influence du steering sur left/right

# ============================================================
# DEBUG
# ============================================================
SHOW_DEBUG = True
CAM_ID = 0
USE_PICAMERA2 = True           # True si tu utilises la PiCam avec Picamera2
