import cv2
import numpy as np
import math
from camera import Camera


class CorridorNavigator:
    def __init__(
        self,
        proto=False,
        video="movement.avi",
        alpha=0.25,
        stop_y_threshold=300,
        start_y_threshold=150,
        candidate_miss_threshold=5,
        lab_lower=(0, 0, 0),
        lab_upper=(255, 109, 255),
        # New tuning for one-side + front-wall behavior
        corner_target_offset_px=20,
        corner_gain_px_per_px=0.6,
        front_tilt_gain_px_per_deg=4.0,
    ):
        # --- Input source configuration ---
        self.proto = proto
        self.video = video

        if self.proto:
            self.cap = cv2.VideoCapture(self.video)
            if not self.cap.isOpened():
                raise RuntimeError(f"Error: Cannot open video '{self.video}'")
        else:
            self.cap = Camera()

        # --- Parameters ---
        self.alpha = alpha
        self.stop_y_threshold = stop_y_threshold
        self.start_y_threshold = start_y_threshold
        self.candidate_miss_threshold = candidate_miss_threshold

        self.lab_lower = np.array(lab_lower, dtype=np.uint8)
        self.lab_upper = np.array(lab_upper, dtype=np.uint8)

        # --- New parameters for one-side + front-wall control ---
        self.corner_target_offset_px = corner_target_offset_px
        self.corner_gain_px_per_px = corner_gain_px_per_px
        self.front_tilt_gain_px_per_deg = front_tilt_gain_px_per_deg

        # --- Temporal smoothing state ---
        self.horizontal_smooth = None
        self.left_smooth = None
        self.right_smooth = None

        self.horizontal_misses = 0
        self.left_misses = 0
        self.right_misses = 0

        self.left_stable_frames = 0
        self.right_stable_frames = 0

        # --- Previous measured representatives (for ghost-angle updates) ---
        self.prev_left_rep = None
        self.prev_right_rep = None

        # --- Ghost TTL: how many frames a ghost line is kept alive ---
        self.ghost_max_frames = 5
        self.left_ghost_frames = 0
        self.right_ghost_frames = 0

        # --- Turn-candidate memory ---
        self.turn_candidate = None   # "LEFT", "RIGHT", or None
        self.candidate_locked = False
        self.last_turn_candidate = "RIGHT"

        # --- Public state/output attributes ---
        self.state = "FORWARD"
        self.target_x = None
        self.target_visible = False
        self.intersection = None

        # --- Frame/cache attributes ---
        self.frame = None
        self.original = None
        self.mask = None
        self.edges = None

        self.H = None
        self.W = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def intersect_hough_lines(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        a2 = np.cos(theta2)
        b2 = np.sin(theta2)

        A = np.array([[a1, b1], [a2, b2]], dtype=np.float32)
        B = np.array([rho1, rho2], dtype=np.float32)

        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None

        x, y = np.linalg.solve(A, B)
        return float(x), float(y)

    @staticmethod
    def hough_y_at_x(line, x):
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)

        if abs(b) < 1e-6:
            return None

        y = (rho - a * x) / b
        return float(y)

    @staticmethod
    def hough_line_endpoints(rho, theta, length=1000):
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))

        return (x1, y1), (x2, y2)

    @staticmethod
    def median_line(lines):
        if not lines:
            return None

        rhos = np.array([line[0] for line in lines], dtype=np.float32)
        thetas = np.array([line[1] for line in lines], dtype=np.float32)

        return float(np.median(rhos)), float(np.median(thetas))

    @staticmethod
    def smooth_line(previous, current, alpha=0.25):
        if current is None:
            return previous
        if previous is None:
            return current

        prev_rho, prev_theta = previous
        curr_rho, curr_theta = current

        rho = alpha * curr_rho + (1.0 - alpha) * prev_rho
        theta = alpha * curr_theta + (1.0 - alpha) * prev_theta

        return (rho, theta)

    @staticmethod
    def wrap_angle_diff(delta):
        return (delta + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # Image processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fill_holes(mask):
        h, w = mask.shape
        flood = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        return cv2.bitwise_or(mask, flood_inv)

    def create_mask(self, frame):
        blur = cv2.GaussianBlur(frame, (3, 3), 1)
        lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)

        mask = cv2.inRange(lab, self.lab_lower, self.lab_upper)
        mask = self.fill_holes(mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((17, 17), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def detect_edges(mask):
        return cv2.Canny(mask, 50, 150)

    @staticmethod
    def detect_hough_lines(edges):
        return cv2.HoughLines(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60
        )

    @staticmethod
    def classify_lines(lines):
        horizontal_lines = []
        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                deg = math.degrees(theta)

                if 80 <= deg <= 100:
                    horizontal_lines.append((rho, theta))
                elif 30 < deg < 80:
                    left_lines.append((rho, theta))
                elif 100 < deg < 150:
                    right_lines.append((rho, theta))

        return horizontal_lines, left_lines, right_lines

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    def update_smoothed_lines(self, horizontal_rep, left_rep, right_rep):
        if horizontal_rep is not None:
            self.horizontal_smooth = self.smooth_line(
                self.horizontal_smooth, horizontal_rep, self.alpha
            )
            self.horizontal_misses = 0
        else:
            self.horizontal_misses += 1

        if left_rep is not None:
            self.left_smooth = self.smooth_line(
                self.left_smooth, left_rep, self.alpha
            )
            self.left_misses = 0
            self.left_stable_frames += 1
        else:
            self.left_misses += 1
            self.left_stable_frames = 0

        if right_rep is not None:
            self.right_smooth = self.smooth_line(
                self.right_smooth, right_rep, self.alpha
            )
            self.right_misses = 0
            self.right_stable_frames += 1
        else:
            self.right_misses += 1
            self.right_stable_frames = 0

    def update_ghost_lines(self, left_rep, right_rep):
        # Left missing, right visible → ghost-update left angle
        if (
            left_rep is None
            and self.left_smooth is not None
            and right_rep is not None
            and self.prev_right_rep is not None
        ):
            if self.left_ghost_frames < self.ghost_max_frames:
                delta_theta = self.wrap_angle_diff(right_rep[1] - self.prev_right_rep[1])
                ghost_rho, ghost_theta = self.left_smooth
                self.left_smooth = (ghost_rho, ghost_theta + delta_theta)
                self.left_ghost_frames += 1
            else:
                self.left_smooth = None

        elif left_rep is not None:
            self.left_ghost_frames = 0

        # Right missing, left visible → ghost-update right angle
        if (
            right_rep is None
            and self.right_smooth is not None
            and left_rep is not None
            and self.prev_left_rep is not None
        ):
            if self.right_ghost_frames < self.ghost_max_frames:
                delta_theta = self.wrap_angle_diff(left_rep[1] - self.prev_left_rep[1])
                ghost_rho, ghost_theta = self.right_smooth
                self.right_smooth = (ghost_rho, ghost_theta + delta_theta)
                self.right_ghost_frames += 1
            else:
                self.right_smooth = None

        elif right_rep is not None:
            self.right_ghost_frames = 0

        if left_rep is not None:
            self.prev_left_rep = left_rep
        if right_rep is not None:
            self.prev_right_rep = right_rep

    def update_state(self, update=True):
        if update:
            if self.state == "FORWARD":
                if self.horizontal_smooth is not None:
                    y_center = self.hough_y_at_x(self.horizontal_smooth, self.W / 2)
                    if y_center is not None and y_center > self.stop_y_threshold:
                        self.state = "TURN"
                        if self.turn_candidate is None and self.last_turn_candidate is not None:
                            self.turn_candidate = self.last_turn_candidate
                        self.candidate_locked = True
                        self.last_turn_candidate = self.turn_candidate

                        self.left_smooth = None
                        self.right_smooth = None
                        self.left_misses = 0
                        self.right_misses = 0
                        self.left_stable_frames = 0
                        self.right_stable_frames = 0
                        self.prev_left_rep = None
                        self.prev_right_rep = None

            if self.state == "TURN":
                N = 2

                one_side_stable = (
                    (self.left_misses == 0 and self.left_stable_frames >= N)
                    or (self.right_misses == 0 and self.right_stable_frames >= N)
                )
                two_sides_stable = (
                    self.left_misses == 0 and self.left_stable_frames >= N
                    and self.right_misses == 0 and self.right_stable_frames >= N
                )
                front_visible = (
                    self.horizontal_smooth is not None
                    and self.horizontal_misses == 0
                )

                # Exit only if we have a coherent corridor view:
                # either both sides, or one side confirmed by a front wall
                if two_sides_stable or (one_side_stable and front_visible):
                    self.state = "FORWARD"
                    self.candidate_locked = False
                    self.turn_candidate = None

    def update_turn_candidate(self, left_rep, right_rep):
        if self.state != "FORWARD" or self.candidate_locked:
            return

        if self.turn_candidate == "LEFT" and left_rep is not None:
            self.turn_candidate = None
        elif self.turn_candidate == "RIGHT" and right_rep is not None:
            self.turn_candidate = None

        if self.turn_candidate is None:
            if (
                self.left_misses >= self.candidate_miss_threshold
                and self.right_misses < self.candidate_miss_threshold
            ):
                self.turn_candidate = "LEFT"
            elif (
                self.right_misses >= self.candidate_miss_threshold
                and self.left_misses < self.candidate_miss_threshold
            ):
                self.turn_candidate = "RIGHT"

    def update_target(self):
        self.target_x = None
        self.target_visible = False
        self.intersection = None

        both_present = self.left_smooth is not None and self.right_smooth is not None
        at_least_one_measured = self.left_misses == 0 or self.right_misses == 0
        front_visible = self.horizontal_smooth is not None and self.horizontal_misses == 0

        # --------------------------------------------------------------
        # Case 1 — Both side lines available
        # --------------------------------------------------------------
        if both_present and at_least_one_measured:
            intersection = self.intersect_hough_lines(self.left_smooth, self.right_smooth)
            if intersection is not None:
                ix, iy = intersection
                ix_clamped = int(round(ix))
                ix_clamped = max(0, min(self.W - 1, ix_clamped))
                self.intersection = (ix, iy)
                self.target_x = ix_clamped
                self.target_visible = True
                return

        # --------------------------------------------------------------
        # Case 2 — One side line + front wall
        # Keep front wall horizontal and constrain the visible corner
        # to remain on the correct side of image center.
        # --------------------------------------------------------------
        if front_visible:
            rho, theta = self.horizontal_smooth
            deg = math.degrees(theta)
            angle_error_deg = deg - 90.0
            tilt_correction = self.front_tilt_gain_px_per_deg * angle_error_deg

            if self.left_smooth is not None and self.left_misses == 0:
                corner = self.intersect_hough_lines(self.left_smooth, self.horizontal_smooth)
                if corner is not None:
                    ix, iy = corner
                    self.intersection = (ix, iy)

                    desired_corner_x = (self.W / 2) - self.corner_target_offset_px
                    corner_error = ix - desired_corner_x
                    corner_correction = -self.corner_gain_px_per_px * corner_error

                    target_x = int(round((self.W / 2) + tilt_correction + corner_correction))
                    self.target_x = max(0, min(self.W - 1, target_x))
                    self.target_visible = True
                    return

            if self.right_smooth is not None and self.right_misses == 0:
                corner = self.intersect_hough_lines(self.right_smooth, self.horizontal_smooth)
                if corner is not None:
                    ix, iy = corner
                    self.intersection = (ix, iy)

                    desired_corner_x = (self.W / 2) + self.corner_target_offset_px
                    corner_error = ix - desired_corner_x
                    corner_correction = -self.corner_gain_px_per_px * corner_error

                    target_x = int(round((self.W / 2) + tilt_correction + corner_correction))
                    self.target_x = max(0, min(self.W - 1, target_x))
                    self.target_visible = True
                    return

        # --------------------------------------------------------------
        # Case 3 — Front wall only
        # --------------------------------------------------------------
        if front_visible:
            rho, theta = self.horizontal_smooth
            deg = math.degrees(theta)

            angle_error_deg = deg - 90.0
            target_x = int(round((self.W / 2) + self.front_tilt_gain_px_per_deg * angle_error_deg))
            target_x = max(0, min(self.W - 1, target_x))

            self.target_x = target_x
            self.target_visible = True
            return

        # --------------------------------------------------------------
        # Case 4 — No info → go straight
        # --------------------------------------------------------------
        return

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def draw_dashed_line(image, pt1, pt2, color, thickness=2, dash_length=20, gap_length=12):
        x1, y1 = pt1
        x2, y2 = pt2

        dx = x2 - x1
        dy = y2 - y1
        line_len = math.hypot(dx, dy)

        if line_len < 1:
            return

        ux = dx / line_len
        uy = dy / line_len

        dist = 0
        while dist < line_len:
            start_x = int(x1 + ux * dist)
            start_y = int(y1 + uy * dist)

            end_dist = min(dist + dash_length, line_len)
            end_x = int(x1 + ux * end_dist)
            end_y = int(y1 + uy * end_dist)

            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            dist += dash_length + gap_length

    def draw_hough_line(self, image, rho, theta, color, thickness=2, dashed=False):
        pt1, pt2 = self.hough_line_endpoints(rho, theta)
        if dashed:
            self.draw_dashed_line(image, pt1, pt2, color, thickness)
        else:
            cv2.line(image, pt1, pt2, color, thickness)

    def draw_overlay(self, image):
        cv2.line(
            image,
            (0, self.stop_y_threshold),
            (self.W - 1, self.stop_y_threshold),
            (0, 0, 0),
            2
        )

        if self.horizontal_smooth is not None:
            self.draw_hough_line(
                image, *self.horizontal_smooth, (0, 255, 255), 3,
                dashed=(self.horizontal_misses > 0)
            )

        if self.left_smooth is not None:
            self.draw_hough_line(
                image, *self.left_smooth, (0, 0, 255), 3,
                dashed=(self.left_misses > 0)
            )

        if self.right_smooth is not None:
            self.draw_hough_line(
                image, *self.right_smooth, (0, 255, 0), 3,
                dashed=(self.right_misses > 0)
            )

        # Optional center guideline for debugging
        cv2.line(image, (self.W // 2, 0), (self.W // 2, self.H - 1), (180, 180, 180), 1)

        if self.intersection is not None:
            ix, iy = self.intersection
            cv2.circle(image, (int(round(ix)), int(round(iy))), 6, (255, 255, 0), -1)

        if self.target_visible:
            target_color = (255, 0, 255) if (self.left_misses == 0 and self.right_misses == 0) else (0, 165, 255)
            cv2.circle(image, (self.target_x, self.H - 1), 8, target_color, -1)
            cv2.line(image, (self.W // 2, self.H - 1), (self.target_x, self.H - 1), target_color, 2)

        cv2.putText(image, self.state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

        candidate_text = f"CANDIDATE: {self.turn_candidate if self.turn_candidate is not None else 'NONE'}"
        cv2.putText(image, candidate_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_frame(self):
        ret, frame = self.cap.read()
        if self.proto and not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def process_frame(self, frame, update=True, draw=True):
        self.frame = frame
        self.original = frame.copy()
        self.H, self.W, _ = self.original.shape

        self.mask = self.create_mask(self.original)

        cv2.imshow("MASK", self.mask)

        self.edges = self.detect_edges(self.mask)
        lines = self.detect_hough_lines(self.edges)

        horizontal_lines, left_lines, right_lines = self.classify_lines(lines)

        horizontal_rep = self.median_line(horizontal_lines)
        left_rep = self.median_line(left_lines)
        right_rep = self.median_line(right_lines)

        self.update_smoothed_lines(horizontal_rep, left_rep, right_rep)
        self.update_ghost_lines(left_rep, right_rep)
        self.update_state(update=update)
        self.update_turn_candidate(left_rep, right_rep)
        self.update_target()

        output = self.original.copy()
        if draw:
            output = self.draw_overlay(output)

        return output

    def step(self, draw=True, update=True):
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return False, None
        output = self.process_frame(frame, draw=draw, update=update)
        return True, output

    def get_control_data(self):
        return {
            "state": self.state,
            "turn_candidate": self.turn_candidate,
            "candidate_locked": self.candidate_locked,
            "target_x": self.target_x,
            "target_visible": self.target_visible,
            "intersection": self.intersection,
            "left_line": self.left_smooth,
            "right_line": self.right_smooth,
            "front_line": self.horizontal_smooth,
            "left_misses": self.left_misses,
            "right_misses": self.right_misses,
            "front_misses": self.horizontal_misses,
            "frame_width": self.W,
            "frame_height": self.H,
        }

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()