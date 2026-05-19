"""
gestures.py - GestureVision AI  [ADVANCED v2.0]
-------------------------------------------------
Detects 11 distinct hand gestures using MediaPipe Tasks landmarks.

Landmark map (21 points):
  0=WRIST | 1-4=THUMB | 5-8=INDEX | 9-12=MIDDLE | 13-16=RING | 17-20=PINKY
  TIP indices : 4, 8, 12, 16, 20
  PIP indices : 3, 6, 10, 14, 18  (proximal joint — finger "knuckle")
"""

import math


# ─────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────
def _dist(a, b):
    """Normalised Euclidean distance between two landmarks (x, y)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def _dist3d(a, b):
    """3D Euclidean distance between two landmarks (x, y, z) using MediaPipe depth."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def _angle(a, b, c):
    """
    Angle (degrees) at landmark b, formed by a-b-c.
    Used for fine-grained joint-angle checks.
    """
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    dot   = v1[0]*v2[0] + v1[1]*v2[1]
    mag1  = math.hypot(*v1)
    mag2  = math.hypot(*v2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def get_finger_states(lm, handedness="RIGHT HAND"):
    """
    Returns [Thumb, Index, Middle, Ring, Pinky] booleans.
    True = extended / up.

    Args:
        lm : list of 21 NormalizedLandmark (from Tasks API result)
        handedness: string indicating "RIGHT HAND" or "LEFT HAND"
    """
    # Thumb: Check 2D distance from tip (4) to pinky base (17) vs IP joint (3) to pinky base (17)
    thumb = _dist(lm[4], lm[17]) > _dist(lm[3], lm[17])

    # Four fingers: combine 2D wrist distance check (rotation invariant) and vertical check (stable fallback)
    fingers = [thumb]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        dist_ok = _dist(lm[tip], lm[0]) > _dist(lm[pip], lm[0]) * 1.04
        vert_ok = lm[tip].y < lm[pip].y
        fingers.append(dist_ok or vert_ok)
    return fingers   # [Thumb, Index, Middle, Ring, Pinky]


def hand_openness(lm):
    """
    Returns 0.0 (closed fist) → 1.0 (fully open palm).
    Based on average fingertip distance from palm centre.
    """
    palm_cx = sum(lm[i].x for i in [0,5,9,13,17]) / 5
    palm_cy = sum(lm[i].y for i in [0,5,9,13,17]) / 5
    class _P:
        def __init__(self, x, y): self.x=x; self.y=y
    palm = _P(palm_cx, palm_cy)
    tips = [lm[i] for i in [4, 8, 12, 16, 20]]
    avg_dist = sum(_dist(t, palm) for t in tips) / 5
    # Typical open hand ≈ 0.35, closed ≈ 0.08
    return min(1.0, max(0.0, (avg_dist - 0.05) / 0.30))


# ─────────────────────────────────────────────────────────────
# Main Gesture Classifier
# ─────────────────────────────────────────────────────────────
def detect_gesture(lm, handedness="RIGHT HAND"):
    """
    Classifies 17 distinct gestures from 21 hand landmarks.
    """
    fingers = get_finger_states(lm, handedness)
    thumb, index, middle, ring, pinky = fingers

    # ── 1. Open Palm: all 5 up ────────────────────────────────────────────
    if all(fingers):
        return "open_palm"

    # ── 2. Fist: all down or hand closed ──────────────────────────────────
    if not any(fingers) or (hand_openness(lm) < 0.13 and not index and not middle):
        return "fist"

    # ── 3. Peace / V-sign: index + middle, rest down ─────────────────────
    if index and middle and not ring and not pinky and not thumb:
        return "peace"

    # ── 4. Thumbs Up: only thumb pointing up ─────────────────────────────
    if thumb and not index and not middle and not ring and not pinky:
        if lm[4].y < lm[3].y:
            return "thumbs_up"

    # ── 5. Thumbs Down: only thumb pointing down ─────────────────────────
    if thumb and not index and not middle and not ring and not pinky:
        if lm[4].y > lm[3].y:
            return "thumbs_down"

    # ── 6. One Finger Up: only index ──────────────────────────────────────
    if not thumb and index and not middle and not ring and not pinky:
        return "one_finger"

    # ── 7. Rock / Horns: index + pinky up, middle + ring down ────────────
    if index and not middle and not ring and pinky:
        return "rock"

    # ── 8. Three Fingers: index + middle + ring (no thumb, no pinky) ─────
    if not thumb and index and middle and ring and not pinky:
        return "three_up"

    # ── 9. Four Fingers: index + middle + ring + pinky (no thumb) ────────
    if not thumb and index and middle and ring and pinky:
        return "four_up"

    # ── 10. Pinky Up only ─────────────────────────────────────────────────
    if not thumb and not index and not middle and not ring and pinky:
        return "pinky_up"

    # ── 11. Spider-Man: thumb + index + pinky extended, middle + ring closed
    if thumb and index and not middle and not ring and pinky:
        return "spiderman"

    # ── 12. Shaka Sign: thumb + pinky extended, index + middle + ring closed
    if thumb and not index and not middle and not ring and pinky:
        return "shaka"

    # ── 13. Gun / L-shape: thumb + index pointing up, rest closed ──────────
    if thumb and index and not middle and not ring and not pinky:
        return "gun"

    # ── 14. Tri-Force: thumb + index + middle extended, ring + pinky closed
    if thumb and index and middle and not ring and not pinky:
        return "triforce"

    # ── 15. Rebel Sign: only middle finger extended ───────────────────────
    if not thumb and not index and middle and not ring and not pinky:
        return "rebel"

    # ── 16. Pinch: thumb + index touching, middle + ring + pinky closed ──
    if not middle and not ring and not pinky:
        if _dist(lm[4], lm[8]) < 0.05:
            return "pinch"

    # ── 17. OK Sign: middle+ring+pinky up; thumb tip ≈ index tip ──────────
    if middle and ring and pinky and not index:
        if _dist(lm[4], lm[8]) < 0.07:
            return "ok"

    return "unknown"


# ─────────────────────────────────────────────────────────────
# Hand skeleton connections (Tasks API — manual drawing)
# ─────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index
    (0,9),(9,10),(10,11),(11,12),    # Middle
    (0,13),(13,14),(14,15),(15,16),  # Ring
    (0,17),(17,18),(18,19),(19,20),  # Pinky
    (5,9),(9,13),(13,17),(5,17),     # Palm
]


# ─────────────────────────────────────────────────────────────
# Gesture Metadata (label, action, color, filter_name)
# ─────────────────────────────────────────────────────────────
GESTURE_INFO = {
    "open_palm":   {"label": "Open Palm",      "action": "Normal Camera",       "color": (80,  255, 130), "filter": "normal"},
    "peace":       {"label": "Peace / V",      "action": "Black & White",       "color": (255, 255, 80),  "filter": "grayscale"},
    "thumbs_up":   {"label": "Thumbs Up",      "action": "Blur Effect",         "color": (80,  180, 255), "filter": "blur"},
    "ok":          {"label": "OK Sign",        "action": "Edge Focus",          "color": (255, 100, 255), "filter": "edges"},
    "rock":        {"label": "Rock / Horns",   "action": "Neon Glow",           "color": (200, 60,  255), "filter": "neon"},
    "three_up":    {"label": "Three Fingers",  "action": "Sepia Tone",          "color": (30,  140, 255), "filter": "sepia"},
    "four_up":     {"label": "Four Fingers",   "action": "Cartoon Effect",      "color": (60,  220, 100), "filter": "cartoon"},
    "pinky_up":    {"label": "Pinky Up",       "action": "Vintage Vignette",    "color": (200, 80,  255), "filter": "vignette"},
    "spiderman":   {"label": "Spider-Man",     "action": "Negative / Invert",   "color": (0,   100, 255), "filter": "negative"},
    "shaka":       {"label": "Shaka Sign",     "action": "Synth Sunset",        "color": (255, 100, 255), "filter": "synthwave"},
    "thumbs_down": {"label": "Thumbs Down",    "action": "Retro Warm",          "color": (120, 200, 255), "filter": "retro"},
    "gun":         {"label": "Gun Shape",      "action": "Thermal Cam",         "color": (50,  50,  255), "filter": "thermal"},
    "triforce":    {"label": "Tri-Force",      "action": "Pop Art",             "color": (255, 180, 50),  "filter": "posterize"},
    "rebel":       {"label": "Rebel Sign",     "action": "Cyber Glitch",        "color": (80,  80,  255), "filter": "glitch"},
    "pinch":       {"label": "Pinch",          "action": "Matrix Rain",         "color": (0,   255, 0),   "filter": "matrix"},
    "one_finger":  {"label": "One Finger",     "action": "Neural Tuner",        "color": (0,   255, 255), "filter": "neural_spectrum"},
    "fist":        {"label": "Fist",           "action": "Clear / Exit",        "color": (0,   0,   255), "filter": "normal"},
    "unknown":     {"label": "No Gesture",     "action": "Waiting...",          "color": (100, 100, 100), "filter": "normal"},
}


# ─────────────────────────────────────────────────────────────
# Gesture Stability Smoother
# ─────────────────────────────────────────────────────────────
class GestureSmoother:
    """
    Requires a gesture to be detected consistently for N frames
    before it becomes the active gesture. Prevents flicker.
    """
    def __init__(self, required_frames=6):
        self.required   = required_frames
        self.candidate  = "unknown"
        self.count      = 0
        self.stable     = "unknown"

    def update(self, raw_gesture):
        """
        Feed raw gesture each frame. Returns stable (confirmed) gesture.
        """
        if raw_gesture == self.candidate:
            self.count += 1
        else:
            self.candidate = raw_gesture
            self.count     = 1

        if self.count >= self.required:
            self.stable = self.candidate

        return self.stable

    @property
    def stability(self):
        """0.0 → 1.0 showing how stable current candidate is."""
        return min(1.0, self.count / self.required)
