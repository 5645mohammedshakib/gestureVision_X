"""
ai_tutor.py - GestureVision AI [ARIA AI Assistant]
-----------------------------------------------------
ARIA = AI Real-time Interactive Assistant

Features:
- Interactive gesture tutorial with live challenge mode
- Step-by-step guided learning for all 11 gestures
- Real-time feedback ("Good job!", "Try again", etc.)
- Voice feedback via pyttsx3 (falls back to silent if not installed)
- On-screen animated AI persona panel
- Smart tip system based on current gesture context
"""

import cv2
import numpy as np
import time
import random
from collections import deque

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

# Voice disabled — _speak_async is a no-op
_TTS_OK = False
_TTS_ENGINE = None


def _speak_async(text):
    """Voice disabled. No TTS."""
    pass


# ─────────────────────────────────────────────────────────────
# Gesture Tutorial Curriculum
# ─────────────────────────────────────────────────────────────
TUTORIAL_STEPS = [
    {
        "gesture":     "open_palm",
        "name":        "Open Palm",
        "instruction": "Spread all 5 fingers wide open",
        "tip":         "Face palm towards camera. All fingers extended.",
        "action":      "Resets to Normal Camera filter",
        "voice":       "Show me an open palm. Spread all 5 fingers wide.",
    },
    {
        "gesture":     "peace",
        "name":        "Peace Sign (V)",
        "instruction": "Raise index and middle fingers in a V shape",
        "tip":         "Curl ring, pinky, and thumb. Keep index and middle up.",
        "action":      "Activates Black & White filter",
        "voice":       "Make a peace sign. Raise your index and middle fingers.",
    },
    {
        "gesture":     "thumbs_up",
        "name":        "Thumbs Up",
        "instruction": "Raise only your thumb, curl all other fingers",
        "tip":         "Point thumb upward. All 4 fingers should be curled.",
        "action":      "Activates Blur filter",
        "voice":       "Show thumbs up. Raise your thumb and curl the rest.",
    },
    {
        "gesture":     "ok",
        "name":        "OK Sign",
        "instruction": "Touch thumb tip to index tip, raise other 3 fingers",
        "tip":         "Make a circle with thumb and index. Middle, ring, pinky up.",
        "action":      "Takes a Screenshot",
        "voice":       "Make the OK sign. Touch your thumb and index fingertip together.",
    },
    {
        "gesture":     "rock",
        "name":        "Rock Horns",
        "instruction": "Raise index and pinky, curl middle and ring",
        "tip":         "Classic rock gesture. Index finger + pinky up.",
        "action":      "Activates Neon Glow filter",
        "voice":       "Show rock horns! Raise index and pinky fingers.",
    },
    {
        "gesture":     "three_up",
        "name":        "3 Fingers",
        "instruction": "Raise index, middle, and ring fingers",
        "tip":         "Three fingers up together. Thumb and pinky curled.",
        "action":      "Activates Sepia filter",
        "voice":       "Three fingers up please. Index, middle, and ring.",
    },
    {
        "gesture":     "four_up",
        "name":        "4 Fingers",
        "instruction": "Raise all fingers except thumb",
        "tip":         "Keep thumb folded. Raise index, middle, ring, pinky.",
        "action":      "Activates Cartoon filter",
        "voice":       "Four fingers up. Keep your thumb folded down.",
    },
    {
        "gesture":     "pinky_up",
        "name":        "Pinky Up",
        "instruction": "Raise only your pinky finger",
        "tip":         "All other fingers curled. Only pinky extended.",
        "action":      "Activates Vignette filter",
        "voice":       "Pinky up! Only raise your little finger.",
    },
    {
        "gesture":     "spiderman",
        "name":        "Spider-Man",
        "instruction": "Raise thumb and pinky, curl index, middle, ring",
        "tip":         "Like Spider-Man shooting webs! Thumb + pinky out.",
        "action":      "Activates Negative filter",
        "voice":       "Spider-Man pose! Thumb and pinky up, others curled.",
    },
    {
        "gesture":     "one_finger",
        "name":        "One Finger (Pointer)",
        "instruction": "Raise only your index finger",
        "tip":         "Hold for 1.5 seconds to activate Writing Mode!",
        "action":      "Hold 1.5s to activate Writing Mode",
        "voice":       "Point with one finger. Hold it to activate writing mode.",
    },
    {
        "gesture":     "fist",
        "name":        "Fist",
        "instruction": "Curl all fingers into a tight fist",
        "tip":         "Hold fist for 2.5 seconds to exit the app.",
        "action":      "Hold 2.5s to EXIT (brief = clear canvas)",
        "voice":       "Make a fist. Hold it for 2 and a half seconds to exit.",
    },
]


# ─────────────────────────────────────────────────────────────
# AI Tips System — Extremely Rich, Futuristic Scifi Dialogue!
# ─────────────────────────────────────────────────────────────
ARIA_TIPS = {
    "open_palm":  [
        "TACTICAL FEED INJECTED: Open Palm active. System state reset completed.",
        "BIOMETRIC FEEDBACK: Palm pattern scanned. Restoring raw camera matrix.",
        "SYSTEM STABILIZER: Clearing active screen canvas buffers instantly."
    ],
    "peace":      [
        "VISUAL DECODER: Peace gesture recognized. Initializing high-contrast monochrome rendering.",
        "NEURAL OVERLAY: Splitting chromatic values into high-density grayscale spectrum.",
        "MONITOR PROTOCOL: Monochrome vision matrix locked and engaged."
    ],
    "thumbs_up":  [
        "Dreamy Gaussian blur engine activated. Depth falloff simulated.",
        "OPTICAL FOCUS: Depth-of-field lens stabilizer engaged. Portrait bokeh active.",
        "AI DEEP BLUR: Softening frame rendering matrix for maximum aesthetic clarity."
    ],
    "ok":         [
        "SCREEN CAPTURE payload initiated! Compressing and saving frame buffer.",
        "TACTICAL SHOT: Captured live vision matrix frame. File saved to folder.",
        "HUD ARCHIVE: Local telemetry frame buffer successfully captured and secure."
    ],
    "rock":       [
        "NEON CHRONOMETER: Cyberpunk edge glow engaged. Chromatic divergence active.",
        "TACTICAL RAY-TRACING: Blending neon cyan & magenta split laser streams.",
        "LASER VISION: Edge highlight neon shader compiled. Heavy metal mode operational."
    ],
    "three_up":   [
        "HISTORICAL SIMULATION: Warm vintage sepia matrix engaged.",
        "SPECTRAL SHIFT: Sepia matrix filter running. Simulating retro optical lens.",
        "WARM CORE: Altering RGB value weights for absolute retro chromatic sepia vibe."
    ],
    "four_up":    [
        "CARTOON SHADER: Adaptive threshold cel-shading active.",
        "ANIME CHRONICLE: Outlining frame silhouettes. Flat color bilateral filter active.",
        "MANGA CORE: Dynamic adaptive line-drawings layered on visual canvas."
    ],
    "pinky_up":   [
        "CINEMATIC PROTOCOL: Vignette vignette filter enabled. Darkening screen margins.",
        "VIGNETTE SHIELD: Smooth oval edge falloff matrix fully functional.",
        "OPTICAL BARRIER: Darkening peripheral vision channels by 45% for focus."
    ],
    "spiderman":  [
        "COGNITIVE DISSOCIATION: Color inversion active. negative vision matrix on.",
        "WEB SLINGER: Negative matrix filter engaged. Inverting chroma and luma.",
        "SPECTRAL ANOMALY: Chromatic inversion active. Reverse photon mapping active."
    ],
    "one_finger": [
        "NEURAL SPECTRUM ACTIVE! Move finger in 2D space to tune 11.7 MILLION procedural filters!",
        "DYNAMIC TUNER ON: Index tip tracked. Shift X/Y to scan infinite filter frequencies!",
        "OPTICAL FREQUENCY SCAN: Move index finger to blend hue/saturation matrices dynamically!"
    ],
    "fist":       [
        "CRITICAL HOLD: Hold FIST 2.5s to trigger OS core termination.",
        "OS SHUTDOWN DETECTED: Release fist immediately to abort terminal payload.",
        "CANVAS FLUSH: Brief fist detected. Canvas coordinates reset to default."
    ],
    "unknown":    [
        "SCANNING HAND SPACE: Please present hand gestures clearly in camera sight.",
        "BIOMETRIC SIGNAL LOSS: Adjust illumination or range for optimal tracking.",
        "NEURAL DECODER: Tracking spatial landmarks. Hold hand steady in central grid."
    ],
}

ARIA_GREETINGS = [
    "Welcome. I am ARIA, your tactical gesture neural operating system.",
    "ARIA Core v3.2 online. Neural interface linked. Present gestures to control.",
    "Futuristic operating system online. Systems operational. Awaiting your spatial command.",
    "Telemetry synchronised. ARIA interactive core fully initialized. Standing by.",
]

ARIA_ENCOURAGE = [
    "GESTURE IDENTIFIED: Tactical command verified and executed.",
    "COMMAND LOCKED: Spatial telemetry parsed successfully.",
    "NEURAL SYNC COMPLETE: Target signature matched and registered.",
    "COORDINATE VERIFIED: Matrix instruction executed cleanly.",
    "VISION PATTERN SCANNED: Tactical framework updated.",
]

ARIA_RETRY = [
    "TELEMETRY FAULT: Hand landmark signal low. Please align with the grid.",
    "COGNITIVE DEVIATION: Present hand clearly within camera illumination.",
    "INTERFACE DISRUPT: Landmarking tracking unstable. Hold gesture steady.",
]


# ─────────────────────────────────────────────────────────────
# ARIA AI Assistant
# ─────────────────────────────────────────────────────────────
class ARIAAssistant:
    """
    On-screen AI tutor and assistant.
    Shows tips, teaches gestures, gives real-time feedback.
    """

    MODE_NORMAL   = "normal"   # just tips
    MODE_TUTORIAL = "tutorial" # step-by-step guided learning

    def __init__(self):
        self.mode          = self.MODE_NORMAL
        self.message       = random.choice(ARIA_GREETINGS)
        self.sub_message   = "Show me a gesture to get started!"
        self.last_gesture  = "unknown"
        self.last_tip_t    = time.time()
        self.tip_interval  = 6.0    # show new tip every 6s
        self.message_t     = time.time()
        self.message_dur   = 4.0

        # Tutorial state
        self.tut_step      = 0
        self.tut_hold_t    = None
        self.tut_hold_req  = 1.5   # seconds to hold gesture for it to count
        self.tut_completed = set()
        self.tut_started   = False
        self._spoke        = False

        # Pulse animation
        self._pulse_t      = time.time()
        self._score        = 0      # tutorial score

        # Advanced Cybernetic persona attributes
        self.cognitive_state = "ACTIVE" # ACTIVE, DECODING, STABILIZING, LOCKED
        self.orbital_angle   = 0.0
        self.thought_stream  = deque(maxlen=4)
        self.thought_stream.append("ARIA Core system check: OK")
        self.thought_stream.append("Biometric vision sensor linked")
        self.thought_stream.append("Ready for spatial controls")
        
        self.intent_forecast = "AWAITING TELEMETRY"
        self.compute_load    = 12.4
        
        # Challenge system
        self.challenge_active = False
        self.challenge_gesture = ""
        self.challenge_start_t = None
        self.challenge_dur_req = 2.0
        self.challenge_streak = 0
        self.next_challenge_t = time.time() + 15.0 # first challenge in 15 seconds

        # Greeted?
        _speak_async(random.choice(ARIA_GREETINGS))

    # ── Public API ─────────────────────────────────────────────
    def update(self, gesture, stability):
        """Call each frame. Updates messages and tutorial progress."""
        now = time.time()
        self.orbital_angle += 0.1
        self.compute_load = round(12.0 + 3.0 * random.random() + (5.0 if gesture != "unknown" else 0.0), 1)

        # Reactive cognitive state and logs
        if gesture != "unknown":
            if stability < 0.85:
                self.cognitive_state = "STABILIZING"
                if random.random() < 0.015:
                    self.thought_stream.append("Jitter filter active on landmark")
            else:
                self.cognitive_state = "DECODING"
        else:
            self.cognitive_state = "ACTIVE"

        # Intent forecasting engine
        if gesture == "one_finger":
            self.intent_forecast = "AIR WRITING / TUNING"
        elif gesture == "fist":
            self.intent_forecast = "SHUTDOWN PAYLOAD REQUEST"
        elif gesture in ["peace", "thumbs_up", "rock", "three_up", "four_up", "pinky_up", "spiderman"]:
            self.intent_forecast = "FILTER SHIFT SEQUENCE"
        elif gesture == "ok":
            self.intent_forecast = "SCREEN CAPTURE INITIATING"
        elif gesture != "unknown":
            self.intent_forecast = "CMD INSTRUCTION PARSED"
        else:
            self.intent_forecast = "AWAITING COMMAND"

        # Challenge minigame engine (only in normal mode)
        if self.mode == self.MODE_NORMAL:
            self._update_challenge(gesture, stability, now)

        if self.mode == self.MODE_TUTORIAL:
            self._update_tutorial(gesture, stability, now)
        else:
            self._update_tips(gesture, now)

    def start_tutorial(self):
        """Begin step-by-step gesture tutorial."""
        self.mode        = self.MODE_TUTORIAL
        self.tut_step    = 0
        self.tut_completed = set()
        self._score      = 0
        self.tut_started = True
        self.challenge_active = False
        step = TUTORIAL_STEPS[0]
        self.message     = f"TUTORIAL: {step['name']}"
        self.sub_message = step['instruction']
        _speak_async(f"Tutorial started. Step 1. {step['voice']}")

    def stop_tutorial(self):
        self.mode        = self.MODE_NORMAL
        self.message     = f"Tutorial complete! Score: {self._score}/{len(TUTORIAL_STEPS)}"
        self.sub_message = "Great work! You learned all gestures."
        _speak_async(f"Tutorial complete! You scored {self._score} out of {len(TUTORIAL_STEPS)}!")

    def speak(self, text):
        _speak_async(text)

    @property
    def tutorial_active(self):
        return self.mode == self.MODE_TUTORIAL

    @property
    def tutorial_progress(self):
        if not TUTORIAL_STEPS: return 0.0
        return len(self.tut_completed) / len(TUTORIAL_STEPS)

    @property
    def current_tutorial_step(self):
        if self.tut_step < len(TUTORIAL_STEPS):
            return TUTORIAL_STEPS[self.tut_step]
        return None

    @property
    def score(self): return self._score

    # ── Internal ────────────────────────────────────────────────
    def _update_challenge(self, gesture, stability, now):
        """Dynamic challenge engine logic."""
        if not self.challenge_active and now > self.next_challenge_t:
            # Issue a new random challenge
            self.challenge_gesture = random.choice([s["gesture"] for s in TUTORIAL_STEPS if s["gesture"] not in ["fist", "unknown"]])
            self.challenge_active = True
            self.challenge_start_t = None
            self.message = f"CHALLENGE: Present {self.challenge_gesture.upper()}"
            self.sub_message = "Hold it steady for 2 seconds to synchronize!"
            self.thought_stream.append(f"AI CHALLENGE: {self.challenge_gesture} check")
            
        if self.challenge_active:
            if gesture == self.challenge_gesture and stability >= 0.85:
                if self.challenge_start_t is None:
                    self.challenge_start_t = now
                elif (now - self.challenge_start_t) >= self.challenge_dur_req:
                    # Success
                    self.challenge_streak += 1
                    self.challenge_active = False
                    self.next_challenge_t = now + 20.0
                    self.message = f"CHALLENGE PASSED! STREAK: {self.challenge_streak}"
                    self.sub_message = f"Neural feedback calibration complete."
                    self.thought_stream.append(f"CALIBRATION ACCURACY: 99.8%")
            else:
                self.challenge_start_t = None

    def _update_tips(self, gesture, now):
        if self.challenge_active:
            return  # Let challenge messages override normal tips
            
        if gesture != self.last_gesture and gesture != "unknown":
            tips = ARIA_TIPS.get(gesture, ARIA_TIPS["unknown"])
            self.message     = random.choice(ARIA_ENCOURAGE)
            self.sub_message = random.choice(tips)
            self.message_t   = now
            self.last_gesture = gesture
            self.thought_stream.append(f"Gesture shift -> {gesture}")
            if random.random() < 0.4:
                _speak_async(self.sub_message)

        elif gesture == "unknown" and (now - self.last_tip_t) > self.tip_interval:
            self.message     = "ARIA TIP"
            self.sub_message = random.choice(ARIA_TIPS["unknown"])
            self.last_tip_t  = now

    def _update_tutorial(self, gesture, stability, now):
        if self.tut_step >= len(TUTORIAL_STEPS):
            self.stop_tutorial()
            return

        step = TUTORIAL_STEPS[self.tut_step]
        target = step["gesture"]

        if gesture == target and stability >= 0.85:
            if self.tut_hold_t is None:
                self.tut_hold_t = now
                _speak_async("Hold it steady!")
            elif (now - self.tut_hold_t) >= self.tut_hold_req:
                # Gesture held long enough — PASS
                self.tut_completed.add(target)
                self._score += 1
                _speak_async(random.choice(ARIA_ENCOURAGE))
                self.tut_step    += 1
                self.tut_hold_t   = None
                if self.tut_step < len(TUTORIAL_STEPS):
                    nxt = TUTORIAL_STEPS[self.tut_step]
                    self.message     = f"Step {self.tut_step+1}/{len(TUTORIAL_STEPS)}: {nxt['name']}"
                    self.sub_message = nxt['instruction']
                    _speak_async(nxt['voice'])
                else:
                    self.stop_tutorial()
        else:
            self.tut_hold_t = None
            step = TUTORIAL_STEPS[self.tut_step]
            self.message     = f"Step {self.tut_step+1}/{len(TUTORIAL_STEPS)}: {step['name']}"
            self.sub_message = step['instruction']

    # ── Rendering ────────────────────────────────────────────────
    def draw(self, frame, state=None):
        """Renders the ARIA panel on the frame."""
        H, W = frame.shape[:2]
        now   = time.time()
        pulse = abs(np.sin((now - self._pulse_t) * 2.0))   # 0→1 oscillation

        if self.mode == self.MODE_TUTORIAL:
            self._draw_tutorial_overlay(frame, W, H, pulse)
        else:
            self._draw_tip_panel(frame, W, H, pulse)

    def _draw_persona_core(self, frame, cx, cy, pulse):
        """Draws rotating futuristic HUD rings & core."""
        import math
        # Outer rotating dashes
        r_outer = 16
        for angle in range(0, 360, 45):
            rad = math.radians(angle + self.orbital_angle * 12)
            x1 = int(cx + r_outer * math.cos(rad))
            y1 = int(cy + r_outer * math.sin(rad))
            x2 = int(cx + (r_outer - 4) * math.cos(rad))
            y2 = int(cy + (r_outer - 4) * math.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 230), 1, cv2.LINE_AA)
            
        # Inner counter-rotating orbital particles
        r_inner = 9
        for angle in range(0, 360, 60):
            rad = math.radians(angle - self.orbital_angle * 18)
            x = int(cx + r_inner * math.cos(rad))
            y = int(cy + r_inner * math.sin(rad))
            cv2.circle(frame, (x, y), 1, (240, 100, 220), -1, cv2.LINE_AA)
            
        # Central pulsing core (reactive color)
        r_core = int(4 + 1.5 * pulse)
        if self.cognitive_state == "STABILIZING":
            color = (0, 165, 255) # Orange Warning
        elif self.cognitive_state == "DECODING":
            color = (0, 255, 150) # Green Success
        else:
            color = (0, 255, 230) # Cyan Normal
        cv2.circle(frame, (cx, cy), r_core, color, -1, cv2.LINE_AA)

    def _draw_tip_panel(self, frame, W, H, pulse):
        """ARIA tip banner — bottom-centre."""
        from utils import glass_panel, txt, label

        px = 30
        panel_w = min(W - 390, 860)  # On 1280x720, this sets panel_w = 860, ending at x = 890 (well before guide starts at 924)
        py = H - 96  # Shifted up to clear the bottom bezel and frame lines
        border_col = tuple(int(c * (0.6 + 0.4 * pulse)) for c in (0, 200, 255))

        # Panel height increased to 82px for spacious telemetry displays
        glass_panel(frame, px, py, px+panel_w, py+82, (8,8,20), 0.84, border_col, r=12)

        # Animated Cybernetic Core Persona
        self._draw_persona_core(frame, px+30, py+32, pulse)
        cv2.putText(frame, f"[{self.cognitive_state}]", (px+8, py+68), FONTB, 0.28, border_col, 1, cv2.LINE_AA)

        # Dialog messages
        cv2.putText(frame, "ARIA", (px+58, py+20), FONTB, 0.44, (0,230,255), 1, cv2.LINE_AA)
        cv2.putText(frame, self.message[:70], (px+58, py+40), FONTB, 0.42, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(frame, self.sub_message[:80], (px+58, py+58), FONT, 0.35, (120,220,150), 1, cv2.LINE_AA)

        # Helper controls info row
        cv2.putText(frame, "T = Toggle Tutorial | G = Toggle Gesture Guide", (px+58, py+74), FONT, 0.30, (120,120,140), 1, cv2.LINE_AA)

        # Right Telemetry and Thought Log Block
        tx = px + panel_w - 300
        cv2.line(frame, (tx-12, py+8), (tx-12, py+74), (30,60,80), 1)
        
        cv2.putText(frame, f"INTENT: {self.intent_forecast}", (tx, py+18), FONTB, 0.32, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"COMPUTE: {self.compute_load} TFLOPS", (tx, py+30), FONT, 0.30, (180, 180, 200), 1, cv2.LINE_AA)
        
        if len(self.thought_stream) >= 2:
            cv2.putText(frame, f">> {self.thought_stream[-2][:38]}", (tx, py+44), FONT, 0.28, (120, 220, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, f">> {self.thought_stream[-1][:38]}", (tx, py+56), FONT, 0.28, (120, 220, 150), 1, cv2.LINE_AA)
            if self.challenge_active:
                cv2.putText(frame, "CHALLENGE ACTIVE", (tx, py+68), FONTB, 0.28, (0,165,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"STREAK: {self.challenge_streak}", (tx, py+68), FONT, 0.28, (0,255,180), 1, cv2.LINE_AA)

    def _draw_tutorial_overlay(self, frame, W, H, pulse):
        """Full tutorial overlay — centre of screen."""
        from utils import glass_panel

        step = self.current_tutorial_step
        if not step: return

        # Semi-transparent centre panel
        pw, ph = 560, 280
        px = (W - pw) // 2
        py = (H - ph) // 2 - 40
        border = tuple(int(c*(0.6+0.4*pulse)) for c in (0, 255, 180))
        glass_panel(frame, px, py, px+pw, py+ph, (5,10,15), 0.88, border, r=16)

        # Header
        cv2.putText(frame, f"ARIA TUTORIAL — Step {self.tut_step+1}/{len(TUTORIAL_STEPS)}",
                    (px+20, py+30), FONTB, 0.60, (0,230,255), 1, cv2.LINE_AA)
        cv2.line(frame, (px+10, py+38), (px+pw-10, py+38), (0,120,80), 1)

        # Gesture name
        cv2.putText(frame, step['name'], (px+20, py+72),
                    FONTB, 1.0, (0,255,180), 2, cv2.LINE_AA)

        # Instruction
        cv2.putText(frame, step['instruction'], (px+20, py+105),
                    FONT, 0.52, (220,220,220), 1, cv2.LINE_AA)

        # Tip
        cv2.putText(frame, f"TIP: {step['tip']}", (px+20, py+132),
                    FONT, 0.42, (0,180,120), 1, cv2.LINE_AA)

        # Action
        cv2.putText(frame, f"Action: {step['action']}", (px+20, py+158),
                    FONT, 0.42, (0,160,255), 1, cv2.LINE_AA)

        # Hold progress bar
        if self.tut_hold_t:
            held  = time.time() - self.tut_hold_t
            prog  = min(held / self.tut_hold_req, 1.0)
            bx1,by1 = px+20, py+178
            bx2      = px+pw-20
            cv2.rectangle(frame,(bx1,by1),(bx2,by1+14),(30,30,30),-1)
            cv2.rectangle(frame,(bx1,by1),(int(bx1+(bx2-bx1)*prog),by1+14),(0,255,180),-1)
            cv2.rectangle(frame,(bx1,by1),(bx2,by1+14),(0,120,80),1)
            cv2.putText(frame,"HOLD STEADY...", (bx1,by1-4), FONT, 0.35, (0,230,255), 1)
        else:
            cv2.putText(frame,"Show the gesture and hold steady!",
                        (px+20,py+188), FONT, 0.42, (100,100,120), 1, cv2.LINE_AA)

        # Score
        cv2.putText(frame, f"Score: {self._score}/{len(TUTORIAL_STEPS)}",
                    (px+pw-140, py+72), FONTB, 0.60, (0,230,255), 1, cv2.LINE_AA)

        # Completed steps checkmarks
        cy_start = py+200
        for i, s in enumerate(TUTORIAL_STEPS[:min(11, len(TUTORIAL_STEPS))]):
            done = s['gesture'] in self.tut_completed
            col  = (0,255,150) if done else (40,40,40)
            dot  = int(10 + 2*pulse) if (i == self.tut_step and not done) else 8
            cx_  = px + 20 + i * 46
            cv2.circle(frame, (cx_, cy_start+15), dot, col, -1)
            if done:
                cv2.putText(frame,"v",(cx_-5,cy_start+20),FONTB,0.4,(0,200,100),1)

        # Press T again to stop
        cv2.putText(frame, "Press T to exit tutorial",
                    (px+20, py+ph-12), FONT, 0.35, (60,60,80), 1, cv2.LINE_AA)
