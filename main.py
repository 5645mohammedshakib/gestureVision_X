"""
main.py - GestureVision X GODMODE [v4.0 LEGENDARY OS EDITION]
-------------------------------------------------------------
A next-generation AI-powered computer control, advanced computer vision,
and surveillance operating system inspired by JARVIS, Cyberpunk 2077, and holographic HUDs.
"""

import cv2
import sys
import time
import os
import math
import random
import threading

# ── HIGH-DPI RAZOR SHARP WINDOW INITIALIZATION ───────────────────────
if sys.platform.startswith('win'):
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
import shutil
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

# Windows System Controls
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    _HAS_PYAUTOGUI = True
except ImportError:
    _HAS_PYAUTOGUI = False

try:
    import keyboard
    _HAS_KEYBOARD = True
except ImportError:
    _HAS_KEYBOARD = False

# Import local modules
from gestures import detect_gesture, GESTURE_INFO, GestureSmoother, get_finger_states
from filters import apply_filter, crossfade
from ai_tutor import ARIAAssistant
from color_detector import draw_color_panel
from color_tracker import ColorTracker
from colorizer import Colorizer
from utils import (
    FPSCounter, GestureCooldown, SessionStats,
    VideoRecorder, WritingCanvas, ParticleSystem,
    draw_hud, draw_skeleton, draw_confidence_ring,
    draw_motion_trail, draw_speed_bar, draw_rainbow_border,
    save_screenshot, show_loading_screen, ensure_dirs,
    glass_panel, txt, label, C_CYAN, C_GREEN, C_PINK, C_WHITE, C_DIM, C_DARK, C_RED, C_YELLOW, C_BORDER
)

# ── Config ──────────────────────────────────────────────────────
WINDOW_NAME       = "GestureVision X GODMODE"
FONT              = cv2.FONT_HERSHEY_SIMPLEX
MODEL_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
CAMERA_INDEX      = 0
# Internal processing speed resolution (1280x720 for native high-definition)
PROC_W, PROC_H    = 1280, 720
# Beautiful HUD presentation resolution (1280x720 for flawless UI)
FRAME_W, FRAME_H  = 1280, 720
DET_CONF          = 0.45
TRACK_CONF        = 0.45
PRES_CONF         = 0.45
FIST_HOLD         = 1.2      # seconds to EXIT
WRITE_HOLD        = 1.0      # seconds to activate writing
SCR_COOLDOWN      = 3.0      # seconds between screenshots

# Screen coordinates for Air Mouse
try:
    if _HAS_PYAUTOGUI:
        SCREEN_W, SCREEN_H = pyautogui.size()
    else:
        SCREEN_W, SCREEN_H = 1920, 1080
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

# ── Dynamic Voice Speech Synth (Disabled) ───────────────────────
def _speak_async(text):
    """Speech synth disabled. Completely silent."""
    pass


class MatrixRain:
    """Matrix Digital Green Code Rain overlay generator."""
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.grid_w = w // 15
        self.grid_h = h // 15
        self.drops = [random.randint(-20, 0) for _ in range(self.grid_w)]

    def draw(self, frame):
        overlay = frame.copy()
        for i in range(self.grid_w):
            char = str(random.randint(0, 1))
            col_x = i * 15
            row_y = self.drops[i] * 15
            
            if row_y >= 0 and row_y < self.h:
                cv2.putText(overlay, char, (col_x, row_y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 240, 100), 1, cv2.LINE_AA)
            
            self.drops[i] += 1
            if self.drops[i] * 15 > self.h or random.random() > 0.975:
                self.drops[i] = 0
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)


class GazeMouseFilter:
    """Smooth cursor control filtering using double exponential smoothing."""
    def __init__(self, alpha=0.18):
        self.alpha = alpha
        self.cx, self.cy = SCREEN_W // 2, SCREEN_H // 2

    def update(self, tx, ty):
        self.cx = self.alpha * tx + (1 - self.alpha) * self.cx
        self.cy = self.alpha * ty + (1 - self.alpha) * self.cy
        return int(self.cx), int(self.cy)


def build_landmarker():
    if not os.path.exists(MODEL_PATH):
        print(f"[ObjectDetector] Retargeting local task path...")
        # Fallback to check relative folders
        alt_path = "hand_landmarker.task"
        if os.path.exists(alt_path):
            shutil.copy2(alt_path, MODEL_PATH)
        else:
            print("[ERROR] hand_landmarker.task is missing in both models/ and base path!")
            sys.exit(1)
            
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=DET_CONF,
        min_hand_presence_confidence=PRES_CONF,
        min_tracking_confidence=TRACK_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def open_camera():
    # Dynamically find the best camera index and configuration that actually returns colorful frames!
    best_cap = None
    best_index = 0
    
    # Check potential camera indices (Optimized to check only index 0 for speed)
    for idx in [0]:
        print(f"[CAMERA DIAGNOSTICS] Probing camera index {idx}...")
        # Try standard Media Foundation API first (usually much more stable than DSHOW on modern Win11/10 laptops)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
            
        if cap.isOpened():
            # Test frame to ensure it is not black/blank
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_val = frame.mean()
                print(f"[CAMERA DIAGNOSTICS] Index {idx} successfully opened. Frame mean: {mean_val:.2f}")
                best_cap = cap
                best_index = idx
                break
            cap.release()
            
    if best_cap is None:
        print("[CAMERA DIAGNOSTICS] Warning: No active colored camera stream detected. Defaulting to index 0.")
        best_cap = cv2.VideoCapture(0)
        if not best_cap.isOpened():
            best_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
    # Skipped setting resolution to avoid black screen issues
        
    return best_cap, best_index


def main():
    ensure_dirs()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_W, FRAME_H)
    show_loading_screen(WINDOW_NAME, duration=0.2)

    print("[SYS] Building landmarker...")
    landmarker  = build_landmarker()
    print("[SYS] Landmarker built.")
    print("[SYS] Opening camera...")
    cap, current_cam_idx = open_camera()
    print("[SYS] Camera opened.")
    
    fps_ctr     = FPSCounter(smoothing=30)
    smoother    = GestureSmoother(required_frames=8)
    stats       = SessionStats()
    recorder    = VideoRecorder(fps=20, w=FRAME_W, h=FRAME_H)
    canvas      = WritingCanvas(FRAME_W, FRAME_H)
    particles   = ParticleSystem(maxp=130)
    aria        = ARIAAssistant()
    from utils import play_beep_async
    play_beep_async("startup")
    
    matrix      = MatrixRain(FRAME_W, FRAME_H)
    gaze_mouse  = GazeMouseFilter()

    # ── Master OS Feature Toggles ──────────────────────────────────
    show_color_pan = False   # K = color panel
    show_tracker   = False   # D = color tracker
    use_colorize   = False   # Z = colorizer
    show_gesture_guide = True  # G = Sleek Gesture Guide manual (On by default!)
    show_full_guide    = False # M = Fullscreen Master Holographic Overlay Index
    
    # Futuristic Operating Modes (Toggled by voice/gestures!)
    hacker_mode     = False  # Matrix falling green binary code
    thermal_mode    = False  # Jet colormap thermal simulation
    night_vision    = False  # Phosphor green night tracking
    red_alert       = False  # Emergency edge-pulsing warning
    air_mouse_mode  = True   # Toggle index-finger Windows control

    # ── App state ────────────────────────────────────────────────
    stable        = "unknown"
    active_filter = "normal"
    prev_filter   = "normal"
    trans_frame   = 0

    fist_start    = None
    write_hold_t  = None
    writing_mode  = False
    
    frame_idx     = 0
    running       = True
    
    # Scrolling System Audit Logs
    system_logs = deque(maxlen=7)
    system_logs.append("[SYS] GestureVision X Core Loaded.")
    system_logs.append("[AI] Holographic Neural OS Active.")
    system_logs.append("[SEC] Gaze biometric security check: OK.")
    
    # Dynamic lock tracker to prevent repeating voice loops
    last_log_t = time.time()
    last_click_t = 0.0
    
    # FRIDAY startup greeting voice
    _speak_async("GestureVision X active. Systems fully operational.")

    while running:
        ret, raw = cap.read()
        if not ret or raw is None:
            # Create a black frame with "NO CAMERA DETECTED" message
            raw = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
            cv2.putText(raw, "NO CAMERA DETECTED", (100, PROC_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(raw, "Please check camera connections or permissions.", (100, PROC_H // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(raw, "Press 'Esc' to exit.", (100, PROC_H // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, raw)
            if cv2.waitKey(28) & 0xFF == 27:
                break
            time.sleep(0.02)
            continue

        # Force resize raw frame to PROC_W and PROC_H if camera defaults to a lower/different resolution
        if raw.shape[1] != PROC_W or raw.shape[0] != PROC_H:
            raw = cv2.resize(raw, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)

        raw = cv2.flip(raw, 1)
        frame_idx += 1
        ts_ms = frame_idx * 33

        # ── MediaPipe pipeline ──────────────────────────────────────
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        rgb.flags.writeable = True

        if result.hand_landmarks and result.handedness:
            # Primary Hand
            landmarks = result.hand_landmarks[0]
            handedness_raw = result.handedness[0][0].category_name
            handedness_label = "RIGHT HAND" if handedness_raw == "Left" else "LEFT HAND"
            raw_gesture = detect_gesture(landmarks, handedness_label)
            fingers = get_finger_states(landmarks, handedness_label)
            finger_count = sum(fingers)

            # Secondary Hand
            if len(result.hand_landmarks) > 1:
                landmarks_sec = result.hand_landmarks[1]
                handedness_raw_sec = result.handedness[1][0].category_name
                handedness_label_sec = "RIGHT HAND" if handedness_raw_sec == "Left" else "LEFT HAND"
                raw_gesture_sec = detect_gesture(landmarks_sec, handedness_label_sec)
                fingers_sec = get_finger_states(landmarks_sec, handedness_label_sec)
                finger_count += sum(fingers_sec)
            else:
                raw_gesture_sec = "unknown"
        else:
            landmarks = None
            handedness_label = "NO HAND"
            raw_gesture = "unknown"
            raw_gesture_sec = "unknown"
            finger_count = 0

        stable    = smoother.update(raw_gesture)
        stability = smoother.stability

        stats.log_gesture(stable)
        aria.update(stable, stability, second_gesture=raw_gesture_sec)

        # ── 1. Gesture System Mode Toggles ──────────────────────────
        info       = GESTURE_INFO.get(stable, GESTURE_INFO["unknown"])
        new_filter = info.get("filter", "normal")
        
        if (stability >= 0.92 and
                new_filter != active_filter and
                stable not in ("fist", "one_finger", "unknown")):
            prev_filter   = active_filter
            active_filter = new_filter
            trans_frame   = 0
            
        if stable == "one_finger" and stability >= 0.85:
            if active_filter != "neural_spectrum":
                prev_filter = active_filter
                active_filter = "neural_spectrum"
                trans_frame = 0
            
        if stable == "open_palm" and stability >= 0.9:
            active_filter = "normal"
            canvas.clear()
            # Turn off emergency alerts on open palm
            red_alert = False

        # Gesture Shortcut: Spiderman/Rock Horns activates Hacker Mode
        if stable == "rock" and stability >= 0.95 and not hacker_mode:
            hacker_mode = True
            system_logs.append("[OS] Matrix Hacker Mode enabled.")
            _speak_async("Hacker protocol engaged")
        elif stable == "thumbs_up" and stability >= 0.95 and hacker_mode:
            hacker_mode = False
            system_logs.append("[OS] Matrix Hacker Mode disabled.")
            _speak_async("Hacker protocol disengaged")

        # ── 2. Windows Air Mouse Controls ──────────────────────────
        if air_mouse_mode and landmarks and not writing_mode:
            # Index Finger coordinates of the main hand (Hand 0)
            ix = landmarks[8].x
            iy = landmarks[8].y
            
            # Map index coordinates to desktop coordinates with smoothing
            screen_x = int(ix * SCREEN_W)
            screen_y = int(iy * SCREEN_H)
            smooth_x, smooth_y = gaze_mouse.update(screen_x, screen_y)
            
            if _HAS_PYAUTOGUI:
                try:
                    pyautogui.moveTo(smooth_x, smooth_y)
                except Exception:
                    pass

            # Click Trigger (Pinch gesture on main hand)
            tx = landmarks[4].x
            ty = landmarks[4].y
            dist = math.dist((ix, iy), (tx, ty))
            clicked = False
            if dist < 0.045:
                clicked = True

            if clicked:
                now_c = time.time()
                # Cooldown of 0.35 seconds to prevent unintended duplicate clicks
                if now_c - last_click_t > 0.35:
                    last_click_t = now_c
                    if _HAS_PYAUTOGUI:
                        try:
                            pyautogui.click()
                            system_logs.append("[MOU] Air Click executed.")
                        except Exception:
                            pass

        # ── 3. Fist Exit Hold Timer ───────────────────────────────
        if stable == "fist":
            if fist_start is None:
                fist_start = time.time()
            elif (time.time() - fist_start) >= FIST_HOLD:
                system_logs.append("[SYS] Termination payload sent.")
                running = False
                # Trigger clean shutdown sound
                _speak_async("Operating system shutting down cleanly")
                time.sleep(1.0)
        else:
            fist_start = None

        # ── 4. Air Writing Canvas Hold Timer ───────────────────────
        if stable == "one_finger":
            if write_hold_t is None:
                write_hold_t = time.time()
            elif not writing_mode and (time.time() - write_hold_t) >= WRITE_HOLD:
                writing_mode = True
                system_logs.append("[SYS] Canvas Writing mode ON.")
                _speak_async("Writing canvas online")
        else:
            write_hold_t = None

        # Clear canvas with Fist gesture
        if stable == "fist" and writing_mode:
            writing_mode = False
            canvas.clear()
            system_logs.append("[SYS] Canvas reset successful.")
            _speak_async("Canvas cleared")

        # Draw / Erase canvas logic
        if writing_mode and landmarks:
            gx_w = int(landmarks[8].x * FRAME_W)
            gy_h = int(landmarks[8].y * FRAME_H)
            if stable == "one_finger":
                canvas.update(True, gx_w, gy_h)
            elif stable == "peace":
                canvas.erase(gx_w, gy_h, radius=32)
                canvas.update(False, 0, 0)
            else:
                canvas.update(False, 0, 0)

        # ── 5. Generate Real-time Screen Filters ───────────────────
        if active_filter == "neural_spectrum" and landmarks:
            ix_coord = landmarks[8].x
            iy_coord = landmarks[8].y
            out = apply_filter(raw, active_filter, ix_coord, iy_coord)
        else:
            out = apply_filter(raw, active_filter)

        if trans_frame < 12:
            trans_frame += 1
            out = crossfade(apply_filter(raw, prev_filter), out, trans_frame / 12.0)

        # Upscale the processed frame from PROC size (640x360) to 1280x720 for razor-sharp presentation!
        out = cv2.resize(out, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

        # Apply Writing Canvas
        out = canvas.blend(out)

        # Draw Neural Tuner dynamic holographic crosshair and stats overlay!
        if stable == "one_finger" and landmarks:
            cx_f = int(landmarks[8].x * FRAME_W)
            cy_f = int(landmarks[8].y * FRAME_H)
            r_ring = int(24 + 6 * math.sin(time.time() * 7.0))
            cv2.circle(out, (cx_f, cy_f), r_ring, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out, (cx_f, cy_f), r_ring - 6, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.circle(out, (cx_f, cy_f), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.line(out, (cx_f - 38, cy_f), (cx_f + 38, cy_f), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(out, (cx_f, cy_f - 38), (cx_f, cy_f + 38), (0, 255, 255), 1, cv2.LINE_AA)
            h_shift = int(landmarks[8].x * 180)
            s_scale = int(landmarks[8].y * 100)
            cv2.putText(out, f"TUNING NEURAL FREQ: #{h_shift}H / {s_scale}%S", (cx_f + 28, cy_f - 14), FONT, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(out, "SPECTRUM RANGE: 11.7M+ COMBINATIONS", (cx_f + 28, cy_f + 20), FONT, 0.28, (0, 255, 180), 1, cv2.LINE_AA)

        # ── 6. Master Surveillance Special Vision Modes ─────────────
        # Z key toggles B&W, H key toggles Hacker Mode
        # T key toggles Thermal Gaze, N key toggles Night Vision
        # E key triggers Emergency Red Alert Mode
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            running = False
        elif key == ord('h') or key == ord('H'):
            hacker_mode = not hacker_mode
            _speak_async("Hacker Mode toggled" if hacker_mode else "Normal Mode restored")
        elif key == ord('t') or key == ord('T'):
            if aria.tutorial_active:
                aria.stop_tutorial()
            else:
                aria.start_tutorial()
        elif key == ord('v') or key == ord('V'):
            thermal_mode = not thermal_mode
            _speak_async("Thermal scanner active" if thermal_mode else "Thermal deactivated")
        elif key == ord('n') or key == ord('N'):
            night_vision = not night_vision
            _speak_async("Night vision active" if night_vision else "Normal vision restored")
        elif key == ord('e') or key == ord('E'):
            red_alert = not red_alert
            if red_alert:
                _speak_async("Warning. Red Alert triggered. Intruder shields active.")
        elif key == ord('a') or key == ord('A'):
            air_mouse_mode = not air_mouse_mode
            _speak_async("Air Mouse active" if air_mouse_mode else "Air Mouse disabled")
        elif key == ord('g') or key == ord('G'):
            show_gesture_guide = not show_gesture_guide
            _speak_async("Gesture manual active" if show_gesture_guide else "Gesture manual closed")
        elif key == ord('m') or key == ord('M'):
            show_full_guide = not show_full_guide
            _speak_async("Master index active" if show_full_guide else "Master index closed")
        elif key == ord('k') or key == ord('K'):
            show_color_pan = not show_color_pan
        elif key == ord('d') or key == ord('D'):
            show_tracker = not show_tracker
        elif key == ord('c') or key == ord('C'):
            current_cam_idx = (current_cam_idx + 1) % 4
            cap.release()
            cap = cv2.VideoCapture(current_cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(current_cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROC_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            system_logs.append(f"[SYS] Switched to camera index {current_cam_idx}")

        # 🎛️ Dynamic Special Modes Overlays
        if hacker_mode:
            # matrix.draw(out) # Disabled to clean the screen as requested
            pass
            
        if thermal_mode:
            # Sleek jet-colored dynamic thermal signature simulation
            out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
            
        if night_vision:
            # Phosphor green military night vision mask
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            green_night = np.zeros_like(out)
            green_night[:, :, 1] = cv2.equalizeHist(gray) # Send high-luminance to green channel
            out = cv2.addWeighted(out, 0.15, green_night, 0.85, 0)
            
            # Night vision circular crosshair overlay
            cv2.circle(out, (FRAME_W // 2, FRAME_H // 2), 160, (0, 255, 120), 1, cv2.LINE_AA)
            cv2.circle(out, (FRAME_W // 2, FRAME_H // 2), 5, (0, 255, 120), -1)
            cv2.line(out, (FRAME_W // 2 - 200, FRAME_H // 2), (FRAME_W // 2 + 200, FRAME_H // 2), (0, 255, 120), 1, cv2.LINE_AA)
            cv2.line(out, (FRAME_W // 2, FRAME_H // 2 - 200), (FRAME_W // 2, FRAME_H // 2 + 200), (0, 255, 120), 1, cv2.LINE_AA)

        if red_alert:
            # Crimson alert glowing borders pulsing in real-time
            pulse = int(128 + 127 * math.sin(time.time() * 8.5))
            cv2.rectangle(out, (0, 0), (FRAME_W, FRAME_H), (0, 0, pulse), 16, cv2.LINE_AA)
            txt(out, "EMERGENCY SYSTEM LOCKDOWN ACTIVE", FRAME_W // 2 - 300, FRAME_H // 2 - 60, (50, 50, 255), 0.95, 3)

        # ── 8. Draw Futuristic Overlay Elements & Particle Trails ─────
        if result.hand_landmarks and result.handedness:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                hand_raw = result.handedness[idx][0].category_name
                hand_lbl = "RIGHT HAND" if hand_raw == "Left" else "LEFT HAND"
                draw_skeleton(out, hand_lms, stability, hand_lbl, show_telemetry=show_tracker)

        if landmarks:
            # Emit neon particles from index tip (8)
            ix_w = int(landmarks[8].x * FRAME_W)
            iy_h = int(landmarks[8].y * FRAME_H)
            particles.emit(ix_w, iy_h, color=(0, 255, 180))
            
            # Display target confidence overlay above the palm MCP
            cx_hand = int(landmarks[9].x * FRAME_W)
            cy_hand = int(landmarks[9].y * FRAME_H)
            draw_confidence_ring(out, cx_hand, cy_hand, stability, (0, 255, 180))

        # Render neon trails
        particles.update_draw(out)

        # ── 9. Interactive HUD and Sci-Fi Panel Overlays ──────────
        fps = fps_ctr.tick()
        hud_state = {
            "gesture_name": stable,
            "gesture_info": info,
            "fps":          fps,
            "fps_history":  fps_ctr.history,
            "stability":    stability,
            "screenshot_flash": False,
            "recording":    False,
            "rec_elapsed":  0,
            "fist_progress": (time.time() - fist_start) / FIST_HOLD if fist_start else 0.0,
            "stats":        stats,
            "filter_name":  active_filter,
            "writing_mode":  writing_mode,
            "writing_color": canvas.color,
            "write_progress": (time.time() - write_hold_t) / WRITE_HOLD if write_hold_t else 0.0,
            "show_gesture_guide": show_gesture_guide,
            "show_full_guide": show_full_guide,
            "handedness_label": handedness_label,
            "finger_count": finger_count
        }
        
        # Render clean HUD
        out = draw_hud(out, hud_state)

        # ── 10. Cyberpunk Gaze Radar Sweep Overlay (Top Center) ──────
        cx_radar, cy_radar = FRAME_W // 2, 23
        r_radar = 18
        # Base ring
        cv2.circle(out, (cx_radar, cy_radar), r_radar, (60, 60, 60), 1, cv2.LINE_AA)
        # Dynamic sweeping radial line
        sweep_ang = time.time() * 2.8
        sx = int(cx_radar + r_radar * math.cos(sweep_ang))
        sy = int(cy_radar + r_radar * math.sin(sweep_ang))
        cv2.line(out, (cx_radar, cy_radar), (sx, sy), (0, 255, 180), 1, cv2.LINE_AA)

        # Clean, borderless minimalist HUD overlay

        # Draw ARIA Assistant panel
        aria.draw(out)

        # Display output frame
        cv2.imshow(WINDOW_NAME, out)

    # Clean Release
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
