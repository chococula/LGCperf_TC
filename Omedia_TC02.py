"""Omedia_TC02 - Samsung TV Plus live stream start time.

Setup:
  1. Home is shown; click the Samsung TV Plus card (route: Down) or its bottom app icon
     (route: Down, Down), as in PACE_01.
  2. TV Plus is launched once; press Enter in the camera window when live TV is playing to capture
     the screen, then drag a box around the live video area on that captured screen.
     Its playing motion sets the "streaming" threshold (area and threshold are reused afterwards).
Each run:
  Power cycle (optional) -> Home -> route -> check Home with TV Plus focused (2 s pre-roll) ->
  OK starts the timer -> stop at the first frame of sustained motion in the live video area.
  Frames, MP4/GIF, CSV row (Stream_Start_ms), and upload are the same as Omedia_TC01.
"""
import cv2
import numpy as np
import os
import pandas as pd
import queue

import subprocess
import threading
import time
import winsound
from datetime import datetime

from config_store import load_saved_configuration, save_configuration
from ir_keys import LG_KEYS, ROKU_KEYS, SAMSUNG_KEYS
from result_uploader import upload_result, wait_for_uploads
from run_video import make_run_videos


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC02_config.json")
TVPLUS_SCREEN_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC02_tvplus_screen.jpg")
DEFAULT_CONFIGURATION = {
    'manufacturer': 'Samsung',
    'SoC': 'U8000',
    'SWV': '1300',
    'LGCV': '8.2.008',
    'num_runs': 5,
    'timeout': 30.0,
    'standby_seconds': 30,
    'route': [],
    'target_point': [],
    'stream_roi': [],
    'motion_threshold': 0.0,
}

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
RAW_IR_KEYSETS = {
    'LG': LG_KEYS,
    'Roku': ROKU_KEYS,
    'Samsung': SAMSUNG_KEYS,
}
APP_NAME = "Samsung TV Plus"
CAMERA_INDEX = 1
CAPTURE_WINDOW = "Omedia_TC02 Capture"
HOME_READY_SIMILARITY = 0.6     # pre-check: Home with TV Plus focused must still match this well
MIN_MOTION_THRESHOLD = 2.0      # mean gray-level change per frame inside the live video area
MOTION_THRESHOLD_RATIO = 0.4    # streaming = motion of at least 40% of the measured playing motion
MIN_STREAM_BRIGHTNESS = 20      # a black screen is not streaming
STREAM_CONFIRM_FRAMES = 3       # consecutive moving frames needed; the first one is the start time
STREAM_EXTRA_SECONDS = 1.0      # keep saving frames this long after the start is confirmed
PRE_ROLL_SECONDS = 2.0
BOOT_WAIT_SECONDS = 20
HOME_SETTLE_SECONDS = 3
KEY_GAP_SECONDS = 0.5
EXIT_WAIT_SECONDS = 3
CSV_PATH = os.path.join(SCRIPT_DIR, "Omedia_TC02_response.csv")


def ask_number(prompt, default, cast, valid):
    while True:
        text = input(f"{prompt} (default: {default:g}): ").strip()
        try:
            value = cast(text) if text else default
        except ValueError:
            print("Please enter a valid number.")
            continue
        if valid(value):
            return value
        print("That value is out of range.")


def get_user_configuration():
    defaults = load_saved_configuration(CONFIG_FILE, DEFAULT_CONFIGURATION)
    print("\n" + "=" * 60)
    print(f"  Omedia_TC02 - {APP_NAME} Live Stream Start Time")
    print("=" * 60)

    while True:
        manufacturer = input(
            f"Enter IR key set {tuple(RAW_IR_KEYSETS)} (default: {defaults['manufacturer']}): "
        ).strip() or defaults['manufacturer']
        if manufacturer in RAW_IR_KEYSETS:
            break
        print(f"Choose one of: {', '.join(RAW_IR_KEYSETS)}")
    SoC = input(f"Enter SoC Model (default: {defaults['SoC']}): ").strip() or defaults['SoC']
    SWV = input(f"Enter Software Version (default: {defaults['SWV']}): ").strip() or defaults['SWV']
    LGCV = input(f"Enter LG CV Version (default: {defaults['LGCV']}): ").strip() or defaults['LGCV']
    num_runs = ask_number("Enter number of runs", defaults['num_runs'], int, lambda v: v > 0)
    timeout = ask_number("Enter stream start timeout in seconds", defaults['timeout'], float, lambda v: v > 0)
    standby_seconds = ask_number("Enter TV standby seconds before each run (0 = no power cycle)",
                                 defaults['standby_seconds'], int, lambda v: v >= 0)

    configuration = dict(defaults)
    configuration.update({
        'manufacturer': manufacturer,
        'SoC': SoC,
        'SWV': SWV,
        'LGCV': LGCV,
        'num_runs': num_runs,
        'timeout': timeout,
        'standby_seconds': standby_seconds,
    })
    print("\nConfiguration:")
    for key in ('manufacturer', 'SoC', 'SWV', 'LGCV', 'num_runs', 'timeout', 'standby_seconds'):
        print(f"  {key}: {configuration[key]}")
    if configuration['route']:
        print(f"  saved route to {APP_NAME}: {', '.join(configuration['route'])}")
    if input("Confirm and proceed? (Y/n): ").strip().lower() == 'n':
        return get_user_configuration()

    save_configuration(CONFIG_FILE, configuration)
    return configuration


def initialize_camera(camera_index=CAMERA_INDEX):
    camera_indices = [camera_index] + [index for index in range(4) if index != camera_index]
    for index in camera_indices:
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Camera initialized (index: {index})")
            return camera
        camera.release()

    raise RuntimeError(
        f"No camera could be opened at indices {camera_indices}. "
        "Check the camera connection or close other apps using the camera."
    )


def position_camera(camera):
    window_name = "Omedia_TC02 Camera Setup"
    print("Point the camera at the TV screen. Press Enter when the image is clear, or Q to cancel.")
    while True:
        ret, frame = camera.read()
        if not ret:
            raise RuntimeError("Could not read a frame from the camera.")
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):
            cv2.destroyWindow(window_name)
            return True
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return False


def read_frame(camera):
    ret, frame = camera.read()
    return frame if ret else None


class CameraFrameCapture:
    def __init__(self, camera, max_queue_size=120):
        self.camera = camera
        self.frames = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.started_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def _capture_loop(self):
        self.started_event.set()
        while not self.stop_event.is_set():
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            captured_at = time.perf_counter()
            try:
                self.frames.put_nowait((captured_at, frame))
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait((captured_at, frame))
                except queue.Full:
                    pass

    def start(self):
        self.thread.start()
        if not self.started_event.wait(timeout=1.0):
            raise RuntimeError("Camera capture worker did not start.")

    def clear_buffer(self):
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def get_frame(self, timeout=0.1):
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def draw_overlay(frame, lines):
    # Sizes are tuned for 1280 px wide frames and scaled to the actual camera resolution.
    # Text sits on a translucent dark box so it stays readable on bright or dark screens.
    ratio = max(frame.shape[1] / 1280.0, 0.5)
    pad = round(12 * ratio)
    thickness = max(1, round(2 * ratio))
    layout = []
    y = pad
    width = 0
    for text, color, scale in lines:
        scale *= ratio
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += text_height
        layout.append((text, color, scale, y))
        y += baseline + round(10 * ratio)
        width = max(width, text_width)
    if not layout:
        return frame

    box_right = min(frame.shape[1], 2 * pad + width)
    box_bottom = min(frame.shape[0], y + pad - round(10 * ratio))
    region = frame[0:box_bottom, 0:box_right]
    region[:] = (region * 0.35).astype(region.dtype)
    for text, color, scale, line_y in layout:
        cv2.putText(frame, text, (pad, line_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame


def wait_with_countdown(camera, seconds, run_idx, title, until):
    print(f"\n{title}: {seconds} seconds")
    end_at = time.perf_counter() + seconds
    last_printed = None
    while True:
        remaining = end_at - time.perf_counter()
        if remaining <= 0:
            break
        if last_printed != int(remaining) + 1:
            last_printed = int(remaining) + 1
            print(f"  {last_printed} seconds remaining...", end="\r")
            if last_printed <= 3:
                winsound.Beep(800, 100)

        ret, frame = camera.read()
        if ret:
            draw_overlay(frame, [
                (f"RUN {run_idx} {title.upper()}", (0, 255, 255), 1.2),
                (f"{int(remaining) // 60}:{remaining % 60:04.1f} to {until}", (0, 255, 255), 1.2),
            ])
            cv2.imshow(CAPTURE_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Measurement canceled by user.")
    print(" " * 50, end="\r")
    print(f"{title} complete.")


def send_key(key_name, manufacturer):
    if manufacturer not in RAW_IR_KEYSETS:
        raise ValueError(f"Unsupported IR key set: {manufacturer}")
    keyset = RAW_IR_KEYSETS[manufacturer]
    if key_name not in keyset:
        raise ValueError(f"{manufacturer} IR key is not configured: {key_name}")

    print(f"Sending '{key_name}' via Flirc ({manufacturer} IR)...")
    action_started_at = time.perf_counter()
    subprocess.run(
        [FLIRC_UTIL_PATH, 'sendir', f"--csv={keyset[key_name]}", '--repeat=1'],
        check=True,
        capture_output=True,
        text=True,
    )
    return action_started_at


class FrameWriter:
    """Save frames on a background thread so disk writes never stall frame processing."""

    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.frames = queue.Queue()
        self.saved = 0
        os.makedirs(frames_dir, exist_ok=True)
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while True:
            item = self.frames.get()
            if item is None:
                return
            filename, frame = item
            if cv2.imwrite(os.path.join(self.frames_dir, filename), frame):
                self.saved += 1

    def save(self, filename, frame):
        self.frames.put((filename, frame))

    def close(self):
        pending = self.frames.qsize()
        if pending:
            print(f"Writing {pending} remaining frames to disk...")
        self.frames.put(None)
        self.thread.join()


def screen_similarity(frame, reference):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if blurred.shape[0] < reference.shape[0] or blurred.shape[1] < reference.shape[1]:
        raise RuntimeError("Camera frame is smaller than the saved Home reference.")
    return float(np.max(cv2.matchTemplate(blurred, reference, cv2.TM_CCOEFF_NORMED)))


# ---------- calibration: route to Samsung TV Plus (same as PACE_01) ----------

def route_for_click(point, frame_shape):
    """D-pad route from Home for a click on the Samsung TV Plus card or its bottom app icon (PACE_01 layout)."""
    x_ratio = point[0] / frame_shape[1]
    y_ratio = point[1] / frame_shape[0]
    if 0.09 <= x_ratio <= 0.35 and 0.47 <= y_ratio <= 0.68:
        return ['Down']
    if 0.09 <= x_ratio <= 0.18 and 0.68 < y_ratio <= 0.78:
        return ['Down', 'Down']
    return None


def select_app_route(camera, saved_point=None):
    """Show Home, let the user click Samsung TV Plus, and return (route, point)."""
    window_name = f"Select {APP_NAME}"
    selection = {'point': tuple(saved_point) if saved_point else None}

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection['point'] = (x, y)

    print(f"\n[SETUP] Click the {APP_NAME} card or its bottom app icon, then press Enter (Q = cancel).")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        preview = frame.copy()
        route = route_for_click(selection['point'], frame.shape) if selection['point'] else None
        if selection['point'] is None:
            lines = [(f"Click {APP_NAME}, then press ENTER", (0, 255, 0), 1.0)]
        elif route:
            lines = [(f"{APP_NAME} selected: {', '.join(route)} + OK.  ENTER = go", (0, 255, 0), 1.0)]
        else:
            lines = [(f"Click the {APP_NAME} card or its app icon", (0, 0, 255), 1.0)]
        if selection['point'] is not None:
            cv2.circle(preview, selection['point'], 12, (0, 255, 0) if route else (0, 0, 255), 3)
        cv2.imshow(window_name, draw_overlay(preview, lines))
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and route:
            cv2.destroyWindow(window_name)
            print(f"[SETUP] Route to {APP_NAME}: {', '.join(route)} + OK")
            return route, selection['point']
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt(f"{APP_NAME} selection canceled.")


def press_home(manufacturer, camera):
    send_key('Home', manufacturer)
    wait_until = time.perf_counter() + HOME_SETTLE_SECONDS
    while time.perf_counter() < wait_until:
        frame = read_frame(camera)
        if frame is not None:
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [("Waiting for Home screen...", (255, 255, 255), 1.0)]))
        cv2.waitKey(1)


def replay_route(route, manufacturer, camera):
    for key in route:
        send_key(key, manufacturer)
        end = time.perf_counter() + KEY_GAP_SECONDS
        while time.perf_counter() < end:
            frame = read_frame(camera)
            if frame is not None:
                cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [(f"Route: {', '.join(route)}", (255, 255, 255), 0.9)]))
            cv2.waitKey(1)


def calibrate_route(configuration, camera):
    """Home first, then the user clicks Samsung TV Plus; the click decides the D-pad route."""
    press_home(configuration['manufacturer'], camera)
    saved_point = configuration['target_point'] or None
    route, point = select_app_route(camera, saved_point)
    configuration['route'] = route
    configuration['target_point'] = [int(point[0]), int(point[1])]
    save_configuration(CONFIG_FILE, configuration)
    return route


# ---------- calibration: where the live TV stream plays ----------

def roi_gray(frame, roi):
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def roi_motion(previous_gray, current_gray):
    return float(np.mean(cv2.absdiff(previous_gray, current_gray)))


def capture_home_ready(camera):
    """Median gray frame of Home with TV Plus focused, used to check each run starts from there."""
    samples = []
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end or not samples:
        frame = read_frame(camera)
        if frame is not None:
            samples.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def capture_tvplus_screen(camera, calibration_dir, home_ready, manufacturer):
    """Live view after OK; Enter captures the TV Plus screen once it is no longer the Home screen."""
    print(f"\n[SETUP] {APP_NAME} is opening. When the live TV stream is playing, press ENTER in the "
          "camera window to capture the screen (O = send OK again, Q = cancel).")
    # Drop key presses left over from earlier windows (e.g. a held Enter) so they cannot capture Home.
    for _ in range(50):
        if cv2.waitKey(1) == -1:
            break
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        on_home = screen_similarity(frame, home_ready) >= HOME_READY_SIMILARITY
        message = (f"Still the Home screen - waiting for {APP_NAME} to open (O = send OK again)" if on_home
                   else f"{APP_NAME}: press ENTER when live TV is playing")
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), [(message, (0, 255, 255) if on_home else (0, 255, 0), 1.0)]))
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            if on_home:
                print(f"[SETUP] The camera still sees the Home screen; wait until {APP_NAME} is showing.")
                continue
            # Capture fresh frames taken after the key press, not the one already on screen.
            for _ in range(3):
                fresh = read_frame(camera)
                if fresh is not None:
                    frame = fresh
            break
        if key == ord('o'):
            send_key('OK', manufacturer)
        if key == ord('q'):
            raise KeyboardInterrupt("Setup canceled.")
    for path in (TVPLUS_SCREEN_FILE, os.path.join(calibration_dir, "tvplus_screen.jpg")):
        cv2.imwrite(path, frame)
    print(f"[SETUP] Captured the {APP_NAME} screen: {TVPLUS_SCREEN_FILE}")
    return frame


def select_stream_area(screen):
    """User drags a box around the live video on the captured TV Plus screen."""
    window_name = "Select live video area"
    print("\n[SETUP] On the captured screen, drag a box around the area where the live TV video plays, "
          "then press Enter (C = cancel and redo).")
    preview = draw_overlay(screen.copy(), [("Drag around the LIVE VIDEO area, then ENTER", (0, 255, 0), 1.0)])
    roi = cv2.selectROI(window_name, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = (int(v) for v in roi)
    if w < 20 or h < 20:
        return None
    return [x, y, w, h]


def measure_playing_motion(camera, roi, seconds=2.0):
    """Typical frame-to-frame change inside the area while live TV is playing."""
    values = []
    previous = None
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame = read_frame(camera)
        if frame is None:
            continue
        current = roi_gray(frame, roi)
        if previous is not None:
            values.append(roi_motion(previous, current))
        previous = current
        preview = frame.copy()
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(preview, [("Measuring live video motion...", (255, 255, 255), 1.0)]))
        cv2.waitKey(1)
    return float(np.median(values)) if values else 0.0


def setup_stream_detection(configuration, camera, route, calibration_dir, home_ready):
    """Capture the TV Plus screen, let the user mark the live video area, and measure its motion.

    A saved area is shown on the saved TV Plus screen and can be reused.
    """
    manufacturer = configuration['manufacturer']
    saved_screen = cv2.imread(TVPLUS_SCREEN_FILE) if os.path.isfile(TVPLUS_SCREEN_FILE) else None
    if configuration['stream_roi'] and configuration['motion_threshold'] > 0:
        x, y, w, h = configuration['stream_roi']
        print(f"\n[SETUP] Saved live video area: x={x} y={y} w={w} h={h}, "
              f"motion threshold {configuration['motion_threshold']:.1f}")
        if saved_screen is not None:
            preview = saved_screen.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.imshow("Saved live video area", draw_overlay(preview, [("Saved live video area", (0, 255, 255), 1.0)]))
            cv2.waitKey(1)
        reuse = input("Reuse the saved live video area? (Y/n): ").strip().lower() in ('', 'y', 'yes')
        if saved_screen is not None:
            cv2.destroyWindow("Saved live video area")
        if reuse:
            return configuration['stream_roi'], configuration['motion_threshold']

    while True:
        print(f"\n[SETUP] Launching {APP_NAME} once (not measured).")
        send_key('OK', manufacturer)
        screen = capture_tvplus_screen(camera, calibration_dir, home_ready, manufacturer)
        roi = select_stream_area(screen)
        if roi is None:
            print("[SETUP] The box is too small. Try again.")
            press_home(manufacturer, camera)
            replay_route(route, manufacturer, camera)
            continue
        playing = measure_playing_motion(camera, roi)
        threshold = max(MIN_MOTION_THRESHOLD, playing * MOTION_THRESHOLD_RATIO)
        print(f"[SETUP] Live video motion {playing:.1f}; streaming counts from motion >= {threshold:.1f}")
        if playing < MIN_MOTION_THRESHOLD:
            print("[SETUP] The area barely moves. Pick the area where the video itself plays.")
        if input("Use this live video area? (Y/n): ").strip().lower() in ('', 'y', 'yes'):
            break
        press_home(manufacturer, camera)
        replay_route(route, manufacturer, camera)

    configuration['stream_roi'] = roi
    configuration['motion_threshold'] = float(threshold)
    save_configuration(CONFIG_FILE, configuration)
    press_home(manufacturer, camera)
    return roi, threshold


# ---------- measurement ----------

def check_home_ready(frame_capture, home_ready, output_dir, seconds=PRE_ROLL_SECONDS):
    """Before OK, confirm Home with TV Plus focused is showing; the frames double as the video pre-roll."""
    frame_capture.clear_buffer()
    scores = []
    last_frame = None
    pre_frames = []
    end_at = time.perf_counter() + seconds
    while time.perf_counter() < end_at or not scores:
        captured = frame_capture.get_frame(timeout=1.0)
        if captured is None:
            if time.perf_counter() > end_at + 2.0:
                raise RuntimeError("No camera frames during the pre-launch check.")
            continue
        captured_at, frame = captured
        score = screen_similarity(frame, home_ready)
        scores.append(score)
        last_frame = frame
        pre_frames.append((captured_at, frame.copy(), score))

    score = float(np.median(scores))
    ready = score >= HOME_READY_SIMILARITY
    last_frame = last_frame.copy()
    draw_overlay(last_frame, [
        (f"PRE-CHECK  Home match {score:.3f} / {HOME_READY_SIMILARITY:.3f}", (0, 255, 255), 1.0),
        ("READY - OK" if ready else f"NOT ON HOME WITH {APP_NAME.upper()} - RUN INVALID",
         (0, 255, 0) if ready else (0, 0, 255), 1.2),
    ])
    cv2.imwrite(os.path.join(output_dir, "PRECHECK_BEFORE_OK.jpg"), last_frame)
    cv2.imshow(CAPTURE_WINDOW, last_frame)
    cv2.waitKey(1)
    return ready, score, pre_frames


def measure_stream_start(frame_capture, roi, motion_threshold, run_idx, output_dir, timeout,
                         ok_pressed_at, pre_frames=()):
    """Time from OK until the live video area shows sustained playback motion."""
    x, y, w, h = roi
    frame_count = 0
    last_elapsed_ms = 0.0
    detected = False
    armed = False
    streak = 0
    candidate_ms = None
    detected_elapsed_ms = None
    previous_roi = None
    stop_at_ms = None
    result_saved = False
    frame_writer = FrameWriter(os.path.join(output_dir, "frames"))

    def annotate(image, lines):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return draw_overlay(image, lines)

    # Pre-roll: the last PRE_ROLL_SECONDS before OK, saved with negative elapsed time for the video.
    pre_index = 0
    for captured_at, frame, _ in pre_frames:
        elapsed_ms = (captured_at - ok_pressed_at) * 1000.0
        if elapsed_ms < -PRE_ROLL_SECONDS * 1000.0 or elapsed_ms >= 0:
            continue
        pre_index += 1
        frame_writer.save(
            f"frame_p{pre_index:04d}_{elapsed_ms:09.1f}ms_sim0.000.jpg",
            annotate(frame, [(f"RUN {run_idx}  BEFORE OK  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4)]),
        )

    print(f"[RUN {run_idx}] Waiting for the live stream to start (motion >= {motion_threshold:.1f} "
          f"for {STREAM_CONFIRM_FRAMES} frames)...")
    try:
        while time.perf_counter() - ok_pressed_at <= timeout:
            captured = frame_capture.get_frame()
            if captured is None:
                continue
            captured_at, frame = captured
            if captured_at < ok_pressed_at:
                continue

            frame_count += 1
            elapsed_ms = (captured_at - ok_pressed_at) * 1000.0
            last_elapsed_ms = elapsed_ms
            current_roi = roi_gray(frame, roi)
            motion = roi_motion(previous_roi, current_roi) if previous_roi is not None else 0.0
            brightness = float(np.mean(current_roi))
            previous_roi = current_roi

            playing = motion >= motion_threshold and brightness >= MIN_STREAM_BRIGHTNESS
            if not playing:
                armed = True
                streak = 0
            elif armed and not detected:
                streak += 1
                if streak == 1:
                    candidate_ms = elapsed_ms
                if streak >= STREAM_CONFIRM_FRAMES:
                    detected = True
                    detected_elapsed_ms = candidate_ms
                    stop_at_ms = elapsed_ms + STREAM_EXTRA_SECONDS * 1000.0
                    print(f"[RUN {run_idx}] Live stream started at {candidate_ms:.2f} ms "
                          f"(motion {motion:.1f} >= {motion_threshold:.1f}).")
                    winsound.Beep(1500, 250)
                    winsound.Beep(2500, 350)

            color = (0, 255, 0) if playing else (0, 255, 255)
            saved_lines = [
                (f"RUN {run_idx}  #{frame_count}  TIMER {elapsed_ms / 1000.0:.3f} s", color, 1.4),
                (f"Motion {motion:.1f} / {motion_threshold:.1f}", color, 1.0),
            ]
            if detected:
                saved_lines.append((f"LIVE STREAM STARTED at {detected_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            frame_writer.save(
                f"frame_{frame_count:05d}_{elapsed_ms:09.1f}ms_sim{min(motion / motion_threshold, 9.999):.3f}.jpg",
                annotate(frame.copy(), saved_lines),
            )

            shown_ms = detected_elapsed_ms if detected else elapsed_ms
            live_lines = [
                (f"RUN {run_idx}  TIMER {shown_ms / 1000.0:.3f} s", (0, 255, 0) if detected else (0, 255, 255), 1.4),
                (f"Motion {motion:.1f} / {motion_threshold:.1f}", color, 1.0),
            ]
            if detected:
                live_lines.append(("LIVE STREAM STARTED", (0, 255, 0), 1.2))
            cv2.imshow(CAPTURE_WINDOW, annotate(frame, live_lines))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")
            if detected and not result_saved:
                cv2.imwrite(os.path.join(output_dir, "RESULT_STREAM_STARTED.jpg"), frame)
                result_saved = True
            if stop_at_ms is not None and elapsed_ms >= stop_at_ms:
                break
    finally:
        frame_writer.close()
        print(f"[RUN {run_idx}] Saved {frame_writer.saved} frames to {frame_writer.frames_dir}")

    if detected:
        cv2.waitKey(1500)
        last_elapsed_ms = detected_elapsed_ms
    else:
        print(f"[RUN {run_idx}] The live stream did not start within {timeout:g} seconds.")
        winsound.Beep(400, 700)

    return {
        'run': run_idx,
        'status': 'PASS' if detected else 'FAIL',
        'response_time_ms': last_elapsed_ms,
        'frame_count': frame_count,
        'directory': output_dir,
    }


def invalid_result(run_idx, output_dir, reason):
    print(f"[RUN {run_idx}] INVALID: {reason}")
    winsound.Beep(400, 300)
    winsound.Beep(400, 300)
    return {'run': run_idx, 'status': 'INVALID', 'response_time_ms': None, 'frame_count': 0,
            'directory': output_dir, 'reason': reason}


def run_once(configuration, camera, run_idx, output_dir, route, home_ready, roi, motion_threshold):
    manufacturer = configuration['manufacturer']

    if configuration['standby_seconds'] > 0:
        print(f"\n[RUN {run_idx}] STEP 1: Power OFF, standby, Power ON.")
        send_key('Power', manufacturer)
        winsound.Beep(1000, 200)
        wait_with_countdown(camera, configuration['standby_seconds'], run_idx, "Standby", "Power ON")
        send_key('Power', manufacturer)
        wait_with_countdown(camera, BOOT_WAIT_SECONDS, run_idx, "Boot wait", "Home")

    print(f"[RUN {run_idx}] STEP 2: Home -> {', '.join(route)} to focus {APP_NAME}.")
    press_home(manufacturer, camera)
    replay_route(route, manufacturer, camera)

    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        ready, score, pre_frames = check_home_ready(frame_capture, home_ready, output_dir)
        if not ready:
            return invalid_result(run_idx, output_dir,
                                  f"Home with {APP_NAME} focused was not showing (match {score:.3f}) before OK.")
        frame_capture.clear_buffer()
        print(f"[RUN {run_idx}] STEP 3: OK launches {APP_NAME}; the timer starts now.")
        ok_pressed_at = send_key('OK', manufacturer)
        winsound.Beep(1500, 300)
        return measure_stream_start(frame_capture, roi, motion_threshold, run_idx, output_dir,
                                    configuration['timeout'], ok_pressed_at, pre_frames)
    finally:
        frame_capture.stop()


def main():
    configuration = get_user_configuration()
    camera = None

    try:
        camera = initialize_camera()
        if not position_camera(camera):
            print("Camera setup canceled.")
            return
        cv2.namedWindow(CAPTURE_WINDOW, cv2.WINDOW_AUTOSIZE)

        calibration_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_dir = os.path.join(
            "C:/Temp",
            f"Omedia_TC02_Calibration_{calibration_timestamp}_{configuration['SoC']}"
        )
        os.makedirs(calibration_dir, exist_ok=True)

        route = calibrate_route(configuration, camera)
        replay_route(route, configuration['manufacturer'], camera)
        home_ready = capture_home_ready(camera)
        cv2.imwrite(os.path.join(calibration_dir, "home_ready.jpg"), home_ready)
        roi, motion_threshold = setup_stream_detection(configuration, camera, route, calibration_dir, home_ready)

        results = []
        for run_idx in range(1, configuration['num_runs'] + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "C:/Temp",
                f"Omedia_TC02_{run_idx:02d}_{timestamp}_{configuration['SoC']}"
                f"_SWV{configuration['SWV']}_LGCV{configuration['LGCV']}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n  [RUN {run_idx}/{configuration['num_runs']}] {APP_NAME} launch\n{'=' * 60}")

            result = run_once(configuration, camera, run_idx, output_dir, route, home_ready, roi, motion_threshold)

            if result['status'] != 'INVALID':
                make_run_videos(output_dir, run_label=f"RUN {run_idx}",
                                end_ms=result['response_time_ms'] if result['status'] == 'PASS' else None)
            # Leave the app so the next run starts from Home.
            send_key('Home', configuration['manufacturer'])
            time.sleep(EXIT_WAIT_SECONDS)

            results.append(result)
            row = {
                'TC': 'Omedia_TC02',
                'Run': result['run'],
                'Timestamp': timestamp,
                'Standby_Seconds': configuration['standby_seconds'],
                'Stream_Start_ms': (
                    round(result['response_time_ms'], 2) if result['response_time_ms'] is not None else None
                ),
                'Frames': result['frame_count'],
                'Status': result['status'],
                'Directory': result['directory'],
            }
            pd.DataFrame([row]).to_csv(
                CSV_PATH,
                mode='a',
                header=not os.path.exists(CSV_PATH),
                index=False,
            )
            upload_result({
                'TC': row['TC'],
                'Session': calibration_timestamp,
                'Run': row['Run'],
                'Timestamp': timestamp,
                'Manufacturer': configuration['manufacturer'],
                'SoC': configuration['SoC'],
                'SWV': configuration['SWV'],
                'LGCV': configuration['LGCV'],
                'Standby_Seconds': configuration['standby_seconds'],
                'Motion_Threshold': round(motion_threshold, 2),
                'Route': ' '.join(route),
                'Stream_Start_ms': row['Stream_Start_ms'],
                'Frames': row['Frames'],
                'Status': row['Status'],
            })

        passed = sum(result['status'] == 'PASS' for result in results)
        invalid = [result for result in results if result['status'] == 'INVALID']
        for result in invalid:
            print(f"Run {result['run']} INVALID: {result.get('reason', '')}")
        print(f"\nOmedia_TC02 complete: {passed}/{len(results)} runs passed.")
        for frequency in (1500, 2000, 2500):
            winsound.Beep(frequency, 200)
        print(f"Results saved to: {CSV_PATH}")
    except KeyboardInterrupt as error:
        print(f"\n{error}")
    except Exception as error:
        print(f"Omedia_TC02 failed: {error}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        wait_for_uploads()


if __name__ == "__main__":
    main()
