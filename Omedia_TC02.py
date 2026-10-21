"""Omedia_TC02 - Samsung TV Plus Live TV channel zapping (Ch Up) time.

Setup (two user inputs):
  1. Home is shown; click Samsung TV Plus. The camera walks the focus there (or finish with the
     arrow keys) and OK opens it. If Home stays on screen, click the tile again on that view.
  2. The opened TV Plus screen is captured; click the Live TV channel to enter. The camera walks the
     focus there (or move it with the keyboard arrow keys), then OK enters the channel.
  The middle of the playing picture is used to detect playback; its motion sets the threshold.
Each run:
  2 s pre-roll while the channel plays -> Ch Up starts the timer -> stop at the first of
  3 consecutive moving frames after the picture went black/frozen. Frames, MP4/GIF, CSV row
  (Zap_Time_ms), and upload are the same as Omedia_TC01.
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

import focus_navigator as nav
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
    'timeout': 15.0,
    'zap_interval_seconds': 8,
    'route': [],
    'target_point': [],
    'channel_route': [],
    'channel_point': [],
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
MIN_STILL_FRAMES = 2            # after Ch Up the picture must stop (black/frozen) this many frames first
ARROW_KEYS = {2424832: 'Left', 2490368: 'Up', 2555904: 'Right', 2621440: 'Down'}   # cv2.waitKeyEx codes
PRE_ROLL_SECONDS = 2.0
HOME_SETTLE_SECONDS = 3
KEY_GAP_SECONDS = 0.5
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
    print(f"  Omedia_TC02 - {APP_NAME} Live TV Channel Zapping (Ch Up)")
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
    timeout = ask_number("Enter zap timeout in seconds", defaults['timeout'], float, lambda v: v > 0)
    zap_interval_seconds = ask_number("Enter seconds to watch each channel before the next Ch Up",
                                      defaults['zap_interval_seconds'], int, lambda v: v >= 0)

    configuration = dict(defaults)
    configuration.update({
        'manufacturer': manufacturer,
        'SoC': SoC,
        'SWV': SWV,
        'LGCV': LGCV,
        'num_runs': num_runs,
        'timeout': timeout,
        'zap_interval_seconds': zap_interval_seconds,
    })
    print("\nConfiguration:")
    for key in ('manufacturer', 'SoC', 'SWV', 'LGCV', 'num_runs', 'timeout', 'zap_interval_seconds'):
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


# ---------- input 1: where Samsung TV Plus is on Home ----------

def select_home_target(camera, saved_point=None):
    """Show Home and let the user click Samsung TV Plus. Returns the clicked point (frame pixels)."""
    window_name = f"Select {APP_NAME}"
    selection = {'point': tuple(saved_point) if saved_point else None}

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection['point'] = (x, y)

    print(f"\n[SETUP] Click {APP_NAME} on the Home screen, then press Enter (Q = cancel).")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        preview = frame.copy()
        if selection['point'] is None:
            lines = [(f"Click {APP_NAME}, then press ENTER", (0, 255, 0), 1.0)]
        else:
            cv2.drawMarker(preview, selection['point'], (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(preview, selection['point'], 16, (0, 255, 0), 2)
            same = saved_point and tuple(selection['point']) == tuple(saved_point)
            lines = [("ENTER = go to the marked spot (saved),  or click another" if same
                      else "ENTER = go there,  click again to change", (0, 255, 0), 1.0)]
        cv2.imshow(window_name, draw_overlay(preview, lines))
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and selection['point'] is not None:
            cv2.destroyWindow(window_name)
            return selection['point']
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


def calibrate_route(configuration, camera, calibration_dir):
    """Home -> user clicks Samsung TV Plus -> the camera walks the focus there. Ends with it focused."""
    manufacturer = configuration['manufacturer']
    press_home(manufacturer, camera)
    saved_point = configuration['target_point'] or None
    point = select_home_target(camera, saved_point)
    if saved_point and tuple(point) == tuple(saved_point) and configuration['route']:
        print(f"[SETUP] Using the saved route to {APP_NAME}: {', '.join(configuration['route'])}")
        replay_route(configuration['route'], manufacturer, camera)
        return configuration['route']
    route = move_to_point(configuration, camera, point, APP_NAME, os.path.join(calibration_dir, "nav_home"))
    configuration['route'] = route
    configuration['target_point'] = [int(point[0]), int(point[1])]
    save_configuration(CONFIG_FILE, configuration)
    return route


# ---------- step 2: open TV Plus and pick the Live TV channel ----------

def roi_gray(frame, roi):
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def roi_motion(previous_gray, current_gray):
    return float(np.mean(cv2.absdiff(previous_gray, current_gray)))


def auto_stream_roi(frame_shape):
    """Middle of the screen: the full-screen live video, clear of the channel banner at the top/bottom."""
    h, w = frame_shape[:2]
    return [int(w * 0.20), int(h * 0.22), int(w * 0.60), int(h * 0.50)]


def capture_home_ready(camera):
    """Median gray frame of Home with the chosen tile focused (tells Home apart from TV Plus)."""
    samples = []
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end or not samples:
        frame = read_frame(camera)
        if frame is not None:
            samples.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def grab_still(camera, label=None):
    """Settled median frame, also shown in the capture window."""
    frame = nav.settle_and_grab(lambda: read_frame(camera))
    preview = frame.copy()
    if label:
        draw_overlay(preview, [(label, (255, 255, 255), 1.0)])
    cv2.imshow(CAPTURE_WINDOW, preview)
    cv2.waitKey(1)
    return frame


def open_clicked_live_tv(configuration, camera, point, frame_shape, context):
    """Home stayed on screen: walk the focus to the tile the user clicked on that view and press OK."""
    manufacturer = configuration['manufacturer']
    print("[SETUP] Going to the clicked tile on Home and pressing OK.")
    press_home(manufacturer, camera)
    route = move_to_point(configuration, camera, point, "the tile", context.get('nav_dir'))
    context['home_ready'] = capture_home_ready(camera)
    configuration['route'] = route
    configuration['target_point'] = [int(point[0]), int(point[1])]
    save_configuration(CONFIG_FILE, configuration)
    send_key('OK', manufacturer)


def capture_tvplus_screen(configuration, camera, calibration_dir, context):
    """Live view after OK; Enter captures the TV Plus screen once it is no longer the Home screen.

    While the camera still sees Home, a click on the view chooses which tile to open.
    """
    manufacturer = configuration['manufacturer']
    print(f"\n[SETUP] {APP_NAME} is opening. When it has loaded, press ENTER in the camera window to "
          "capture the screen. If Home stays on screen, click the tile to open (O = send OK again, Q = cancel).")
    clicks = []

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(CAPTURE_WINDOW, on_mouse)
    # Drop key presses left over from earlier windows (e.g. a held Enter) so they cannot capture Home.
    for _ in range(50):
        if cv2.waitKey(1) == -1:
            break
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        on_home = screen_similarity(frame, context['home_ready']) >= HOME_READY_SIMILARITY
        if on_home and clicks:
            open_clicked_live_tv(configuration, camera, clicks.pop(), frame.shape, context)
            clicks.clear()
            continue
        clicks.clear()
        if on_home:
            lines = [(f"Still the Home screen - waiting for {APP_NAME} to open", (0, 255, 255), 1.0),
                     ("Click the tile to open,  O = send OK again", (0, 255, 255), 0.9)]
        else:
            lines = [(f"{APP_NAME}: press ENTER when it has loaded", (0, 255, 0), 1.0)]
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), lines))
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            if on_home:
                print(f"[SETUP] The camera still sees the Home screen; wait until {APP_NAME} is showing.")
                continue
            for _ in range(3):              # fresh frames taken after the key press
                fresh = read_frame(camera)
                if fresh is not None:
                    frame = fresh
            break
        if key == ord('o'):
            send_key('OK', manufacturer)
        if key == ord('q'):
            raise KeyboardInterrupt("Setup canceled.")
    cv2.setMouseCallback(CAPTURE_WINDOW, lambda *a: None)
    for path in (TVPLUS_SCREEN_FILE, os.path.join(calibration_dir, "tvplus_screen.jpg")):
        cv2.imwrite(path, frame)
    print(f"[SETUP] Captured the {APP_NAME} screen: {TVPLUS_SCREEN_FILE}")
    return frame


def click_channel(screen):
    """On the captured TV Plus screen, the user clicks the Live TV channel to enter."""
    window_name = "Select Live TV channel"
    selection = {'point': None}

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection['point'] = (x, y)

    print("\n[SETUP] Click the Live TV channel to enter on the captured screen, then press Enter (Q = cancel).")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        preview = screen.copy()
        if selection['point'] is None:
            lines = [("Click the Live TV channel to enter, then ENTER", (0, 255, 0), 1.0)]
        else:
            cv2.drawMarker(preview, selection['point'], (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(preview, selection['point'], 16, (0, 255, 0), 2)
            lines = [("ENTER = go to this channel,  click again to change", (0, 255, 0), 1.0)]
        cv2.imshow(window_name, draw_overlay(preview, lines))
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and selection['point'] is not None:
            cv2.destroyWindow(window_name)
            return selection['point']
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Channel selection canceled.")


def keyboard_drive(configuration, camera, pressed):
    """Arrow keys in the camera window move the TV focus (sent as IR); Enter when on the channel."""
    print("[SETUP] Use the arrow keys in the camera window to move to the channel, then press Enter "
          "(Backspace = undo last move).")
    manufacturer = configuration['manufacturer']
    while True:
        frame = read_frame(camera)
        if frame is not None:
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), [
                ("Arrow keys = move,  ENTER = this channel,  Backspace = undo", (0, 255, 0), 1.0),
                (f"Keys: {' '.join(pressed[-12:])}", (255, 255, 255), 0.8),
            ]))
        code = cv2.waitKeyEx(1)
        if code in ARROW_KEYS:
            key = ARROW_KEYS[code]
            send_key(key, manufacturer)
            pressed.append(key)
        elif code in (8, 0x2E0000) and pressed:          # Backspace / Delete: undo the last move
            undo = nav.OPPOSITE.get(pressed[-1])
            if undo:
                send_key(undo, manufacturer)
                pressed.append(undo)
        elif code in (10, 13):
            return pressed


def move_to_point(configuration, camera, point, what, debug_dir=None):
    """The camera walks the focus to the clicked point; the user confirms or finishes with arrow keys.

    Returns every key pressed (replaying them from the same start lands on the same tile).
    """
    manufacturer = configuration['manufacturer']
    pressed = []

    def press(key):
        send_key(key, manufacturer)
        pressed.append(key)

    focus_box = None
    try:
        _, frame, focus_box = nav.navigate_to_target(press, lambda: grab_still(camera, f"Moving to {what}..."),
                                                     point, debug_dir=debug_dir)
    except RuntimeError as error:
        print(f"[SETUP] Automatic move failed: {error}")
        frame = grab_still(camera)

    preview = frame.copy()
    if focus_box is not None:
        x, y, w, h = focus_box
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.drawMarker(preview, tuple(int(v) for v in point), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    cv2.imshow("Focus check", draw_overlay(preview, [(f"Is {what} focused?  Answer in the console", (0, 255, 255), 1.0)]))
    cv2.waitKey(1)
    answer = input(f"Is {what} focused? (Y = yes / k = move with arrow keys): ").strip().lower()
    cv2.destroyWindow("Focus check")
    if answer == 'k':
        keyboard_drive(configuration, camera, pressed)
    print(f"[SETUP] Keys to {what}: {', '.join(pressed) or '(no move)'}")
    return pressed


def choose_channel(configuration, camera, calibration_dir, screen):
    """Click the channel on the captured TV Plus screen; the camera walks the focus there."""
    point = click_channel(screen)
    pressed = move_to_point(configuration, camera, point, "the Live TV channel",
                            os.path.join(calibration_dir, "nav_channel"))
    return pressed, point


def wait_until_playing(camera, message):
    print(f"[SETUP] {message} Press ENTER in the camera window when it is playing.")
    for _ in range(50):
        if cv2.waitKey(1) == -1:
            break
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), [(f"{message}  ENTER when playing", (0, 255, 0), 1.0)]))
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            return frame
        if key == ord('q'):
            raise KeyboardInterrupt("Setup canceled.")


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


def enter_live_channel(configuration, camera, calibration_dir, context):
    """TV Plus is opening: capture it, choose (or reuse) the channel, enter it, and calibrate motion."""
    manufacturer = configuration['manufacturer']
    screen = capture_tvplus_screen(configuration, camera, calibration_dir, context)
    reuse = False
    if configuration['channel_point']:
        preview = screen.copy()
        cv2.drawMarker(preview, tuple(configuration['channel_point']), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.imshow("Saved channel", draw_overlay(preview, [("Saved channel", (0, 255, 255), 1.0)]))
        cv2.waitKey(1)
        print(f"[SETUP] Saved channel route: {', '.join(configuration['channel_route']) or '(no move)'} + OK")
        reuse = input("Use the saved channel? (Y/n): ").strip().lower() in ('', 'y', 'yes')
        cv2.destroyWindow("Saved channel")
    if reuse:
        replay_route(configuration['channel_route'], manufacturer, camera)
        channel_route = configuration['channel_route']
    else:
        channel_route, point = choose_channel(configuration, camera, calibration_dir, screen)
        configuration['channel_route'] = channel_route
        configuration['channel_point'] = [int(point[0]), int(point[1])]
    send_key('OK', manufacturer)

    frame = wait_until_playing(camera, "Entering the Live TV channel.")
    roi = auto_stream_roi(frame.shape)
    playing = measure_playing_motion(camera, roi)
    threshold = max(MIN_MOTION_THRESHOLD, playing * MOTION_THRESHOLD_RATIO)
    print(f"[SETUP] Live video motion {playing:.1f}; playing counts from motion >= {threshold:.1f}")
    if playing < MIN_MOTION_THRESHOLD:
        print("[SETUP] The picture barely moves; pick a channel with moving video for reliable timing.")
    configuration['stream_roi'] = roi
    configuration['motion_threshold'] = float(threshold)
    save_configuration(CONFIG_FILE, configuration)
    cv2.imwrite(os.path.join(calibration_dir, "live_channel.jpg"), frame)
    return roi, threshold


# ---------- measurement: Ch Up zapping ----------

def check_playing(frame_capture, roi, motion_threshold, output_dir, seconds=PRE_ROLL_SECONDS):
    """Before Ch Up, confirm the channel is playing; the frames double as the video pre-roll."""
    frame_capture.clear_buffer()
    motions = []
    previous = None
    last_frame = None
    pre_frames = []
    end_at = time.perf_counter() + seconds
    while time.perf_counter() < end_at or not pre_frames:
        captured = frame_capture.get_frame(timeout=1.0)
        if captured is None:
            if time.perf_counter() > end_at + 2.0:
                raise RuntimeError("No camera frames before Ch Up.")
            continue
        captured_at, frame = captured
        current = roi_gray(frame, roi)
        if previous is not None:
            motions.append(roi_motion(previous, current))
        previous = current
        last_frame = frame
        pre_frames.append((captured_at, frame.copy(), 0.0))

    motion = float(np.median(motions)) if motions else 0.0
    playing = motion >= motion_threshold
    preview = last_frame.copy()
    x, y, w, h = roi
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
    draw_overlay(preview, [
        (f"PRE-CHECK  motion {motion:.1f} / {motion_threshold:.1f}", (0, 255, 255), 1.0),
        ("PLAYING - CH UP" if playing else "NOT PLAYING - RUN INVALID", (0, 255, 0) if playing else (0, 0, 255), 1.2),
    ])
    cv2.imwrite(os.path.join(output_dir, "PRECHECK_BEFORE_CHUP.jpg"), preview)
    cv2.imshow(CAPTURE_WINDOW, preview)
    cv2.waitKey(1)
    return playing, motion, pre_frames, previous


def measure_zap_time(frame_capture, roi, motion_threshold, run_idx, output_dir, timeout,
                     key_pressed_at, pre_frames=(), last_roi=None):
    """Time from Ch Up until the new channel plays: after a blank/frozen picture, sustained motion again."""
    x, y, w, h = roi
    frame_count = 0
    last_elapsed_ms = 0.0
    detected = False
    still_frames = 0            # consecutive frames with no playback (black or frozen)
    streak = 0
    candidate_ms = None
    detected_elapsed_ms = None
    previous_roi = last_roi
    stop_at_ms = None
    result_saved = False
    frame_writer = FrameWriter(os.path.join(output_dir, "frames"))

    def annotate(image, lines):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return draw_overlay(image, lines)

    # Pre-roll: the last PRE_ROLL_SECONDS before Ch Up, saved with negative elapsed time for the video.
    pre_index = 0
    for captured_at, frame, _ in pre_frames:
        elapsed_ms = (captured_at - key_pressed_at) * 1000.0
        if elapsed_ms < -PRE_ROLL_SECONDS * 1000.0 or elapsed_ms >= 0:
            continue
        pre_index += 1
        frame_writer.save(
            f"frame_p{pre_index:04d}_{elapsed_ms:09.1f}ms_sim0.000.jpg",
            annotate(frame, [(f"RUN {run_idx}  BEFORE CH UP  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4)]),
        )

    print(f"[RUN {run_idx}] Waiting for the new channel to play (motion >= {motion_threshold:.1f} for "
          f"{STREAM_CONFIRM_FRAMES} frames after the picture stops)...")
    try:
        while time.perf_counter() - key_pressed_at <= timeout:
            captured = frame_capture.get_frame()
            if captured is None:
                continue
            captured_at, frame = captured
            if captured_at < key_pressed_at:
                continue

            frame_count += 1
            elapsed_ms = (captured_at - key_pressed_at) * 1000.0
            last_elapsed_ms = elapsed_ms
            current_roi = roi_gray(frame, roi)
            motion = roi_motion(previous_roi, current_roi) if previous_roi is not None else 0.0
            brightness = float(np.mean(current_roi))
            previous_roi = current_roi

            playing = motion >= motion_threshold and brightness >= MIN_STREAM_BRIGHTNESS
            if not playing:
                still_frames += 1
                streak = 0
            elif still_frames >= MIN_STILL_FRAMES and not detected:
                # The old channel stopped (black or frozen) and the new one is moving.
                streak += 1
                if streak == 1:
                    candidate_ms = elapsed_ms
                if streak >= STREAM_CONFIRM_FRAMES:
                    detected = True
                    detected_elapsed_ms = candidate_ms
                    stop_at_ms = elapsed_ms + STREAM_EXTRA_SECONDS * 1000.0
                    print(f"[RUN {run_idx}] New channel playing at {candidate_ms:.2f} ms "
                          f"(motion {motion:.1f} >= {motion_threshold:.1f}).")
                    winsound.Beep(1500, 250)
                    winsound.Beep(2500, 350)

            color = (0, 255, 0) if playing else (0, 255, 255)
            saved_lines = [
                (f"RUN {run_idx}  #{frame_count}  TIMER {elapsed_ms / 1000.0:.3f} s", color, 1.4),
                (f"Motion {motion:.1f} / {motion_threshold:.1f}", color, 1.0),
            ]
            if detected:
                saved_lines.append((f"NEW CHANNEL PLAYING at {detected_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
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
                live_lines.append(("NEW CHANNEL PLAYING", (0, 255, 0), 1.2))
            cv2.imshow(CAPTURE_WINDOW, annotate(frame, live_lines))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")
            if detected and not result_saved:
                cv2.imwrite(os.path.join(output_dir, "RESULT_NEW_CHANNEL.jpg"), frame)
                result_saved = True
            if stop_at_ms is not None and elapsed_ms >= stop_at_ms:
                break
    finally:
        frame_writer.close()
        print(f"[RUN {run_idx}] Saved {frame_writer.saved} frames to {frame_writer.frames_dir}")

    if detected:
        last_elapsed_ms = detected_elapsed_ms
    else:
        print(f"[RUN {run_idx}] The new channel did not start playing within {timeout:g} seconds.")
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


def zap_once(configuration, camera, run_idx, output_dir, roi, motion_threshold):
    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        playing, motion, pre_frames, last_roi = check_playing(frame_capture, roi, motion_threshold, output_dir)
        if not playing:
            return invalid_result(run_idx, output_dir,
                                  f"the channel was not playing before Ch Up (motion {motion:.1f}).")
        frame_capture.clear_buffer()
        print(f"[RUN {run_idx}] Ch Up; the timer starts now.")
        key_pressed_at = send_key('ChUp', configuration['manufacturer'])
        winsound.Beep(1500, 300)
        return measure_zap_time(frame_capture, roi, motion_threshold, run_idx, output_dir,
                                configuration['timeout'], key_pressed_at, pre_frames, last_roi)
    finally:
        frame_capture.stop()


def watch(camera, seconds, run_idx):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame = read_frame(camera)
        if frame is not None:
            remaining = end - time.perf_counter()
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [
                (f"RUN {run_idx} next Ch Up in {remaining:.1f} s", (0, 255, 255), 1.0)]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Measurement canceled by user.")


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

        # Input 1: where Samsung TV Plus is on Home.
        calibrate_route(configuration, camera, calibration_dir)      # ends with TV Plus focused
        context = {'home_ready': capture_home_ready(camera), 'nav_dir': os.path.join(calibration_dir, "nav_reopen")}
        send_key('OK', configuration['manufacturer'])
        # Input 2: which Live TV channel to enter inside TV Plus.
        roi, motion_threshold = enter_live_channel(configuration, camera, calibration_dir, context)

        results = []
        for run_idx in range(1, configuration['num_runs'] + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "C:/Temp",
                f"Omedia_TC02_{run_idx:02d}_{timestamp}_{configuration['SoC']}"
                f"_SWV{configuration['SWV']}_LGCV{configuration['LGCV']}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n  [RUN {run_idx}/{configuration['num_runs']}] Ch Up zap\n{'=' * 60}")

            result = zap_once(configuration, camera, run_idx, output_dir, roi, motion_threshold)
            if result['status'] != 'INVALID':
                make_run_videos(output_dir, run_label=f"RUN {run_idx}",
                                end_ms=result['response_time_ms'] if result['status'] == 'PASS' else None)

            results.append(result)
            row = {
                'TC': 'Omedia_TC02',
                'Run': result['run'],
                'Timestamp': timestamp,
                'Zap_Time_ms': (
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
                'Motion_Threshold': round(motion_threshold, 2),
                'Zap_Time_ms': row['Zap_Time_ms'],
                'Frames': row['Frames'],
                'Status': row['Status'],
            })
            if run_idx < configuration['num_runs']:
                watch(camera, configuration['zap_interval_seconds'], run_idx + 1)

        passed = sum(result['status'] == 'PASS' for result in results)
        for result in results:
            if result['status'] == 'INVALID':
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
