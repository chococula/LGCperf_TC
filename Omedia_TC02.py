"""Omedia_TC02 - Samsung TV Plus launch time.

Calibration (once, reused afterwards):
  1. Home is shown; click Samsung TV Plus on the camera image.
  2. The program walks the focus there with D-pad keys, watching the focus highlight on camera,
     and saves the key route plus an image of the focused tile.
  3. TV Plus is launched once to capture the "fully loaded" landing reference.
Each run:
  Power cycle (optional) -> Home -> replay the route -> confirm TV Plus is focused ->
  record 2 s of pre-roll -> OK starts the timer -> stop when the landing similarity reaches the
  threshold. Frames, MP4/GIF, CSV row, and upload are the same as Omedia_TC01.
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
LANDING_REFERENCE_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC02_landing_reference.jpg")
FOCUS_REFERENCE_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC02_focus_reference.jpg")
DEFAULT_CONFIGURATION = {
    'manufacturer': 'Samsung',
    'SoC': 'U8000',
    'SWV': '1300',
    'LGCV': '8.2.008',
    'num_runs': 5,
    'timeout': 30.0,
    'landing_similarity_threshold': 0.7,
    'standby_seconds': 30,
    'route': [],
    'target_point': [],
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
FRAME_CAPTURE_SIMILARITY = 0.85
PRE_ROLL_SECONDS = 2.0
BOOT_WAIT_SECONDS = 20
HOME_SETTLE_SECONDS = 5
KEY_GAP_SECONDS = 1.0
EXIT_WAIT_SECONDS = 3
FOCUS_MATCH_THRESHOLD = 0.8
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
    print(f"  Omedia_TC02 - {APP_NAME} Launch Time")
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
    timeout = ask_number("Enter launch timeout in seconds", defaults['timeout'], float, lambda v: v > 0)
    threshold = ask_number("Enter landing similarity threshold (0-1)",
                           defaults['landing_similarity_threshold'], float, lambda v: 0 < v <= 1)
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
        'landing_similarity_threshold': threshold,
        'standby_seconds': standby_seconds,
    })
    print("\nConfiguration:")
    for key in ('manufacturer', 'SoC', 'SWV', 'LGCV', 'num_runs', 'timeout',
                'landing_similarity_threshold', 'standby_seconds'):
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


def grab_still(camera, label=None):
    """Settled median frame, also shown in the capture window."""
    frame = nav.settle_and_grab(lambda: read_frame(camera))
    preview = frame.copy()
    if label:
        draw_overlay(preview, [(label, (255, 255, 255), 1.0)])
    cv2.imshow(CAPTURE_WINDOW, preview)
    cv2.waitKey(1)
    return frame


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
        raise RuntimeError("Camera frame is smaller than the saved landing reference.")
    return float(np.max(cv2.matchTemplate(blurred, reference, cv2.TM_CCOEFF_NORMED)))


# ---------- calibration: route to Samsung TV Plus ----------

def click_target(camera, saved_point=None):
    """Show the Home screen and ask where to go. Returns the point in frame pixels.

    A previously chosen spot is pre-marked; Enter keeps it, a click picks a new one.
    """
    window_name = f"Click {APP_NAME}"
    still = grab_still(camera)
    selection = {'point': tuple(saved_point) if saved_point else None}

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection['point'] = (x, y)

    print(f"\n[SETUP] Home screen shown. Click where to go ({APP_NAME}), then press Enter "
          "(R = refresh image, Q = cancel).")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        preview = still.copy()
        if selection['point'] is None:
            message = f"Click {APP_NAME}, then press ENTER"
        elif saved_point and tuple(selection['point']) == tuple(saved_point):
            message = "ENTER = go to the marked spot,  or click another"
        else:
            message = "ENTER = go there,  click again to change"
        draw_overlay(preview, [(message, (0, 255, 0), 1.0)])
        if selection['point'] is not None:
            cv2.drawMarker(preview, selection['point'], (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(preview, selection['point'], 16, (0, 255, 0), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and selection['point'] is not None:
            cv2.destroyWindow(window_name)
            return selection['point']
        if key == ord('r'):
            still = grab_still(camera)
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Target selection canceled.")


def show_and_confirm(frame, title, lines, box=None, point=None):
    preview = frame.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
    if point is not None:
        cv2.drawMarker(preview, tuple(int(v) for v in point), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    draw_overlay(preview, lines)
    cv2.imshow(title, preview)
    cv2.waitKey(1)


def parse_route(text):
    aliases = {'u': 'Up', 'up': 'Up', 'd': 'Down', 'dn': 'Down', 'down': 'Down',
               'l': 'Left', 'left': 'Left', 'r': 'Right', 'rt': 'Right', 'right': 'Right'}
    route = []
    for part in text.replace(' ', ',').split(','):
        part = part.strip().lower()
        if not part:
            continue
        if part not in aliases:
            raise ValueError(f"Unknown key '{part}'. Use Up, Down, Left, Right.")
        route.append(aliases[part])
    return route


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


def auto_navigate(configuration, camera, target_point):
    manufacturer = configuration['manufacturer']

    def press(key):
        send_key(key, manufacturer)

    return nav.navigate_to_target(press, lambda: grab_still(camera, f"Finding {APP_NAME}..."), target_point)


def calibrate_route(configuration, camera, calibration_dir):
    """Show Home, ask the user where to go, and move there with the D-pad.

    If the user keeps the previously chosen spot, the saved route is replayed and checked on camera;
    otherwise (or if it misses) the route is found again by watching the focus move.
    """
    manufacturer = configuration['manufacturer']
    saved_focus = cv2.imread(FOCUS_REFERENCE_FILE) if os.path.isfile(FOCUS_REFERENCE_FILE) else None
    saved_route = configuration['route'] if saved_focus is not None else []
    saved_point = tuple(configuration['target_point']) if saved_route and configuration['target_point'] else None

    while True:
        press_home(manufacturer, camera)
        target_point = click_target(camera, saved_point)
        if saved_point and tuple(target_point) == saved_point:
            print(f"[SETUP] Going to the marked spot with the saved route: {', '.join(saved_route)}")
            replay_route(saved_route, manufacturer, camera)
            frame = grab_still(camera)
            score = nav.region_similarity(saved_focus, frame, target_point)
            print(f"[SETUP] Focus match {score:.2f} (needs {FOCUS_MATCH_THRESHOLD:.2f})")
            if score >= FOCUS_MATCH_THRESHOLD:
                return saved_route, target_point, saved_focus
            print("[SETUP] The saved route missed; finding the spot again with the camera.")
            press_home(manufacturer, camera)
        route, focus_frame, focus_box = [], None, None
        try:
            route, focus_frame, focus_box = auto_navigate(configuration, camera, target_point)
        except RuntimeError as error:
            print(f"[SETUP] Automatic navigation failed: {error}")
            focus_frame = grab_still(camera)

        lines = [(f"Route: {', '.join(route) or '(none)'}", (0, 255, 255), 1.0),
                 (f"Is {APP_NAME} focused?  Answer in the console.", (255, 255, 255), 0.9)]
        show_and_confirm(focus_frame, "Route Check", lines, focus_box, target_point)
        answer = input(f"Is {APP_NAME} focused now? (Y = save / n = try again / m = type the route): ").strip().lower()
        cv2.destroyWindow("Route Check")
        if answer == 'm':
            while True:
                try:
                    route = parse_route(input("Route from Home, e.g. Down,Down,Right: "))
                    break
                except ValueError as error:
                    print(error)
            press_home(manufacturer, camera)
            replay_route(route, manufacturer, camera)
            focus_frame = grab_still(camera)
            show_and_confirm(focus_frame, "Route Check", [(f"Route: {', '.join(route)}", (0, 255, 255), 1.0)],
                             point=target_point)
            answer = input(f"Is {APP_NAME} focused now? (Y/n): ").strip().lower()
            cv2.destroyWindow("Route Check")
        if answer in ('', 'y', 'yes') and focus_frame is not None:
            break
        print("[SETUP] Let's try again from the Home screen.")

    # The calibration route was found from wherever the focus started; check that replaying it
    # from a fresh Home press lands on the target too.
    press_home(manufacturer, camera)
    replay_route(route, manufacturer, camera)
    replay_frame = grab_still(camera)
    score = nav.region_similarity(focus_frame, replay_frame, target_point)
    print(f"[SETUP] Replaying the route from Home: focus match {score:.2f} (needs {FOCUS_MATCH_THRESHOLD:.2f})")
    if score < FOCUS_MATCH_THRESHOLD:
        print("[SETUP] The route does not land on the same tile after Home; each run will re-find "
              f"{APP_NAME} with the camera if the replayed route misses.")

    for path in (FOCUS_REFERENCE_FILE, os.path.join(calibration_dir, "focus_reference.jpg")):
        cv2.imwrite(path, focus_frame)
    configuration['route'] = route
    configuration['target_point'] = [int(target_point[0]), int(target_point[1])]
    save_configuration(CONFIG_FILE, configuration)
    print(f"[SETUP] Saved route to {APP_NAME}: {', '.join(route)}")
    return route, target_point, focus_frame


def focus_on_target(configuration, camera, route, target_point, focus_reference, run_idx):
    """Home -> route; confirm the focus with the camera, re-finding the target once if needed."""
    manufacturer = configuration['manufacturer']
    press_home(manufacturer, camera)
    replay_route(route, manufacturer, camera)
    frame = grab_still(camera, f"RUN {run_idx}: checking focus on {APP_NAME}")
    score = nav.region_similarity(focus_reference, frame, target_point)
    print(f"[RUN {run_idx}] Focus match on {APP_NAME}: {score:.2f}")
    if score >= FOCUS_MATCH_THRESHOLD:
        return True, frame
    print(f"[RUN {run_idx}] Route missed {APP_NAME}; finding it with the camera...")
    try:
        press_home(manufacturer, camera)
        _, frame, _ = auto_navigate(configuration, camera, target_point)
    except RuntimeError as error:
        print(f"[RUN {run_idx}] {error}")
        return False, grab_still(camera)
    score = nav.region_similarity(focus_reference, frame, target_point)
    print(f"[RUN {run_idx}] Focus match after re-finding: {score:.2f}")
    return score >= FOCUS_MATCH_THRESHOLD, frame


# ---------- calibration: landing reference ----------

def capture_landing_reference(camera, manufacturer, output_dir):
    reference_path = os.path.join(output_dir, "landing_reference.jpg")
    while True:
        print(f"\n[SETUP] Launching {APP_NAME} once (not measured) to capture its loaded screen.")
        send_key('OK', manufacturer)
        print(f"Wait until {APP_NAME} has fully loaded (video/guide showing).")
        input("Then press Enter to capture it: ")

        samples = []
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < 2.0:
            ret, frame = camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(cv2.GaussianBlur(gray, (5, 5), 0))
            time.sleep(0.05)
        if not samples:
            raise RuntimeError("Could not capture the landing reference.")

        reference = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
        cv2.imshow("Landing Reference Preview", reference)
        cv2.waitKey(1)
        approval = input(f"Use this as the {APP_NAME} loaded reference? (Y/n): ").strip().lower()
        cv2.destroyWindow("Landing Reference Preview")
        if approval in ('', 'y', 'yes'):
            for image_path in (reference_path, LANDING_REFERENCE_FILE):
                if not cv2.imwrite(image_path, reference):
                    raise RuntimeError(f"Could not save landing reference: {image_path}")
            print(f"Landing reference saved: {LANDING_REFERENCE_FILE}")
            return reference
        print("Reference rejected. Going back to Home to try again.")
        send_key('Home', manufacturer)
        time.sleep(HOME_SETTLE_SECONDS)
        input(f"Make sure {APP_NAME} is focused on the Home screen, then press Enter: ")


def review_saved_landing_reference():
    if not os.path.isfile(LANDING_REFERENCE_FILE):
        print(f"No saved {APP_NAME} landing reference. One will be captured now.")
        return None
    reference = cv2.imread(LANDING_REFERENCE_FILE, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        print("Saved landing reference could not be read; a new one will be captured.")
        return None
    cv2.imshow("Saved Landing Reference", reference)
    cv2.waitKey(1)
    reuse = input(f"Reuse this saved {APP_NAME} landing screen? (Y/n): ").strip().lower()
    cv2.destroyWindow("Saved Landing Reference")
    if reuse in ('', 'y', 'yes'):
        print(f"Reusing saved landing reference: {LANDING_REFERENCE_FILE}")
        return reference
    print("Saved reference rejected. A new landing screen will be captured.")
    return None


# ---------- measurement ----------

def check_not_launched(frame_capture, reference, similarity_threshold, output_dir, seconds=PRE_ROLL_SECONDS):
    """Before OK, confirm TV Plus is not already showing; the frames double as the video pre-roll."""
    frame_capture.clear_buffer()
    max_score = None
    last_frame = None
    pre_frames = []
    end_at = time.perf_counter() + seconds
    while time.perf_counter() < end_at or max_score is None:
        captured = frame_capture.get_frame(timeout=1.0)
        if captured is None:
            if time.perf_counter() > end_at + 2.0:
                raise RuntimeError("No camera frames during the pre-launch check.")
            continue
        captured_at, frame = captured
        score = screen_similarity(frame, reference)
        max_score = score if max_score is None else max(max_score, score)
        last_frame = frame
        pre_frames.append((captured_at, frame.copy(), score))

    ready = max_score < similarity_threshold
    last_frame = last_frame.copy()
    draw_overlay(last_frame, [
        (f"PRE-CHECK  similarity {max_score:.3f} / {similarity_threshold:.3f}", (0, 255, 255), 1.0),
        ("READY - OK" if ready else f"{APP_NAME} ALREADY SHOWING - RUN INVALID",
         (0, 255, 0) if ready else (0, 0, 255), 1.2),
    ])
    cv2.imwrite(os.path.join(output_dir, "PRECHECK_BEFORE_OK.jpg"), last_frame)
    cv2.imshow(CAPTURE_WINDOW, last_frame)
    cv2.waitKey(1)
    return ready, max_score, pre_frames


def measure_launch_time(frame_capture, reference, run_idx, output_dir, timeout,
                        similarity_threshold, ok_pressed_at, pre_frames=()):
    frame_count = 0
    last_elapsed_ms = 0.0
    detected = False
    armed = False
    detected_elapsed_ms = None
    frame_writer = FrameWriter(os.path.join(output_dir, "frames"))

    # Pre-roll: the last PRE_ROLL_SECONDS before OK, saved with negative elapsed time for the video.
    pre_index = 0
    for captured_at, frame, score in pre_frames:
        elapsed_ms = (captured_at - ok_pressed_at) * 1000.0
        if elapsed_ms < -PRE_ROLL_SECONDS * 1000.0 or elapsed_ms >= 0:
            continue
        pre_index += 1
        frame_writer.save(
            f"frame_p{pre_index:04d}_{elapsed_ms:09.1f}ms_sim{score:.3f}.jpg",
            draw_overlay(frame, [
                (f"RUN {run_idx}  BEFORE OK  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4),
                (f"Similarity {score:.3f} / {similarity_threshold:.3f}", (0, 255, 255), 1.0),
            ]),
        )

    print(f"[RUN {run_idx}] Waiting for {APP_NAME} to load (saving every frame until similarity "
          f"{FRAME_CAPTURE_SIMILARITY:.2f})...")
    try:
        while time.perf_counter() - ok_pressed_at <= timeout:
            captured = frame_capture.get_frame()
            if captured is None:
                continue

            captured_at, frame = captured
            if captured_at < ok_pressed_at:
                continue

            frame_count += 1
            score = screen_similarity(frame, reference)
            elapsed_ms = (captured_at - ok_pressed_at) * 1000.0
            last_elapsed_ms = elapsed_ms
            # Only count a match after a frame below the threshold, so a stale frame cannot pass.
            if score < similarity_threshold:
                armed = True
            newly_detected = armed and not detected and score >= similarity_threshold
            if newly_detected:
                detected = True
                detected_elapsed_ms = elapsed_ms

            saved_color = (0, 255, 0) if score >= similarity_threshold else (0, 255, 255)
            saved_overlay = [
                (f"RUN {run_idx}  #{frame_count}  TIMER {elapsed_ms / 1000.0:.3f} s", saved_color, 1.4),
                (f"Similarity {score:.3f} / {similarity_threshold:.3f}", saved_color, 1.0),
            ]
            if detected:
                saved_overlay.append((f"LOADED at {detected_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            frame_writer.save(
                f"frame_{frame_count:05d}_{elapsed_ms:09.1f}ms_sim{score:.3f}.jpg",
                draw_overlay(frame.copy(), saved_overlay),
            )

            shown_ms = detected_elapsed_ms if detected else elapsed_ms
            status_color = (0, 255, 0) if detected else (0, 255, 255)
            overlay = [
                (f"RUN {run_idx}  TIMER {shown_ms / 1000.0:.3f} s", status_color, 1.4),
                (f"Similarity {score:.3f} / {similarity_threshold:.3f}", status_color, 1.0),
            ]
            if detected:
                overlay.append((f"{APP_NAME.upper()} LOADED", (0, 255, 0), 1.2))
                overlay.append((f"Saving frames until {FRAME_CAPTURE_SIMILARITY:.2f}...", (255, 255, 255), 0.8))
            draw_overlay(frame, overlay)
            cv2.imshow(CAPTURE_WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")

            if newly_detected:
                cv2.imwrite(os.path.join(output_dir, "RESULT_LOADED.jpg"), frame)
                print(f"[RUN {run_idx}] {APP_NAME} loaded in {elapsed_ms:.2f} ms "
                      f"(match {score:.3f}, threshold {similarity_threshold:.3f}).")
                winsound.Beep(1500, 250)
                winsound.Beep(2500, 350)

            if detected and score >= FRAME_CAPTURE_SIMILARITY:
                print(f"[RUN {run_idx}] Similarity {score:.3f} reached {FRAME_CAPTURE_SIMILARITY:.2f}; "
                      f"frame capture stopped at {elapsed_ms:.2f} ms.")
                break
    finally:
        frame_writer.close()
        print(f"[RUN {run_idx}] Saved {frame_writer.saved} frames to {frame_writer.frames_dir}")

    if detected:
        cv2.waitKey(2000)
        last_elapsed_ms = detected_elapsed_ms
    else:
        print(f"[RUN {run_idx}] {APP_NAME} did not load within {timeout:g} seconds.")
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


def run_once(configuration, camera, run_idx, output_dir, route, target_point, focus_reference, landing_reference):
    manufacturer = configuration['manufacturer']
    threshold = configuration['landing_similarity_threshold']

    if configuration['standby_seconds'] > 0:
        print(f"\n[RUN {run_idx}] STEP 1: Power OFF, standby, Power ON.")
        send_key('Power', manufacturer)
        winsound.Beep(1000, 200)
        wait_with_countdown(camera, configuration['standby_seconds'], run_idx, "Standby", "Power ON")
        send_key('Power', manufacturer)
        wait_with_countdown(camera, BOOT_WAIT_SECONDS, run_idx, "Boot wait", "Home")

    print(f"[RUN {run_idx}] STEP 2: Home -> {', '.join(route)} to focus {APP_NAME}.")
    focused, focus_frame = focus_on_target(configuration, camera, route, target_point, focus_reference, run_idx)
    cv2.imwrite(os.path.join(output_dir, "FOCUS_BEFORE_OK.jpg"), focus_frame)
    if not focused:
        return invalid_result(run_idx, output_dir, f"{APP_NAME} could not be focused.")

    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        ready, precheck_score, pre_frames = check_not_launched(frame_capture, landing_reference, threshold, output_dir)
        if not ready:
            return invalid_result(run_idx, output_dir,
                                  f"{APP_NAME} was already showing (similarity {precheck_score:.3f}) before OK.")
        frame_capture.clear_buffer()
        print(f"[RUN {run_idx}] STEP 3: OK launches {APP_NAME}; the timer starts now.")
        ok_pressed_at = send_key('OK', manufacturer)
        winsound.Beep(1500, 300)
        return measure_launch_time(frame_capture, landing_reference, run_idx, output_dir,
                                   configuration['timeout'], threshold, ok_pressed_at, pre_frames)
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

        route, target_point, focus_reference = calibrate_route(configuration, camera, calibration_dir)
        landing_reference = review_saved_landing_reference()
        if landing_reference is None:
            press_home(configuration['manufacturer'], camera)
            replay_route(route, configuration['manufacturer'], camera)
            landing_reference = capture_landing_reference(camera, configuration['manufacturer'], calibration_dir)
        else:
            cv2.imwrite(os.path.join(calibration_dir, "landing_reference.jpg"), landing_reference)

        results = []
        threshold = configuration['landing_similarity_threshold']
        for run_idx in range(1, configuration['num_runs'] + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "C:/Temp",
                f"Omedia_TC02_{run_idx:02d}_{timestamp}_{configuration['SoC']}"
                f"_SWV{configuration['SWV']}_LGCV{configuration['LGCV']}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n  [RUN {run_idx}/{configuration['num_runs']}] {APP_NAME} launch\n{'=' * 60}")

            result = run_once(configuration, camera, run_idx, output_dir, route, target_point,
                              focus_reference, landing_reference)

            if result['status'] != 'INVALID':
                make_run_videos(output_dir, threshold=threshold, run_label=f"RUN {run_idx}")
            # Leave the app so the next run starts from Home.
            send_key('Home', configuration['manufacturer'])
            time.sleep(EXIT_WAIT_SECONDS)

            results.append(result)
            row = {
                'TC': 'Omedia_TC02',
                'Run': result['run'],
                'Timestamp': timestamp,
                'Standby_Seconds': configuration['standby_seconds'],
                'Launch_Time_ms': (
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
                'Similarity_Threshold': threshold,
                'Route': ' '.join(route),
                'Launch_Time_ms': row['Launch_Time_ms'],
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
