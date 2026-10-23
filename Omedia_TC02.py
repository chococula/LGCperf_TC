"""Omedia_TC02 - Samsung TV Plus Live TV channel change time.

Setup (three inputs, each a click on the live camera view, then Enter):
  1. Home: click the Samsung TV Plus card (Down) or its bottom app icon (Down, Down), as in PACE_01; OK.
  2. TV Plus: click the highlighted item, then the Live TV channel. The planned keys (e.g. Down x1,
     Right x2) are drawn on screen first; arrow keys adjust them; Enter sends exactly those keys + OK.
  3. Live TV: the same as step 2 for one more item, then OK.
  Detection (asked each session): EmptyArea = an area picked on screen that is empty while the next
  channel loads; Motion = the middle of the picture moves again after going black/frozen.
Each run:
  Network method (default): Live TV plays, CH+/CH- is sent over Wi-Fi (Samsung remote API; IR channel
  keys are ignored in TV Plus), 2 s pre-roll, the key starts the timer -> stop when the picture lands.
  Guide method: OK, OK opens the GUIDE, Down moves to the
  next channel, 2 s pre-roll, then OK starts the timer -> stop when the new channel's picture appears.
  Ch Up method: 2 s pre-roll while the channel plays -> Ch Up starts the timer -> stop at the first of
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

from config_store import choose_option, load_saved_configuration, save_configuration
from ir_keys import HISENSEVIDAA_KEYS, LG_KEYS, ROKU_KEYS, SAMSUNG_KEYS
from result_uploader import upload_result, wait_for_uploads
from run_video import make_run_videos
from samsung_remote import CHANNEL_KEYS, NetworkRemote


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC02_config.json")
DEFAULT_CONFIGURATION = {
    'manufacturer': 'Samsung',
    'SoC': 'U8000',
    'SWV': '1300',
    'LGCV': '8.2.008',
    'num_runs': 5,
    'timeout': 15.0,
    'zap_interval_seconds': 8,
    'zap_method': 'Network',
    'tv_ip': '192.168.4.140',
    'zap_direction': 'Alternate',
    'detector': 'EmptyArea',
    'route': [],
    'target_point': [],
    'plan_steps': {},
    'navigation': {},            # per SoC: the setup that worked (route, plan_steps, stream_roi, motion_threshold)
    'stream_roi': [],
    'motion_threshold': 0.0,
    'screen_rois': {},           # per SoC: [x, y, w, h] of the TV screen; black is judged on the whole screen
}

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
FLIRC_IDLE_WAKE_SECONDS = 20    # the Flirc sleeps when idle: after this long without traffic, wake it first
LAST_FLIRC_AT = 0.0
RAW_IR_KEYSETS = {
    'LG': LG_KEYS,
    'Roku': ROKU_KEYS,
    'Samsung': SAMSUNG_KEYS,
    'HisenseVidaa': HISENSEVIDAA_KEYS,
}
APP_NAME = "Samsung TV Plus"
CAMERA_INDEX = 1
CAPTURE_WINDOW = "Omedia_TC02 Capture"
MIN_MOTION_THRESHOLD = 2.0      # mean gray-level change per frame inside the live video area
MOTION_THRESHOLD_RATIO = 0.4    # streaming = motion of at least 40% of the measured playing motion
MIN_STREAM_BRIGHTNESS = 20      # a black screen is not streaming
PLAYBACK_WINDOW_FRAMES = 15     # playback = in this many frames (~0.5 s) ...
PLAYBACK_MIN_FRACTION = 0.7     # ... at least this share move; the start is the window's first frame
STREAM_EXTRA_SECONDS = 1.0      # keep saving frames this long after the start is confirmed
MIN_STILL_FRAMES = 2            # after the key the picture must stop (black/frozen) this many frames first
METHODS = ('Network', 'Guide', 'ChUp')
DETECTORS = {
    'EmptyArea': "pick an area that is empty while the next channel loads",
    'Motion': "the picture moves again after going black/frozen",
}
SAVED_DETECTOR = {'EmptyArea': 'landing_area', 'Motion': 'motion'}   # name kept in saved navigation
GUIDE_KEY_GAP_SECONDS = 1.0     # Guide method: wait between OK, OK and Down
GUIDE_OPEN_SECONDS = 5.0        # Guide method: stay on the GUIDE this long after the second OK
ARROW_KEYS = {2424832: 'Left', 2490368: 'Up', 2555904: 'Right', 2621440: 'Down'}   # cv2.waitKeyEx codes
TILE_STEP_RATIO = 1.15         # fallback step (tile size + gap) when the next tile cannot be measured
PRE_ROLL_SECONDS = 2.0
STEP_FRAMES = 6                 # landing area: change is measured against the frame this many frames earlier
ARM_STEP = 10                   # change of the area when the GUIDE closes
LANDING_STEP = 15               # change of the area when the picture lands (loading drift stays below this)
QUIET_FRAMES = 9                # ~0.3 s without change after the GUIDE closed = loading screen
PERSIST_SECONDS = 0.5           # the landed picture must stay different from the loading screen this long
NO_GUIDE_CHANGE_MS = 3000       # if the area never changed when the GUIDE closed, start waiting anyway
LOADING_DARK = 60               # area brightness below this = the old picture went black (loading screen)
# Black screen of a CH+/CH- change, judged on the whole TV screen: a sudden drop below BLACK_SCREEN, to under
# BLACK_RATIO of and BLACK_DROP below the brightest frame of the last BLACK_WINDOW_MS (LG: black 2-53,
# playing pictures 55-190). Checked on the LG NU7000 session of 2026-10-22 16:25: it is the first dark frame
# before every correct landing, and comes after every early (wrong) one.
BLACK_SCREEN = 60
BLACK_RATIO = 0.7
BLACK_DROP = 15
BLACK_WINDOW_MS = 200
LANDING_PATTERN = 20            # the landing must also change the area's pattern this much (mean removed):
                                # LG 2026-10-22 real landings 24-81; loading screen brightening / Pluto logo 9-16
MIN_REACTION_MS = 300          # the TV does not react faster; darkness before this is the old channel
BLACK_OLD_DARK = 40             # old channel darker than this: black cannot be told apart, not required
ONSET_STEP = 8                  # the landing time is the first frame that moved this far from the loading screen
STEP_LOAD_SECONDS = 6           # saved navigation: wait after each OK for the next screen to load
HOME_SETTLE_SECONDS = 3
KEY_GAP_SECONDS = 0.5
CSV_PATH = os.path.join(SCRIPT_DIR, "Omedia_TC02_response.csv")
NETWORK_REMOTE = None            # NetworkRemote, connected in main() for the Network method


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
    print(f"  Omedia_TC02 - {APP_NAME} Live TV Channel Change")
    print("=" * 60)

    manufacturer = choose_option("IR key set (TV maker)", RAW_IR_KEYSETS, defaults['manufacturer'])
    SoC = input(f"Enter SoC Model (default: {defaults['SoC']}): ").strip() or defaults['SoC']
    SWV = input(f"Enter Software Version (default: {defaults['SWV']}): ").strip() or defaults['SWV']
    LGCV = input(f"Enter LG CV Version (default: {defaults['LGCV']}): ").strip() or defaults['LGCV']
    num_runs = ask_number("Enter number of runs", defaults['num_runs'], int, lambda v: v > 0)
    timeout = ask_number("Enter zap timeout in seconds", defaults['timeout'], float, lambda v: v > 0)
    zap_interval_seconds = ask_number("Enter seconds to watch each channel before the next change",
                                      defaults['zap_interval_seconds'], int, lambda v: v >= 0)
    default_method = defaults['zap_method'] if defaults['zap_method'] in METHODS else 'Network'
    if manufacturer != 'Samsung' and default_method == 'Network':
        default_method = 'ChUp'          # the network remote API is Samsung only
    zap_method = choose_option("Channel change method", {
        'Network': "Samsung only, CH+/- over Wi-Fi",
        'Guide': "OK, OK, Down, OK",
        'ChUp': "IR CH+/-",
    }, default_method)
    tv_ip, zap_direction = defaults['tv_ip'], defaults['zap_direction']
    if zap_method == 'Network':
        tv_ip = input(f"TV IP address (default: {defaults['tv_ip']}): ").strip() or defaults['tv_ip']
    if zap_method in ('Network', 'ChUp'):
        zap_direction = choose_option("Channel direction", ('Up', 'Down', 'Alternate'), defaults['zap_direction'])
    detector = choose_option("Channel change detection", DETECTORS, defaults['detector'])

    configuration = dict(defaults)
    configuration.update({
        'manufacturer': manufacturer,
        'SoC': SoC,
        'SWV': SWV,
        'LGCV': LGCV,
        'num_runs': num_runs,
        'timeout': timeout,
        'zap_interval_seconds': zap_interval_seconds,
        'zap_method': zap_method,
        'tv_ip': tv_ip,
        'zap_direction': zap_direction,
        'detector': detector,
    })
    print("\nConfiguration:")
    for key in ('manufacturer', 'SoC', 'SWV', 'LGCV', 'num_runs', 'timeout', 'zap_interval_seconds', 'zap_method',
                'tv_ip', 'zap_direction', 'detector'):
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


def wake_flirc():
    """Talk to the Flirc without sending IR (`flirc_util version`, ~2.5 s) so it is awake for the next send."""
    global LAST_FLIRC_AT
    try:
        subprocess.run([FLIRC_UTIL_PATH, 'version'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[FLIRC] Wake-up call failed: {error}")
    LAST_FLIRC_AT = time.perf_counter()


def wake_flirc_if_idle():
    if time.perf_counter() - LAST_FLIRC_AT >= FLIRC_IDLE_WAKE_SECONDS:
        print("[FLIRC] Idle for a while; waking it up first (no IR).")
        wake_flirc()


def send_key(key_name, manufacturer):
    global LAST_FLIRC_AT
    if manufacturer not in RAW_IR_KEYSETS:
        raise ValueError(f"Unsupported IR key set: {manufacturer}")
    keyset = RAW_IR_KEYSETS[manufacturer]
    if key_name not in keyset:
        raise ValueError(f"{manufacturer} IR key is not configured: {key_name}")

    wake_flirc_if_idle()
    print(f"Sending '{key_name}' via Flirc ({manufacturer} IR)...")
    action_started_at = time.perf_counter()
    subprocess.run(
        [FLIRC_UTIL_PATH, 'sendir', f"--csv={keyset[key_name]}", '--repeat=1'],
        check=True,
        capture_output=True,
        text=True,
    )
    LAST_FLIRC_AT = time.perf_counter()
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


# ---------- input 1: Samsung TV Plus on Home (same as PACE_01) ----------

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


# ---------- input 2: the Live TV channel inside TV Plus ----------

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


def flood_box(frame, point):
    """Box of the uniform-colour area under `point` if it looks like a tile, else None."""
    h, w = frame.shape[:2]
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < w and 0 <= y < h):
        return None
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(blurred, mask, (x, y), 0, (14, 14, 14), (14, 14, 14), flags)
    ys, xs = np.nonzero(mask[1:-1, 1:-1])
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if 0.03 * w <= bw <= 0.5 * w and 0.03 * h <= bh <= 0.5 * h:
        return int(x0), int(y0), int(bw), int(bh)
    return None


def tile_box_at(frame, point):
    """Box of the tile under `point`; a typical tile size centred on the click if it cannot be isolated."""
    box = flood_box(frame, point)
    if box is not None:
        return box
    h, w = frame.shape[:2]
    bw, bh = int(0.15 * w), int(0.18 * h)
    return int(point[0] - bw / 2), int(point[1] - bh / 2), bw, bh


def measure_step(frame, tile, axis, sign):
    """Distance from the focused tile to the next tile of similar size along one axis (tile + gap)."""
    x, y, w, h = tile
    size = w if axis == 0 else h
    centre = (x + w / 2.0, y + h / 2.0)
    edge = (x + w if sign > 0 else x) if axis == 0 else (y + h if sign > 0 else y)
    for offset in range(3, int(size * 1.5), 3):
        probe = [centre[0], centre[1]]
        probe[axis] = edge + sign * offset
        box = flood_box(frame, probe)
        if box is None:
            continue
        other = box[2] if axis == 0 else box[3]
        if 0.6 * size <= other <= 1.6 * size:
            start = box[0] if axis == 0 else box[1]
            own = x if axis == 0 else y
            if sign < 0:
                start, own = own, start
            step = start - own
            if step > 0.8 * size:
                return float(step)
    return size * TILE_STEP_RATIO


def plan_moves(frame, focus_point, target_point, tile_box):
    """Rows and columns from the focused item to the target, stepping by the measured tile pitch."""
    dx = target_point[0] - focus_point[0]
    dy = target_point[1] - focus_point[1]
    step_x = measure_step(frame, tile_box, 0, 1 if dx >= 0 else -1)
    step_y = measure_step(frame, tile_box, 1, 1 if dy >= 0 else -1)
    rows = int(round(dy / step_y))
    cols = int(round(dx / step_x))
    return rows, cols, (step_x, step_y)


def route_from_plan(rows, cols):
    """Vertical moves first, then horizontal (the same order the plan is drawn)."""
    return (['Down'] * rows if rows > 0 else ['Up'] * -rows) + (['Right'] * cols if cols > 0 else ['Left'] * -cols)


def draw_plan(image, focus_point, rows, cols, steps):
    """Dots on every tile the focus will pass, arrows between them, a ring on the end tile."""
    step_x, step_y = steps
    points = [tuple(int(v) for v in focus_point)]
    x, y = focus_point
    for _ in range(abs(rows)):
        y += step_y if rows > 0 else -step_y
        points.append((int(x), int(y)))
    for _ in range(abs(cols)):
        x += step_x if cols > 0 else -step_x
        points.append((int(x), int(y)))
    for start, end in zip(points, points[1:]):
        cv2.arrowedLine(image, start, end, (0, 255, 255), 3, tipLength=0.25)
    for point in points[1:-1]:
        cv2.circle(image, point, 6, (0, 255, 255), -1)
    cv2.circle(image, points[0], 10, (255, 255, 255), 3)
    cv2.circle(image, points[-1], 18, (0, 255, 0), 4)
    return image


def select_plan(camera, step_no, what, saved=None):
    """Click the highlighted item, then the target; the planned keys are drawn first.

    Arrow keys adjust the plan, R starts over, Enter sends exactly the planned keys.
    Returns (route, focus_point, target_point).
    """
    window_name = f"Step {step_no}: select {what}"
    clicks = list(saved or [])
    plan = {'rows': 0, 'cols': 0, 'steps': None, 'tile': None}

    def replan(frame):
        plan['tile'] = tile_box_at(frame, clicks[0])
        plan['rows'], plan['cols'], plan['steps'] = plan_moves(frame, clicks[0], clicks[1], plan['tile'])

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) >= 2:
                clicks.clear()
                plan['steps'] = None
            clicks.append((x, y))

    print(f"\n[SETUP] Step {step_no}: when the screen has loaded, click (1) the item that is highlighted now, "
          f"then (2) {what}. Check the planned moves on screen; arrow keys adjust, R = start over, "
          "Enter = go (Q = cancel).")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        preview = frame.copy()
        if len(clicks) == 0:
            lines = [("1) Click the item that is HIGHLIGHTED now", (0, 255, 0), 1.0)]
        elif len(clicks) == 1:
            cv2.circle(preview, clicks[0], 10, (255, 255, 255), 3)
            lines = [(f"2) Click {what}", (0, 255, 0), 1.0)]
        else:
            if plan['steps'] is None:
                replan(frame)
            x, y, w, h = plan['tile']
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 255), 1)
            draw_plan(preview, clicks[0], plan['rows'], plan['cols'], plan['steps'])
            route = route_from_plan(plan['rows'], plan['cols'])
            lines = [(f"Planned: {describe_route(route)} + OK", (0, 255, 255), 1.0),
                     ("Arrow keys adjust,  R = start over,  S = skip step,  ENTER = go", (255, 255, 255), 0.8)]
        cv2.imshow(window_name, draw_overlay(preview, lines))
        code = cv2.waitKeyEx(30)
        if code in ARROW_KEYS and len(clicks) == 2:
            key = ARROW_KEYS[code]
            plan['rows'] += {'Down': 1, 'Up': -1}.get(key, 0)
            plan['cols'] += {'Right': 1, 'Left': -1}.get(key, 0)
        elif code in (10, 13) and len(clicks) == 2:
            cv2.destroyWindow(window_name)
            route = route_from_plan(plan['rows'], plan['cols'])
            print(f"[SETUP] Step {step_no}: planned keys to {what}: {describe_route(route)} + OK")
            return route, clicks[0], clicks[1]
        elif code in (ord('r'), ord('R')):
            clicks.clear()
            plan['steps'] = None
        elif code in (ord('s'), ord('S')):
            cv2.destroyWindow(window_name)
            print(f"[SETUP] Step {step_no} skipped.")
            return None
        elif code in (ord('q'), ord('Q')):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt(f"Step {step_no} selection canceled.")


def describe_route(route):
    """['Down', 'Right', 'Right'] -> 'Down x1, Right x2'."""
    parts = []
    for key in route:
        if parts and parts[-1][0] == key:
            parts[-1][1] += 1
        else:
            parts.append([key, 1])
    return ', '.join(f"{key} x{count}" for key, count in parts) or '(no move)'


def plan_step(configuration, camera, step_no, what):
    """One planned step: plan on screen, then send exactly the planned keys and OK.

    No camera decisions while moving. The clicks and keys are saved per step (plan_steps[step_no]).
    """
    saved_steps = configuration['plan_steps']
    saved = saved_steps.get(str(step_no), {}).get('clicks') or None
    chosen = select_plan(camera, step_no, what, [tuple(p) for p in saved] if saved else None)
    if chosen is None:
        return None
    route, focus_point, target_point = chosen
    step = {'clicks': [[int(v) for v in focus_point], [int(v) for v in target_point]], 'route': route}
    saved_steps[str(step_no)] = step
    save_configuration(CONFIG_FILE, configuration)
    replay_route(route, configuration['manufacturer'], camera)
    send_key('OK', configuration['manufacturer'])
    return step


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


def calibrate_playback(configuration, camera, calibration_dir):
    """After the last precondition step the channel plays; set up what marks the next channel's start.

    EmptyArea: the user picks an area that is empty while loading (landing area).
    Motion: the motion of the middle of the playing picture sets the threshold.
    """
    frame = wait_until_playing(camera, "Waiting for the channel to play.")
    cv2.imwrite(os.path.join(calibration_dir, "live_channel.jpg"), frame)
    if configuration['detector'] == 'EmptyArea':
        roi = select_landing_area(configuration, camera, calibration_dir)
        configuration['stream_roi'] = roi
        save_configuration(CONFIG_FILE, configuration)
        wait_until_playing(camera, "Waiting for the channel to play again.")
        return roi, LANDING_STEP
    roi = auto_stream_roi(frame.shape)
    playing = measure_playing_motion(camera, roi)
    threshold = max(MIN_MOTION_THRESHOLD, playing * MOTION_THRESHOLD_RATIO)
    print(f"[SETUP] Live video motion {playing:.1f}; playing counts from motion >= {threshold:.1f}")
    if playing < MIN_MOTION_THRESHOLD:
        print("[SETUP] The picture barely moves; pick a channel with moving video for reliable timing.")
    configuration['stream_roi'] = roi
    configuration['motion_threshold'] = float(threshold)
    save_configuration(CONFIG_FILE, configuration)
    return roi, threshold


def confirm_live_tv_ready(camera):
    """Before the first channel change: the user confirms Live TV is playing (ENTER) or cancels (Q)."""
    print("[READY] Is Live TV playing and ready for the first channel change? "
          "Press ENTER in the camera window to start, Q to cancel.")
    while True:
        frame = read_frame(camera)
        if frame is not None:
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [
                ("Is Live TV playing and ready?", (0, 255, 255), 1.0),
                ("ENTER = start measuring,  Q = cancel", (255, 255, 255), 0.8)]))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            print("[READY] Confirmed; starting the measurement.")
            return
        if key in (ord('q'), ord('Q')):
            raise KeyboardInterrupt("Canceled before the first channel change.")


def show_for(camera, seconds, label):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame = read_frame(camera)
        if frame is not None:
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [(label, (255, 255, 255), 1.0)]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Canceled by user.")


def replay_saved_navigation(configuration, camera, saved):
    """Run the saved navigation for this model with no clicks: Home, TV Plus, each planned step, each + OK."""
    manufacturer = configuration['manufacturer']
    print("[SETUP] Replaying the saved navigation: Home", end='')
    press_home(manufacturer, camera)
    if saved['route']:                                  # Samsung: PACE step to TV Plus
        print(f" -> {', '.join(saved['route'])} + OK", end='')
        replay_route(saved['route'], manufacturer, camera)
        send_key('OK', manufacturer)
        show_for(camera, STEP_LOAD_SECONDS, f"Opening {APP_NAME}...")
    for step_no in sorted(saved['plan_steps'], key=int):
        keys = saved['plan_steps'][step_no]['route']
        print(f" -> step {step_no}: {describe_route(keys)} + OK", end='')
        replay_route(keys, manufacturer, camera)
        send_key('OK', manufacturer)
        show_for(camera, STEP_LOAD_SECONDS, f"Step {step_no} done, waiting for the screen...")
    print()
    return saved['stream_roi'], saved['motion_threshold']


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
                     key_pressed_at, pre_frames=(), last_roi=None, key_label="CH UP", start_still=0):
    """Time from Ch Up until the new channel plays: after a blank/frozen picture, sustained motion again."""
    x, y, w, h = roi
    frame_count = 0
    last_elapsed_ms = 0.0
    detected = False
    still_frames = start_still  # consecutive frames with no playback (black or frozen)
    window = []                 # (elapsed_ms, moving, after_still) for the last PLAYBACK_WINDOW_FRAMES
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
            annotate(frame, [(f"RUN {run_idx}  BEFORE {key_label}  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4)]),
        )

    print(f"[RUN {run_idx}] Waiting for the new channel to play (motion >= {motion_threshold:.1f} in "
          f"{PLAYBACK_MIN_FRACTION:.0%} of {PLAYBACK_WINDOW_FRAMES} frames after the picture stops)...")
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
            # Playback = sustained motion: most frames of a short window move. Short bursts (the GUIDE
            # closing, a banner sliding, a loading animation) do not qualify.
            window.append((elapsed_ms, playing, still_frames >= MIN_STILL_FRAMES))
            if len(window) > PLAYBACK_WINDOW_FRAMES:
                window.pop(0)
            if (not detected and len(window) == PLAYBACK_WINDOW_FRAMES and window[0][1] and window[0][2]
                    and sum(moving for _, moving, _ in window) >= PLAYBACK_MIN_FRACTION * PLAYBACK_WINDOW_FRAMES):
                detected = True
                detected_elapsed_ms = window[0][0]
                stop_at_ms = elapsed_ms + STREAM_EXTRA_SECONDS * 1000.0
                print(f"[RUN {run_idx}] New channel playing at {detected_elapsed_ms:.2f} ms "
                      f"(confirmed by {PLAYBACK_WINDOW_FRAMES} frames of motion).")
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


def capture_pre_roll(frame_capture, roi, seconds=PRE_ROLL_SECONDS):
    """Frames for the video pre-roll (no check); returns them, the last area image, and the median motion."""
    frame_capture.clear_buffer()
    pre_frames, motions, previous = [], [], None
    end_at = time.perf_counter() + seconds
    while time.perf_counter() < end_at or not pre_frames:
        captured = frame_capture.get_frame(timeout=1.0)
        if captured is None:
            if time.perf_counter() > end_at + 2.0:
                raise RuntimeError("No camera frames before the key press.")
            continue
        captured_at, frame = captured
        current = roi_gray(frame, roi)
        if previous is not None:
            motions.append(roi_motion(previous, current))
        previous = current
        pre_frames.append((captured_at, frame.copy(), 0.0))
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [("GUIDE: next channel selected, OK follows", (0, 255, 255), 1.0)]))
        cv2.waitKey(1)
    return pre_frames, previous, (float(np.median(motions)) if motions else 0.0)


def region_small(frame, roi):
    """Blurred, reduced grayscale copy of the area (small details and noise do not count)."""
    x, y, w, h = roi
    gray = cv2.GaussianBlur(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)


def guide_keys(configuration, camera, label):
    """OK, OK opens the GUIDE, Down selects the next channel (the timed OK is sent by the caller)."""
    manufacturer = configuration['manufacturer']
    for key, pause in (('OK', GUIDE_KEY_GAP_SECONDS), ('OK', GUIDE_OPEN_SECONDS), ('Down', GUIDE_KEY_GAP_SECONDS)):
        send_key(key, manufacturer)
        end = time.perf_counter() + pause
        while time.perf_counter() < end:
            frame = read_frame(camera)
            if frame is not None:
                cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [(f"{label}  {key} sent", (255, 255, 255), 1.0)]))
            cv2.waitKey(1)


def pattern_change(area, reference):
    """Change of the picture inside the area with each one's mean brightness removed.

    A landed picture brings new shapes; a loading screen whose backlight brightens (or a logo screen
    behind the area) only lifts the same room reflections, which this ignores.
    """
    return float(np.mean(np.abs((area - area.mean()) - (reference - reference.mean()))))


def screen_brightness(frame, screen_roi):
    """Mean gray level of the TV screen (blurred), used to see the black screen of a channel change."""
    x, y, w, h = screen_roi
    return float(cv2.GaussianBlur(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (9, 9), 0).mean())


def screen_area(configuration, camera, calibration_dir):
    """The TV screen in the camera view, saved per SoC; the user drags it once."""
    saved = configuration['screen_rois'].get(configuration['SoC'])
    if saved:
        print(f"[SETUP] Using the saved TV screen area for {configuration['SoC']}: {saved}")
        return saved
    frame = None
    while frame is None:
        frame = read_frame(camera)
    window_name = "Select the TV screen"
    while True:
        print("[SETUP] Drag a box around the whole TV screen (picture area only, not the wall), then press Enter.")
        preview = draw_overlay(frame.copy(), [("Drag around the WHOLE TV screen, then ENTER", (0, 255, 0), 1.0)])
        x, y, w, h = (int(v) for v in cv2.selectROI(window_name, preview, showCrosshair=False, fromCenter=False))
        cv2.destroyWindow(window_name)
        if w * h > 0.85 * frame.shape[0] * frame.shape[1]:
            print("[SETUP] The box covers almost the whole camera view; drag around the TV screen only "
                  "(the lit wall around it hides the black screen).")
        elif w >= 60 and h >= 40:
            break
        else:
            print("[SETUP] The box is too small; drag around the whole TV screen.")
    configuration['screen_rois'][configuration['SoC']] = [x, y, w, h]
    save_configuration(CONFIG_FILE, configuration)
    marked = frame.copy()
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(calibration_dir, "screen_area.jpg"), marked)
    return [x, y, w, h]


def select_landing_area(configuration, camera, calibration_dir):
    """One unmeasured channel change; the user drags an area that is empty (black) while it loads.

    That area is black/empty until the full-screen picture lands, so a lasting change there is the landing.
    """
    print("\n[SETUP] One unmeasured channel change to choose the landing area.")
    if configuration['zap_method'] == 'Network':
        NETWORK_REMOTE.send(channel_key(configuration, 0))
    elif configuration['zap_method'] == 'ChUp':
        send_key(ir_channel_key(configuration, 0), configuration['manufacturer'])
    else:
        guide_keys(configuration, camera, "SETUP")
        send_key('OK', configuration['manufacturer'])
    print("[SETUP] Press ENTER in the camera window while the screen is black/loading (before the picture appears).")
    for _ in range(50):
        if cv2.waitKey(1) == -1:
            break
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), [
            ("ENTER while the screen is black / loading", (0, 255, 255), 1.0)]))
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            break
        if key == ord('q'):
            raise KeyboardInterrupt("Setup canceled.")
    cv2.imwrite(os.path.join(calibration_dir, "loading_screen.jpg"), frame)
    window_name = "Select the landing area"
    while True:
        print("[SETUP] Drag over an area that is EMPTY (black) on this loading screen and filled by the picture "
              "when it lands, then press Enter.")
        preview = draw_overlay(frame.copy(), [("Drag an EMPTY (black) area, then ENTER", (0, 255, 0), 1.0)])
        x, y, w, h = (int(v) for v in cv2.selectROI(window_name, preview, showCrosshair=True, fromCenter=False))
        cv2.destroyWindow(window_name)
        if w >= 20 and h >= 20:
            return [x, y, w, h]
        print("[SETUP] The box is too small; drag a larger area.")


def measure_landing(frame_capture, roi, run_idx, output_dir, timeout, key_pressed_at, pre_frames=(),
                    require_black=False, screen_roi=None):
    """Time from OK until the full-screen picture lands, judged in an area that is empty while loading.

    Phases: 'closing' (the GUIDE disappears: the area changes once, then goes quiet), 'loading' (quiet),
    then the first large change that lasts PERSIST_SECONDS is the landing; its first frame is the time.
    require_black (CH+/CH-): nothing counts until the screen has gone black, so movement of the old channel
    before the black screen can never be taken as the new one. With screen_roi the whole TV screen is
    judged (the landing area may sit in a dark part of the old picture); otherwise the landing area.
    """
    x, y, w, h = roi
    # (elapsed_ms, area) for the last STEP_FRAMES frames
    history = [((captured_at - key_pressed_at) * 1000.0, region_small(frame, roi))
               for captured_at, frame, _ in list(pre_frames)[-STEP_FRAMES:]]
    phase, seen_change, quiet = 'closing', False, 0
    # Brightness of the old channel before the key: the whole screen when known, else the landing area.
    def darkness(frame, area):
        return screen_brightness(frame, screen_roi) if screen_roi else float(area.mean())
    pre_list = list(pre_frames)[-STEP_FRAMES:]
    # (elapsed_ms, brightness) of recent frames: the black screen is a sudden drop against these
    levels = [((captured_at - key_pressed_at) * 1000.0, darkness(frame, region_small(frame, roi)))
              for captured_at, frame, _ in pre_list]
    before = float(np.mean([level for _, level in levels])) if levels else None   # None: no pre-roll
    black_level = BLACK_SCREEN if screen_roi else LOADING_DARK
    old_dark = BLACK_OLD_DARK if screen_roi else LOADING_DARK + ARM_STEP
    black_seen = True
    loading_ref, loading_min = None, float('inf')   # darkest area of the loading screen (after the black)
    if require_black:
        black_seen, phase = False, 'waiting for black'
    candidate = None                 # (elapsed_ms, area before the change)
    detected, detected_elapsed_ms, stop_at_ms, result_saved = False, None, None, False
    frame_count, last_elapsed_ms = 0, 0.0
    frame_writer = FrameWriter(os.path.join(output_dir, "frames"))

    def annotate(image, lines):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return draw_overlay(image, lines)

    pre_index = 0
    for captured_at, frame, _ in pre_frames:
        elapsed_ms = (captured_at - key_pressed_at) * 1000.0
        if elapsed_ms < -PRE_ROLL_SECONDS * 1000.0 or elapsed_ms >= 0:
            continue
        pre_index += 1
        frame_writer.save(
            f"frame_p{pre_index:04d}_{elapsed_ms:09.1f}ms_sim0.000.jpg",
            annotate(frame, [(f"RUN {run_idx}  BEFORE OK  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4)]),
        )

    print(f"[RUN {run_idx}] Waiting for the picture to land in the selected area "
          f"(change >= {LANDING_STEP:.0f} lasting {PERSIST_SECONDS:g} s after the loading screen)...")
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
            current = region_small(frame, roi)
            reference = history[0][1] if len(history) >= STEP_FRAMES else None
            step = float(np.mean(np.abs(current - reference))) if reference is not None else 0.0
            history.append((elapsed_ms, current))
            history = history[-STEP_FRAMES:]

            if not detected:
                if loading_ref is not None and candidate is None:
                    # The darkest frame of the loading screen is what the landed picture is compared with.
                    level = darkness(frame, current)
                    if level < loading_min:
                        loading_ref, loading_min = current, level
                if not black_seen:
                    level = darkness(frame, current)
                    if before is None:
                        before = level
                    recent = [lv for t, lv in levels if elapsed_ms - BLACK_WINDOW_MS <= t < elapsed_ms]
                    levels = (levels + [(elapsed_ms, level)])[-30:]
                    top = max(recent) if recent else before
                    if before < old_dark:
                        # The old picture is already about as dark as the black screen: it cannot be told
                        # apart, so the black screen is not required for this run.
                        print(f"[RUN {run_idx}] The old picture is already dark ({before:.0f}); "
                              "the black screen is not required for this run.")
                        black_seen, phase = True, 'closing'
                    elif (elapsed_ms >= MIN_REACTION_MS and level < black_level
                          and level < top * BLACK_RATIO and level < top - BLACK_DROP):
                        # A sudden drop to dark: the old channel is gone (a dark scene cut of the old
                        # channel right at the key is not the TV's reaction, hence MIN_REACTION_MS).
                        # The old channel is gone: compare the landing against the black screen from here.
                        black_seen, phase = True, 'loading'
                        history, candidate = [(elapsed_ms, current)], None
                        loading_ref, loading_min = current, level
                elif phase == 'closing':
                    if step >= ARM_STEP:
                        seen_change, quiet = True, 0
                    elif seen_change or elapsed_ms > NO_GUIDE_CHANGE_MS:
                        quiet += 1
                        if quiet >= QUIET_FRAMES:
                            phase = 'loading'
                elif (reference is not None and current.mean() < LOADING_DARK
                      and current.mean() < reference.mean() - ARM_STEP):
                    # The old picture is going black (e.g. Hisense after CH+/-): that is the loading screen
                    # starting, not the new picture. Forget the old picture and wait for the landing from here.
                    history, candidate = [(elapsed_ms, current)], None
                elif candidate is None:
                    if step >= LANDING_STEP:
                        # The change started at the first frame since the reference that moved away from it.
                        onset = next(t for t, area in history[1:]
                                     if float(np.mean(np.abs(area - reference))) >= ONSET_STEP)
                        candidate = (onset, reference)
                elif elapsed_ms - candidate[0] >= PERSIST_SECONDS * 1000.0:
                    if (float(np.mean(np.abs(current - candidate[1]))) >= LANDING_STEP * 0.6
                            and pattern_change(current, candidate[1] if loading_ref is None else loading_ref)
                            >= LANDING_PATTERN):
                        detected, detected_elapsed_ms = True, candidate[0]
                        stop_at_ms = elapsed_ms + STREAM_EXTRA_SECONDS * 1000.0
                        print(f"[RUN {run_idx}] Landed at {detected_elapsed_ms:.2f} ms.")
                        winsound.Beep(1500, 250)
                        winsound.Beep(2500, 350)
                    else:
                        candidate = None          # a flash, not the picture: keep waiting

            color = (0, 255, 0) if detected else (0, 255, 255)
            lines = [
                (f"RUN {run_idx}  #{frame_count}  TIMER {elapsed_ms / 1000.0:.3f} s", color, 1.4),
                (f"Change {step:.0f} / {LANDING_STEP:.0f}  [{'landed' if detected else phase}]", color, 1.0),
            ]
            if detected:
                lines.append((f"LANDED at {detected_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            frame_writer.save(f"frame_{frame_count:05d}_{elapsed_ms:09.1f}ms_sim{min(step / LANDING_STEP, 9.999):.3f}.jpg",
                              annotate(frame.copy(), lines))
            shown = detected_elapsed_ms if detected else elapsed_ms
            live = [(f"RUN {run_idx}  TIMER {shown / 1000.0:.3f} s", color, 1.4), lines[1]]
            cv2.imshow(CAPTURE_WINDOW, annotate(frame, live))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")
            if detected and not result_saved:
                cv2.imwrite(os.path.join(output_dir, "RESULT_LANDED.jpg"), frame)
                result_saved = True
            if stop_at_ms is not None and elapsed_ms >= stop_at_ms:
                break
    finally:
        frame_writer.close()
        print(f"[RUN {run_idx}] Saved {frame_writer.saved} frames to {frame_writer.frames_dir}")

    if not detected:
        print(f"[RUN {run_idx}] No landing within {timeout:g} seconds."
              + ("" if black_seen else " The area never went black (did the channel change?)."))
        winsound.Beep(400, 700)
    return {
        'run': run_idx,
        'status': 'PASS' if detected else 'FAIL',
        'response_time_ms': detected_elapsed_ms if detected else last_elapsed_ms,
        'frame_count': frame_count,
        'directory': output_dir,
    }


def measure_after_key(configuration, frame_capture, roi, motion_threshold, run_idx, output_dir,
                      key_label, press, check_playing_first=True):
    """Pre-roll, press the key (timer starts), then wait for the new channel with the chosen detection."""
    motion = configuration['detector'] == 'Motion'
    if motion and check_playing_first:
        playing, level, pre_frames, last_roi = check_playing(frame_capture, roi, motion_threshold, output_dir)
        if not playing:
            return invalid_result(run_idx, output_dir,
                                  f"the channel was not playing before {key_label} (motion {level:.1f}).")
    else:
        pre_frames, last_roi, _ = capture_pre_roll(frame_capture, roi)
    frame_capture.clear_buffer()
    print(f"[RUN {run_idx}] {key_label}; the timer starts now.")
    key_pressed_at = press()
    winsound.Beep(1500, 300)
    if motion:
        return measure_zap_time(frame_capture, roi, motion_threshold, run_idx, output_dir, configuration['timeout'],
                                key_pressed_at, pre_frames, last_roi, key_label=key_label)
    # CH+/CH- (IR or network): the old channel goes black first; the GUIDE (OK) closes without black.
    return measure_landing(frame_capture, roi, run_idx, output_dir, configuration['timeout'],
                           key_pressed_at, pre_frames, require_black=configuration['zap_method'] != 'Guide',
                           screen_roi=configuration['screen_rois'].get(configuration['SoC']))


def zap_by_guide(configuration, camera, run_idx, output_dir, roi, motion_threshold):
    """Samsung: OK, OK opens the GUIDE, Down picks the next channel; OK starts the timer."""
    manufacturer = configuration['manufacturer']
    print(f"[RUN {run_idx}] OK, OK (GUIDE), Down (next channel)...")
    guide_keys(configuration, camera, f"RUN {run_idx}")
    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        # The GUIDE is on screen, so the picture is not checked for motion before OK.
        return measure_after_key(configuration, frame_capture, roi, motion_threshold, run_idx, output_dir,
                                 "OK", lambda: send_key('OK', manufacturer), check_playing_first=False)
    finally:
        frame_capture.stop()


def channel_key(configuration, run_idx):
    """KEY_CHUP / KEY_CHDOWN for this run (Alternate: Up on odd runs, Down on even runs)."""
    direction = configuration['zap_direction']
    if direction == 'Alternate':
        direction = 'Down' if run_idx % 2 == 0 else 'Up'
    return CHANNEL_KEYS[direction]


def zap_by_network(configuration, camera, run_idx, output_dir, roi, motion_threshold):
    """Live TV playing: pre-roll, then CH+/CH- over the network starts the timer; stop at the landing."""
    key = channel_key(configuration, run_idx)
    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        return measure_after_key(configuration, frame_capture, roi, motion_threshold, run_idx, output_dir,
                                 key, lambda: NETWORK_REMOTE.send(key))
    finally:
        frame_capture.stop()


def ir_channel_key(configuration, run_idx):
    """IR key name ChUp / ChDown for this run (same direction rule as the network keys)."""
    return 'ChUp' if channel_key(configuration, run_idx) == CHANNEL_KEYS['Up'] else 'ChDown'


def zap_by_ir(configuration, camera, run_idx, output_dir, roi, motion_threshold):
    """Live TV playing: pre-roll, then IR CH+/CH- starts the timer; stop at the landing."""
    key = ir_channel_key(configuration, run_idx)
    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        return measure_after_key(configuration, frame_capture, roi, motion_threshold, run_idx, output_dir,
                                 f"IR {key}", lambda: send_key(key, configuration['manufacturer']))
    finally:
        frame_capture.stop()


def zap_once(configuration, camera, run_idx, output_dir, roi, motion_threshold):
    if configuration['zap_method'] != 'Network':
        wake_flirc_if_idle()             # before the pre-roll, so the timed key is never delayed by a wake-up
    zap = {'Network': zap_by_network, 'ChUp': zap_by_ir, 'Guide': zap_by_guide}[configuration['zap_method']]
    return zap(configuration, camera, run_idx, output_dir, roi, motion_threshold)


def watch(camera, seconds, run_idx):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame = read_frame(camera)
        if frame is not None:
            remaining = end - time.perf_counter()
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [
                (f"RUN {run_idx} next channel change in {remaining:.1f} s", (0, 255, 255), 1.0)]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Measurement canceled by user.")


def connect_network_remote(configuration):
    """Network method: connect to the TV now (first time ever: press Allow on the TV) and show what it is."""
    global NETWORK_REMOTE
    NETWORK_REMOTE = NetworkRemote(configuration['tv_ip'])
    model, name, power = NETWORK_REMOTE.info()
    print(f"[TV] {configuration['tv_ip']}: {name} ({model}), power {power}")
    print("[TV] Connecting to the remote API (if the TV asks, press Allow)...")
    NETWORK_REMOTE.open()
    print("[TV] Connected.")


def main():
    configuration = get_user_configuration()
    if configuration['zap_method'] == 'Network':
        connect_network_remote(configuration)
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

        model = configuration['SoC']
        saved_navigation = configuration['navigation'].get(model)
        expected_detector = SAVED_DETECTOR[configuration['detector']]
        if saved_navigation and saved_navigation.get('detector') != expected_detector:
            print(f"[SETUP] The saved navigation for {model} was made for another detection method; set it up again.")
            saved_navigation = None
        if saved_navigation and input(f"Use the saved navigation for {model} (no clicks needed)? (Y/n): "
                                      ).strip().lower() in ('', 'y', 'yes'):
            roi, motion_threshold = replay_saved_navigation(configuration, camera, saved_navigation)
        else:
            steps = {}
            if configuration['manufacturer'] == 'Samsung':
                # Input 1: Samsung TV Plus on Home (PACE_01), then OK.
                route = calibrate_route(configuration, camera)
                replay_route(route, configuration['manufacturer'], camera)
                send_key('OK', configuration['manufacturer'])
            else:
                # Input 1 on other TVs (e.g. Hisense VIDAA): planned keys from Home to the app, then OK.
                route = []
                configuration['route'] = []
                press_home(configuration['manufacturer'], camera)
                steps['1'] = plan_step(configuration, camera, 1, "the app on Home (e.g. Live TV)")
            # Input 2: the Live TV channel, then OK.  Input 3: one more step, then OK.  (S skips a step.)
            steps['2'] = plan_step(configuration, camera, 2, "the Live TV channel")
            steps['3'] = plan_step(configuration, camera, 3, "the next item in Live TV")
            steps = {number: step for number, step in steps.items() if step}
            roi, motion_threshold = calibrate_playback(configuration, camera, calibration_dir)
            # Keep this model's navigation so later sessions run without clicks.
            configuration['navigation'][model] = {
                'route': route,
                'plan_steps': steps,
                'stream_roi': [int(v) for v in roi],
                'motion_threshold': float(motion_threshold),
                'zap_method': configuration['zap_method'],
                'detector': expected_detector,
            }
            save_configuration(CONFIG_FILE, configuration)
            print(f"[SETUP] Saved the navigation for {model}; next time it can run without clicks.")

        if configuration['zap_method'] != 'Guide':
            screen_area(configuration, camera, calibration_dir)     # black is judged on the whole TV screen
        confirm_live_tv_ready(camera)
        results = []
        for run_idx in range(1, configuration['num_runs'] + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "C:/Temp",
                f"Omedia_TC02_{run_idx:02d}_{timestamp}_{configuration['SoC']}"
                f"_SWV{configuration['SWV']}_LGCV{configuration['LGCV']}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n  [RUN {run_idx}/{configuration['num_runs']}] channel change "
                  f"({configuration['zap_method']})\n{'=' * 60}")

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
        if NETWORK_REMOTE is not None:
            NETWORK_REMOTE.close()
        wait_for_uploads()


if __name__ == "__main__":
    main()
