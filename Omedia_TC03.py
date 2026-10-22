"""Omedia_TC03 - YouTube app launch time after an AC power cycle.

Each run:
  1. AC OFF (10 s) -> AC ON (TP-Link smart plug).
  2. Wait until the camera sees the Home screen, then stay on Home for 30 s.
  3. Move to YouTube with the saved keys (planned like PACE: shown on screen first), OK starts the timer.
  4. Stop when the app lands: an area that is empty while YouTube loads gets filled.
Setup (once per model, saved and reused):
  - First AC cycle: press Enter when Home is showing (saves the Home reference).
  - After 30 s on Home: click the item that is highlighted, then YouTube; the planned keys are drawn;
    arrow keys adjust; Enter sends them + OK (this launch is not measured).
  - While YouTube loads (black/splash), press Enter and drag an area that is empty then.
Frames, MP4/GIF, CSV row (Launch_Time_ms), and upload are the same as Omedia_TC01.
"""
import asyncio
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

import kasa
from config_store import load_saved_configuration, save_configuration
from ir_keys import LG_KEYS, ROKU_KEYS, SAMSUNG_KEYS
from result_uploader import upload_result, wait_for_uploads
from run_video import make_run_videos


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC03_config.json")
DEFAULT_CONFIGURATION = {
    'manufacturer': 'Samsung',
    'SoC': 'U8000',
    'SWV': '1300',
    'LGCV': '8.2.008',
    'num_runs': 5,
    'timeout': 30.0,
    'plug_ip': '192.168.5.3',
    'ac_off_seconds': 10,
    'home_wait_seconds': 30,
    'boot_timeout': 120.0,
    'navigation': {},            # per SoC: route to the app, landing area, Home reference file
}

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
RAW_IR_KEYSETS = {
    'LG': LG_KEYS,
    'Roku': ROKU_KEYS,
    'Samsung': SAMSUNG_KEYS,
}
APP_NAME = "YouTube"
CAMERA_INDEX = 1
CAPTURE_WINDOW = "Omedia_TC03 Capture"
ARROW_KEYS = {2424832: 'Left', 2490368: 'Up', 2555904: 'Right', 2621440: 'Down'}   # cv2.waitKeyEx codes
TILE_STEP_RATIO = 1.15         # fallback step (tile size + gap) when the next tile cannot be measured
HOME_SIMILARITY = 0.6          # the camera frame matches the saved Home reference at least this well
POWER_KEY_AFTER_SECONDS = 40   # if Home has not appeared this long after AC ON, send Power once
PRE_ROLL_SECONDS = 2.0
STEP_FRAMES = 6                # landing area: change is measured against the frame this many frames earlier
ARM_STEP = 10                  # change of the area when Home disappears
LANDING_STEP = 15              # change of the area when the app lands (loading drift stays below this)
QUIET_FRAMES = 9               # ~0.3 s without change after Home disappeared = loading screen
PERSIST_SECONDS = 0.5          # the landed app must stay different from the loading screen this long
NO_GUIDE_CHANGE_MS = 3000      # if the area never changed when Home disappeared, start waiting anyway
ONSET_STEP = 8                 # the landing time is the first frame that moved this far from the loading screen
STREAM_EXTRA_SECONDS = 1.0     # keep saving frames this long after the landing is confirmed
HOME_SETTLE_SECONDS = 3
KEY_GAP_SECONDS = 0.5
CSV_PATH = os.path.join(SCRIPT_DIR, "Omedia_TC03_response.csv")


def get_user_configuration():
    defaults = load_saved_configuration(CONFIG_FILE, DEFAULT_CONFIGURATION)
    print("\n" + "=" * 60)
    print(f"  Omedia_TC03 - {APP_NAME} launch after AC power cycle")
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
    plug_ip = input(f"Enter smart plug IP (default: {defaults['plug_ip']}): ").strip() or defaults['plug_ip']
    num_runs = ask_number("Enter number of runs", defaults['num_runs'], int, lambda v: v > 0)
    timeout = ask_number("Enter launch timeout in seconds", defaults['timeout'], float, lambda v: v > 0)
    ac_off_seconds = ask_number("Enter AC OFF seconds", defaults['ac_off_seconds'], int, lambda v: v > 0)
    home_wait_seconds = ask_number("Enter seconds to stay on Home before launching",
                                   defaults['home_wait_seconds'], int, lambda v: v >= 0)

    configuration = dict(defaults)
    configuration.update({
        'manufacturer': manufacturer, 'SoC': SoC, 'SWV': SWV, 'LGCV': LGCV, 'plug_ip': plug_ip,
        'num_runs': num_runs, 'timeout': timeout, 'ac_off_seconds': ac_off_seconds,
        'home_wait_seconds': home_wait_seconds,
    })
    print("\nConfiguration:")
    for key in ('manufacturer', 'SoC', 'SWV', 'LGCV', 'plug_ip', 'num_runs', 'timeout',
                'ac_off_seconds', 'home_wait_seconds'):
        print(f"  {key}: {configuration[key]}")
    if input("Confirm and proceed? (Y/n): ").strip().lower() == 'n':
        return get_user_configuration()

    save_configuration(CONFIG_FILE, configuration)
    return configuration


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


def roi_gray(frame, roi):
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def roi_motion(previous_gray, current_gray):
    return float(np.mean(cv2.absdiff(previous_gray, current_gray)))


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
                     ("Arrow keys adjust,  R = start over,  ENTER = go", (255, 255, 255), 0.8)]
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
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [(f"{APP_NAME} selected, OK follows", (0, 255, 255), 1.0)]))
        cv2.waitKey(1)
    return pre_frames, previous, (float(np.median(motions)) if motions else 0.0)


def region_small(frame, roi):
    """Blurred, reduced grayscale copy of the area (small details and noise do not count)."""
    x, y, w, h = roi
    gray = cv2.GaussianBlur(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)


def measure_landing(frame_capture, roi, run_idx, output_dir, timeout, key_pressed_at, pre_frames=()):
    """Time from OK until the full-screen picture lands, judged in an area that is empty while loading.

    Phases: 'closing' (the previous screen disappears: the area changes once, then goes quiet), 'loading' (quiet),
    then the first large change that lasts PERSIST_SECONDS is the landing; its first frame is the time.
    """
    x, y, w, h = roi
    # (elapsed_ms, area) for the last STEP_FRAMES frames
    history = [((captured_at - key_pressed_at) * 1000.0, region_small(frame, roi))
               for captured_at, frame, _ in list(pre_frames)[-STEP_FRAMES:]]
    phase, seen_change, quiet = 'closing', False, 0
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
                if phase == 'closing':
                    if step >= ARM_STEP:
                        seen_change, quiet = True, 0
                    elif seen_change or elapsed_ms > NO_GUIDE_CHANGE_MS:
                        quiet += 1
                        if quiet >= QUIET_FRAMES:
                            phase = 'loading'
                elif candidate is None:
                    if step >= LANDING_STEP:
                        # The change started at the first frame since the reference that moved away from it.
                        onset = next(t for t, area in history[1:]
                                     if float(np.mean(np.abs(area - reference))) >= ONSET_STEP)
                        candidate = (onset, reference)
                elif elapsed_ms - candidate[0] >= PERSIST_SECONDS * 1000.0:
                    if float(np.mean(np.abs(current - candidate[1]))) >= LANDING_STEP * 0.6:
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
        print(f"[RUN {run_idx}] No landing within {timeout:g} seconds.")
        winsound.Beep(400, 700)
    return {
        'run': run_idx,
        'status': 'PASS' if detected else 'FAIL',
        'response_time_ms': detected_elapsed_ms if detected else last_elapsed_ms,
        'frame_count': frame_count,
        'directory': output_dir,
    }



# ---------- AC power and Home ----------

async def _ac_cycle(ip, off_seconds):
    plug = await kasa.Device.connect(host=ip)
    try:
        await plug.turn_off()
        print(f"[AC] OFF ({off_seconds} s)")
        await asyncio.sleep(off_seconds)
        await plug.turn_on()
        print("[AC] ON")
    finally:
        await plug.disconnect()


def ac_power_cycle(configuration):
    asyncio.run(_ac_cycle(configuration['plug_ip'], configuration['ac_off_seconds']))
    return time.perf_counter()


def gray_reference(camera, seconds=1.0):
    """Median blurred gray frame (removes noise) - the Home reference."""
    samples = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or not samples:
        frame = read_frame(camera)
        if frame is not None:
            samples.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def home_similarity(frame, home_reference):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return float(np.max(cv2.matchTemplate(gray, home_reference, cv2.TM_CCOEFF_NORMED)))


def wait_for_home(configuration, camera, home_reference, ac_on_at, run_label):
    """After AC ON, wait until the camera sees Home (sends Power once if the TV stays off)."""
    power_sent = False
    while True:
        elapsed = time.perf_counter() - ac_on_at
        if elapsed > configuration['boot_timeout']:
            return False
        frame = read_frame(camera)
        if frame is None:
            continue
        score = home_similarity(frame, home_reference)
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), [
            (f"{run_label}  waiting for Home  {elapsed:.0f} s", (0, 255, 255), 1.0),
            (f"Home match {score:.2f} / {HOME_SIMILARITY:.2f}", (0, 255, 255), 0.9)]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Canceled by user.")
        if score >= HOME_SIMILARITY:
            print(f"[{run_label}] Home is showing {elapsed:.1f} s after AC ON.")
            return True
        if not power_sent and elapsed > POWER_KEY_AFTER_SECONDS:
            print(f"[{run_label}] No Home yet; sending Power once.")
            send_key('Power', configuration['manufacturer'])
            power_sent = True


def stay_on_home(camera, seconds, run_label):
    print(f"[{run_label}] Staying on Home for {seconds} s.")
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame = read_frame(camera)
        if frame is not None:
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [
                (f"{run_label}  on Home, {end - time.perf_counter():.1f} s to {APP_NAME}", (0, 255, 255), 1.0)]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Canceled by user.")


def wait_for_enter(camera, message):
    print(f"[SETUP] {message} Press ENTER in the camera window.")
    for _ in range(50):
        if cv2.waitKey(1) == -1:
            break
    while True:
        frame = read_frame(camera)
        if frame is None:
            continue
        cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame.copy(), [(f"{message}  ENTER", (0, 255, 0), 1.0)]))
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            return frame
        if key == ord('q'):
            raise KeyboardInterrupt("Setup canceled.")


# ---------- setup (once per model) ----------

def select_landing_area(camera, calibration_dir):
    """Right after OK: Enter while the app is loading (black/splash), then drag an area that is empty then."""
    frame = wait_for_enter(camera, f"{APP_NAME} is loading: while the screen is black / splash,")
    cv2.imwrite(os.path.join(calibration_dir, "loading_screen.jpg"), frame)
    window_name = "Select the landing area"
    while True:
        print(f"[SETUP] Drag over an area that is EMPTY on this loading screen and filled when {APP_NAME} "
              "has launched, then press Enter.")
        preview = draw_overlay(frame.copy(), [("Drag an EMPTY area, then ENTER", (0, 255, 0), 1.0)])
        x, y, w, h = (int(v) for v in cv2.selectROI(window_name, preview, showCrosshair=True, fromCenter=False))
        cv2.destroyWindow(window_name)
        if w >= 20 and h >= 20:
            return [x, y, w, h]
        print("[SETUP] The box is too small; drag a larger area.")


def setup_navigation(configuration, camera, calibration_dir):
    """First AC cycle for this model: Home reference, the planned keys to YouTube, and the landing area."""
    model = configuration['SoC']
    print(f"\n[SETUP] Setting up {model}: AC OFF/ON once (not measured).")
    ac_power_cycle(configuration)
    wait_for_enter(camera, "When the Home screen is showing,")
    home_reference = gray_reference(camera)
    home_file = os.path.join(SCRIPT_DIR, f"Omedia_TC03_home_{model}.jpg")
    cv2.imwrite(home_file, home_reference)
    stay_on_home(camera, configuration['home_wait_seconds'], "SETUP")
    route, focus_point, target_point = select_plan(camera, 1, APP_NAME)
    replay_route(route, configuration['manufacturer'], camera)
    send_key('OK', configuration['manufacturer'])
    roi = select_landing_area(camera, calibration_dir)
    navigation = {'route': route, 'clicks': [list(focus_point), list(target_point)],
                  'landing_roi': roi, 'home_file': home_file}
    configuration['navigation'][model] = navigation
    save_configuration(CONFIG_FILE, configuration)
    print(f"[SETUP] Saved the navigation for {model}: {describe_route(route)} + OK; next time no clicks.")
    return navigation, home_reference


# ---------- measurement ----------

def launch_once(configuration, camera, run_idx, output_dir, navigation, home_reference):
    run_label = f"RUN {run_idx}"
    print(f"\n[{run_label}] STEP 1: AC OFF {configuration['ac_off_seconds']} s -> AC ON")
    winsound.Beep(1000, 200)
    ac_on_at = ac_power_cycle(configuration)
    print(f"[{run_label}] STEP 2: waiting for Home, then {configuration['home_wait_seconds']} s on Home")
    if not wait_for_home(configuration, camera, home_reference, ac_on_at, run_label):
        return invalid_result(run_idx, output_dir, f"Home did not appear within {configuration['boot_timeout']:g} s.")
    stay_on_home(camera, configuration['home_wait_seconds'], run_label)

    print(f"[{run_label}] STEP 3: {describe_route(navigation['route'])} to {APP_NAME}, then OK (timer).")
    replay_route(navigation['route'], configuration['manufacturer'], camera)
    roi = navigation['landing_roi']
    frame_capture = CameraFrameCapture(camera)
    frame_capture.start()
    try:
        pre_frames, _, _ = capture_pre_roll(frame_capture, roi)
        frame_capture.clear_buffer()
        key_pressed_at = send_key('OK', configuration['manufacturer'])
        winsound.Beep(1500, 300)
        return measure_landing(frame_capture, roi, run_idx, output_dir, configuration['timeout'],
                               key_pressed_at, pre_frames)
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

        session = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_dir = os.path.join("C:/Temp", f"Omedia_TC03_Calibration_{session}_{configuration['SoC']}")
        os.makedirs(calibration_dir, exist_ok=True)

        model = configuration['SoC']
        navigation = configuration['navigation'].get(model)
        home_reference = None
        if navigation and os.path.isfile(navigation.get('home_file', '')):
            print(f"[SETUP] Saved navigation for {model}: {describe_route(navigation['route'])} + OK")
            if input(f"Use the saved navigation for {model} (no clicks needed)? (Y/n): ").strip().lower() in ('', 'y', 'yes'):
                home_reference = cv2.imread(navigation['home_file'], cv2.IMREAD_GRAYSCALE)
        if home_reference is None:
            navigation, home_reference = setup_navigation(configuration, camera, calibration_dir)

        results = []
        for run_idx in range(1, configuration['num_runs'] + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "C:/Temp",
                f"Omedia_TC03_{run_idx:02d}_{timestamp}_{configuration['SoC']}"
                f"_SWV{configuration['SWV']}_LGCV{configuration['LGCV']}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n{'=' * 60}\n  [RUN {run_idx}/{configuration['num_runs']}] {APP_NAME} launch after AC cycle\n{'=' * 60}")

            result = launch_once(configuration, camera, run_idx, output_dir, navigation, home_reference)
            if result['status'] != 'INVALID':
                make_run_videos(output_dir, run_label=f"RUN {run_idx}",
                                end_ms=result['response_time_ms'] if result['status'] == 'PASS' else None)

            results.append(result)
            row = {
                'TC': 'Omedia_TC03',
                'Run': result['run'],
                'Timestamp': timestamp,
                'Launch_Time_ms': (
                    round(result['response_time_ms'], 2) if result['response_time_ms'] is not None else None
                ),
                'Frames': result['frame_count'],
                'Status': result['status'],
                'Directory': result['directory'],
            }
            pd.DataFrame([row]).to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
            upload_result({
                'TC': row['TC'], 'Session': session, 'Run': row['Run'], 'Timestamp': timestamp,
                'Manufacturer': configuration['manufacturer'], 'SoC': configuration['SoC'],
                'SWV': configuration['SWV'], 'LGCV': configuration['LGCV'],
                'Launch_Time_ms': row['Launch_Time_ms'], 'Frames': row['Frames'], 'Status': row['Status'],
            })

        passed = sum(result['status'] == 'PASS' for result in results)
        for result in results:
            if result['status'] == 'INVALID':
                print(f"Run {result['run']} INVALID: {result.get('reason', '')}")
        print(f"\nOmedia_TC03 complete: {passed}/{len(results)} runs passed.")
        for frequency in (1500, 2000, 2500):
            winsound.Beep(frequency, 200)
        print(f"Results saved to: {CSV_PATH}")
    except KeyboardInterrupt as error:
        print(f"\n{error}")
    except Exception as error:
        print(f"Omedia_TC03 failed: {error}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        wait_for_uploads()


if __name__ == "__main__":
    main()
