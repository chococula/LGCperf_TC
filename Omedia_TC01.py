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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC01_config.json")
REFERENCE_CHECK_SECONDS = 3.0   # at start: how long the live Home is compared with the reference
SCREEN_MARGIN = 12              # the screen area is searched this many pixels around (small camera shifts)
SCREEN_ROI = None               # [x, y, w, h] of the TV screen; only the screen is compared with Home
# Second measurement in the same run: the screen came on and then stopped moving (still for SETTLE_SECONDS).
SETTLE_ON_CHANGE = 10.0         # the screen differs this much from the off screen = it is on
SETTLE_STILL_DIFF = 2.0         # frame-to-frame change below this = still (camera noise is ~0.5)
SETTLE_SECONDS = 1.0            # still this long = movement stopped; the time is the first still frame
SETTLE_EXTRA_SECONDS = 8.0      # after Home is detected, keep recording up to this long for the settle
SETTLE_HOME_LIKE = 0.9          # still frames count only when they look like Home (>= 90% of the threshold),
                                # so a still boot logo is not taken as the end
DEFAULT_CONFIGURATION = {
    'manufacturer': 'LG',
    'SoC': 'O22N3',
    'SWV': '33.31.22',
    'LGCV': '4.0.18-1',
    'num_runs': 5,
    'timeout': 180.0,
    'home_similarity_threshold': 0.7,
    'standby_seconds': 300,      # wait with the TV off before Power ON
    'screen_rois': {},           # per SoC: [x, y, w, h] of the TV screen in the camera view
}

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
FLIRC_KEEPALIVE_SECONDS = 30    # the Flirc sleeps when idle (sleep detection is always on): talk to it this often
FLIRC_IDLE_WAKE_SECONDS = 20    # before a send after this long without Flirc traffic, wake it first
LAST_FLIRC_AT = 0.0
RAW_IR_KEYSETS = {
    'LG': LG_KEYS,
    'Roku': ROKU_KEYS,
    'Samsung': SAMSUNG_KEYS,
    'HisenseVidaa': HISENSEVIDAA_KEYS,
}
CAMERA_INDEX = 1
CAPTURE_WINDOW = "Omedia_TC01 Capture"
FRAME_CAPTURE_SIMILARITY = 0.85
PRE_ROLL_SECONDS = 2.0
CSV_PATH = os.path.join(SCRIPT_DIR, "Omedia_TC01_response.csv")


def get_user_configuration():
    defaults = load_saved_configuration(CONFIG_FILE, DEFAULT_CONFIGURATION)
    print("\n" + "=" * 60)
    print("  Omedia_TC01 - Home Screen Wake Time")
    print("=" * 60)

    manufacturer = choose_option("IR key set (TV maker)", RAW_IR_KEYSETS, defaults['manufacturer'])
    SoC = input(f"Enter SoC Model (default: {defaults['SoC']}): ").strip() or defaults['SoC']
    SWV = input(f"Enter Software Version (default: {defaults['SWV']}): ").strip() or defaults['SWV']
    LGCV = input(f"Enter LG CV Version (default: {defaults['LGCV']}): ").strip() or defaults['LGCV']

    while True:
        try:
            num_runs = int(input(f"Enter number of runs (default: {defaults['num_runs']}): ").strip() or str(defaults['num_runs']))
            if num_runs > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            standby_seconds = int(input(f"Enter standby wait with the TV off, in seconds "
                                        f"(default: {defaults['standby_seconds']}): ").strip()
                                  or str(defaults['standby_seconds']))
            if standby_seconds >= 0:
                break
            print("Please enter 0 or more.")
        except ValueError:
            print("Please enter a whole number of seconds.")

    while True:
        try:
            timeout = float(input(f"Enter wake timeout in seconds (default: {defaults['timeout']:g}): ").strip() or str(defaults['timeout']))
            if timeout > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            home_similarity_threshold = float(
                input(
                    "Enter Home screen similarity threshold (0-1, "
                    f"default: {defaults['home_similarity_threshold']:g}): "
                ).strip() or str(defaults['home_similarity_threshold'])
            )
            if 0 < home_similarity_threshold <= 1:
                break
            print("Enter a value greater than 0 and at most 1.")
        except ValueError:
            print("Please enter a valid number.")

    configuration = {
        'manufacturer': manufacturer,
        'SoC': SoC,
        'SWV': SWV,
        'LGCV': LGCV,
        'num_runs': num_runs,
        'timeout': timeout,
        'home_similarity_threshold': home_similarity_threshold,
        'standby_seconds': standby_seconds,
        'screen_rois': defaults['screen_rois'],
    }
    print("\nConfiguration:")
    for key, value in configuration.items():
        if key != 'screen_rois':
            print(f"  {key}: {value}")
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
    window_name = "Omedia_TC01 Camera Setup"
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


def home_reference_file(SoC):
    """Each TV model has its own Home screen, so each SoC keeps its own reference image."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in SoC)
    return os.path.join(SCRIPT_DIR, f"Omedia_TC01_home_reference_{safe}.jpg")


def capture_home_reference(camera, output_dir, reference_file):
    reference_path = os.path.join(output_dir, "home_reference.jpg")
    while True:
        print("\nEnsure the TV is ON and showing its Home screen.")
        input("When the Home screen is stable, press Enter to capture it: ")

        samples = []
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < 2.0:
            ret, frame = camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(cv2.GaussianBlur(gray, (5, 5), 0))
            time.sleep(0.05)

        if not samples:
            raise RuntimeError("Could not capture the Home screen reference.")

        reference = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
        cv2.imshow("Home Reference Preview", reference)
        cv2.waitKey(1)
        approval = input("Use this captured Home screen as the landing reference? (Y/n): ").strip().lower()
        cv2.destroyWindow("Home Reference Preview")
        if approval in ('', 'y', 'yes'):
            for image_path in (reference_path, reference_file):
                if not cv2.imwrite(image_path, reference):
                    raise RuntimeError(f"Could not save Home screen reference: {image_path}")
            print(f"Home landing reference saved: {reference_file}")
            return reference
        print("Reference rejected. Capture the Home screen again.")


def review_saved_home_reference(reference_file, SoC):
    if not os.path.isfile(reference_file):
        print(f"No saved Home landing reference for {SoC}. One will be captured now.")
        return None

    reference = cv2.imread(reference_file, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        print("Saved Home landing reference could not be read; a new one will be captured.")
        return None

    cv2.imshow("Saved Home Reference", reference)
    cv2.waitKey(1)
    reuse = input(f"Reuse this saved Home landing page for {SoC}? (Y/n): ").strip().lower()
    cv2.destroyWindow("Saved Home Reference")
    if reuse in ('', 'y', 'yes'):
        print(f"Reusing saved Home landing reference: {reference_file}")
        return reference

    print("Saved reference rejected. A new Home screen will be captured.")
    return None


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


def wait_with_countdown(camera, seconds, run_idx):
    print(f"\nTV standby wait: {seconds} seconds")
    end_at = time.perf_counter() + seconds
    last_printed = None
    keepalive = None
    while True:
        remaining = end_at - time.perf_counter()
        if remaining <= 0:
            break
        # Keep the Flirc awake during long waits (in the background, so the camera view keeps running).
        if (time.perf_counter() - LAST_FLIRC_AT >= FLIRC_KEEPALIVE_SECONDS
                and (keepalive is None or not keepalive.is_alive())):
            keepalive = threading.Thread(target=wake_flirc, daemon=True)
            keepalive.start()
        if last_printed != int(remaining) + 1:
            last_printed = int(remaining) + 1
            print(f"  {last_printed} seconds remaining...", end="\r")
            if last_printed <= 3:
                winsound.Beep(800, 100)

        ret, frame = camera.read()
        if ret:
            draw_overlay(frame, [
                (f"RUN {run_idx} STANDBY", (0, 255, 255), 1.2),
                (f"{int(remaining) // 60}:{remaining % 60:04.1f} to Power ON", (0, 255, 255), 1.2),
            ])
            cv2.imshow(CAPTURE_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Measurement canceled by user.")
    if keepalive is not None:
        keepalive.join()
    print(" " * 50, end="\r")
    print("Standby wait complete.")


def send_key(key_name, manufacturer):
    if manufacturer not in RAW_IR_KEYSETS:
        raise ValueError(f"Unsupported IR key set: {manufacturer}")
    keyset = RAW_IR_KEYSETS[manufacturer]
    if key_name not in keyset:
        raise ValueError(f"{manufacturer} IR key is not configured: {key_name}")

    global LAST_FLIRC_AT
    if time.perf_counter() - LAST_FLIRC_AT >= FLIRC_IDLE_WAKE_SECONDS:
        print("[FLIRC] Idle for a while; waking it up first (no IR).")
        wake_flirc()
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


def home_similarity(frame, reference):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if blurred.shape[0] < reference.shape[0] or blurred.shape[1] < reference.shape[1]:
        raise RuntimeError("Camera frame is smaller than the saved Home reference.")
    if SCREEN_ROI:
        # Only the TV screen: the wall and table around it look the same with the TV off or on.
        x, y, w, h = SCREEN_ROI
        m = SCREEN_MARGIN
        area = blurred[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        return float(np.max(cv2.matchTemplate(area, reference[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)))
    return float(np.max(cv2.matchTemplate(blurred, reference, cv2.TM_CCOEFF_NORMED)))


def screen_small(frame):
    """Blurred, reduced grayscale TV screen area for the movement measurement."""
    x, y, w, h = SCREEN_ROI or (0, 0, frame.shape[1], frame.shape[0])
    gray = cv2.GaussianBlur(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)


def select_screen_area(reference, configuration, calibration_dir):
    """The user drags a box around the TV screen (picture area) on the Home reference; saved per SoC."""
    global SCREEN_ROI
    print("\n[SETUP] Drag a box around the TV screen only (not the wall or stand), then press ENTER.")
    while True:
        x, y, w, h = cv2.selectROI("Select the TV screen, then ENTER", cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR),
                                   showCrosshair=False)
        cv2.destroyWindow("Select the TV screen, then ENTER")
        if w >= 40 and h >= 40:
            break
        print("[SETUP] The box is too small; drag around the whole TV screen.")
    SCREEN_ROI = [int(x), int(y), int(w), int(h)]
    configuration['screen_rois'][configuration['SoC']] = SCREEN_ROI
    save_configuration(CONFIG_FILE, {k: v for k, v in configuration.items() if k != 'calibration_dir'})
    preview = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(calibration_dir, "screen_area.jpg"), preview)
    print(f"[SETUP] TV screen area saved for {configuration['SoC']}: {SCREEN_ROI}")


def check_reference_matches_home(camera, reference, configuration):
    """At start the TV shows Home: the live Home must score above the threshold, or every run would FAIL.

    Home screens with rotating banners (e.g. Hisense VIDAA) score lower; the user can recapture,
    lower the threshold to what was measured, or continue. Returns the reference to use.
    """
    while True:
        threshold = configuration['home_similarity_threshold']
        scores = []
        end_at = time.perf_counter() + REFERENCE_CHECK_SECONDS
        while time.perf_counter() < end_at:
            ret, frame = camera.read()
            if not ret:
                continue
            score = home_similarity(frame, reference)
            scores.append(score)
            color = (0, 255, 0) if score >= threshold else (0, 0, 255)
            cv2.imshow(CAPTURE_WINDOW, draw_overlay(frame, [
                ("CHECK: live Home vs reference", (255, 255, 255), 1.0),
                (f"Similarity {score:.3f} / {threshold:.3f}", color, 1.0)]))
            cv2.waitKey(1)
        if not scores:
            raise RuntimeError("No camera frames for the Home reference check.")
        lowest = min(scores)
        print(f"[CHECK] Live Home vs reference: lowest similarity {lowest:.3f}, threshold {threshold:.3f}.")
        if lowest >= threshold:
            print("[CHECK] OK: the Home screen will be recognized.")
            return reference
        suggested = max(0.3, round(lowest - 0.05, 2))
        print("[CHECK] The live Home scores below the threshold, so Home would not be detected after Power ON.")
        print("        Is the TV on Home? (After Power ON the TV must open Home, not the last input.)")
        answer = input(f"  R = recapture the reference, L = lower the threshold to {suggested:.2f}, "
                       "C = continue anyway, Enter = check again: ").strip().lower()
        if answer == 'r':
            reference = capture_home_reference(camera, configuration['calibration_dir'],
                                               home_reference_file(configuration['SoC']))
            select_screen_area(reference, configuration, configuration['calibration_dir'])
        elif answer == 'l':
            configuration['home_similarity_threshold'] = suggested
            save_configuration(CONFIG_FILE, {k: v for k, v in configuration.items() if k != 'calibration_dir'})
            print(f"[CHECK] Threshold set to {suggested:.2f}.")
        elif answer == 'c':
            return reference


def check_tv_off(frame_capture, reference, similarity_threshold, output_dir, seconds=PRE_ROLL_SECONDS):
    """Before Power ON, confirm the screen does not already match Home (i.e. the TV really is off).

    The frames seen here are returned too, so the run video can start PRE_ROLL_SECONDS before Power ON.
    """
    frame_capture.clear_buffer()
    max_score = None
    last_frame = None
    pre_frames = []
    end_at = time.perf_counter() + seconds
    while time.perf_counter() < end_at or max_score is None:
        captured = frame_capture.get_frame(timeout=1.0)
        if captured is None:
            if time.perf_counter() > end_at + 2.0:
                raise RuntimeError("No camera frames during the TV-off check.")
            continue
        captured_at, frame = captured
        score = home_similarity(frame, reference)
        max_score = score if max_score is None else max(max_score, score)
        last_frame = frame
        pre_frames.append((captured_at, frame.copy(), score))

    tv_off = max_score < similarity_threshold
    last_frame = last_frame.copy()
    draw_overlay(last_frame, [
        (f"PRE-CHECK  similarity {max_score:.3f} / {similarity_threshold:.3f}", (0, 255, 255), 1.0),
        ("TV OFF - OK" if tv_off else "TV NOT OFF - RUN INVALID", (0, 255, 0) if tv_off else (0, 0, 255), 1.2),
    ])
    cv2.imwrite(os.path.join(output_dir, "PRECHECK_BEFORE_POWER_ON.jpg"), last_frame)
    cv2.imshow(CAPTURE_WINDOW, last_frame)
    cv2.waitKey(1)
    return tv_off, max_score, pre_frames


def measure_home_time(frame_capture, reference, run_idx, output_dir, timeout,
                      similarity_threshold, power_on_at, pre_frames=()):
    frame_count = 0
    last_elapsed_ms = 0.0
    detected = False
    armed = False
    detected_elapsed_ms = None
    # movement stop: off screen (last pre-roll frame), previous frame, current still streak
    off_small = screen_small(pre_frames[-1][1]) if pre_frames else None
    previous_small = None
    still_start = None              # (elapsed_ms, frame) of the first frame of the current still streak
    settled_elapsed_ms = None
    settle_frames = max(1, int(round(SETTLE_SECONDS * 30)))
    still_count = 0
    frame_writer = FrameWriter(os.path.join(output_dir, "frames"))

    # Pre-roll: the last PRE_ROLL_SECONDS before Power ON, saved with negative elapsed time for the video.
    pre_index = 0
    for captured_at, frame, score in pre_frames:
        elapsed_ms = (captured_at - power_on_at) * 1000.0
        if elapsed_ms < -PRE_ROLL_SECONDS * 1000.0 or elapsed_ms >= 0:
            continue
        pre_index += 1
        frame_writer.save(
            f"frame_p{pre_index:04d}_{elapsed_ms:09.1f}ms_sim{score:.3f}.jpg",
            draw_overlay(frame, [
                (f"RUN {run_idx}  BEFORE POWER ON  {elapsed_ms / 1000.0:.3f} s", (255, 255, 255), 1.4),
                (f"Similarity {score:.3f} / {similarity_threshold:.3f}", (0, 255, 255), 1.0),
            ]),
        )

    print(f"[RUN {run_idx}] Waiting for Home screen (saving every frame until similarity "
          f"{FRAME_CAPTURE_SIMILARITY:.2f})...")
    try:
        while time.perf_counter() - power_on_at <= timeout:
            captured = frame_capture.get_frame()
            if captured is None:
                continue

            captured_at, frame = captured
            if captured_at < power_on_at:
                continue

            frame_count += 1
            score = home_similarity(frame, reference)
            elapsed_ms = (captured_at - power_on_at) * 1000.0
            last_elapsed_ms = elapsed_ms
            # Only count a Home match after at least one frame below the threshold (TV off / booting),
            # so a stale buffered frame or a TV that never turned off cannot pass instantly.
            if score < similarity_threshold:
                armed = True
            newly_detected = armed and not detected and score >= similarity_threshold
            if newly_detected:
                detected = True
                detected_elapsed_ms = elapsed_ms

            current_small = screen_small(frame)
            if off_small is None:
                off_small = current_small
            movement = float(np.mean(np.abs(current_small - previous_small))) if previous_small is not None else 0.0
            previous_small = current_small
            screen_on = float(np.mean(np.abs(current_small - off_small))) >= SETTLE_ON_CHANGE
            newly_settled = False
            if settled_elapsed_ms is None:
                if screen_on and movement < SETTLE_STILL_DIFF and score >= SETTLE_HOME_LIKE * similarity_threshold:
                    if still_count == 0:
                        still_start = (elapsed_ms, frame.copy())
                    still_count += 1
                    if still_count >= settle_frames:
                        settled_elapsed_ms = still_start[0]
                        newly_settled = True
                else:
                    still_count = 0

            # Saved frames show this frame's own elapsed time from Power ON and its similarity.
            saved_color = (0, 255, 0) if score >= similarity_threshold else (0, 255, 255)
            saved_overlay = [
                (f"RUN {run_idx}  #{frame_count}  TIMER {elapsed_ms / 1000.0:.3f} s", saved_color, 1.4),
                (f"Similarity {score:.3f} / {similarity_threshold:.3f}", saved_color, 1.0),
                (f"Movement {movement:.1f} / {SETTLE_STILL_DIFF:.1f}", (255, 255, 255), 0.8),
            ]
            if detected:
                saved_overlay.append((f"HOME DETECTED at {detected_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            if settled_elapsed_ms is not None:
                saved_overlay.append((f"MOVEMENT STOPPED at {settled_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
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
                overlay.append(("HOME DETECTED", (0, 255, 0), 1.2))
            if settled_elapsed_ms is not None:
                overlay.append((f"MOVEMENT STOPPED at {settled_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            draw_overlay(frame, overlay)
            cv2.imshow(CAPTURE_WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")

            if newly_detected:
                cv2.imwrite(os.path.join(output_dir, "RESULT_HOME.jpg"), frame)
                print(
                    f"[RUN {run_idx}] Home similarity threshold reached in "
                    f"{elapsed_ms:.2f} ms (match {score:.3f}, threshold {similarity_threshold:.3f})."
                )
                winsound.Beep(1500, 250)
                winsound.Beep(2500, 350)

            if newly_settled:
                settled_image = draw_overlay(still_start[1], [
                    (f"RUN {run_idx}  MOVEMENT STOPPED  {settled_elapsed_ms / 1000.0:.3f} s", (0, 255, 0), 1.4),
                    (f"Still for {SETTLE_SECONDS:g} s from this frame", (0, 255, 0), 1.0)])
                cv2.imwrite(os.path.join(output_dir, "RESULT_SETTLED.jpg"), settled_image)
                print(f"[RUN {run_idx}] Movement stopped at {settled_elapsed_ms:.2f} ms "
                      f"(still for {SETTLE_SECONDS:g} s).")

            # Stop once both are measured, or the settle did not come within SETTLE_EXTRA_SECONDS of Home.
            if detected and settled_elapsed_ms is not None and (score >= FRAME_CAPTURE_SIMILARITY
                                                                or elapsed_ms >= settled_elapsed_ms + 500):
                print(f"[RUN {run_idx}] Both measured; frame capture stopped at {elapsed_ms:.2f} ms.")
                break
            if detected and elapsed_ms - detected_elapsed_ms > SETTLE_EXTRA_SECONDS * 1000.0:
                print(f"[RUN {run_idx}] The screen kept moving for {SETTLE_EXTRA_SECONDS:g} s after Home; "
                      "no movement-stop time for this run.")
                break
    finally:
        frame_writer.close()
        print(f"[RUN {run_idx}] Saved {frame_writer.saved} frames to {frame_writer.frames_dir}")

    if detected:
        cv2.waitKey(2000)
        last_elapsed_ms = detected_elapsed_ms

    if not detected:
        print(f"[RUN {run_idx}] Home screen not detected within {timeout:g} seconds.")
        winsound.Beep(400, 700)

    return {
        'run': run_idx,
        'status': 'PASS' if detected else 'FAIL',
        'response_time_ms': last_elapsed_ms,
        'settle_time_ms': settled_elapsed_ms,
        'frame_count': frame_count,
        'directory': output_dir,
    }


def main():
    configuration = get_user_configuration()
    camera = None

    try:
        camera = initialize_camera()
        if not position_camera(camera):
            print("Camera setup canceled.")
            return

        calibration_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_dir = os.path.join(
            "C:/Temp",
            f"Omedia_TC01_Calibration_{calibration_timestamp}_{configuration['SoC']}"
        )
        os.makedirs(calibration_dir, exist_ok=True)
        global SCREEN_ROI
        reference_file = home_reference_file(configuration['SoC'])
        reference = review_saved_home_reference(reference_file, configuration['SoC'])
        saved_roi = configuration['screen_rois'].get(configuration['SoC'])
        if reference is None:
            reference = capture_home_reference(camera, calibration_dir, reference_file)
            saved_roi = None
        else:
            cv2.imwrite(os.path.join(calibration_dir, "home_reference.jpg"), reference)
        configuration['calibration_dir'] = calibration_dir
        if saved_roi:
            SCREEN_ROI = saved_roi
            print(f"[SETUP] Using the saved TV screen area for {configuration['SoC']}: {SCREEN_ROI}")
        else:
            select_screen_area(reference, configuration, calibration_dir)
        reference = check_reference_matches_home(camera, reference, configuration)
        configuration.pop('calibration_dir')

        results = []
        for run_idx in range(1, configuration['num_runs'] + 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "C:/Temp",
                f"Omedia_TC01_{run_idx:02d}_{timestamp}_{configuration['SoC']}"
                f"_SWV{configuration['SWV']}_LGCV{configuration['LGCV']}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n[RUN {run_idx}/{configuration['num_runs']}] STEP 1: Send remote Power to turn the TV OFF.")
            send_key('Power', configuration['manufacturer'])
            winsound.Beep(1000, 200)
            print("STEP 2: Starting the standby wait.")
            wait_with_countdown(camera, configuration['standby_seconds'], run_idx)

            wake_flirc()                     # awake for the timed Power ON (the pre-check below takes 2 s)
            frame_capture = CameraFrameCapture(camera)
            frame_capture.start()
            try:
                tv_off, precheck_score, pre_frames = check_tv_off(
                    frame_capture, reference, configuration['home_similarity_threshold'], output_dir
                )
                if not tv_off:
                    print(
                        f"[RUN {run_idx}] INVALID: the screen already matches Home (similarity "
                        f"{precheck_score:.3f}) before Power ON, so the TV did not turn off. "
                        "Skipping this run; the next run's Power OFF should restore the state."
                    )
                    winsound.Beep(400, 300)
                    winsound.Beep(400, 300)
                    result = {
                        'run': run_idx,
                        'status': 'INVALID',
                        'response_time_ms': None,
                        'frame_count': 0,
                        'directory': output_dir,
                    }
                else:
                    frame_capture.clear_buffer()
                    print("STEP 3: Send remote Power to turn the TV ON; measurement starts now.")
                    power_on_at = send_key('Power', configuration['manufacturer'])
                    winsound.Beep(1500, 300)
                    result = measure_home_time(
                        frame_capture,
                        reference,
                        run_idx,
                        output_dir,
                        configuration['timeout'],
                        configuration['home_similarity_threshold'],
                        power_on_at,
                        pre_frames,
                    )
            finally:
                frame_capture.stop()

            if result['status'] != 'INVALID':
                ends = [t for t in (result['response_time_ms'], result.get('settle_time_ms')) if t is not None]
                make_run_videos(output_dir, threshold=configuration['home_similarity_threshold'],
                                end_ms=max(ends) if result['status'] == 'PASS' and ends else None,
                                run_label=f"RUN {run_idx}")

            results.append(result)
            row = {
                'TC': 'Omedia_TC01',
                'Run': result['run'],
                'Timestamp': timestamp,
                'Standby_Seconds': configuration['standby_seconds'],
                'Home_Time_ms': (
                    round(result['response_time_ms'], 2) if result['response_time_ms'] is not None else None
                ),
                'Settle_Time_ms': (
                    round(result['settle_time_ms'], 2) if result.get('settle_time_ms') is not None else None
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
                'Similarity_Threshold': configuration['home_similarity_threshold'],
                'Home_Time_ms': row['Home_Time_ms'],
                'Settle_Time_ms': row['Settle_Time_ms'],
                'Frames': row['Frames'],
                'Status': row['Status'],
            })

        passed = sum(result['status'] == 'PASS' for result in results)
        invalid = sum(result['status'] == 'INVALID' for result in results)
        if invalid:
            print(f"{invalid} run(s) were INVALID because the TV was not off before Power ON.")
        print(f"\nOmedia_TC01 complete: {passed}/{len(results)} runs passed.")
        for frequency in (1500, 2000, 2500):
            winsound.Beep(frequency, 200)
        print(f"Results saved to: {CSV_PATH}")
    except KeyboardInterrupt as error:
        print(f"\n{error}")
    except Exception as error:
        print(f"Omedia_TC01 failed: {error}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        wait_for_uploads()


if __name__ == "__main__":
    main()
