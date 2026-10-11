import cv2
import numpy as np
import time
import pandas as pd
import os
import json
import shutil
from datetime import datetime
import winsound
import keyboard
import sys
import subprocess
from ir_keys import LG_KEYS, ROKU_KEYS, SAMSUNG_KEYS

# ===== Configuration Input =====
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PACE_01_Launch_config.json")
LANDING_REFERENCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "PACE_01_Launch_landing_reference.jpg"
)
DEFAULT_CONFIGURATION = {
    'manufacturer': 'LG',
    'SoC': 'K25Lp',
    'SWV': '33.30.98',
    'LGCV': '4.0.18-2',
    'num_runs': 5,
    'timeout': 20.0
}


def load_saved_configuration(config_path=None, default_configuration=None):
    if config_path is None:
        config_path = CONFIG_FILE
    if default_configuration is None:
        default_configuration = DEFAULT_CONFIGURATION

    config = default_configuration.copy()
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            saved_config = json.load(config_file)
    except FileNotFoundError:
        return config
    except (OSError, ValueError) as e:
        print(f"Could not load saved configuration: {e}. Using built-in defaults.")
        return config

    if not isinstance(saved_config, dict):
        return config

    for key, default_value in default_configuration.items():
        value = saved_config.get(key)
        if type(value) is type(default_value):
            config[key] = value

    if config['manufacturer'] not in ('LG', 'Samsung', 'Roku'):
        config['manufacturer'] = default_configuration['manufacturer']
    if config['num_runs'] <= 0:
        config['num_runs'] = default_configuration['num_runs']
    if config['timeout'] <= 0:
        config['timeout'] = default_configuration['timeout']
    return config


def save_configuration(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=2)
    except OSError as e:
        print(f"Could not save configuration: {e}")


def select_tv_manufacturer(default_manufacturer):
    manufacturers = {
        '1': 'LG',
        '2': 'Samsung',
        '3': 'Roku',
    }
    default_choice = next(
        (key for key, value in manufacturers.items() if value == default_manufacturer),
        '1'
    )

    print("Select TV manufacturer:")
    for key, value in manufacturers.items():
        marker = " (default)" if key == default_choice else ""
        print(f"  {key}. {value}{marker}")

    while True:
        choice = input(f"Enter selection [1-3] (default: {default_choice}): ").strip() or default_choice
        if choice in manufacturers:
            return manufacturers[choice]
        print("Invalid selection. Enter 1, 2, or 3.")


def get_user_configuration():
    """
    Interactive configuration input from user
    """
    print("\n" + "="*60)
    print("  PACE_01_Launch - Configuration")
    print("="*60 + "\n")

    defaults = load_saved_configuration()
    manufacturer = select_tv_manufacturer(defaults['manufacturer'])

    # SoC
    SoC = input(f"Enter SoC or Model (default: {defaults['SoC']}): ").strip() or defaults['SoC']

    # Software Version
    SWV = input(f"Enter Software Version (default: {defaults['SWV']}): ").strip() or defaults['SWV']

    # LG CV
    LGCV = input(f"Enter FAST Version (default: {defaults['LGCV']}): ").strip() or defaults['LGCV']

    # Number of runs
    while True:
        try:
            num_runs = int(input(f"Enter number of test runs (default: {defaults['num_runs']}): ").strip() or str(defaults['num_runs']))
            if num_runs > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # Timeout
    while True:
        try:
            timeout_default = f"{defaults['timeout']:g}"
            timeout = float(input(f"Enter timeout for motion detection in seconds (default: {timeout_default}): ").strip() or timeout_default)
            if timeout > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")

    config = {
        'manufacturer': manufacturer,
        'SoC': SoC,
        'SWV': SWV,
        'LGCV': LGCV,
        'num_runs': num_runs,
        'timeout': timeout
    }

    print("\n" + "="*60)
    print("  Configuration Summary")
    print("="*60)
    for key, value in config.items():
        print(f"  {key.upper():<15}: {value}")
    print("="*60 + "\n")

    confirm = input("Confirm and proceed? (Y/n): ").strip().lower()
    if confirm != 'n':
        save_configuration(config)
        return config
    else:
        return get_user_configuration()


FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
RAW_IR_KEYSETS = {
    'LG': LG_KEYS,
    'Samsung': SAMSUNG_KEYS,
    'Roku': ROKU_KEYS,
}
RAW_IR_KEY_ALIASES = {
    'DpadUp': 'Up',
    'DpadDn': 'Down',
    'DpadLt': 'Left',
    'DpadRt': 'Right',
}

# ===== Measurement Thresholds =====
GRAY_THRESHOLD = 150
MOTION_THRESHOLD = 150
STABLE_FRAMES = 5
LANDING_TEMPLATE_THRESHOLD = 0.95
LANDING_STABLE_COUNT = 5

# ===== Helper Functions =====
def send_key(key_name, delay=2, manufacturer='LG'):
    """Send a key using the selected TV manufacturer's IR source."""
    if manufacturer not in RAW_IR_KEYSETS:
        raise ValueError(f"Unsupported TV manufacturer: {manufacturer}")
    keyset = RAW_IR_KEYSETS[manufacturer]
    raw_key_name = RAW_IR_KEY_ALIASES.get(key_name, key_name)
    if raw_key_name not in keyset:
        raise ValueError(f"{manufacturer} key is not configured: {raw_key_name}")

    print(f"[ACTION] Sending key '{key_name}' via {manufacturer} IR source...")
    action_started_at = time.perf_counter()
    subprocess.run(
        [FLIRC_UTIL_PATH, 'sendir', f"--csv={keyset[raw_key_name]}", '--repeat=1'],
        check=True,
        capture_output=True,
        text=True,
    )
    time.sleep(delay)
    return action_started_at


def wait_with_countdown_noKeyInput(seconds, description="Timer", cap=None, recorder=None):
    """
    Countdown timer without key input requirement
    """
    winsound.Beep(1000, 200)
    print(f"\n[WAIT] {description}: {seconds} seconds")

    for remaining in range(seconds, 0, -1):
        print(f"  ⏱ {remaining} seconds remaining ({description})...", end="\r")
        interval_end = time.perf_counter() + 1
        while time.perf_counter() < interval_end:
            if recorder is not None:
                recorder.pump()
            elif cap is not None:
                cap.grab()
            time.sleep(0.01)

    print(" " * 80, end="\r")
    winsound.Beep(1500, 300)
    print(f"[WAIT] ✓ {description} Complete!\n")


def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Camera could not be opened (index: {camera_index}).")
    print(f"✓ Camera initialized (index: {camera_index})")
    return cap


def lock_camera_focus(cap):
    """Disable autofocus after the user confirms a sharp camera preview."""
    accepted = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    autofocus_mode = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    if not accepted or autofocus_mode == 1:
        print(f"[CAMERA] Could not confirm autofocus lock (reported mode: {autofocus_mode}).")
        return False

    print(f"[CAMERA] Autofocus disabled; focus locked (reported mode: {autofocus_mode}).")
    return True


class CycleFrameRecorder:
    def __init__(self, cap, dir_path, interval_seconds=0.25):
        self.cap = cap
        self.dir_path = dir_path
        self.interval_seconds = interval_seconds
        self.frame_count = 0
        self.last_frame_time = None

    def record_frame(self, frame, force=False):
        now = time.perf_counter()
        if not force and self.last_frame_time is not None:
            if now - self.last_frame_time < self.interval_seconds:
                return False

        next_frame = self.frame_count + 1
        frame_path = os.path.join(self.dir_path, f"cycle_frame_{next_frame:05d}.jpg")
        if not cv2.imwrite(frame_path, frame):
            return False
        self.frame_count = next_frame
        self.last_frame_time = now
        return True

    def capture(self, force=False):
        ret, frame = self.cap.read()
        return self.record_frame(frame, force) if ret else False

    def pump(self):
        now = time.perf_counter()
        if self.last_frame_time is None or now - self.last_frame_time >= self.interval_seconds:
            self.capture(force=True)
        else:
            self.cap.grab()


def create_cycle_gif(dir_path, frame_count):
    """Create a looping GIF from the images captured during one complete cycle."""
    if frame_count == 0:
        print("[GIF] No cycle frames were captured; skipping GIF generation.")
        return None

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("[GIF] ffmpeg unavailable. Install project requirements to enable cycle GIFs.")
            return None

    frame_pattern = os.path.join(dir_path, "cycle_frame_%05d.jpg")
    gif_path = os.path.join(dir_path, "cycle.gif")
    command = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "4", "-start_number", "1", "-i", frame_pattern,
        "-filter_complex",
        "[0:v]scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", gif_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[GIF] Failed to create cycle GIF: {error}")
        return None

    print(f"[GIF] Cycle animation saved: {gif_path}")
    return gif_path


def flush_camera_buffer(cap, max_duration=1.0, max_frames=120, recorder=None):
    """Discard queued frames so the next read reflects the current TV screen."""
    start_time = time.perf_counter()
    flushed_frames = 0
    while flushed_frames < max_frames and time.perf_counter() - start_time < max_duration:
        if recorder is not None:
            recorder.pump()
            flushed_frames += 1
            continue
        if not cap.grab():
            break
        flushed_frames += 1
    print(f"[CAMERA] Flushed {flushed_frames} queued frame(s) before launch.")
    return flushed_frames


def position_camera(cap):
    """Show a live preview so the user can position the camera before testing."""
    window_name = "Camera Position Setup"
    print("\n[SETUP] Adjust the camera and wait until the TV screen looks sharp.")
    print("[SETUP] Press Enter to lock focus and continue, or Q to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera preview failed: no frame received.")
            cv2.destroyAllWindows()
            return False

        cv2.putText(frame, "Adjust camera; press ENTER when ready (Q to cancel)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (10, 13):
            lock_camera_focus(cap)
            cv2.destroyWindow(window_name)
            print("[SETUP] Camera position confirmed.")
            return True
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            print("[SETUP] Camera positioning canceled.")
            return False


def select_tv_plus_navigation(cap):
    """Let the user select a visible Samsung TV Plus tile and return its D-pad path."""
    window_name = "Select Samsung TV Plus"
    selection = {'point': None, 'route': None}

    def on_mouse(event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        frame_width, frame_height = parameter
        x_ratio = x / frame_width
        y_ratio = y / frame_height
        selection['point'] = (x, y)

        if 0.09 <= x_ratio <= 0.35 and 0.47 <= y_ratio <= 0.68:
            selection['route'] = ['DpadDn']
        elif 0.09 <= x_ratio <= 0.18 and 0.68 < y_ratio <= 0.78:
            selection['route'] = ['DpadDn', 'DpadDn']
        else:
            selection['route'] = None

    print("\n[APP SELECT] Click the Samsung TV Plus card or its bottom app icon.")
    print("[APP SELECT] Press Enter to confirm the highlighted choice, or Q to cancel.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read a frame while selecting the app.")

        cv2.setMouseCallback(window_name, on_mouse, (frame.shape[1], frame.shape[0]))
        preview = frame.copy()
        cv2.putText(preview, "Click Samsung TV Plus, then press ENTER",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if selection['point'] is not None:
            color = (0, 255, 0) if selection['route'] else (0, 0, 255)
            cv2.circle(preview, selection['point'], 9, color, 2)
            message = "TV Plus target selected" if selection['route'] else "Click a TV Plus tile"
            cv2.putText(preview, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30) & 0xFF

        if key in (10, 13) and selection['route']:
            cv2.destroyWindow(window_name)
            print(f"[APP SELECT] Selected TV Plus navigation: {selection['route']}")
            return selection['route'] + ['OK']
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            raise RuntimeError("TV Plus selection canceled.")


def launch_selected_app(cap, manufacturer, navigation, recorder=None):
    """Send the app navigation, flush stale frames, and trigger OK without a post-key delay."""
    for key_name in navigation:
        if key_name == 'OK':
            flush_camera_buffer(cap, recorder=recorder)
            return send_key(key_name, 0, manufacturer)
        send_key(key_name, 1, manufacturer)
    raise ValueError("App navigation must end with an OK key.")


def capture_reference_landing_frame(cap, dir_path, capture_seconds=2.0):
    """Capture and get user approval for a stable landing-page reference frame."""
    reference_path = os.path.join(dir_path, "landing_reference.jpg")
    window_name = "Landing Reference Capture"

    while True:
        print("\n[SETUP] Live camera is showing the TV. Press Enter in its window when the landing page is complete.")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        selected_page = False
        while not selected_page:
            ret, frame = cap.read()
            if not ret:
                cv2.destroyWindow(window_name)
                raise RuntimeError("Reference capture failed: no camera frame was available.")

            preview = frame.copy()
            cv2.putText(preview, "Live TV: press ENTER when landing is complete (Q to cancel)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (10, 13):
                selected_page = True
            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                raise RuntimeError("Reference capture canceled by user.")

        samples = []
        start_time = time.perf_counter()
        print("[SETUP] Landing page selected. Showing and sampling stable frames...")

        while time.perf_counter() - start_time < capture_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            preview = frame.copy()
            cv2.putText(preview, "Capturing landing reference...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(window_name)
                raise RuntimeError("Reference capture canceled by user.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            samples.append(gray)
            time.sleep(0.1)

        cv2.destroyWindow(window_name)

        if not samples:
            raise RuntimeError("Reference frame capture failed: no camera frames were available.")

        reference_gray = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
        cv2.imshow("Landing Reference Preview", reference_gray)
        cv2.waitKey(1)
        approval = input("Use this image as the landing reference? (Y/n): ").strip().lower()
        cv2.destroyWindow("Landing Reference Preview")

        if approval in ("", "y", "yes"):
            for image_path in (reference_path, LANDING_REFERENCE_FILE):
                if not cv2.imwrite(image_path, reference_gray):
                    raise RuntimeError(f"Could not save landing reference image: {image_path}")
            print(f"[SETUP] User approved reference image: {LANDING_REFERENCE_FILE}")
            return reference_gray

        print("[SETUP] Reference rejected. You can correct the TV screen and capture again.")


def review_saved_landing_reference():
    """Show the previously approved landing page and let the user reuse or replace it."""
    if not os.path.isfile(LANDING_REFERENCE_FILE):
        print("[SETUP] No saved landing reference. One will be captured during calibration.")
        return None

    reference_gray = cv2.imread(LANDING_REFERENCE_FILE, cv2.IMREAD_GRAYSCALE)
    if reference_gray is None:
        print("[SETUP] Saved landing reference could not be read; a new one will be captured.")
        return None

    cv2.imshow("Saved Landing Reference", reference_gray)
    cv2.waitKey(1)
    reuse = input("Reuse this saved landing page? (Y/n): ").strip().lower()
    cv2.destroyWindow("Saved Landing Reference")

    if reuse in ("", "y", "yes"):
        print(f"[SETUP] Reusing saved landing reference: {LANDING_REFERENCE_FILE}")
        return reference_gray

    print("[SETUP] Saved reference rejected. A new image will be captured during calibration.")
    return None


def measure_landing_time(cap, run_idx, dir_path, timeout, config, reference_gray, launch_started_at,
                         recorder=None):
    """
    Measure landing time by comparing live frames against a pre-confirmed landing page reference.
    The run finishes only when the screen remains above the similarity threshold for several frames.
    """
    print(f"\n[RUN {run_idx}] Waiting for landing screen similarity against saved reference...")

    landed = False
    stable_count = 0
    frame_count = 0
    elapsed_ms = 0.0
    threshold_crossing_ms = None
    start_time = launch_started_at
    last_frame = None

    print(f"[RUN {run_idx}] Monitoring for landing screen match...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Frame read failed.")
            break

        if recorder is not None:
            recorder.record_frame(frame)

        frame_count += 1
        last_frame = frame
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        score_map = cv2.matchTemplate(curr_gray_blur, reference_gray, cv2.TM_CCOEFF_NORMED)
        best_score = float(np.max(score_map))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        top_text = f"Time: {elapsed_ms:.2f}ms | Similarity: {best_score:.4f} | Frame: {frame_count}"
        cv2.putText(curr_gray, top_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,), 1)

        file_name = os.path.join(dir_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(file_name, curr_gray)

        if best_score >= LANDING_TEMPLATE_THRESHOLD:
            if stable_count == 0:
                threshold_crossing_ms = elapsed_ms
            stable_count += 1
        else:
            stable_count = 0
            threshold_crossing_ms = None

        if stable_count >= LANDING_STABLE_COUNT:
            landed = True
            cv2.putText(curr_gray, "✓ LANDING DETECTED!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,), 2)
            cv2.imwrite(os.path.join(dir_path, "RESULT_LANDING.jpg"), curr_gray)
            print(f"✓ [RUN {run_idx}] Landing confirmed! Threshold-crossing time: {threshold_crossing_ms:.2f}ms | Similarity: {best_score:.4f}")
            winsound.Beep(1500, 250)
            winsound.Beep(2500, 350)
            break

        if (time.perf_counter() - start_time) > timeout:
            print(f"❌ [RUN {run_idx}] Timeout: No landing match in {timeout} seconds")
            break

        cv2.imshow('Landing Detection', curr_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User exit requested.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    if recorder is not None and last_frame is not None:
        recorder.record_frame(last_frame, force=True)

    return {
        'run': run_idx,
        'detected': landed,
        'response_time_ms': threshold_crossing_ms if landed else elapsed_ms,
        'frame_count': frame_count,
        'dir': dir_path
    }


# ===== Main Execution =====
def main():
    """Main test execution"""

    cap = initialize_camera()
    if not position_camera(cap):
        cap.release()
        cv2.destroyAllWindows()
        return

    reference_gray = review_saved_landing_reference()
    config = get_user_configuration()

    SoC = config['SoC']
    SWV = config['SWV']
    LGCV = config['LGCV']
    manufacturer = config['manufacturer']
    num_runs = config['num_runs']
    timeout = config['timeout']

    csv_path = "pace_launch_response.csv"
    results = []

    print(f"\n{'='*60}")
    print(f"  Starting PACE_01_Launch - {manufacturer} - {num_runs} Test Runs")
    print(f"{'='*60}\n")

    calibration_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_dir = os.path.join(
        "C:/Temp",
        f"PACE_01_Launch_Calibration_{calibration_ts}_{SoC}_SWV{SWV}_LGCV{LGCV}"
    )
    os.makedirs(calibration_dir, exist_ok=True)
    print(f"\n[CALIBRATION] Reference setup output: {calibration_dir}")

    try:
        if manufacturer != 'Samsung':
            raise RuntimeError("TV Plus mouse navigation is currently configured for Samsung Smart Hub only.")
        print(f"[CALIBRATION] Preparing {manufacturer} home screen for an unmeasured calibration launch.")
        send_key('Home', 2, manufacturer)
        wait_with_countdown_noKeyInput(5, "Calibration Home Screen Stabilization", cap)
        print("[INFO] Power cycling is reserved for measured runs.")
        launch_navigation = select_tv_plus_navigation(cap)
        launch_selected_app(cap, manufacturer, launch_navigation)
        if reference_gray is None:
            print("[CALIBRATION] When the landing-complete page appears, confirm it for reference capture.")
            reference_gray = capture_reference_landing_frame(cap, calibration_dir)
        else:
            print("[CALIBRATION] Approved saved landing reference will be used for this run.")
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return

    print(f"\n[CALIBRATION] Reference approved. Starting {num_runs} measured runs.\n")

    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"  [RUN {run_idx}/{num_runs}] Starting")
        print(f"{'='*60}")

        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join(
            "C:/Temp",
            f"PACE_01_Launch_{run_idx:02d}_{ts_run}_{SoC}_SWV{SWV}_LGCV{LGCV}"
        )
        os.makedirs(dir_path, exist_ok=True)
        print(f"Output directory: {dir_path}")
        recorder = CycleFrameRecorder(cap, dir_path)
        recorder.capture(force=True)

        try:
            print(f"\n[STATUS] Run {run_idx}/{num_runs}: preparing {manufacturer} home screen")
            print("[STEP] Turning TV OFF with the remote Power button...")
            send_key('Power', 0, manufacturer)
            wait_with_countdown_noKeyInput(30, "TV Power OFF", recorder=recorder)

            print("[STEP] Turning TV ON with the remote Power button...")
            send_key('Power', 0, manufacturer)
            wait_with_countdown_noKeyInput(5, "TV Power ON Stabilization", recorder=recorder)

            print("[STEP] Sending Home...")
            send_key('Home', 0, manufacturer)
            wait_with_countdown_noKeyInput(5, "Home Screen Stabilization", recorder=recorder)

            print(f"[STEP] Launching app sequence for run {run_idx}...")
            launch_started_at = launch_selected_app(cap, manufacturer, launch_navigation, recorder)

            print(f"[LAUNCH] Landing detection started for run {run_idx}...")
            result = measure_landing_time(
                cap, run_idx, dir_path, timeout, config, reference_gray, launch_started_at, recorder
            )

            if result:
                results.append(result)

                if result['detected']:
                    df = pd.DataFrame({
                        'Run': [result['run']],
                        'Timestamp': [ts_run],
                        'Landing_Time_ms': [round(result['response_time_ms'], 2)],
                        'Frames': [result['frame_count']],
                        'Status': ['PASS'],
                        'Directory': [result['dir']]
                    })
                    write_header = not os.path.exists(csv_path)
                    df.to_csv(csv_path, mode='a', header=write_header, index=False)

        except Exception as e:
            print(f"❌ Error in run {run_idx}: {e}")

        create_cycle_gif(dir_path, recorder.frame_count)

        if run_idx < num_runs:
            wait_with_countdown_noKeyInput(10, "Interval between runs", recorder=recorder)

    print(f"\n{'='*60}")
    print(f"  Test Complete - Summary")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r['detected'])
    print(f"Total runs: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")

    if results:
        avg_time = np.mean([r['response_time_ms'] for r in results if r['detected']])
        print(f"Average landing time: {avg_time:.2f}ms")

    print(f"Results saved to: {csv_path}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("✓ All resources released.")


if __name__ == "__main__":
    main()
