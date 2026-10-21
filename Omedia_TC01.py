import cv2
import numpy as np
import os
import pandas as pd
import queue
import subprocess
import threading
import time
from datetime import datetime

from config_store import load_saved_configuration, save_configuration
from ir_keys import LG_KEYS, ROKU_KEYS, SAMSUNG_KEYS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "Omedia_TC01_config.json")
DEFAULT_CONFIGURATION = {
    'manufacturer': 'LG',
    'SoC': 'O22N3',
    'SWV': '33.31.22',
    'LGCV': '4.0.18-1',
    'num_runs': 5,
    'timeout': 180.0,
    'home_similarity_threshold': 0.92,
}

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
RAW_IR_KEYSETS = {
    'LG': LG_KEYS,
    'Roku': ROKU_KEYS,
    'Samsung': SAMSUNG_KEYS,
}
CAMERA_INDEX = 1
STANDBY_SECONDS = 20
CSV_PATH = os.path.join(SCRIPT_DIR, "Omedia_TC01_response.csv")


def get_user_configuration():
    defaults = load_saved_configuration(CONFIG_FILE, DEFAULT_CONFIGURATION)
    print("\n" + "=" * 60)
    print("  Omedia_TC01 - Home Screen Wake Time")
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
    }
    print("\nConfiguration:")
    for key, value in configuration.items():
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


def capture_home_reference(camera, output_dir):
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
            if not cv2.imwrite(reference_path, reference):
                raise RuntimeError(f"Could not save Home screen reference: {reference_path}")
            return reference
        print("Reference rejected. Capture the Home screen again.")


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


def wait_with_countdown(seconds):
    print(f"\nTV standby wait: {seconds} seconds")
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining} seconds remaining...", end="\r")
        time.sleep(1)
    print(" " * 50, end="\r")
    print("Standby wait complete.")


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


def measure_home_time(frame_capture, reference, run_idx, output_dir, timeout,
                      similarity_threshold, power_on_at):
    frame_count = 0
    last_elapsed_ms = 0.0
    detected = False

    print(f"[RUN {run_idx}] Waiting for Home screen...")
    while time.perf_counter() - power_on_at <= timeout:
        captured = frame_capture.get_frame()
        if captured is None:
            continue

        captured_at, frame = captured
        if captured_at < power_on_at:
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if blurred.shape[0] < reference.shape[0] or blurred.shape[1] < reference.shape[1]:
            raise RuntimeError("Camera frame is smaller than the saved Home reference.")

        match = cv2.matchTemplate(blurred, reference, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(match))
        elapsed_ms = (captured_at - power_on_at) * 1000.0
        last_elapsed_ms = elapsed_ms

        cv2.putText(gray, f"{elapsed_ms:.0f} ms | Home match {score:.3f}/{similarity_threshold:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
        cv2.imshow("Omedia_TC01 Home Detection", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Measurement canceled by user.")

        if score >= similarity_threshold:
            detected = True
            cv2.putText(gray, "HOME DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,), 2)
            cv2.imwrite(os.path.join(output_dir, "RESULT_HOME.jpg"), gray)
            print(
                f"[RUN {run_idx}] Home similarity threshold reached in "
                f"{elapsed_ms:.2f} ms (match {score:.3f}, threshold {similarity_threshold:.3f})."
            )
            break

    if not detected:
        print(f"[RUN {run_idx}] Home screen not detected within {timeout:g} seconds.")

    return {
        'run': run_idx,
        'detected': detected,
        'response_time_ms': last_elapsed_ms,
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
        reference = capture_home_reference(camera, calibration_dir)

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
            print("STEP 2: Starting the 5-minute standby wait.")
            wait_with_countdown(STANDBY_SECONDS)

            frame_capture = CameraFrameCapture(camera)
            frame_capture.start()
            frame_capture.clear_buffer()
            print("STEP 3: Send remote Power to turn the TV ON; measurement starts now.")
            power_on_at = send_key('Power', configuration['manufacturer'])
            try:
                result = measure_home_time(
                    frame_capture,
                    reference,
                    run_idx,
                    output_dir,
                    configuration['timeout'],
                    configuration['home_similarity_threshold'],
                    power_on_at,
                )
            finally:
                frame_capture.stop()

            results.append(result)
            row = {
                'TC': 'Omedia_TC01',
                'Run': result['run'],
                'Timestamp': timestamp,
                'Standby_Seconds': STANDBY_SECONDS,
                'Home_Time_ms': round(result['response_time_ms'], 2),
                'Frames': result['frame_count'],
                'Status': 'PASS' if result['detected'] else 'FAIL',
                'Directory': result['directory'],
            }
            pd.DataFrame([row]).to_csv(
                CSV_PATH,
                mode='a',
                header=not os.path.exists(CSV_PATH),
                index=False,
            )

        passed = sum(result['detected'] for result in results)
        print(f"\nOmedia_TC01 complete: {passed}/{len(results)} runs passed.")
        print(f"Results saved to: {CSV_PATH}")
    except KeyboardInterrupt as error:
        print(f"\n{error}")
    except Exception as error:
        print(f"Omedia_TC01 failed: {error}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()