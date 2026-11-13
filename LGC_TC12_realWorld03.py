import cv2
from camera_util import open_camera
import numpy as np
import time
import pandas as pd
import serial
import os
import config as cfg
from datetime import datetime
import winsound
import keyboard
import asyncio
from kasa import SmartPlug
import sys
from config_store import remembered_input
import lgc_similarity
import lgc_apps
import suite_progress

# The last answer to each "(default: ...)" prompt is the next default (saved in LGC_TC12_realWorld03_inputs.json).
input = remembered_input("LGC_TC12_realWorld03_inputs.json")

# ===== Configuration Input =====
def get_user_configuration():
    """
    Interactive configuration input from user
    """
    print("\n" + "="*60)
    print("  LG TV Cold Boot Performance Test - Configuration")
    print("="*60 + "\n")
    
    # Device IP
    while True:
        ip = input("Enter Device IP Address (default: 192.168.5.3): ").strip()
        if not ip:
            ip = "192.168.5.3"
            break
        if len(ip.split('.')) == 4:
            break
        print("❌ Invalid IP format. Please try again.")
    
    # Serial Port
    while True:
        port = input("Enter Serial Port (default: COM10): ").strip()
        if not port:
            port = "COM10"
            break
        if port.upper().startswith("COM"):
            break
        print("❌ Invalid port format. Use format like COM10.")
    
    # SoC
    SoC = input("Enter SoC Model (default: O22N3): ").strip() or "O22N3"
    
    # Software Version
    SWV = input("Enter Software Version (default: 33.31.24): ").strip() or "33.31.24"
    
    # LG CV
    LGCV = input("Enter LG CV Version (default: 4.0.18-1): ").strip() or "4.0.18-1"
    
    # Number of runs
    while True:
        try:
            num_runs = int(input("Enter number of test runs (default: 5): ").strip() or "5")
            if num_runs > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Timeout
    while True:
        try:
            timeout = float(input("Enter timeout for motion detection in seconds (default: 10): ").strip() or "10")
            if timeout > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Live TV channel (IP 195 or 597)
    channel = input("Live TV channel (IP 195 / 597) (default: 195): ").strip() or "195"
    if not all(c.isdigit() or c == '-' for c in channel):
        print(f"⚠ Invalid channel {channel}: using 195.")
        channel = "195"

    config = {
        'channel': channel,
        'ip': ip,
        'port': port,
        'SoC': SoC,
        'SWV': SWV,
        'LGCV': LGCV,
        'num_runs': num_runs,
        'timeout': timeout
    }
    
    # Display summary
    print("\n" + "="*60)
    print("  Configuration Summary")
    print("="*60)
    for key, value in config.items():
        print(f"  {key.upper():<15}: {value}")
    print("="*60 + "\n")
    
    confirm = input("Confirm configuration and proceed? (Y/n): ").strip().lower()
    if confirm != 'n':
        return config
    else:
        return get_user_configuration()


# ===== Serial Communication =====
def initialize_serial(port):
    """
    Initialize serial connection with error handling
    """
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        print(f"✓ Serial connection established on {port}")
        return ser
    except serial.SerialException as e:
        print(f"❌ Failed to open serial port {port}: {e}")
        sys.exit(1)


# ===== IR Key Definitions =====
IR_KEYS = {
    'P_TOG': b'mc 00 08\n',
    'P_ON': b'ka 00 01\n',
    'P_OFF': b'ka 00 00\n',
    'ChUp': b'mc 00 00\n',
    'ChDown': b'mc 00 01\n',
    'VolUp': b'mc 00 02\n',
    'VolDown': b'mc 00 03\n',
    'DpadUp': b'mc 00 40\n',
    'DpadDn': b'mc 00 41\n',
    'DpadLt': b'mc 00 07\n',
    'DpadRt': b'mc 00 06\n',
    'Back': b'mc 00 28\n',
    'Home': b'mc 00 7C\n',
    'OK': b'mc 00 44\n',
    'LiveTV': b'mc 00 D6\n',
    'Exit': b'mc 00 5B\n',
    'Num_00': b'mc 00 10\n',
    'Num_01': b'mc 00 11\n',
    'Num_02': b'mc 00 12\n',
    'Num_03': b'mc 00 13\n',
    'Num_04': b'mc 00 14\n',
    'Num_05': b'mc 00 15\n',
    'Num_06': b'mc 00 16\n',
    'Num_07': b'mc 00 17\n',
    'Num_08': b'mc 00 18\n',
    'Num_09': b'mc 00 19\n',
    'DASH': b'mc 00 4C\n',
    'KeyNetflix': b'mc 00 56\n',
    'KeyAmazon': b'mc 00 5C\n',
    'RED': b'mc 00 72\n',
    'GRN': b'mc 00 71\n',
    'YEL': b'mc 00 63\n',
    'BLU': b'mc 00 61\n',
}

# ===== Measurement Thresholds =====
GRAY_THRESHOLD = 150
MOTION_THRESHOLD = 150
STABLE_FRAMES = 5

# ===== Helper Functions =====
def send_key(ser, key_name, delay=2):
    """Send IR key command"""
    if key_name in IR_KEYS:
        ser.write(IR_KEYS[key_name])
        ser.flush()
        time.sleep(delay)
    else:
        print(f"⚠ Unknown key: {key_name}")


def send_shell_command_with_debug(ser, command):
    """Execute shell command via debug mode"""
    try:
        print(f"[DEBUG] Sending command: {command}")
        
        ser.write(b'debug\n')
        time.sleep(3)
        print("✓ Entered debug mode.")
        
        ser.write(b's\n')
        time.sleep(3)
        print("✓ Entered shell.")
        
        ser.write((command + '\n').encode())
        time.sleep(2)
        print("✓ Command executed.")
        
        ser.write(b'exit\n')
        time.sleep(2)
        ser.flush()
        
        ser.write(b'x\n')
        time.sleep(2)
        ser.flush()
        print("✓ Exited debug mode.")
        
    except Exception as e:
        print(f"❌ Error executing shell command: {e}")


def wait_with_countdown_noKeyInput(seconds, description="Timer"):
    """Timed wait: camera view with run / step / time left, sounds at start, last 3 s and end
    (shortened in the quick flow check)."""
    suite_progress.countdown(seconds, description)


def wait_for_screen_stable(cap, timeout=60, stable_threshold=3.0, min_stable_duration=4.0):
    """Wait until the screen is continuously stable for min_stable_duration seconds.
    Time-based to avoid false positives from brief stable moments (e.g. YouTube loading UI)."""
    print(f"  [Waiting for screen stable, timeout={timeout}s, need {min_stable_duration}s stable...]")
    prev_gray = None
    stable_start = None
    start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        if prev_gray is not None:
            diff_score = np.mean(cv2.absdiff(prev_gray, curr_blur))
            elapsed = time.perf_counter() - start

            if diff_score < stable_threshold:
                if stable_start is None:
                    stable_start = time.perf_counter()
                stable_dur = time.perf_counter() - stable_start
                print(f"  ⏱ {elapsed:.1f}s | Diff: {diff_score:.2f} | Stable: {stable_dur:.1f}/{min_stable_duration}s", end="\r")
                if stable_dur >= min_stable_duration:
                    print(f"\n  ✓ Screen stable after {elapsed:.1f}s")
                    return True
            else:
                stable_start = None
                print(f"  ⏱ {elapsed:.1f}s | Diff: {diff_score:.2f} | (motion)", end="\r")

        prev_gray = curr_blur

        if (time.perf_counter() - start) > timeout:
            print(f"\n  ⚠ Stable wait timeout after {timeout}s")
            return False

    return False


def wait_for_app_ready(cap, motion_timeout=15, stable_timeout=60):
    """Phase 1: wait for screen to start moving (app loading).
       Phase 2: wait for screen to stabilize (app ready for input)."""
    print(f"  [Waiting for app activity (motion_timeout={motion_timeout}s)...]")
    prev_gray = None
    start = time.perf_counter()

    while (time.perf_counter() - start) < motion_timeout:
        ret, frame = cap.read()
        if not ret:
            continue
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        if prev_gray is not None:
            diff_score = np.mean(cv2.absdiff(prev_gray, curr_blur))
            elapsed = time.perf_counter() - start
            print(f"  ⏱ {elapsed:.1f}s | Diff: {diff_score:.2f} (waiting for motion>5)", end="\r")
            if diff_score > 5:
                print(f"\n  ✓ Screen active at {elapsed:.1f}s — now waiting for stable...")
                result = wait_for_screen_stable(cap, timeout=stable_timeout)
                time.sleep(1)
                return result
        prev_gray = curr_blur

    print(f"\n  ⚠ No motion in {motion_timeout}s — waiting for stable anyway...")
    result = wait_for_screen_stable(cap, timeout=stable_timeout)
    time.sleep(1)
    return result


async def run_ac_power_cycle(ip, off_seconds):
    """
    AC Power cycle via SmartPlug
    """
    try:
        dev = SmartPlug(ip)
        await dev.update()
        await dev.turn_off()
        suite_progress.sound('ac_off')
        print(f"✓ Device powered OFF")
        
        suite_progress.countdown(off_seconds, "AC power OFF", as_step=False)
        
        await dev.turn_on()
        suite_progress.sound('ac_on')
        print(f"✓ Device powered ON")
    except Exception as e:
        print(f"❌ Power cycle failed: {e}")


def initialize_camera(camera_index=0):
    # the configured index first, then the others; only a camera that streams real pictures is used
    return open_camera(camera_index)


def perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK', mode=None,
                             target=None, before_key=None):
    """End detection like the Omedia TCs (lgc_similarity.detect): CH+/CH- = the screen goes black, then the
    picture lands; screens = similarity with a saved reference (+ movement stopped); scrolls = movement stopped.
    before_key(): sent after the pre-roll, right before the measured key (PIP -> LG Channels)."""
    def press():
        winsound.Beep(800, 200)
        if before_key:
            before_key()
        sent_at = time.perf_counter()
        send_key(ser, trigger_key, 0)
        return sent_at
    return lgc_similarity.detect(cap, press, lambda: send_key(ser, trigger_key, 0), config['SoC'], run_idx,
                                 dir_path, timeout, trigger_key, mode, target)


# ===== Main Execution =====
def main():
    """Main test execution"""
    
    # Get configuration
    config = get_user_configuration()
    
    # Extract config
    ip = config['ip']
    port = config['port']
    SoC = config['SoC']
    SWV = config['SWV']
    LGCV = config['LGCV']
    num_runs = config['num_runs']
    timeout = config['timeout']
    
    # Initialize serial
    ser = initialize_serial(port)
    
    # Initialize camera
    cap = initialize_camera()
    suite_progress.begin_tc(1, 1, "TC12_realWorld03", num_runs, cap)
    quick = suite_progress.ask_quick()

    # The apps this TC uses: navigator route from Home + installed / signed in, checked on the camera
    lgc_apps.check_apps(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                        lgc_apps.apps_for(['TC12']))
    # the run ends with the PIP focus (10 s), then LG Channels: the route between them is checked now
    lgc_apps.plan_pip_to_lg_channels(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'])
    
    # CSV path
    csv_path = "tv_response_QUICK.csv" if quick else "tv_response.csv"
    results = []
    
    print(f"\n{'='*60}")
    print(f"  Starting {num_runs} Test Runs")
    print(f"{'='*60}\n")
    
    # Main test loop
    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"  [RUN {run_idx}/{num_runs}] Starting")
        print(f"{'='*60}")
        suite_progress.state.update(run_idx=run_idx, steps=[], step=0)
        if run_idx > 1:
            suite_progress.sound('tc_start')
        
        # Create run directory
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join(
            "C:/Temp",
            f"LGC_Perf_TC12_{run_idx:02d}_{ts_run}_{SoC}_SWV{SWV}_LGCV{LGCV}"
        )
        os.makedirs(dir_path, exist_ok=True)
        print(f"Output directory: {dir_path}")
        
        try:

            # Step 1: DC off 2 min -> DC on, 1 min
            suite_progress.announce("\n[STEP 1] DC Power cycle (OFF 2 min)...")
            send_key(ser, 'P_OFF', 0)
            wait_with_countdown_noKeyInput(120, "DC power OFF 2 min")
            send_key(ser, 'P_ON', 0)
            wait_with_countdown_noKeyInput(60, "DC power ON 1 min")

            # Step 2: Live TV channel (IP 195 / 597), 1 min
            channel = config['channel']
            suite_progress.announce(f"\n[STEP 2] Live TV channel {channel}...")
            send_key(ser, 'Exit', 2)
            send_key(ser, 'LiveTV', 5)
            for char in channel:
                send_key(ser, 'DASH' if char == '-' else f'Num_0{char}', 1)
            send_key(ser, 'OK', 5)
            wait_with_countdown_noKeyInput(60, f"Live TV {channel}")

            # Step 3: Netflix, play 1 min
            suite_progress.announce("\n[STEP 3] Launching Netflix...")
            send_key(ser, 'Home', 2)
            lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
            lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix',
                                 legacy=lambda: (send_key(ser, 'OK', 0), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                                 send_key(ser, 'OK', 0), wait_for_screen_stable(cap, timeout=30)))
            wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

            # Step 4: YouTube, play 3 min
            suite_progress.announce("\n[STEP 4] Launching YouTube...")
            send_key(ser, 'Home', 2)
            lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
            lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube',
                                 legacy=lambda: (send_key(ser, 'OK', 0), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                                 send_key(ser, 'OK', 0), wait_for_screen_stable(cap, timeout=30)))
            wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

            # Step 5: Home, 30 s
            suite_progress.announce("\n[STEP 5] Returning to Home...")
            send_key(ser, 'Home', 2)
            wait_with_countdown_noKeyInput(30, "Home Screen")

            # Step 6: Home PIP focus 10 s (the measurement's pre-roll is the last part of it)
            suite_progress.announce("\n[STEP 6] Home PIP focus 10 s...")
            lgc_apps.focus_pip(lambda key, delay: send_key(ser, key, delay), config['SoC'])
            wait_with_countdown_noKeyInput(10 - int(lgc_similarity.PRE_ROLL_SECONDS), "Home PIP focus")

            # Step 7: PIP -> LG Channels: measured
            suite_progress.announce("\n[STEP 7] PIP -> LG Channels - measuring load time...")
            result = perform_motion_detection(
                ser, cap, run_idx, dir_path, timeout, config,
                before_key=lambda: lgc_apps.lg_channels_from_pip(lambda key, delay: send_key(ser, key, delay),
                                                                 config['SoC']))

            if result:
                results.append(result)

                if result['detected']:
                    df = pd.DataFrame({
                        'Run': [result['run']],
                        'Timestamp': [ts_run],
                        'Response_Time_ms': [round(result['response_time_ms'], 2)],
                        'Frames': [result['frame_count']],
                        'Status': ['PASS'],
                        'Directory': [result['dir']]
                    })
                    write_header = not os.path.exists(csv_path)
                    df.to_csv(csv_path, mode='a', header=write_header, index=False)

        except Exception as e:
            print(f"❌ Error in run {run_idx}: {e}")
        
        # the next run starts right away with the DC power cycle (no interval in the TC12 scenario)

    suite_progress.sound('all_done')

    # Summary
    print(f"\n{'='*60}")
    print(f"  Test Complete - Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r['detected'])
    print(f"Total runs: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    
    if results:
        avg_time = np.mean([r['response_time_ms'] for r in results if r['detected']])
        print(f"Average response time: {avg_time:.2f}ms")
    
    print(f"Results saved to: {csv_path}\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("✓ All resources released.")


if __name__ == "__main__":
    main()