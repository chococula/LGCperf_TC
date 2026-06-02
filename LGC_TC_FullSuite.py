
import cv2
import numpy as np
import time
import pandas as pd
import serial
import os
import random
import config as cfg
from datetime import datetime
import winsound
import keyboard
import asyncio
from kasa import SmartPlug
import sys

# =====================================================================
# ===== Shared Constants ==============================================
# =====================================================================
IR_KEYS = {
    'P_TOG': b'mc 00 08\n',
    'P_ON':  b'ka 00 01\n',
    'P_OFF': b'ka 00 00\n',
    'ChUp':      b'mc 00 00\n',
    'ChDown':    b'mc 00 01\n',
    'VolUp':     b'mc 00 02\n',
    'VolDown':   b'mc 00 03\n',
    'DpadUp':    b'mc 00 40\n',
    'DpadDn':    b'mc 00 41\n',
    'DpadLt':    b'mc 00 07\n',
    'DpadRt':    b'mc 00 06\n',
    'Back':      b'mc 00 28\n',
    'Home':      b'mc 00 7C\n',
    'OK':        b'mc 00 44\n',
    'LiveTV':    b'mc 00 D6\n',
    'Exit':      b'mc 00 5B\n',
    'Num_00':    b'mc 00 10\n',
    'Num_01':    b'mc 00 11\n',
    'Num_02':    b'mc 00 12\n',
    'Num_03':    b'mc 00 13\n',
    'Num_04':    b'mc 00 14\n',
    'Num_05':    b'mc 00 15\n',
    'Num_06':    b'mc 00 16\n',
    'Num_07':    b'mc 00 17\n',
    'Num_08':    b'mc 00 18\n',
    'Num_09':    b'mc 00 19\n',
    'DASH':      b'mc 00 4C\n',
    'KeyNetflix':b'mc 00 56\n',
    'KeyAmazon': b'mc 00 5C\n',
    'RED':       b'mc 00 72\n',
    'GRN':       b'mc 00 71\n',
    'YEL':       b'mc 00 63\n',
    'BLU':       b'mc 00 61\n',
}

CHANNEL_KEY_MAP = {
    '0': 'Num_00', '1': 'Num_01', '2': 'Num_02', '3': 'Num_03',
    '4': 'Num_04', '5': 'Num_05', '6': 'Num_06', '7': 'Num_07',
    '8': 'Num_08', '9': 'Num_09', '-': 'DASH',
}

GRAY_THRESHOLD   = 150
MOTION_THRESHOLD = 150

ALL_TC_LABELS = ['TC01', 'TC02', 'TC03', 'TC04',
                 'TC05_NtN', 'TC05_NtP', 'TC05_PtP',
                 'TC06_07_08',
                 'TC10', 'TC11', 'TC12']

# =====================================================================
# ===== Shared Helper Functions =======================================
# =====================================================================

def send_key(ser, key_name, delay=2):
    if key_name in IR_KEYS:
        ser.write(IR_KEYS[key_name])
        ser.flush()
        time.sleep(delay)
    else:
        print(f"⚠ Unknown key: {key_name}")


def send_key_human_like(ser, key_name, delay_range=(0.25, 0.75)):
    send_key(ser, key_name, 0)
    time.sleep(random.uniform(*delay_range))


def channel_to_keys(channel):
    keys = []
    for char in str(channel):
        if char not in CHANNEL_KEY_MAP:
            raise ValueError(f"Unsupported channel character: {char}")
        keys.append(CHANNEL_KEY_MAP[char])
    return keys


def prepare_channel_input(ser, channel, delay_range=(0.35, 0.9), confirm_with_ok=True):
    keys = channel_to_keys(channel)
    if confirm_with_ok:
        for key_name in keys:
            send_key_human_like(ser, key_name, delay_range)
        return keys, 'OK'
    if len(keys) <= 1:
        return [], keys[0]
    for key_name in keys[:-1]:
        send_key_human_like(ser, key_name, delay_range)
    return keys[:-1], keys[-1]


def send_shell_command_with_debug(ser, command):
    try:
        print(f"[DEBUG] Sending command: {command}")
        ser.write(b'debug\n'); time.sleep(3); print("✓ Entered debug mode.")
        ser.write(b's\n');     time.sleep(3); print("✓ Entered shell.")
        ser.write((command + '\n').encode()); time.sleep(2); print("✓ Command executed.")
        ser.write(b'exit\n');  time.sleep(2); ser.flush()
        ser.write(b'x\n');     time.sleep(2); ser.flush(); print("✓ Exited debug mode.")
    except Exception as e:
        print(f"❌ Error executing shell command: {e}")


def wait_with_countdown_noKeyInput(seconds, description="Timer"):
    winsound.Beep(1000, 200)
    print(f"\n[TIMER] {description}: {seconds} seconds")
    for remaining in range(seconds, 0, -1):
        print(f"  ⏱ {remaining} seconds remaining...", end="\r")
        time.sleep(1)
    print(" " * 50, end="\r")
    winsound.Beep(1500, 300)
    print(f"[TIMER] ✓ {description} Complete!\n")


def wait_for_screen_stable(cap, timeout=60, stable_threshold=5.0, min_stable_duration=4.0):
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


def wait_for_app_ready(cap, motion_timeout=15, stable_timeout=30):
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


async def _ac_power_cycle(ip, off_seconds):
    try:
        dev = SmartPlug(ip)
        await dev.update()
        await dev.turn_off()
        print("✓ Device powered OFF")
        await asyncio.sleep(off_seconds)
        await dev.turn_on()
        print("✓ Device powered ON")
    except Exception as e:
        print(f"❌ Power cycle failed: {e}")


def run_ac_power_cycle(ip, off_seconds):
    asyncio.run(_ac_power_cycle(ip, off_seconds))


def initialize_serial(port):
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        print(f"✓ Serial connection established on {port}")
        return ser
    except serial.SerialException as e:
        print(f"❌ Failed to open serial port {port}: {e}")
        sys.exit(1)


def initialize_camera(camera_index=1):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"✓ Camera initialized (index: {camera_index})")
    return cap

def perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK'):
    print(f"\n[RUN {run_idx}] Motion detection starting...")

    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to read initial frame. Skipping run.")
        return None

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    motion_detected = False
    frame_count = 0
    elapsed_ms = 0.0

    winsound.Beep(800, 200)
    send_key(ser, trigger_key, 0)
    ser.flush()
    start_time = time.perf_counter()

    print(f"[RUN {run_idx}] Trigger key: {trigger_key}")
    print(f"[RUN {run_idx}] Monitoring for motion...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Frame read failed.")
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        diff = cv2.absdiff(prev_gray, curr_gray_blur)
        diff_score = np.mean(diff)
        mean_val = np.mean(curr_gray)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(f"  ⏱ Elapsed: {elapsed_ms/1000:.2f}s | Diff: {diff_score:.2f} | Mean: {mean_val:.1f}", end="\r")

        top_text = f"Time: {elapsed_ms:.2f}ms | Diff: {diff_score:.2f} | Frame: {frame_count}"
        cv2.putText(curr_gray, top_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,), 1)

        file_name = os.path.join(dir_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(file_name, curr_gray)

        if diff_score > MOTION_THRESHOLD and mean_val > GRAY_THRESHOLD:
            motion_detected = True
            cv2.putText(curr_gray, "✓ MOTION DETECTED!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,), 2)
            cv2.imwrite(os.path.join(dir_path, "RESULT_HIT.jpg"), curr_gray)
            print(f"✓ [RUN {run_idx}] Motion detected! Response time: {elapsed_ms:.2f}ms")
            winsound.Beep(1500, 250)
            winsound.Beep(2500, 350)
            break

        if (time.perf_counter() - start_time) > timeout:
            print(f"❌ [RUN {run_idx}] Timeout: No motion in {timeout} seconds")
            break

        cv2.imshow('Motion Detection', curr_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User exit requested.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        prev_gray = curr_gray_blur

    return {
        'run': run_idx,
        'detected': motion_detected,
        'response_time_ms': elapsed_ms,
        'frame_count': frame_count,
        'dir': dir_path,
    }


def save_result(result, ts_run, csv_path, tc_label, extra_cols=None):
    if not result or not result['detected']:
        return
    row = {
        'TC':                [tc_label],
        'Run':               [result['run']],
        'Timestamp':         [ts_run],
        'Response_Time_ms':  [round(result['response_time_ms'], 2)],
        'Frames':            [result['frame_count']],
        'Status':            ['PASS'],
        'Directory':         [result['dir']],
    }
    if extra_cols:
        for k, v in extra_cols.items():
            row[k] = [v]
    df = pd.DataFrame(row)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


def make_dir(base_root, label, run_idx, ts_run, config, suffix=""):
    SoC  = config['SoC']
    SWV  = config['SWV']
    LGCV = config['LGCV']
    name = f"{label}_{run_idx:02d}_{ts_run}_{SoC}_SWV{SWV}_LGCV{LGCV}{suffix}"
    path = os.path.join(base_root, name)
    os.makedirs(path, exist_ok=True)
    return path


# =====================================================================
# ===== TC01 – Regular Boot ==========================================
# =====================================================================

def run_tc01(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = 20
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC01", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    print("\n[PRE-CONDITION] Setting up test environment...")
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(30, "Pre-load Channel")

    send_key(ser, 'Home', 0)
    wait_with_countdown_noKeyInput(60, "Power Stabilization")

    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(30, "Pre-load Channel")

    print("\n[AC POWER CYCLE] Starting power cycle...")
    run_ac_power_cycle(ip, 60)
    wait_with_countdown_noKeyInput(60, "Power Stabilization")

    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 2)

    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK')
    save_result(result, ts_run, csv_path, 'TC01')
    return result


# =====================================================================
# ===== TC02 – Cold Boot =============================================
# =====================================================================

def run_tc02(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC02", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    print("\n[PRE-CONDITION] Setting up test environment...")
    send_key(ser, 'LiveTV', 0.5)
    send_key(ser, 'Num_03', 0.5)
    send_key(ser, 'Num_06', 0.5)
    send_key(ser, 'DASH', 0.5)
    send_key(ser, 'Num_01', 0.5)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(10, "Pre-load Channel")

    print("\n[AC POWER CYCLE] Starting power cycle...")
    run_ac_power_cycle(ip, 10)
    wait_with_countdown_noKeyInput(60, "Power Stabilization")

    send_key(ser, 'Home', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'OK', 20)

    send_shell_command_with_debug(ser, "stop preload-manager")
    send_shell_command_with_debug(ser,
        "luna-send -n 1 -f luna://com.webos.applicationManager/closeByAppId "
        "'{\"id\": \"com.webos.app.lgchannels\"}'")

    send_key(ser, 'Home', 1)
    wait_with_countdown_noKeyInput(30, "Cool-down")
    send_key(ser, 'DpadRt', 2)

    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK')
    save_result(result, ts_run, csv_path, 'TC02')
    return result


def run_tc02_between_runs(ser):
    send_key(ser, 'Home', 2)
    send_key(ser, 'Back', 2)
    send_key(ser, 'DpadUp', 2)
    send_key(ser, 'DpadUp', 2)
    send_key(ser, 'OK', 2)


# =====================================================================
# ===== TC03 – Ch Zapping ============================================
# =====================================================================

def setup_tc03(ser, cap, config):
    ip = config['ip']
    print("\n[TC03 SETUP] Setting up test environment...")
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(30, "Pre-load Channel")

    print("\n[AC POWER CYCLE] Starting power cycle...")
    run_ac_power_cycle(ip, 60)
    wait_with_countdown_noKeyInput(60, "Power Stabilization")

    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadDn', 2)
    send_key(ser, 'OK', 10)

    send_key(ser, 'Num_01', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)

    send_key(ser, 'Num_04', 1)
    send_key(ser, 'Num_04', 1)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'OK')


def run_tc03(ser, cap, config, run_idx, csv_path):
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC03", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    send_key(ser, 'ChDown', 10)
    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='ChUp')
    save_result(result, ts_run, csv_path, 'TC03')
    return result


# =====================================================================
# ===== TC04 – Ch Num Press ==========================================
# =====================================================================

TC04_SOURCE_CHANNEL       = "11-1"
TC04_TARGET_CHANNEL       = "450"
TC04_USE_OK_TO_CONFIRM    = True


def setup_tc04(ser, cap, config):
    ip = config['ip']
    print("\n[TC04 SETUP] Setting up test environment...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(30, "Pre-load Channel")

    print("\n[AC POWER CYCLE] Starting power cycle...")
    run_ac_power_cycle(ip, 60)
    wait_with_countdown_noKeyInput(180, "Power Stabilization")

    send_key(ser, 'Exit', 2)
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadDn', 2)
    send_key(ser, 'OK', 10)


def run_tc04(ser, cap, config, run_idx, csv_path):
    timeout = config['timeout']
    source  = TC04_SOURCE_CHANNEL
    target  = TC04_TARGET_CHANNEL

    ts_run   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC04", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    print(f"[RUN {run_idx}] Source channel: {source}")
    for key_name in channel_to_keys(source):
        send_key_human_like(ser, key_name)
    send_key(ser, 'OK', 0)
    wait_with_countdown_noKeyInput(180, f"Source channel {source} stabilization")

    print(f"[RUN {run_idx}] Target channel: {target}")
    _, trigger_key = prepare_channel_input(ser, target, confirm_with_ok=TC04_USE_OK_TO_CONFIRM)

    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key)
    if result:
        result['source_channel'] = source
        result['target_channel'] = target
    save_result(result, ts_run, csv_path, 'TC04',
                extra_cols={'Source_Channel': source, 'Target_Channel': target})
    return result


# =====================================================================
# ===== TC05_NtN – Native to Native ==================================
# =====================================================================

def setup_tc05_ntn(ser, cap, config):
    native_previous = config.get('native_previous_channel', '443')
    print("\n[TC05_NtN SETUP] Setting up test environment...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(10, "Pre-load Channel")

    send_key(ser, 'Exit', 2)
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 10)

    for key_name in channel_to_keys(native_previous):
        send_key(ser, key_name, 2)
    send_key(ser, 'OK', 5)
    send_key(ser, 'Back', 5)


def run_tc05_ntn(ser, cap, config, run_idx, csv_path):
    timeout         = config['timeout']
    native_previous = config.get('native_previous_channel', '443')

    ts_run   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC05_NtN", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    send_key(ser, 'ChDown', 10)
    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='ChUp')
    if result:
        result['source_channel'] = native_previous
        result['target_channel'] = 'Native+1'
    save_result(result, ts_run, csv_path, 'TC05_NtN',
                extra_cols={'Source_Channel': native_previous, 'Target_Channel': 'Native+1'})
    return result


# =====================================================================
# ===== TC05_NtP – Native to Pluto ===================================
# =====================================================================

def setup_tc05_ntp(ser, cap, config):
    pluto_ch = config.get('pluto_channel_ntp', '151')
    print("\n[TC05_NtP SETUP] Setting up test environment...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(10, "Pre-load Channel")

    send_key(ser, 'Exit', 2)
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 10)

    for key_name in channel_to_keys(pluto_ch):
        send_key(ser, key_name, 2)
    send_key(ser, 'OK', 5)
    send_key(ser, 'Back', 5)


def run_tc05_ntp(ser, cap, config, run_idx, csv_path):
    timeout  = config['timeout']
    pluto_ch = config.get('pluto_channel_ntp', '151')

    ts_run   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC05_NtP", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    send_key(ser, 'ChDown', 10)
    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='ChUp')
    if result:
        result['source_channel'] = 'Native'
        result['target_channel'] = pluto_ch
    save_result(result, ts_run, csv_path, 'TC05_NtP',
                extra_cols={'Source_Channel': 'Native', 'Target_Channel': pluto_ch})
    return result


# =====================================================================
# ===== TC05_PtP – Pluto to Pluto ====================================
# =====================================================================

def setup_tc05_ptp(ser, cap, config):
    pluto_ch = config.get('pluto_channel_ptp', '220')
    print("\n[TC05_PtP SETUP] Setting up test environment...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(10, "Pre-load Channel")

    send_key(ser, 'Exit', 2)
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 10)

    for key_name in channel_to_keys(pluto_ch):
        send_key(ser, key_name, 2)
    send_key(ser, 'OK', 5)
    send_key(ser, 'Back', 5)


def run_tc05_ptp(ser, cap, config, run_idx, csv_path):
    timeout  = config['timeout']
    pluto_ch = config.get('pluto_channel_ptp', '220')

    ts_run   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC05_PtP", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    send_key(ser, 'ChDown', 10)
    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='ChUp')
    if result:
        result['source_channel'] = pluto_ch
        result['target_channel'] = 'Pluto+1'
    save_result(result, ts_run, csv_path, 'TC05_PtP',
                extra_cols={'Source_Channel': pluto_ch, 'Target_Channel': 'Pluto+1'})
    return result


# =====================================================================
# ===== TC06/07/08 – Movies and TV Scroll ============================
# =====================================================================

def run_tc06_07_08(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC06", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    results = []

    print("\n[STEP 1] Live TV 36-1 pre-load...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(60, "Live TV 36-1 Stabilization")

    print("\n[STEP 2] AC Power Cycle...")
    run_ac_power_cycle(ip, 60)

    wait_with_countdown_noKeyInput(180, "Boot Stabilization")

    print("\n[STEP 4] Entering LG Channels...")
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(30, "LG Channels Loading")

    print("\n[STEP 5] Navigating to Movies and TV...")
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadUp', 1)

    # TC06 – OK to select Movies and TV
    r1 = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK')
    if r1:
        r1['source_channel'] = 'LG Channels'
        r1['target_channel'] = 'Movies and TV'
        results.append(r1)
        save_result(r1, ts_run, csv_path, 'TC06',
                    extra_cols={'Source_Channel': 'LG Channels', 'Target_Channel': 'Movies and TV'})

    # TC07 – scroll down
    dir_scroll = dir_path + "_Scroll"
    os.makedirs(dir_scroll, exist_ok=True)
    r2 = perform_motion_detection(ser, cap, run_idx, dir_scroll, timeout, config, trigger_key='DpadDn')
    if r2:
        r2['source_channel'] = 'Movies and TV (top)'
        r2['target_channel'] = 'Movies and TV (scroll down)'
        results.append(r2)
        save_result(r2, ts_run, csv_path, 'TC07',
                    extra_cols={'Source_Channel': 'Movies and TV (top)',
                                'Target_Channel': 'Movies and TV (scroll down)'})

    # TC08 – scroll right
    dir_scroll_rt = dir_path + "_ScrollRight"
    os.makedirs(dir_scroll_rt, exist_ok=True)
    r3 = perform_motion_detection(ser, cap, run_idx, dir_scroll_rt, timeout, config, trigger_key='DpadRt')
    if r3:
        r3['source_channel'] = 'Movies and TV (scroll down)'
        r3['target_channel'] = 'Movies and TV (scroll right)'
        results.append(r3)
        save_result(r3, ts_run, csv_path, 'TC08',
                    extra_cols={'Source_Channel': 'Movies and TV (scroll down)',
                                'Target_Channel': 'Movies and TV (scroll right)'})

    return results


# =====================================================================
# ===== TC10 – Real World 01 =========================================
# =====================================================================

def run_tc10(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC10", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    print("\n[STEP 1] Live TV Fox 36-1...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 5)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(60, "Live TV Fox 36-1")

    print("\n[STEP 2] DC Power Cycle (OFF 2 min)...")
    run_ac_power_cycle(ip, 120)

    wait_with_countdown_noKeyInput(60, "Boot Stabilization")

    print("\n[STEP 4] Launching Amazon...")
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(60, "Amazon Video Playback")

    print("\n[STEP 5] Launching Netflix...")
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

    print("\n[STEP 6] Launching YouTube...")
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'DpadRt', 1)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

    print("\n[STEP 7] Returning to Home...")
    send_key(ser, 'Home', 2)
    wait_with_countdown_noKeyInput(30, "Home Screen")

    print("\n[STEP 8] Entering LG Channels - measuring load time...")
    send_key(ser, 'DpadRt', 2)

    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK')
    save_result(result, ts_run, csv_path, 'TC10')
    return result


# =====================================================================
# ===== TC11 – Real World 02 =========================================
# =====================================================================

def run_tc11(ser, cap, config, run_idx, csv_path):
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC11", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    send_key(ser, 'P_OFF', 0)
    wait_with_countdown_noKeyInput(120, "DC Power cycle for 2m")

    send_key(ser, 'P_ON', 0)

    wait_with_countdown_noKeyInput(180, "Boot Stabilization")

    print("\n[STEP 1] Live TV Fox 36-1...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 5)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(60, "Live TV Fox 36-1")

    print("\n[STEP 5] Launching Netflix...")
    send_key(ser, 'Home', 3)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

    print("\n[STEP 6] Launching YouTube...")
    send_key(ser, 'Home', 3)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    send_key(ser, 'OK', 0); wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

    print("\n[STEP 7] Returning to Home...")
    send_key(ser, 'Home', 2)
    wait_with_countdown_noKeyInput(30, "Home Screen")

    print("\n[STEP 8] Entering PIP...")
    send_key(ser, 'DpadDn', 2)
    send_key(ser, 'OK', 1)

    print("\n[STEP 9] Entering LG Channels - measuring load time...")
    send_key(ser, 'Home', 3)
    send_key(ser, 'DpadRt', 2)

    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK')
    save_result(result, ts_run, csv_path, 'TC11')
    return result


# =====================================================================
# ===== TC12 – Real World 03 =========================================
# =====================================================================

def run_tc12(ser, cap, config, run_idx, csv_path):
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC12", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    send_key(ser, 'P_OFF', 0)
    wait_with_countdown_noKeyInput(120, "DC Power cycle for 2m")

    send_key(ser, 'P_ON', 0)

    wait_with_countdown_noKeyInput(180, "Boot Stabilization")

    print("\n[STEP 1] Live TV Fox 36-1...")
    send_key(ser, 'Exit', 2)
    send_key(ser, 'LiveTV', 5)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(60, "Live TV Fox 36-1")

    print("\n[STEP 5] Launching Netflix...")
    send_key(ser, 'Home', 3)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=45)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

    print("\n[STEP 6] Launching YouTube...")
    send_key(ser, 'Home', 3)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=45)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    send_key(ser, 'OK', 0); wait_for_screen_stable(cap, timeout=30)
    wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

    print("\n[STEP 7] Returning to Home...")
    send_key(ser, 'Home', 2)
    wait_with_countdown_noKeyInput(60, "Home Screen")

    print("\n[STEP 8] PIP → Full Screen...")
    send_key(ser, 'DpadDn', 2)
    send_key(ser, 'OK', 1)
    wait_with_countdown_noKeyInput(10, "PIP to Full Screen")

    print("\n[STEP 9] Entering LG Channels - measuring load time...")
    send_key(ser, 'Home', 2)
    send_key(ser, 'DpadRt', 2)

    result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK')
    save_result(result, ts_run, csv_path, 'TC12')
    return result


# =====================================================================
# ===== Configuration ================================================
# =====================================================================

def select_tcs():
    print("\n" + "="*60)
    print("  Select Test Cases to Run")
    print("="*60)
    for i, label in enumerate(ALL_TC_LABELS, 1):
        print(f"  {i:>2}. {label}")
    print("="*60)
    print("  Enter numbers (comma-separated), or press Enter for ALL")
    raw = input("  Selection: ").strip()

    if not raw:
        return list(ALL_TC_LABELS)

    selected = []
    for token in raw.split(','):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(ALL_TC_LABELS):
                selected.append(ALL_TC_LABELS[idx])
            else:
                print(f"⚠ Invalid number: {token}")
        else:
            match = [t for t in ALL_TC_LABELS if token.upper() in t.upper()]
            if match:
                selected.extend(match)
            else:
                print(f"⚠ Unknown TC: {token}")

    seen = set()
    return [t for t in selected if not (t in seen or seen.add(t))]


def get_user_configuration(selected_tcs):
    print("\n" + "="*60)
    print("  LGC Perf Full Test Suite – Configuration")
    print("="*60 + "\n")

    while True:
        ip = input("Enter Device IP Address (default: 192.168.5.3): ").strip()
        if not ip:
            ip = "192.168.5.3"; break
        if len(ip.split('.')) == 4:
            break
        print("❌ Invalid IP format.")

    while True:
        port = input("Enter Serial Port (default: COM10): ").strip()
        if not port:
            port = "COM10"; break
        if port.upper().startswith("COM"):
            break
        print("❌ Invalid port format.")

    SoC  = input("Enter SoC Model (default: K25Lp): ").strip()  or "K25Lp"
    SWV  = input("Enter Software Version (default: 33.30.98): ").strip() or "33.30.98"
    LGCV = input("Enter LG CV Version (default: 4.0.18-2): ").strip()   or "4.0.18-2"

    while True:
        try:
            num_runs = int(input("Enter number of test runs (default: 5): ").strip() or "5")
            if num_runs > 0: break
            print("❌ Positive number required.")
        except ValueError:
            print("❌ Invalid number.")

    while True:
        try:
            timeout = float(input("Enter motion detection timeout in seconds (default: 10): ").strip() or "10")
            if timeout > 0: break
            print("❌ Positive number required.")
        except ValueError:
            print("❌ Invalid number.")

    config = {
        'ip': ip, 'port': port,
        'SoC': SoC, 'SWV': SWV, 'LGCV': LGCV,
        'num_runs': num_runs, 'timeout': timeout,
    }

    if 'TC05_NtN' in selected_tcs:
        config['native_previous_channel'] = (
            input("TC05_NtN – Native Previous Channel (default: 443): ").strip() or "443"
        )

    if 'TC05_NtP' in selected_tcs:
        config['pluto_channel_ntp'] = (
            input("TC05_NtP – Pluto Channel (default: 151): ").strip() or "151"
        )

    if 'TC05_PtP' in selected_tcs:
        config['pluto_channel_ptp'] = (
            input("TC05_PtP – Pluto Channel (default: 220): ").strip() or "220"
        )

    print("\n" + "="*60)
    print("  Configuration Summary")
    print("="*60)
    for k, v in config.items():
        print(f"  {k.upper():<30}: {v}")
    print("="*60)
    print(f"  TCs to run: {', '.join(selected_tcs)}")
    print("="*60 + "\n")

    confirm = input("Confirm and proceed? (Y/n): ").strip().lower()
    if confirm == 'n':
        return get_user_configuration(selected_tcs)
    return config


# =====================================================================
# ===== Master Orchestrator ==========================================
# =====================================================================

TC_SETUP_MAP = {
    'TC03':      setup_tc03,
    'TC04':      setup_tc04,
    'TC05_NtN':  setup_tc05_ntn,
    'TC05_NtP':  setup_tc05_ntp,
    'TC05_PtP':  setup_tc05_ptp,
}

TC_RUN_MAP = {
    'TC01':      run_tc01,
    'TC02':      run_tc02,
    'TC03':      run_tc03,
    'TC04':      run_tc04,
    'TC05_NtN':  run_tc05_ntn,
    'TC05_NtP':  run_tc05_ntp,
    'TC05_PtP':  run_tc05_ptp,
    'TC06_07_08': run_tc06_07_08,
    'TC10':      run_tc10,
    'TC11':      run_tc11,
    'TC12':      run_tc12,
}

TC_BETWEEN_RUNS_MAP = {
    'TC02': run_tc02_between_runs,
}


def main():
    selected_tcs = select_tcs()
    if not selected_tcs:
        print("No TCs selected. Exiting.")
        sys.exit(0)

    config = get_user_configuration(selected_tcs)

    num_runs = config['num_runs']
    port     = config['port']

    ser = initialize_serial(port)
    cap = initialize_camera()

    # ===== Camera Test =====
    print(f"\n{'='*60}")
    print("  Camera Test — Check framing before starting TCs")
    print(f"{'='*60}")
    print("  Live preview is ON. Press 'q' to quit, Enter to confirm OK.\n")

    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera frame read failed.")
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        display = frame.copy()
        if prev_gray is not None:
            diff_score = np.mean(cv2.absdiff(prev_gray, curr_blur))
            mean_val   = np.mean(curr_gray)
            h = display.shape[0]
            cv2.putText(display, f"Diff: {diff_score:.2f}  Mean: {mean_val:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "Press 'q' to quit | Enter to start",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        prev_gray = curr_blur

        cv2.imshow("Camera Test — Confirm framing then press Enter", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Camera test cancelled. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            ser.close()
            sys.exit(0)
        if key == 13:  # Enter
            break

    cv2.destroyAllWindows()
    print("✓ Camera confirmed. Starting TCs.\n")
    # ===== End Camera Test =====

    ts_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"C:/Temp/LGC_FullSuite_{ts_session}.csv"
    os.makedirs("C:/Temp", exist_ok=True)

    all_results = []

    for tc_label in selected_tcs:
        print(f"\n{'='*60}")
        print(f"  ▶▶▶  {tc_label}  ({num_runs} runs)")
        print(f"{'='*60}")

        # One-time setup if needed
        if tc_label in TC_SETUP_MAP:
            try:
                TC_SETUP_MAP[tc_label](ser, cap, config)
            except Exception as e:
                print(f"❌ Setup failed for {tc_label}: {e}")
                continue

        run_fn = TC_RUN_MAP.get(tc_label)
        if run_fn is None:
            print(f"⚠ No run function for {tc_label}, skipping.")
            continue

        tc_results = []
        for run_idx in range(1, num_runs + 1):
            print(f"\n{'='*60}")
            print(f"  [{tc_label}] RUN {run_idx}/{num_runs}")
            print(f"{'='*60}")
            try:
                result = run_fn(ser, cap, config, run_idx, csv_path)
                if isinstance(result, list):
                    tc_results.extend([r for r in result if r])
                elif result:
                    tc_results.append(result)
            except Exception as e:
                print(f"❌ Error in {tc_label} run {run_idx}: {e}")

            if run_idx < num_runs:
                wait_with_countdown_noKeyInput(10, "Interval between runs")
                if tc_label in TC_BETWEEN_RUNS_MAP:
                    try:
                        TC_BETWEEN_RUNS_MAP[tc_label](ser)
                    except Exception as e:
                        print(f"⚠ Between-run action failed: {e}")

        all_results.extend(tc_results)

        # Per-TC summary
        passed = sum(1 for r in tc_results if r.get('detected'))
        print(f"\n[{tc_label}] Runs: {len(tc_results)} | Pass: {passed} | Fail: {len(tc_results)-passed}")
        if passed:
            avg = np.mean([r['response_time_ms'] for r in tc_results if r.get('detected')])
            print(f"[{tc_label}] Average response time: {avg:.2f}ms")

    # ===== Overall Summary =====
    print(f"\n{'='*60}")
    print(f"  ===  FULL SUITE COMPLETE  ===")
    print(f"{'='*60}")
    total_passed = sum(1 for r in all_results if r.get('detected'))
    print(f"Total measurements : {len(all_results)}")
    print(f"Passed             : {total_passed}")
    print(f"Failed             : {len(all_results) - total_passed}")
    if total_passed:
        avg_all = np.mean([r['response_time_ms'] for r in all_results if r.get('detected')])
        print(f"Overall avg RT     : {avg_all:.2f}ms")
    print(f"Results saved to   : {csv_path}")

    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("✓ All resources released.")


if __name__ == "__main__":
    main()
