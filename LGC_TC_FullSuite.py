
import cv2
from camera_util import open_camera
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
import lgc_similarity
import lgc_lineup
import lgc_apps
import suite_progress
from config_store import remembered_input

# The last answer to each "(default: ...)" prompt is the next default (saved in LGC_TC_FullSuite_inputs.json).
input = remembered_input("LGC_TC_FullSuite_inputs.json")

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
        suite_progress.before_key(key_name)
        ser.write(IR_KEYS[key_name])
        ser.flush()
        time.sleep(delay)
        # after Num keys + OK: check on the camera that the channel really changed (quick check: confirm)
        suite_progress.after_key(key_name, lambda key, wait: send_key(ser, key, wait))
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


def validate_channel_value(channel, default):
    if not channel:
        return default
    if all(char in CHANNEL_KEY_MAP for char in str(channel)):
        return channel
    print(f"⚠ Invalid channel format: {channel}. Using default {default}.")
    return default


def go_to_channel(ser, channel, delay=1):
    print(f"  ▶ Navigating to channel {channel}")
    for key_name in channel_to_keys(channel):
        send_key(ser, key_name, delay)
    send_key(ser, 'OK', 0)


def verify_t2_pga_channel_settings(config):
    t2_channel = config.get('t2_channel', '443')
    pga_channel = config.get('pga_channel', '442')
    print(f"\n[CHANNEL CHECK] T2 Tennis channel: {t2_channel}")
    print(f"[CHANNEL CHECK] PGA channel: {pga_channel}")
    if t2_channel == pga_channel:
        print("⚠ Warning: T2 and PGA channels are configured to the same value.")
    if not channel_to_keys(t2_channel):
        print(f"⚠ Unable to verify T2 channel format: {t2_channel}")
    if not channel_to_keys(pga_channel):
        print(f"⚠ Unable to verify PGA channel format: {pga_channel}")


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
    # one step of the TC: camera view and console show TC i/N, step j/M and the time left
    suite_progress.countdown(seconds, description)


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
        suite_progress.sound('ac_off')
        suite_progress.countdown(off_seconds, "AC power OFF", as_step=False)
        await dev.turn_on()
        print("✓ Device powered ON")
        suite_progress.sound('ac_on')
    except Exception as e:
        print(f"❌ Power cycle failed: {e}")


def run_ac_power_cycle(ip, off_seconds):
    suite_progress.step('ac', f"AC power OFF {off_seconds} s -> ON")
    asyncio.run(_ac_power_cycle(ip, off_seconds))


def initialize_serial(port):
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        print(f"✓ Serial connection established on {port}")
        return ser
    except serial.SerialException as e:
        print(f"❌ Failed to open serial port {port}: {e}")
        sys.exit(1)


def initialize_camera(camera_index=0):
    # the configured index first, then the others; only a camera that streams real pictures is used
    return open_camera(camera_index)


def perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='OK', mode=None,
                             target=None):
    """End detection like the Omedia TCs (lgc_similarity.detect): CH+/CH- = the screen goes black, then the
    picture lands; screens = similarity with a saved reference (+ movement stopped); scrolls = movement stopped."""
    suite_progress.step('measure', f"Measure ({trigger_key})")
    suite_progress.state['measuring'] = True
    def press():
        winsound.Beep(800, 200)
        sent_at = time.perf_counter()
        send_key(ser, trigger_key, 0)
        return sent_at
    try:
        return lgc_similarity.detect(cap, press, lambda: send_key(ser, trigger_key, 0), config['SoC'], run_idx,
                                     dir_path, timeout, trigger_key, mode, target)
    finally:
        suite_progress.state['measuring'] = False
        suite_progress._entry['digits'] = ''   # digits typed for a measured entry (TC04)


def open_lg_channels(ser):
    """Home -> Right -> OK: opens LG Channels (the end screen of TC01 / TC02)."""
    send_key(ser, 'Home', 3)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 2)


def measure_lgc_home(ser, cap, run_idx, out_dir, timeout, soc, label):
    """OK on LG Channels starts the timer; similarity with the saved LG Channels home stops it."""
    suite_progress.step('measure', "Measure (OK)")
    def press():
        winsound.Beep(800, 200)
        sent_at = time.perf_counter()
        send_key(ser, 'OK', 0)
        return sent_at
    result = lgc_similarity.measure(cap, press, soc, out_dir, timeout, run_label=label)
    result['run'] = run_idx
    return result


def prepare_channel_landing(ser, cap, config):
    """Once per SoC: the TV screen box and the landing area, with an unmeasured ChDown and ChUp back, so the
    channel the test runs on is the same afterwards. Does nothing when both are saved."""
    lgc_similarity.ensure_screen(cap, config['SoC'])
    if not lgc_similarity.has_landing(config['SoC']):
        lgc_similarity.ensure_landing(cap, config['SoC'], lambda: send_key(ser, 'ChDown', 0))
        send_key(ser, 'ChUp', 0)
        wait_with_countdown_noKeyInput(10, "Back on the test channel")


def save_result(result, ts_run, csv_path, tc_label, extra_cols=None):
    if not result or not result['detected']:
        return
    row = {
        'TC':                tc_label,
        'Run':               result['run'],
        'Timestamp':         ts_run,
        'Response_Time_ms':  round(result['response_time_ms'], 2),
        'Frames':            result['frame_count'],
        'Status':            'PASS',
        'Directory':         result['dir'],
    }
    if 'settle_time_ms' in result:          # similarity TCs: the second time (movement stopped)
        settle = result['settle_time_ms']
        row['Settle_Time_ms'] = round(settle, 2) if settle is not None else ''
    if extra_cols:
        row.update(extra_cols)
    lgc_similarity.append_row(csv_path, row)   # keeps columns aligned when TCs add different ones


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
    TC01_REPEAT_COUNT = 5
    ip      = config['ip']
    timeout = 30
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = make_dir("C:/Temp", "LGC_Perf_TC01", run_idx, ts_run, config)
    print(f"Output directory: {base_dir}")

    # PRECONDITION (one time): LiveTV 36-1 for 1 minute, then Home for 1 minute.
    print("\n[PRE-CONDITION] LiveTV 36-1 (60s) -> Home (60s)")
    send_key(ser, 'LiveTV', 3)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(60, "Pre-load Channel")

    send_key(ser, 'Home', 0)
    wait_with_countdown_noKeyInput(60, "Home Screen Wait")

    # REPEAT SCENARIO (five times):
    # LiveTV 36-1 (60s) -> AC Off (60s) -> AC On (180s) -> LG Channels.
    results = []
    for cycle_idx in range(1, TC01_REPEAT_COUNT + 1):
        print(f"\n[TC01] Scenario cycle {cycle_idx}/{TC01_REPEAT_COUNT}")

        send_key(ser, 'LiveTV', 3)
        send_key(ser, 'Num_03', 1)
        send_key(ser, 'Num_06', 1)
        send_key(ser, 'DASH', 1)
        send_key(ser, 'Num_01', 1)
        send_key(ser, 'OK', 2)
        wait_with_countdown_noKeyInput(60, "Pre-load Channel")

        print("\n[AC POWER CYCLE] Power OFF for 60 seconds...")
        run_ac_power_cycle(ip, 60)
        wait_with_countdown_noKeyInput(180, "Power ON Stabilization")

        send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
        send_key(ser, 'DpadRt', 2)

        cycle_dir = os.path.join(base_dir, f"cycle_{cycle_idx:02d}")
        os.makedirs(cycle_dir, exist_ok=True)
        result = measure_lgc_home(ser, cap, run_idx, cycle_dir, timeout, config['SoC'],
                                  f"TC01 CYCLE {cycle_idx}")
        save_result(
            result, ts_run, csv_path, 'TC01', extra_cols={'Cycle': cycle_idx}
        )
        if result:
            result['cycle'] = cycle_idx
            results.append(result)

    return results


# =====================================================================
# ===== TC02 – Cold Boot =============================================
# =====================================================================

def run_tc02(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = 20
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = make_dir("C:/Temp", "LGC_Perf_TC02", run_idx, ts_run, config)
    print(f"Output directory: {base_dir}")

    print("\n[PRE-CONDITION] LiveTV 36-1 (60s) -> Home (60s)")
    send_key(ser, 'LiveTV', 0.5)
    send_key(ser, 'Num_03', 0.5)
    send_key(ser, 'Num_06', 0.5)
    send_key(ser, 'DASH', 0.5)
    send_key(ser, 'Num_01', 0.5)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(60, "Pre-load Channel")

    send_key(ser, 'Home', 0)
    wait_with_countdown_noKeyInput(60, "Home Screen Wait")

    TC02_REPEAT_COUNT = 5
    results = []
    for cycle_idx in range(1, TC02_REPEAT_COUNT + 1):
        print(f"\n[TC02] Scenario cycle {cycle_idx}/{TC02_REPEAT_COUNT}")

        send_key(ser, 'LiveTV', 0.5)
        send_key(ser, 'Num_03', 0.5)
        send_key(ser, 'Num_06', 0.5)
        send_key(ser, 'DASH', 0.5)
        send_key(ser, 'Num_01', 0.5)
        send_key(ser, 'OK', 2)
        wait_with_countdown_noKeyInput(60, "Pre-load Channel")

        print("\n[AC POWER CYCLE] Power OFF for 60 seconds...")
        run_ac_power_cycle(ip, 60)
        wait_with_countdown_noKeyInput(180, "Power ON Stabilization")

        send_key(ser, 'Home', 1)
        send_key(ser, 'DpadRt', 1)
        send_key(ser, 'OK', 20)

        send_shell_command_with_debug(ser, "stop preload-manager")
        send_shell_command_with_debug(ser,
            "luna-send -n 1 -f luna://com.webos.applicationManager/closeByAppId "
            "'{\"id\": \"com.webos.app.lgchannels\"}'")

        send_key(ser, 'Home', 1)
        wait_with_countdown_noKeyInput(30, "Home wait")
        send_key(ser, 'DpadRt', 2)

        cycle_dir = os.path.join(base_dir, f"cycle_{cycle_idx:02d}")
        os.makedirs(cycle_dir, exist_ok=True)

        result = measure_lgc_home(ser, cap, run_idx, cycle_dir, timeout, config['SoC'], f"TC02 CYCLE {cycle_idx}")
        save_result(result, ts_run, csv_path, 'TC02', extra_cols={'Cycle': cycle_idx})
        if result:
            result['cycle'] = cycle_idx
            results.append(result)

    return results


def run_tc02_between_runs(ser):
    send_key(ser, 'Home', 2)
    send_key(ser, 'Back', 2)
    send_key(ser, 'DpadUp', 2)
    send_key(ser, 'DpadUp', 2)
    send_key(ser, 'OK', 2)


# =====================================================================
# ===== TC03 – Ch Zapping ============================================
# =====================================================================

def run_tc03(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = config['timeout']
    t2_channel = config.get('t2_channel', '450')
    pga_channel = config.get('pga_channel', '449')
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = make_dir("C:/Temp", "LGC_Perf_TC03", run_idx, ts_run, config)
    print(f"Output directory: {base_dir}")

    print("\n[PRE-CONDITION] LiveTV 36-1 (60 s) -> AC off (60 s) -> AC on (180 s) -> Home PIP Live TV "
          "-> 11-1 NBC (180 s) -> T2")
    send_key(ser, 'LiveTV', 3)
    for key_name in channel_to_keys('36-1'):
        send_key(ser, key_name, 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(60, "LiveTV 36-1 (Fox News)")

    print("\n[AC POWER CYCLE] Power OFF for 60 seconds...")
    run_ac_power_cycle(ip, 60)
    wait_with_countdown_noKeyInput(180, "Power ON Stabilization")

    print("\n[STEP] Home -> PIP Live TV")
    send_key(ser, 'Home', 3)
    lgc_similarity.navigate_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                      'tc03_pip_live_tv', "the Live TV (PIP) on Home", ['Down'])
    wait_with_countdown_noKeyInput(10, "Live TV")

    print("\n[STEP] Channel 11-1 (NBC)")
    for key_name in channel_to_keys('11-1'):
        send_key(ser, key_name, 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(180, "NBC 11-1")

    print(f"\n[STEP] Channel {t2_channel} (T2 Tennis)")
    go_to_channel(ser, t2_channel, delay=1)
    wait_with_countdown_noKeyInput(10, "T2 Tennis")

    # One-time per SoC: TV screen box and the landing area (one unmeasured ChDown, then back to T2).
    prepare_channel_landing(ser, cap, config)

    TC03_REPEAT_COUNT = 5
    results = []
    for cycle_idx in range(1, TC03_REPEAT_COUNT + 1):
        print(f"\n[TC03] Scenario cycle {cycle_idx}/{TC03_REPEAT_COUNT}")
        send_key(ser, 'ChDown', 0)
        wait_with_countdown_noKeyInput(10, f"PGA {pga_channel}")

        print(f"[TC03] Measuring ChUp: {pga_channel} -> {t2_channel}")
        cycle_dir = os.path.join(base_dir, f"cycle_{cycle_idx:02d}")
        os.makedirs(cycle_dir, exist_ok=True)
        result = perform_motion_detection(
            ser, cap, run_idx, cycle_dir, timeout, config, trigger_key='ChUp'
        )
        save_result(result, ts_run, csv_path, 'TC03',
                    extra_cols={'Cycle': cycle_idx, 'From_Ch': pga_channel, 'To_Ch': t2_channel})
        if result:
            result['cycle'] = cycle_idx
            result['pga_channel'] = pga_channel
            results.append(result)
        wait_with_countdown_noKeyInput(5, f"T2 Tennis {t2_channel}")

    return results


# =====================================================================
# ===== TC04 – Ch Num Press ==========================================
# =====================================================================

TC04_SOURCE_CHANNEL       = "11-1"
TC04_TARGET_CHANNEL       = "450"
TC04_USE_OK_TO_CONFIRM    = True
TC04_REPEAT_COUNT         = 5


def run_tc04(ser, cap, config, run_idx, csv_path):
    """LiveTV 36-1 (60 s) -> AC off (60 s) -> AC on (180 s) -> Home PIP Live TV, then 5x:
    11-1 NBC (180 s) -> number keys of T2 (IP450) + OK; OK starts the timer, the new channel cutting in ends it."""
    ip      = config['ip']
    timeout = config['timeout']
    source  = TC04_SOURCE_CHANNEL
    target  = config.get('t2_channel', TC04_TARGET_CHANNEL)
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = make_dir("C:/Temp", "LGC_Perf_TC04", run_idx, ts_run, config)
    print(f"Output directory: {base_dir}")

    print("\n[PRE-CONDITION] LiveTV 36-1 (60 s) -> AC off (60 s) -> AC on (180 s) -> Home PIP Live TV")
    send_key(ser, 'LiveTV', 3)
    for key_name in channel_to_keys('36-1'):
        send_key(ser, key_name, 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(60, "LiveTV 36-1 (Fox News)")

    print("\n[AC POWER CYCLE] Power OFF for 60 seconds...")
    run_ac_power_cycle(ip, 60)
    wait_with_countdown_noKeyInput(180, "Power ON Stabilization")

    print("\n[STEP] Home -> PIP Live TV")
    send_key(ser, 'Home', 3)
    lgc_similarity.navigate_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                      'tc03_pip_live_tv', "the Live TV (PIP) on Home", ['Down'])
    wait_with_countdown_noKeyInput(10, "Live TV")

    # One-time per SoC: TV screen box and how the channel changes (one unmeasured ChDown, then ChUp back).
    prepare_channel_landing(ser, cap, config)

    results = []
    for cycle_idx in range(1, TC04_REPEAT_COUNT + 1):
        print(f"\n[TC04] Scenario cycle {cycle_idx}/{TC04_REPEAT_COUNT}")
        print(f"[STEP] Channel {source} (NBC)")
        go_to_channel(ser, source, delay=1)
        wait_with_countdown_noKeyInput(180, f"NBC {source}")

        print(f"[TC04] Number keys {target} (T2 Tennis), OK is measured")
        _, trigger_key = prepare_channel_input(ser, target, confirm_with_ok=TC04_USE_OK_TO_CONFIRM)
        cycle_dir = os.path.join(base_dir, f"cycle_{cycle_idx:02d}")
        os.makedirs(cycle_dir, exist_ok=True)
        result = perform_motion_detection(ser, cap, run_idx, cycle_dir, timeout, config, trigger_key,
                                          mode='channel')
        save_result(result, ts_run, csv_path, 'TC04',
                    extra_cols={'Cycle': cycle_idx, 'From_Ch': source, 'To_Ch': target})
        if result:
            result['source_channel'] = source
            result['target_channel'] = target
            result['cycle'] = cycle_idx
            results.append(result)
        wait_with_countdown_noKeyInput(10, f"T2 Tennis {target}")

    return results


# =====================================================================
# ===== TC05_NtN – Native to Native ==================================
# =====================================================================

TC05_NTN_REPEAT_COUNT = 5
TC05_APP_ROUTE = ['Right']                                          # Home -> LG Channels (default keys)
TC05_LIVE_ROUTE = ['Left', 'Left', 'Left', 'Left', 'Down', 'Down']   # LG Channels home -> LIVE (default keys)


def open_lg_channels_home(ser, soc):
    """Home -> the saved route to LG Channels + OK, waiting until its home has loaded (route planning)."""
    send_key(ser, 'Home', 3)
    lgc_similarity.replay_route(lambda key, delay: send_key(ser, key, delay), soc, 'tc05_lgc_app', TC05_APP_ROUTE)
    send_key(ser, 'OK', 2)
    time.sleep(13)


TC05_KINDS = {'TC05_NtN': ('Native', 'Native'), 'TC05_NtP': ('Native', 'Pluto'), 'TC05_PtP': ('Pluto', 'Pluto')}
TC05_REPEAT = {'TC05_NtN': TC05_NTN_REPEAT_COUNT, 'TC05_NtP': 5, 'TC05_PtP': 5}
TC05_PAIR_KEY = {'TC05_NtN': 'ntn_pair', 'TC05_NtP': 'ntp_pair', 'TC05_PtP': 'ptp_pair'}
OPPOSITE_KEY = {'ChUp': 'ChDown', 'ChDown': 'ChUp'}
TC05_AFTER_MEASURE_SECONDS = 5   # after each measurement: the channel plays before the next keys
TC05_DIGIT_GAP = 0.5               # seconds between the digits of a channel number in LG Channels


def _tc05_live_precondition(ser, cap, config, first_channel):
    """LiveTV 36-1 (60 s) -> AC off (60 s) -> AC on (180 s) -> LG Channels -> LIVE -> first_channel (10 s)."""
    print("\n[PRE-CONDITION] LiveTV 36-1 (60 s) -> AC off (60 s) -> AC on (180 s) -> LG Channels -> LIVE "
          f"-> {first_channel}")
    send_key(ser, 'LiveTV', 3)
    for key_name in channel_to_keys('36-1'):
        send_key(ser, key_name, 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(60, "LiveTV 36-1 (Fox News)")

    print("\n[AC POWER CYCLE] Power OFF for 60 seconds...")
    run_ac_power_cycle(config['ip'], 60)
    wait_with_countdown_noKeyInput(180, "Power ON Stabilization")

    print("\n[STEP] Home -> LG Channels")
    send_key(ser, 'Home', 3)
    lgc_similarity.navigate_from_home(lambda k, delay: send_key(ser, k, delay), cap, config['SoC'],
                                      'tc05_lgc_app', "LG Channels on Home", TC05_APP_ROUTE)
    wait_with_countdown_noKeyInput(15, "LG Channels")

    print("\n[STEP] LIVE")
    lgc_similarity.navigate_from_home(lambda k, delay: send_key(ser, k, delay), cap, config['SoC'],
                                      'tc05_lgc_live', "LIVE in LG Channels", TC05_LIVE_ROUTE,
                                      open_start=lambda: open_lg_channels_home(ser, config['SoC']))
    wait_with_countdown_noKeyInput(10, "LIVE")

    print(f"\n[STEP] Channel {first_channel}")
    send_key(ser, 'Num_00', 1.5)   # throwaway digit: the first key lands before the entry field is ready
    go_to_channel(ser, first_channel, delay=TC05_DIGIT_GAP)
    wait_with_countdown_noKeyInput(10, f"Channel {first_channel}")


TC05_BLACK_AREA = ('TC05_NtP', 'TC05_PtP')   # Pluto shows a black / logo screen first: judged by the black area


def show_tc05_channels(label, pair, from_kind, to_kind):
    """The channels a TC05 case uses, checked on the lineup (printed before it runs)."""
    lgc_lineup.show(label, pair, from_kind, to_kind)


def prepare_black_area(ser, cap, config, label, key, src):
    """Once per SoC: one unmeasured key from the From channel, the user picks the black / logo screen and the
    black area; back on the From channel with the opposite key."""
    def back():
        send_key(ser, OPPOSITE_KEY[key], 0)
        wait_with_countdown_noKeyInput(10, f"Back on {src}")
    lgc_similarity.ensure_black_area(cap, config['SoC'], label.lower(), f"{label} {key}",
                                     lambda: send_key(ser, key, 0), back)


def _run_tc05(ser, cap, config, run_idx, csv_path, label):
    """One TC05 case with the pair chosen from the lineup: From --key--> To is measured 5 times.
    The From channel is entered by number once (after LIVE); every later cycle returns to it with the
    opposite key (ChUp <-> ChDown), 10 s, then the key into To is measured."""
    timeout = config['timeout']
    pair = config[TC05_PAIR_KEY[label]]
    src, key, dst = pair['from'], pair['key'], pair['to']
    from_kind, to_kind = TC05_KINDS[label]
    show_tc05_channels(label, pair, from_kind, to_kind)       # the channels used, checked on the lineup
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = make_dir("C:/Temp", f"LGC_Perf_{label}", run_idx, ts_run, config)
    print(f"Output directory: {base_dir}")

    _tc05_live_precondition(ser, cap, config, src)
    black_area = label in TC05_BLACK_AREA
    if black_area:
        # One-time per SoC: the black area of the black / logo screen (one unmeasured key, then back).
        prepare_black_area(ser, cap, config, label, key, src)
    else:
        # One-time per SoC: TV screen box and how the channel changes (one unmeasured ChDown, then ChUp back).
        prepare_channel_landing(ser, cap, config)

    results = []
    for cycle_idx in range(1, TC05_REPEAT[label] + 1):
        print(f"\n[{label}] Scenario cycle {cycle_idx}/{TC05_REPEAT[label]}")
        if cycle_idx > 1:
            send_key(ser, OPPOSITE_KEY[key], 0)            # back to From: ChUp -> ChDown, ChDown -> ChUp
        wait_with_countdown_noKeyInput(10, f"{from_kind} {src}")

        print(f"[{label}] Measuring {key}: {from_kind} {src} -> {to_kind} {dst}")
        cycle_dir = os.path.join(base_dir, f"cycle_{cycle_idx:02d}")
        os.makedirs(cycle_dir, exist_ok=True)
        result = perform_motion_detection(ser, cap, run_idx, cycle_dir, timeout, config, trigger_key=key,
                                          mode='area' if black_area else None,
                                          target=label.lower() if black_area else None)
        save_result(result, ts_run, csv_path, label,
                    extra_cols={'Cycle': cycle_idx, 'From_Ch': src, 'To_Ch': dst, 'Key': key})
        if result:
            result['cycle'] = cycle_idx
            result['source_channel'] = src
            result['target_channel'] = dst
            results.append(result)
        # Let the new channel finish starting before the next key.
        wait_with_countdown_noKeyInput(TC05_AFTER_MEASURE_SECONDS, f"{to_kind} {dst} playing")

    return results


def run_tc05_ntn(ser, cap, config, run_idx, csv_path):
    """Native -> Native (e.g. PGA TOUR 442 --ChUp--> T2 443), channels from the lineup."""
    return _run_tc05(ser, cap, config, run_idx, csv_path, 'TC05_NtN')


def run_tc05_ntp(ser, cap, config, run_idx, csv_path):
    """Native -> Pluto with one ChUp / ChDown, channels from the lineup."""
    return _run_tc05(ser, cap, config, run_idx, csv_path, 'TC05_NtP')


def run_tc05_ptp(ser, cap, config, run_idx, csv_path):
    """Pluto -> Pluto with one ChUp / ChDown, channels from the lineup."""
    return _run_tc05(ser, cap, config, run_idx, csv_path, 'TC05_PtP')


# =====================================================================
# ===== TC06/07/08 – Movies and TV Scroll ============================
# =====================================================================

def run_tc06_07_08(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
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
    send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    send_key(ser, 'DpadRt', 2)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(30, "LG Channels Loading")

    print("\n[STEP 5] Navigating to Movies and TV...")
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadLt', 2)
    send_key(ser, 'DpadDn', 1)

    # TC06 – OK to select Movies and TV
    r1 = perform_motion_detection(ser, cap, run_idx, dir_path, timeout=25, config=config, trigger_key='OK',
                                  target='MovTV')
    if r1:
        r1['source_channel'] = 'LG Channels'
        r1['target_channel'] = 'Movies and TV'
        results.append(r1)
        save_result(r1, ts_run, csv_path, 'TC06',
                    extra_cols={'Source_Channel': 'LG Channels', 'Target_Channel': 'Movies and TV'})

    # TC07 – scroll down
    dir_scroll = dir_path + "_Scroll"


    os.makedirs(dir_scroll, exist_ok=True)
    r2 = perform_motion_detection(ser, cap, run_idx, dir_scroll, timeout=5, config=config, trigger_key='DpadDn')
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
    r3 = perform_motion_detection(ser, cap, run_idx, dir_scroll_rt, timeout=5, config=config, trigger_key='DpadRt')
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

TC10_REPEAT_COUNT = 5


def run_tc10(ser, cap, config, run_idx, csv_path):
    ip      = config['ip']
    timeout = config['timeout']
    ts_run  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = make_dir("C:/Temp", "LGC_Perf_TC10", run_idx, ts_run, config)
    print(f"Output directory: {dir_path}")

    print("\n[SETUP] Installing/launching Amazon...")
    send_key(ser, 'Home', 2);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Amazon'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Amazon',
                         legacy=lambda: (send_key(ser, 'OK', 2), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                         send_key(ser, 'OK', 2), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(60, "Amazon Video Playback")

    print("\n[SETUP] Installing/launching Netflix...")
    send_key(ser, 'Home', 2);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    wait_for_app_ready(cap, motion_timeout=3, stable_timeout=7)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix',
                         legacy=lambda: (send_key(ser, 'OK', 3), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                         send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

    print("\n[SETUP] Launching YouTube...")
    send_key(ser, 'Home', 2);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube',
                         legacy=lambda: (send_key(ser, 'OK', 2), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                         send_key(ser, 'OK', 2), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(60, "YouTube Video Playback")

    results = []
    for cycle_idx in range(1, TC10_REPEAT_COUNT + 1):
        print(f"\n[TC10] Scenario cycle {cycle_idx}/{TC10_REPEAT_COUNT}")
        cycle_dir = os.path.join(dir_path, f"cycle_{cycle_idx:02d}")
        os.makedirs(cycle_dir, exist_ok=True)

        print("\n[STEP 1] Live TV Fox 36-1...")
        send_key(ser, 'Exit', 2)
        send_key(ser, 'LiveTV', 5);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
        send_key(ser, 'Num_03', 1)
        send_key(ser, 'Num_06', 1)
        send_key(ser, 'DASH', 1)
        send_key(ser, 'Num_01', 1)
        send_key(ser, 'OK', 5)
        wait_with_countdown_noKeyInput(60, "Live TV Fox 36-1")

        print("\n[STEP 2] DC Power Cycle (OFF 2 min)...")
        send_key(ser, 'P_OFF', 0)
        wait_with_countdown_noKeyInput(120, "DC Power cycle for 2m")
        send_key(ser, 'P_ON', 0)
        wait_with_countdown_noKeyInput(60, "DC Power cycle for 1m")

        print("\n[STEP 4] Launching Amazon...")
        send_key(ser, 'Home', 2);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
        lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Amazon'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
        lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Amazon',
                             legacy=lambda: (send_key(ser, 'OK', 2), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                             send_key(ser, 'OK', 2), wait_for_screen_stable(cap, timeout=30)))
        wait_with_countdown_noKeyInput(60, "Amazon Video Playback")

        print("\n[STEP 5] Launching Netflix...")
        send_key(ser, 'Home', 2);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
        lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
        lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix',
                             legacy=lambda: (send_key(ser, 'OK', 3), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                             send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30)))
        wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

        print("\n[STEP 6] Launching YouTube...")
        send_key(ser, 'Home', 2);wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
        lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
        lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube',
                             legacy=lambda: (send_key(ser, 'OK', 2), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                             send_key(ser, 'OK', 2), wait_for_screen_stable(cap, timeout=30)))
        wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

        print("\n[STEP 7] Returning to Home...")
        send_key(ser, 'Home', 2)
        wait_with_countdown_noKeyInput(30, "Home Screen")

        print("\n[STEP 8] Entering LG Channels - measuring load time...")
        lgc_apps.focus_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'LG Channels')

        result = perform_motion_detection(ser, cap, run_idx, cycle_dir, timeout, config, trigger_key='OK')
        if result:
            result['cycle'] = cycle_idx
            results.append(result)
        save_result(result, ts_run, csv_path, 'TC10', extra_cols={'Cycle': cycle_idx})

    return results


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

    wait_with_countdown_noKeyInput(60, "Boot Stabilization")

    print("\n[STEP 1] Live TV Fox 36-1...")
    send_key(ser, 'Exit', 2); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    send_key(ser, 'LiveTV', 5)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(60, "Live TV Fox 36-1")

    print("\n[STEP 5] Launching Netflix...")
    send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix',
                         legacy=lambda: (send_key(ser, 'OK', 3), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                         send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

    print("\n[STEP 6] Launching YouTube...")
    send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube'); wait_for_app_ready(cap, motion_timeout=20, stable_timeout=60)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube',
                         legacy=lambda: (send_key(ser, 'OK', 3), wait_for_app_ready(cap, motion_timeout=10, stable_timeout=30),
                                         send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

    print("\n[STEP 7] Returning to Home...")
    send_key(ser, 'Home', 2); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    wait_with_countdown_noKeyInput(30, "Home Screen")

    print("\n[STEP 8] Before PIP is on...")
    lgc_apps.focus_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Live TV PIP')
    send_key(ser, 'OK', 3)
    

    print("\n[STEP 9] Entering LG Channels - measuring load time...")
    send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.focus_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'LG Channels')

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

    wait_with_countdown_noKeyInput(60, "Boot Stabilization")

    print("\n[STEP 1] Live TV Fox 36-1...")
    send_key(ser, 'Exit', 2); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    send_key(ser, 'LiveTV', 5)
    send_key(ser, 'Num_03', 1)
    send_key(ser, 'Num_06', 1)
    send_key(ser, 'DASH', 1)
    send_key(ser, 'Num_01', 1)
    send_key(ser, 'OK', 5)
    wait_with_countdown_noKeyInput(60, "Live TV Fox 36-1")

    print("\n[STEP 5] Launching Netflix...")
    send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix'); wait_for_screen_stable(cap, timeout=45)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Netflix',
                         legacy=lambda: (send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30),
                                         send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(60, "Netflix Video Playback")

    print("\n[STEP 6] Launching YouTube...")
    send_key(ser, 'Home', 3); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.open_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube'); wait_for_screen_stable(cap, timeout=45)
    lgc_apps.play_in_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'YouTube',
                         legacy=lambda: (send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30),
                                         send_key(ser, 'OK', 3), wait_for_screen_stable(cap, timeout=30)))
    wait_with_countdown_noKeyInput(180, "YouTube Video Playback")

    print("\n[STEP 7] Returning to Home...")
    send_key(ser, 'Home', 2)
    wait_with_countdown_noKeyInput(60, "Home Screen")

    print("\n[STEP 8] PIP → Full Screen...")
    lgc_apps.focus_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'Live TV PIP')
    wait_with_countdown_noKeyInput(50, "PIP to Alive")
    send_key(ser, 'OK', 3)
    wait_with_countdown_noKeyInput(10, "PIP to Full Screen")

    print("\n[STEP 9] Entering LG Channels - measuring load time...")
    send_key(ser, 'Home', 2); wait_for_app_ready(cap, motion_timeout=5, stable_timeout=10)
    lgc_apps.focus_app(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'], 'LG Channels')

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
        ip = input("Enter AC Smart Plug IP Address (default: 192.168.4.214): ").strip()
        if not ip:
            ip = "192.168.4.214"; break
        if len(ip.split('.')) == 4:
            break
        print("❌ Invalid IP format.")

    while True:
        port = input("Enter Serial Port (default: COM13): ").strip()
        if not port:
            port = "COM13"; break
        if port.upper().startswith("COM"):
            break
        print("❌ Invalid port format.")

    SoC  = input("Enter SoC Model (default: K25Lp): ").strip()  or "K25Lp"
    SWV  = input("Enter Software Version (default: 33.31.61): ").strip() or "33.31.61"
    LGCV = input("Enter LG CV Version (default: 4.0.18-9): ").strip()   or "4.0.18-9"

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

    if 'TC03' in selected_tcs or 'TC04' in selected_tcs:
        t2_channel = input("T2 Tennis Channel (default: 450): ").strip() or "450"
        pga_channel = input("PGA Channel (default: 449): ").strip() or "449"
        config['t2_channel'] = validate_channel_value(t2_channel, "450")
        config['pga_channel'] = validate_channel_value(pga_channel, "449")


    # TC05: the channels come from the LG Channels lineup (channel server); the check is shown and confirmed
    if 'TC05_NtN' in selected_tcs:
        config['ntn_pair'] = lgc_lineup.choose_pair(input, 'TC05_NtN', 'Native', 'Native', prefer=('T2', 'PGA'))
    if 'TC05_NtP' in selected_tcs:
        config['ntp_pair'] = lgc_lineup.choose_pair(input, 'TC05_NtP', 'Native', 'Pluto')

    if 'TC05_PtP' in selected_tcs:
        config['ptp_pair'] = lgc_lineup.choose_pair(input, 'TC05_PtP', 'Pluto', 'Pluto')

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
    # ===== Camera Test (optional) =====
    do_cam_test = input("Run camera test? (y/N): ").strip().lower()
    if do_cam_test == 'y':
        print("\n" + "="*60)
        print("  [CAMERA TEST]")
        print("="*60)
        print("  Adjust camera so the TV screen is fully visible.")
        print("    SPACE  ->  Confirm and proceed")
        print("    Q      ->  Quit")
        print("="*60 + "\n")

    cap = initialize_camera()
    suite_progress.state['camera'] = cap

    if do_cam_test == 'y':
        prev_gray = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera frame read failed.")
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
            frame_count += 1

            display = frame.copy()
            diff_score = 0.0
            mean_val   = np.mean(curr_gray)

            if prev_gray is not None:
                diff_score = np.mean(cv2.absdiff(prev_gray, curr_blur))

            h = display.shape[0]
            cv2.putText(display, f"Diff: {diff_score:.2f}  Mean: {mean_val:.1f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display, "SPACE: Confirm  |  Q: Quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("[CAMERA TEST] Adjust framing, then press SPACE", display)

            if frame_count % 30 == 0:
                print(f"  ⏱ Camera live | Diff: {diff_score:.2f}  Mean: {mean_val:.1f}  (SPACE to confirm)", end="\r")

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nCamera test cancelled. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            if key == ord(' '):
                break

            prev_gray = curr_blur

        cv2.destroyAllWindows()
        print("\n✓ Camera position confirmed.\n")
    # ===== End Camera Test =====

    selected_tcs = select_tcs()
    if not selected_tcs:
        print("No TCs selected. Exiting.")
        sys.exit(0)

    config = get_user_configuration(selected_tcs)
    quick = suite_progress.ask_quick()

    num_runs = config['num_runs']
    port     = config['port']

    ser = initialize_serial(port)

    if 'TC03' in selected_tcs or 'TC04' in selected_tcs:
        verify_t2_pga_channel_settings(config)

    if 'TC01' in selected_tcs or 'TC02' in selected_tcs:
        # TC01 / TC02 end = similarity with the LG Channels home (reference and TV screen area saved per SoC)
        lgc_similarity.prepare(cap, config['SoC'], lambda: open_lg_channels(ser))

    if 'TC03' in selected_tcs or 'TC04' in selected_tcs:
        # TC03 / TC04 enter Live TV from Home ~4 min into the run (after the AC cycle): ask the route now
        lgc_similarity.plan_route_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                        'tc03_pip_live_tv', "the Live TV (PIP) on Home", ['Down'])
    real_world = [label for label in ('TC10', 'TC11', 'TC12') if label in selected_tcs]
    if real_world:
        # TC10-12: every app they use is opened once by the navigator and confirmed installed / signed in
        lgc_apps.check_apps(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                            lgc_apps.apps_for(real_world))
    if any(label in selected_tcs for label in ('TC05_NtN', 'TC05_NtP', 'TC05_PtP')):
        # TC05_NtN / NtP open LG Channels, then LIVE, ~4 min into the run: check both routes now
        lgc_similarity.plan_route_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                            'tc05_lgc_app', "LG Channels on Home", TC05_APP_ROUTE)
        lgc_similarity.plan_route_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                            'tc05_lgc_live', "LIVE in LG Channels", TC05_LIVE_ROUTE,
                                            open_start=lambda: open_lg_channels_home(ser, config['SoC']),
                                            start_name='LG Channels')

    ts_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"C:/Temp/LGC_FullSuite_{ts_session}{'_QUICK' if quick else ''}.csv"
    os.makedirs("C:/Temp", exist_ok=True)

    all_results = []
    suite_progress.state['camera'] = cap
    lgc_similarity.STATUS = suite_progress.status_lines   # TC / step line in the measurement windows

    for tc_number, tc_label in enumerate(selected_tcs, 1):
        tc_runs = 1 if tc_label in ('TC01', 'TC02', 'TC03', 'TC04', 'TC05_NtN', 'TC05_NtP', 'TC05_PtP', 'TC10') else num_runs
        suite_progress.begin_tc(tc_number, len(selected_tcs), tc_label, tc_runs, cap)

        # One-time setup if needed
        if tc_label in TC_SETUP_MAP:
            suite_progress.begin_run(1, suite_progress.plan_steps(
                sys.modules[__name__], TC_SETUP_MAP[tc_label], (ser, cap, config)), phase='Setup')
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
        for run_idx in range(1, tc_runs + 1):
            print(f"\n{'='*60}")
            print(f"  [{tc_label}] RUN {run_idx}/{tc_runs}")
            print(f"{'='*60}")
            suite_progress.begin_run(run_idx, suite_progress.plan_steps(
                sys.modules[__name__], run_fn, (ser, cap, config, run_idx, csv_path),
                [(lgc_similarity, 'navigate_from_home', lambda *a, **k: None)]))
            try:
                result = run_fn(ser, cap, config, run_idx, csv_path)
                if isinstance(result, list):
                    tc_results.extend([r for r in result if r])
                elif result:
                    tc_results.append(result)
            except Exception as e:
                print(f"❌ Error in {tc_label} run {run_idx}: {e}")

            if run_idx < tc_runs:
                wait_with_countdown_noKeyInput(10, "Interval between runs")
                if tc_label in TC_BETWEEN_RUNS_MAP:
                    try:
                        TC_BETWEEN_RUNS_MAP[tc_label](ser)
                    except Exception as e:
                        print(f"⚠ Between-run action failed: {e}")

        all_results.extend(tc_results)
        suite_progress.sound('tc_done')

        # Per-TC summary
        passed = sum(1 for r in tc_results if r.get('detected'))
        print(f"\n[{tc_label}] Runs: {len(tc_results)} | Pass: {passed} | Fail: {len(tc_results)-passed}")
        if passed:
            avg = np.mean([r['response_time_ms'] for r in tc_results if r.get('detected')])
            print(f"[{tc_label}] Average response time: {avg:.2f}ms")

    # ===== Overall Summary =====
    print(f"\n{'='*60}")
    print(f"  ===  FULL SUITE COMPLETE  ===")
    suite_progress.sound('all_done')
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
