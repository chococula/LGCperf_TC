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
import lgc_lineup
import suite_progress

# The last answer to each "(default: ...)" prompt is the next default (saved in LGC_TC5_NtN_inputs.json).
input = remembered_input("LGC_TC5_NtN_inputs.json")

# ===== Configuration Input =====
def get_user_configuration():
    """
    Interactive configuration input from user
    """
    print("\n" + "="*60)
    print("  LG TV Native to Native Channel Change Test - Configuration")
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
        print("❌ Invalid port format. Use format like COM4.")
    
    # SoC
    SoC = input("Enter SoC Model (default: O22N3): ").strip() or "O22N3"
    
    # Software Version
    SWV = input("Enter Software Version (default: 33.31.22): ").strip() or "33_30_97"
    
    # LG CV
    LGCV = input("Enter LG CV Version (default: 4.0.18-1): ").strip() or "4_0_7-2"
    
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
            timeout = float(input("Enter timeout for motion detection in seconds (default: 20): ").strip() or "20")
            if timeout > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    config = {
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
    
    confirm = input("Confirm and proceed? (Y/n): ").strip().lower()
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
        suite_progress.before_key(key_name)
        ser.write(IR_KEYS[key_name])
        ser.flush()
        time.sleep(delay)
        # after Num keys + OK: check on the camera that the channel really changed (quick check: confirm)
        suite_progress.after_key(key_name, lambda key, wait: send_key(ser, key, wait))
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


def initialize_camera(camera_index=1):
    # the configured index first, then the others; only a camera that streams real pictures is used
    return open_camera(camera_index)


def perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key='ChUp', mode=None,
                             target=None):
    """End detection like the Omedia TCs (lgc_similarity.detect): CH+/CH- = the screen goes black, then the
    picture lands; screens = similarity with a saved reference (+ movement stopped); scrolls = movement stopped."""
    def press():
        winsound.Beep(800, 200)
        sent_at = time.perf_counter()
        send_key(ser, trigger_key, 0)
        return sent_at
    return lgc_similarity.detect(cap, press, lambda: send_key(ser, trigger_key, 0), config['SoC'], run_idx,
                                 dir_path, timeout, trigger_key, mode, target)


# ===== Main Execution =====

CHANNEL_KEY_MAP = {str(d): f'Num_0{d}' for d in range(10)}
CHANNEL_KEY_MAP['-'] = 'DASH'


def channel_to_keys(channel):
    return [CHANNEL_KEY_MAP[char] for char in str(channel)]


def go_to_channel(ser, channel, delay=1):
    print(f"  > Navigating to channel {channel}")
    for key_name in channel_to_keys(channel):
        send_key(ser, key_name, delay)
    send_key(ser, 'OK', 0)


APP_ROUTE = ['Right']                                          # Home -> LG Channels (default keys)
LIVE_ROUTE = ['Left', 'Left', 'Left', 'Left', 'Down', 'Down']   # LG Channels home -> LIVE (default keys)


def open_lg_channels_home(ser, soc):
    """Home -> the saved route to LG Channels + OK, waiting until its home has loaded (route planning)."""
    send_key(ser, 'Home', 3)
    lgc_similarity.replay_route(lambda key, delay: send_key(ser, key, delay), soc, 'tc05_lgc_app', APP_ROUTE)
    send_key(ser, 'OK', 2)
    time.sleep(13)


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
    suite_progress.state['camera'] = cap
    suite_progress.begin_tc(1, 1, "TC5_NtN", num_runs, cap)

    # LG Channels, then LIVE, are opened ~4 min into the run (after the AC cycle): check both routes now
    lgc_similarity.plan_route_from_home(lambda key, delay: send_key(ser, key, delay), cap, SoC,
                                        'tc05_lgc_app', "LG Channels on Home", APP_ROUTE)
    lgc_similarity.plan_route_from_home(lambda key, delay: send_key(ser, key, delay), cap, SoC,
                                        'tc05_lgc_live', "LIVE in LG Channels", LIVE_ROUTE,
                                        open_start=lambda: open_lg_channels_home(ser, SoC), start_name='LG Channels')
    
    # CSV path
    quick = suite_progress.ask_quick()
    csv_path = "tv_response_QUICK.csv" if quick else "tv_response.csv"
    results = []
    
    print(f"\n{'='*60}")
    print(f"  Starting {num_runs} Test Runs")
    print(f"{'='*60}\n")
    
    # Native -> Native one key apart, from the lineup (e.g. PGA TOUR 442 --ChUp--> T2 443)
    pair = lgc_lineup.choose_pair(input, 'TC05_NtN', 'Native', 'Native', prefer=('T2', 'PGA'))
    pga_channel, t2_channel, measure_key = pair['from'], pair['to'], pair['key']
    back_key = 'ChDown' if measure_key == 'ChUp' else 'ChUp'

    # the channels used, checked on the lineup, shown again right before the run
    lgc_lineup.show('TC05_NtN', pair, 'Native', 'Native')

    print("\n[PRE-CONDITION] LiveTV 36-1 (60 s) -> AC off (60 s) -> AC on (180 s) -> LG Channels -> LIVE -> To channel")
    send_key(ser, 'LiveTV', 3)
    for key_name in channel_to_keys('36-1'):
        send_key(ser, key_name, 1)
    send_key(ser, 'OK', 2)
    wait_with_countdown_noKeyInput(60, "LiveTV 36-1 (Fox News)")

    print("\n[AC POWER CYCLE] Power OFF for 60 seconds...")
    asyncio.run(run_ac_power_cycle(ip, 60))
    wait_with_countdown_noKeyInput(180, "Power ON Stabilization")

    print("\n[STEP] Home -> LG Channels")
    send_key(ser, 'Home', 3)
    lgc_similarity.navigate_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                      'tc05_lgc_app', "LG Channels on Home", APP_ROUTE)
    wait_with_countdown_noKeyInput(15, "LG Channels")

    print("\n[STEP] LIVE")
    lgc_similarity.navigate_from_home(lambda key, delay: send_key(ser, key, delay), cap, config['SoC'],
                                      'tc05_lgc_live', "LIVE in LG Channels", LIVE_ROUTE,
                                      open_start=lambda: open_lg_channels_home(ser, config['SoC']))
    wait_with_countdown_noKeyInput(10, "LIVE")

    print(f"\n[STEP] Channel {t2_channel}")
    send_key(ser, 'Num_00', 1.5)   # throwaway digit: the first key lands before the entry field is ready
    go_to_channel(ser, t2_channel, delay=1)
    wait_with_countdown_noKeyInput(10, f"Native {t2_channel}")

    # One-time per SoC: TV screen box and how the channel changes (one unmeasured ChDown, then ChUp back).
    lgc_similarity.ensure_screen(cap, config['SoC'])
    if not lgc_similarity.has_landing(config['SoC']):
        lgc_similarity.ensure_landing(cap, config['SoC'], lambda: send_key(ser, 'ChDown', 0))
        send_key(ser, 'ChUp', 0)
        wait_with_countdown_noKeyInput(10, f"Back on {t2_channel}")

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
            f"LGC_Perf_TC05_NtN_{run_idx:02d}_{ts_run}_{SoC}_SWV{SWV}_LGCV{LGCV}"
        )
        os.makedirs(dir_path, exist_ok=True)
        print(f"Output directory: {dir_path}")
        
        try:
            send_key(ser, back_key, 0)
            wait_with_countdown_noKeyInput(10, f"Native {pga_channel}")

            # Measured: the key from the From channel back to the To channel
            print(f"[RUN {run_idx}] Measuring {measure_key}: Native {pga_channel} -> Native {t2_channel}")
            result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config, trigger_key=measure_key)
            
            if result:
                results.append(result)
                
                # Save to CSV
                if result['detected']:
                    df = pd.DataFrame({
                        'Run': [run_idx],
                        'Timestamp': [ts_run],
                        'From_Ch': [pga_channel],
                        'To_Ch': [t2_channel],
                        'Key': [measure_key],
                        'Response_Time_ms': [round(result['response_time_ms'], 2)],
                        'Frames': [result['frame_count']],
                        'Status': ['PASS'],
                        'Directory': [result['dir']]
                    })
                    write_header = not os.path.exists(csv_path)
                    df.to_csv(csv_path, mode='a', header=write_header, index=False)
        
        except Exception as e:
            print(f"❌ Error in run {run_idx}: {e}")
        
        # Wait before next run
        if run_idx < num_runs:
            
            wait_with_countdown_noKeyInput(5, f"Native {t2_channel}")

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
