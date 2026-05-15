import cv2
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
        ip = input("Enter Device IP Address (default: 192.168.4.208): ").strip()
        if not ip:
            ip = "192.168.4.208"
            break
        if len(ip.split('.')) == 4:
            break
        print("❌ Invalid IP format. Please try again.")
    
    # Serial Port
    while True:
        port = input("Enter Serial Port (default: COM4): ").strip()
        if not port:
            port = "COM4"
            break
        if port.upper().startswith("COM"):
            break
        print("❌ Invalid port format. Use format like COM4.")
    
    # SoC
    SoC = input("Enter SoC Model (default: O22N3): ").strip() or "O22N3"
    
    # Software Version
    SWV = input("Enter Software Version (default: 33_30_97): ").strip() or "33_30_97"
    
    # LG CV
    LGCV = input("Enter LG CV Version (default: 4_0_7-2): ").strip() or "4_0_7-2"
    
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
    """
    Countdown timer without key input requirement
    """
    winsound.Beep(1000, 200)
    print(f"\n[TIMER] {description}: {seconds} seconds")
    
    for remaining in range(seconds, 0, -1):
        print(f"  ⏱ {remaining} seconds remaining...", end="\r")
        time.sleep(1)
    
    print(" " * 50, end="\r")
    winsound.Beep(1500, 300)
    print(f"[TIMER] ✓ {description} Complete!\n")


async def run_ac_power_cycle(ip, off_seconds):
    """
    AC Power cycle via SmartPlug
    """
    try:
        dev = SmartPlug(ip)
        await dev.update()
        await dev.turn_off()
        print(f"✓ Device powered OFF")
        
        await asyncio.sleep(off_seconds)
        
        await dev.turn_on()
        print(f"✓ Device powered ON")
    except Exception as e:
        print(f"❌ Power cycle failed: {e}")


def initialize_camera(camera_index=1):
    """
    Initialize camera with error handling
    """
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"✓ Camera initialized (index: {camera_index})")
        return cap
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        sys.exit(1)


def perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config):
    """
    Perform motion detection measurement
    """
    print(f"\n[RUN {run_idx}] Motion detection starting...")
    
    # Get baseline frame
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to read initial frame. Skipping run.")
        return None
    
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    motion_detected = False
    frame_count = 0
    elapsed_ms = 0.0
    
    # Skip first few frames
    #for _ in range(3):
    #    cap.grab()
    
    # Send OK command to trigger action
    send_key(ser, 'OK', 0)
    ser.flush()
    start_time = time.perf_counter()
    
    print(f"[RUN {run_idx}] Monitoring for motion...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Frame read failed.")
            break
        
        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # Calculate motion metrics
        diff = cv2.absdiff(prev_gray, curr_gray_blur)
        diff_score = np.mean(diff)
        mean_val = np.mean(curr_gray)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Display info on frame
        top_text = f"Time: {elapsed_ms:.2f}ms | Diff: {diff_score:.2f} | Frame: {frame_count}"
        cv2.putText(curr_gray, top_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,), 1)
        
        # Save frame
        file_name = os.path.join(dir_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(file_name, curr_gray)
        
        # Check for motion detection
        if diff_score > MOTION_THRESHOLD and mean_val > GRAY_THRESHOLD:
            motion_detected = True
            cv2.putText(curr_gray, "✓ MOTION DETECTED!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,), 2)
            cv2.imwrite(os.path.join(dir_path, "RESULT_HIT.jpg"), curr_gray)
            print(f"✓ [RUN {run_idx}] Motion detected! Response time: {elapsed_ms:.2f}ms")
            break
        
        # Timeout check
        if (time.perf_counter() - start_time) > timeout:
            print(f"❌ [RUN {run_idx}] Timeout: No motion in {timeout} seconds")
            break
        
        # Display
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
        'dir': dir_path
    }


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
    cap = initialize_camera(1)
    
    # CSV path
    csv_path = "tv_response.csv"
    results = []
    
    print(f"\n{'='*60}")
    print(f"  Starting {num_runs} Test Runs")
    print(f"{'='*60}\n")
    
    # Main test loop
    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"  [RUN {run_idx}/{num_runs}] Starting")
        print(f"{'='*60}")
        
        # Create run directory
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join(
            "C:/Temp",
            f"LGC_Perf_TC01_{run_idx:02d}_{ts_run}_{SoC}_SWV{SWV}_LGCV{LGCV}"
        )
        os.makedirs(dir_path, exist_ok=True)
        print(f"Output directory: {dir_path}")
        
        try:
            # Pre-conditions
            print("\n[PRE-CONDITION] Setting up test environment...")
            send_key(ser, 'LiveTV', 3)
            send_key(ser, 'Num_03', 1)
            send_key(ser, 'Num_06', 1)
            send_key(ser, 'DASH', 1)
            send_key(ser, 'Num_01', 1)
            send_key(ser, 'OK', 2)
            wait_with_countdown_noKeyInput(30, "Pre-load Channel")


            # Home for 1 min
            send_key(ser, 'Home', 0)
            wait_with_countdown_noKeyInput(60, "Power Stabilization")


            # Pre-conditions
            print("\n[PRE-CONDITION] Setting up test environment...")
            send_key(ser, 'LiveTV', 3)
            send_key(ser, 'Num_03', 1)
            send_key(ser, 'Num_06', 1)
            send_key(ser, 'DASH', 1)
            send_key(ser, 'Num_01', 1)
            send_key(ser, 'OK', 2)
            wait_with_countdown_noKeyInput(30, "Pre-load Channel")


            # AC Power cycle
            print("\n[AC POWER CYCLE] Starting power cycle...")
            asyncio.run(run_ac_power_cycle(ip, 60))
            wait_with_countdown_noKeyInput(60, "Power Stabilization")
            

            
            # Navigate to LGC
            send_key(ser, 'Home', 2)
            send_key(ser, 'DpadRt', 2)
            
            # Motion detection measurement
            result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, config)
            
            if result:
                results.append(result)
                
                # Save to CSV
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
        
        # Wait before next run
        if run_idx < num_runs:
            wait_with_countdown_noKeyInput(10, "Interval between runs")
    
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
