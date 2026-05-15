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
    print("  LG TV Test Case 01 - Feb10 Performance Test - Configuration")
    print("="*60 + "\n")

    # Device IP
    while True:
        ip = input("Enter Device IP Address (default: 192.168.4.44): ").strip()
        if not ip:
            ip = "192.168.4.44"
            break
        if len(ip.split('.')) == 4:
            break
        print("❌ Invalid IP format. Please try again.")

    # Serial Port
    while True:
        port = input("Enter Serial Port (default: COM8): ").strip()
        if not port:
            port = "COM8"
            break
        if port.upper().startswith("COM"):
            break
        print("❌ Invalid port format. Use format like COM8.")

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

    # Gray Threshold
    while True:
        try:
            gray_threshold = int(input("Enter gray threshold (default: 150): ").strip() or "150")
            if gray_threshold > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # Motion Threshold
    while True:
        try:
            motion_threshold = int(input("Enter motion threshold (default: 150): ").strip() or "150")
            if motion_threshold > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")

    config = {
        'ip': ip,
        'port': port,
        'num_runs': num_runs,
        'timeout': timeout,
        'gray_threshold': gray_threshold,
        'motion_threshold': motion_threshold
    }

    # Display summary
    print("\n" + "="*60)
    print("  Configuration Summary")
    print("="*60)
    for key, value in config.items():
        print(f"  {key.upper():<20}: {value}")
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
    'RED': b'mc 00 72\n',
    'GRN': b'mc 00 71\n',
    'YEL': b'mc 00 63\n',
    'BLU': b'mc 00 61\n',
}


# ===== Helper Functions =====
def send_key(ser, key_name, delay=2):
    """Send IR key command"""
    if key_name in IR_KEYS:
        ser.write(IR_KEYS[key_name])
        ser.flush()
        time.sleep(delay)
    else:
        print(f"⚠ Unknown key: {key_name}")


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


def perform_motion_detection(ser, cap, run_idx, dir_path, timeout, gray_threshold, motion_threshold):
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
        if diff_score > motion_threshold and mean_val > gray_threshold:
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
    num_runs = config['num_runs']
    timeout = config['timeout']
    gray_threshold = config['gray_threshold']
    motion_threshold = config['motion_threshold']

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

    # Pre-test setup
    print("Pre-test: Live TV for 60s")
    send_key(ser, 'LiveTV')
    send_key(ser, 'Num_03')
    send_key(ser, 'Num_06')
    send_key(ser, 'DASH')
    send_key(ser, 'Num_01')
    wait_with_countdown_noKeyInput(60, "Live TV")

    print("Pre-test: Home for 60s")
    send_key(ser, 'Home', 0)
    wait_with_countdown_noKeyInput(60, "Home")

    # Main test loop
    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"  [RUN {run_idx}/{num_runs}] Starting")
        print(f"{'='*60}")

        # Create run directory
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join(
            "C:/Temp",
            f"LG_TestCase01_run_{run_idx:02d}_{ts_run}"
        )
        os.makedirs(dir_path, exist_ok=True)
        print(f"Output directory: {dir_path}")

        try:
            # Test sequence
            print("\n[Test Sequence] Live TV for 60s")
            send_key(ser, 'LiveTV')
            send_key(ser, 'Num_03')
            send_key(ser, 'Num_06')
            send_key(ser, 'DASH')
            send_key(ser, 'Num_01')
            wait_with_countdown_noKeyInput(60, "Live TV Loop")

            # AC Power cycle
            print("\n[AC POWER CYCLE] Power cycle for 60s")
            asyncio.run(run_ac_power_cycle(ip, 60))

            # Power on stabilization
            print("\n[POWER ON] Stabilization for 3 minutes")
            wait_with_countdown_noKeyInput(180, "Power Stabilization")

            # Navigate to LGC
            print("\n[NAVIGATION] Navigating to LGC app")
            send_key(ser, 'Home', 2)
            send_key(ser, 'DpadRt', 1)

            # Motion detection measurement
            result = perform_motion_detection(ser, cap, run_idx, dir_path, timeout, gray_threshold, motion_threshold)

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
                        'Directory': [result['dir']],
                        'Gray_Threshold': [gray_threshold],
                        'Motion_Threshold': [motion_threshold]
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
    """
    Flow:
      1) AC OFF 안내 + 비프
      2) Space/Enter 입력되면 OFF 카운트다운 시작
      3) 카운트다운 완료 후 다른 비프 + ON 안내
    """
    # 1) OFF 안내 비프 + 메시지
    winsound.Beep(1000, 600)  # AC OFF 안내음 (1.0kHz, 0.6s)
    print("[INFO] Turn AC OFF. Press Spacebar or Enter to continue...")

    # 2) Space/Enter 입력 대기
    while True:
        if keyboard.is_pressed("space") or keyboard.is_pressed("enter"):
            print("[INFO] Key pressed. Starting OFF timer...")
            time.sleep(0.2)  # 디바운스
            break
        time.sleep(0.03)     # CPU 점유 완화

    # 3) OFF 카운트다운
    print(f"[INFO] AC OFF in Progress for {secondsOFF} seconds")
    for remaining in range(secondsOFF, 0, -1):
        print(f"[INFO] Waiting... {remaining} seconds remaining.", end="\r")
        time.sleep(1)
    print(" " * 80, end="\r")  # 라인 클리어

    # 완료 알림 + 다른 비프
    print("\n[INFO] Time is up! Beep! Turn ON AC Powerstrip")
    winsound.Beep(1500, 800)  # 완료음 (1.5kHz, 0.8s)
     



def wait_with_countdown(secondsOFF):
    """
    Flow:
      1) AC OFF 안내 + 비프
      2) Space/Enter 입력되면 OFF 카운트다운 시작
      3) 카운트다운 완료 후 다른 비프 + ON 안내
    """
    # 1) OFF 안내 비프 + 메시지
    winsound.Beep(1000, 600)  # AC OFF 안내음 (1.0kHz, 0.6s)
    print("[INFO] Timer. Press Spacebar or Enter to continue...")

    # 2) Space/Enter 입력 대기
    while True:
        if keyboard.is_pressed("space") or keyboard.is_pressed("enter"):
            print("[INFO] Key pressed. Starting OFF timer...")
            time.sleep(0.2)  # 디바운스  
            break
        time.sleep(0.03)     # CPU 점유 완화

    # 3) OFF 카운트다운
    print(f"[INFO] Timer in Progress for {secondsOFF} seconds")
    for remaining in range(secondsOFF, 0, -1):
        print(f"[INFO] Waiting... {remaining} seconds remaining.", end="\r")
        time.sleep(1)
    print(" " * 80, end="\r")  # 라인 클리어

    # 완료 알림 + 다른 비프
    print("\n[INFO] Time is up! Beep! Exiting Timer")
    winsound.Beep(1500, 800)  # 완료음 (1.5kHz, 0.8s)



def wait_with_countdown_noKeyInput(secondsOFF):
    """
    Flow:
      1) AC OFF 안내 + 비프
      2) Space/Enter 입력되면 OFF 카운트다운 시작
      3) 카운트다운 완료 후 다른 비프 + ON 안내
    """
    # 1) OFF 안내 비프 + 메시지
    winsound.Beep(1000, 600)  # AC OFF 안내음 (1.0kHz, 0.6s)
    print("[INFO] Timer. Press Spacebar or Enter to continue...")

    # 3) OFF 카운트다운
    print(f"[INFO] Timer in Progress for {secondsOFF} seconds")
    for remaining in range(secondsOFF, 0, -1):
        print(f"[INFO] Waiting... {remaining} seconds remaining.", end="\r")
        time.sleep(1)
    print(" " * 80, end="\r")  # 라인 클리어

    # 완료 알림 + 다른 비프
    print("\n[INFO] Time is up! Beep! Exiting Timer")
    winsound.Beep(1500, 800)  # 완료음 (1.5kHz, 0.8s)




shell_command = (
    "luna-send -n 1 -f -a com.webos.service.preloadmanager "
    "luna://com.webos.applicationManager/launch "
    "'{\"id\": \"com.webos.app.lgchannels\", \"preload\": \"partial\"}'"
)




##Live TV 60s 
send_key(LiveTV)
send_key(Num_03)
send_key(Num_06)
send_key(DASH)
send_key(Num_01)
print('Live TV for 60s')
wait_with_countdown_noKeyInput(60)


## Home 60s
send_key(Home,0)
print('Home for 60s')
wait_with_countdown_noKeyInput(60)



# === 5회 반복 실행 ===
for run_idx in range(1, 6):
    print(f"\n========== [RUN {run_idx}] 시작 ==========")

    # 1) 매 실행(run)마다 새로운 저장 디렉토리 생성 (타임스탬프 포함)
    ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join("C:/", "Temp", f"LG_TestCase01_run_{run_idx:02d}_{ts_run}")
    os.makedirs(dir_path, exist_ok=True)
    print(f"[RUN {run_idx}] 저장 디렉토리: {dir_path}")

    # 2) 시나리오: 상태 정렬 (필요시 조정)
    #    - Home → 오른쪽 → OK → OK
    #    - 전원 Off 60s → On 180s → Home → 오른쪽 → OK (예시 그대로 적용)
    #    원하는 시나리오가 다르면 이 블록만 수정하세요.
 

    ##Live TV 60s
    send_key(LiveTV)
    print('Live TV Loop for 60s')
    wait_with_countdown_noKeyInput(60)

    ##AC Power OFF 60s
    print('AC Power Cycle')
    dev = SmartPlug(ip)
    async def runAC_PowerCycle(ACseconds):
        await dev.update()    # 장치 상태 업데이트 (필수)
        await dev.turn_off()   # 켜기
        print("Device turned OFF")
        
        await asyncio.sleep(ACseconds) # 2초 대기
        await dev.turn_on()  # 끄기
        print("Device turned ON")
    asyncio.run(runAC_PowerCycle(60))



    ##AC Power On 3m
    print('Power on for 3m')
    wait_with_countdown_noKeyInput(180)
  


    ##LGC
    send_key(Home, delay=2)
    send_key(DpadRt, delay=1)
 
    



    
    # 3) 측정 시작: 기준 프레임 확보
    
    ret, frame = cap.read()
    if not ret:
        print(f"[RUN {run_idx}] 카메라 프레임을 읽지 못했습니다. 스킵합니다.")
        continue
 

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    # 4) 측정 트리거: OK 전송 후 타이머 시작
    
    
    print(f"[RUN {run_idx}] OK 전송. 움직임 감시 중...")

    motion_detected = False
    frame_count = 0
    elapsed_ms = 0.0
    
    send_key(OK, 0)
    start_time = time.perf_counter()

    # 5) 분석 루프
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[RUN {run_idx}] 프레임 read 실패. 루프 종료.")
            break


        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    
    

        # 차분/지표 계산
        diff = cv2.absdiff(prev_gray, curr_gray_blur)
        diff_score = np.mean(diff)
        mean_val = np.mean(curr_gray)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 화면 상단 텍스트
        top_text = f"Elapsed: {elapsed_ms:.2f}ms | Diff: {diff_score:.2f} | Frame: {frame_count}"
        cv2.putText(curr_gray, top_text, (10, 400), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,), 1)

        # 프레임 저장
        file_name = os.path.join(dir_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(file_name, curr_gray)

        # 판정: 변화량 + 밝기 기준 넘으면 HIT
        if diff_score > MOTION_THRESHOLD and mean_val > GRAY_THRESHOLD:
            motion_detected = True
            cv2.putText(curr_gray, "MOTION DETECTED!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,), 2)
            cv2.imwrite(os.path.join(dir_path, "RESULT_HIT.jpg"), curr_gray)
            print(f"[RUN {run_idx}] 움직임 포착! 소요 시간: {elapsed_ms:.2f}ms")
            break

        # 타임아웃 (10초)
        if (time.perf_counter() - start_time) > 20.0:
            print(f"[RUN {run_idx}] 측정 실패: 10초간 움직임 없음")
            break

        # 디버깅용 실시간 표시 (원하면 유지)
        cv2.imshow('Motion Detection', curr_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자 종료 요청(q).")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

    # 6) 결과 CSV 누적
    if motion_detected:
        df = pd.DataFrame({
            'Run': [run_idx],
            'Dir': [dir_path],
            'Timestamp': [ts_run],
            'Response_Time_ms': [round(elapsed_ms, 2)],
            'Gray_Threshold': [GRAY_THRESHOLD],
            'Motion_Threshold': [MOTION_THRESHOLD]
        })
        # 최초 생성 시 헤더 포함, 이후 append
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=write_header, index=False)

    # 다음 run까지 대기 (TV 화면 완전 준비 시간 제공)
    time.sleep(10)

if __name__ == "__main__":
    main()