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
ip="192.168.4.208" 
SoC="O22N3"
SWV="33_30_97"
LGCV="4_0_7-2"


# === Serial ===
ser = serial.Serial('COM4', 115200, timeout=0.1)

# === IR KEYS ===
P_TOG=b'mc 00 08\n'
P_ON = b'ka 00 01\n'
P_OFF = b'ka 00 00\n'
ChUp = b'mc 00 00\n'
ChDown = b'mc 00 01\n'

VolUp = b'mc 00 02\n'
VolDown = b'mc 00 03\n'

DpadUp = b'mc 00 40\n'
DpadDn = b'mc 00 41\n'
DpadLt = b'mc 00 07\n'
DpadRt = b'mc 00 06\n'

Back =  b'mc 00 28\n'
Home = b'mc 00 7C\n'
OK = b'mc 00 44\n'
LiveTV = b'mc 00 D6\n'
Exit = b'mc 00 5B\n'

Num_00 =b'mc 00 10\n'
Num_01 =b'mc 00 11\n'
Num_02 =b'mc 00 12\n'
Num_03 =b'mc 00 13\n'
Num_04 =b'mc 00 14\n'
Num_05 =b'mc 00 15\n'
Num_06 =b'mc 00 16\n'
Num_07 =b'mc 00 17\n'
Num_08 =b'mc 00 18\n'
Num_09 =b'mc 00 19\n'
DASH = b'mc 00 4C\n'



KeyNetflix = b'mc 00 56\n'
KeyAmazon = b'mc 00 5C\n'


RED=b'mc 00 72\n'
GRN=b'mc 00 71\n'
YEL=b'mc 00 63\n'
BLU=b'mc 00 61\n'

# === 측정 임계값 ===
GRAY_THRESHOLD = 150
MOTION_THRESHOLD = 150
STABLE_FRAMES = 5  # (현재 로직에선 사용하지 않음)

# === 헬퍼: 키 전송 ===
def send_key(key, delay=2):
    ser.write(key)
    ser.flush()
    time.sleep(delay)




import time
import winsound
import keyboard



# Shell Command to Send
shell_command = (
    "luna-send -n 1 -f -a com.webos.service.preloadmanager "
    "luna://com.webos.applicationManager/launch "
    "'{\"id\": \"com.webos.app.lgchannels\", \"preload\": \"partial\"}'"
)

shell_command_closeAppID = (
    "luna-send -n 1 -f luna://com.webos.applicationManager/closeByAppId "
    "'{\"id\": \"com.webos.app.lgchannels\"}'"
)

stop_preloadManager = ("stop preload-manager")


# Debug Mode Access and Command Execution
def send_shell_command_with_debug(command):
    try:
        # Step 1: Enter debug mode
        ser.write(b'debug\n')
        time.sleep(3)  # Wait for debug mode to activate
        print("Entered debug mode.")

        # Step 3: Press 's'
        ser.write(b's\n')
        time.sleep(3)
        print("Pressed 's' to enter shell.")

        # Step 3: Send the actual shell command
        print(f"Sending command: {command}")
        ser.write((command + '\n').encode())
        print("Luna Command Sent")
        time.sleep(2)  # Give some time for the command to execute

        # Step 5: Exit shell
        print("Exiting shell...")
        ser.write(b'exit\n')
        time.sleep(2)
        ser.flush()  # Ensure buffer is cleared
        print("Sent 'exit' command.")


        # Step 6: Send 'x' to finalize exit
        ser.write(b'x\n')
        time.sleep(2)
        ser.flush()  # Ensure all data is sent
        print("Sent 'x' to finalize exit.")

    except Exception as e:
        print(f"Error during shell command execution: {e}")


def send_command(command, delay=2):
    ser.write(command)
    time.sleep(delay)



def wait_with_countdown_and_key_AC_OFF(secondsOFF):
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
     


def wait_with_countdown_and_key_DC_OFF(secondsOFF):
    """
    Flow:
      1) AC OFF 안내 + 비프
      2) Space/Enter 입력되면 OFF 카운트다운 시작
      3) 카운트다운 완료 후 다른 비프 + ON 안내
    """
    # 1) OFF 안내 비프 + 메시지
    winsound.Beep(1000, 600)  # AC OFF 안내음 (1.0kHz, 0.6s)
    print("[INFO] Turn AC OFF. Press Spacebar or Enter to continue...")



    # 3) OFF 카운트다운
    print(f"[INFO] AC OFF in Progress for {secondsOFF} seconds")
    for remaining in range(secondsOFF, 0, -1):
        print(f"[INFO] Waiting... {remaining} seconds remaining.", end="\r")
        time.sleep(1)
    print(" " * 80, end="\r")  # 라인 클리어

    # 완료 알림 + 다른 비프
    print("\n[INFO] Time is up! Beep! Turn ON AC Powerstrip")
    winsound.Beep(1100, 800)  # 완료음 (1.5kHz, 0.8s)
    

def wait_with_countdown_and_key_DC_OFF(secondsOFF):
    """
    Flow:
      1) AC OFF 안내 + 비프
      2) Space/Enter 입력되면 OFF 카운트다운 시작
      3) 카운트다운 완료 후 다른 비프 + ON 안내
    """
    # 1) OFF 안내 비프 + 메시지
    winsound.Beep(1500, 1000)  # AC OFF 안내음 (1.5kHz, 1.0s)
    print("[INFO] Turn DC OFF. Press Spacebar or Enter to continue...")

    # 3) OFF 카운트다운
    print(f"[INFO] DC OFF in Progress for {secondsOFF} seconds")
    for remaining in range(secondsOFF, 0, -1):
        print(f"[INFO] Waiting... {remaining} seconds remaining.", end="\r")
        time.sleep(1)
    print(" " * 80, end="\r")  # 라인 클리어

    # 완료 알림 + 다른 비프
    print("\n[INFO] Time is up! Beep!")
    winsound.Beep(1100, 1500)  # 완료음 (800kHz, 1.8s)





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


"""
##Live TV 60s
print('Live TV 60s')
send_key(LiveTV)
send_key(Num_03)
send_key(Num_06)
send_key(DASH)
send_key(Num_01)
send_key(OK) 
wait_with_countdown_noKeyInput(60)
"""


## Home 30s
send_key(Home)
wait_with_countdown_noKeyInput(10)
  


# === 카메라 준비 ===
try:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Windows + DirectShow
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception as e:
    print(f"연결 오류: {e}")
    raise SystemExit(1)

# === 결과 CSV 경로 ===
csv_path = "tv_response.csv"


# === 5회 반복 실행 ===
for run_idx in range(1, 6):
    print(f"\n========== [RUN {run_idx}] 시작 ==========")

    # 1) 매 실행(run)마다 새로운 저장 디렉토리 생성 (타임스탬프 포함)
    ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(
    "C:/",
    "Temp",
    f"LGC_Perf_TC02_PreloadDisabled_{run_idx:02d}_{ts_run}_{SoC}_SWV{SWV}_LGCV{LGCV}"
    )
    os.makedirs(dir_path, exist_ok=True)
    print(f"[RUN {run_idx}] 저장 디렉토리: {dir_path}")

    # 2) 시나리오: 상태 정렬 (필요시 조정)
    ## Test Scenario : Pre-conditions

    ##Live TV 60s
    send_key(LiveTV)
    send_key(Num_03)
    send_key(Num_06)
    send_key(DASH)
    send_key(Num_01)
    send_key(OK) 
    wait_with_countdown_noKeyInput(10)

 
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

    ##Power On
    wait_with_countdown_noKeyInput(18)


    #LG Channel
    send_key(Home)
    #send_key(DpadUp)
    send_key(DpadRt)
    send_key(OK,10)

    #Stop Preload Manager (to ensure clean state for next runs)
    send_shell_command_with_debug(stop_preloadManager)
    print("Preload Manager Stopped to ensure clean state for next runs.")

    ## LGC Exit
    send_shell_command_with_debug(shell_command_closeAppID)

    ## Home 30s
    send_key(Home)
    wait_with_countdown_noKeyInput(30)

    ##Launch LGC
    #send_key(DpadUp)
    send_key(DpadRt)





    
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

    for _ in range(3):
        cap.grab()

    send_key(OK, 0)
    ser.flush()
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
        top_text = f"TC2 ColdBoot_Lunch: {elapsed_ms:.2f}ms | Diff: {diff_score:.2f} | Frame: {frame_count}"
        cv2.putText(curr_gray, top_text, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,), 1)

        

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

# === 자원 정리 ===
cap.release()
cv2.destroyAllWindows()
print("\n모든 실행 완료.")