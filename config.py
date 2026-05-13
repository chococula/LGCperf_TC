# config.py
import serial

# Serial Setting up
PORT = 'COM12'
BAUD = 115200

# IR KEYS (명령어 세트)
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

RED=b'mc 00 72\n'
GRN=b'mc 00 71\n'
YEL=b'mc 00 63\n'
BLU=b'mc 00 61\n'


# 측정 임계값 (여기서 관리하면 튜닝하기 편함)
GRAY_THRESHOLD = 40
DIFF_THRESHOLD = 1.2
STABLE_FRAMES = 2