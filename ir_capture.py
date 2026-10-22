"""Capture a whole remote's IR keymap with the Flirc USB receiver, one button at a time.

Run:  python ir_capture.py Hisense
For each button the tool asks you to press it (twice, to confirm the two readings match).
Press Enter instead to skip a button the remote does not have. Results are written to
ir_keys_<brand>.json and to ir_keys.py as <BRAND>_KEYS (same CSV timing format the TCs send with Flirc).
"""
import json
import msvcrt
import os
import queue
import re
import subprocess
import sys
import threading
import time

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_PULSES = 16            # a real key frame has at least this many timings (NEC repeat frames have 3-4)
FRAME_GAP_SECONDS = 0.25   # timings arriving closer than this belong to the same button press
MATCH_TOLERANCE = 0.25     # two presses match when every timing is within 25%
MAX_TIMING_US = 20000      # longer marks/spaces are idle gaps between presses (frame separators)

# The keys the TCs already use (same names as LG_KEYS / ROKU_KEYS / SAMSUNG_KEYS in ir_keys.py).
KEYMAP = [
    ('Power', 'Power'), ('Home', 'Home'), ('Back', 'Back / Return'), ('OK', 'OK / Enter'),
    ('Up', 'Up'), ('Down', 'Down'), ('Left', 'Left'), ('Right', 'Right'),
    ('ChUp', 'CH +'), ('ChDown', 'CH -'),
]


class FlircReceiver:
    """Runs `flirc_util device_log -i -p` and turns its output into frames of IR timings."""

    def __init__(self, raw_log_path):
        self.frames = queue.Queue()
        self.raw_log = open(raw_log_path, 'a', encoding='utf-8')
        self.process = subprocess.Popen([FLIRC_UTIL_PATH, 'device_log', '-i', '-p'], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        """The IR debug log prints '+mark -space ...' in microseconds; a long idle shows up as one large
        value (e.g. -14139000), which separates button presses."""
        pending, last_at = [], 0.0
        for line in self.process.stdout:
            self.raw_log.write(line)
            self.raw_log.flush()
            tokens = re.findall(r'[+-]\d+', line)
            if not tokens:
                continue
            now = time.perf_counter()
            if pending and now - last_at > FRAME_GAP_SECONDS:
                self._flush(pending)
                pending = []
            last_at = now
            for token in tokens:
                value = abs(int(token))
                if value > MAX_TIMING_US:          # idle gap: the previous press is complete
                    self._flush(pending)
                    pending = []
                else:
                    pending.append(value)
        if pending:
            self._flush(pending)

    def _flush(self, timings):
        # Noise bursts are short; a real key frame has a header and many bits.
        if len(timings) >= MIN_PULSES:
            self.frames.put(list(timings))

    def drain(self):
        while not self.frames.empty():
            self.frames.get_nowait()

    def next_frame_or_enter(self):
        """Wait for a key frame; returns None if the user presses Enter (skip)."""
        while True:
            try:
                frame = self.frames.get(timeout=0.05)
            except queue.Empty:
                frame = None
            if frame is not None:
                time.sleep(FRAME_GAP_SECONDS)       # let repeat frames of the same press arrive
                self.drain()
                return frame
            if msvcrt.kbhit() and msvcrt.getwch() in ('\r', '\n'):
                return None

    def close(self):
        self.process.terminate()
        self.raw_log.close()


def frames_match(a, b):
    if abs(len(a) - len(b)) > 2:
        return False
    return all(abs(x - y) <= MATCH_TOLERANCE * max(x, y) for x, y in zip(a, b))


def decode_nec(timings):
    """Best-effort NEC/Samsung32 decode (address/command) for the summary; None if it does not look like it."""
    bits = []
    for i in range(2, len(timings) - 1, 2):
        mark, space = timings[i], timings[i + 1]
        if not 300 <= mark <= 900:
            break
        bits.append(1 if space > 1000 else 0)
    if len(bits) < 32:
        return None
    data = [sum(bits[byte * 8 + k] << k for k in range(8)) for byte in range(4)]
    return ' '.join(f"{b:02X}" for b in data)


def write_ir_keys_py(brand, keys):
    """Add or replace <BRAND>_KEYS in ir_keys.py."""
    path = os.path.join(SCRIPT_DIR, 'ir_keys.py')
    source = open(path, encoding='utf-8').read()
    name = f"{brand.upper()}_KEYS"
    block = f"{name} = {{\n" + ''.join(f"    {key!r}: {csv!r},\n" for key, csv in keys.items()) + "}\n"
    pattern = re.compile(rf"^{name} = \{{.*?^\}}\n", re.S | re.M)
    source = pattern.sub(block, source) if pattern.search(source) else source.rstrip('\n') + "\n\n\n" + block
    open(path, 'w', encoding='utf-8', newline='\n').write(source)


def main():
    brand = sys.argv[1] if len(sys.argv) > 1 else input("Remote brand (e.g. Hisense): ").strip() or 'Hisense'
    json_path = os.path.join(SCRIPT_DIR, f"ir_keys_{brand.lower()}.json")
    keys = json.load(open(json_path, encoding='utf-8')) if os.path.exists(json_path) else {}
    receiver = FlircReceiver(os.path.join(SCRIPT_DIR, f"ir_capture_{brand.lower()}_raw.log"))
    time.sleep(1.0)
    receiver.drain()
    print(f"\nCapturing the {brand} remote. Point it at the Flirc and press each button when asked.")
    print("Press Enter to skip a button the remote does not have. Ctrl+C stops (captured keys are kept).\n")
    try:
        for index, (key, label) in enumerate(KEYMAP, 1):
            done = ' (already captured - press again to replace, Enter keeps it)' if key in keys else ''
            print(f"[{index:2d}/{len(KEYMAP)}] Press  {label}{done}")
            first = receiver.next_frame_or_enter()
            if first is None:
                print("        skipped")
                continue
            while True:
                print("        got it - press the same button once more to confirm")
                second = receiver.next_frame_or_enter()
                if second is None:
                    print("        kept the first reading")
                    break
                if frames_match(first, second):
                    break
                print("        the two readings differ - press it again")
                first = second
            keys[key] = ','.join(str(t) for t in first)
            code = decode_nec(first)
            print(f"        saved {key}: {len(first)} timings" + (f", code {code}" if code else ""))
            json.dump(keys, open(json_path, 'w', encoding='utf-8'), indent=1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        receiver.close()
    write_ir_keys_py(brand, keys)
    print(f"\n{len(keys)} keys saved to {json_path} and ir_keys.py ({brand.upper()}_KEYS).")


if __name__ == "__main__":
    main()
