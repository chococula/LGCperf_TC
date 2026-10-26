"""Capture a whole remote's IR keymap with the Flirc USB receiver, one button at a time.

Run:  python ir_capture.py Hisense            (all keys)
      python ir_capture.py LG Input           (only the named keys; keys already in ir_keys.py are kept)
A window shows which button to press, flashes when the Flirc receives IR (green = key signal,
orange = too weak / noise), and confirms each button with two matching presses.
S skips a button the remote does not have, R redoes the current key, B goes back, Q finishes. Results are written to
ir_keys_<brand>.json and to ir_keys.py as <BRAND>_KEYS (same CSV timing format the TCs send with Flirc).
"""
import json
import os
import queue
import re
import sys
import threading
import time
import winsound

import cv2
import numpy as np
from winpty import PtyProcess

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_PULSES = 16            # a real key frame has at least this many timings (NEC repeat frames have 3-4)
FRAME_GAP_SECONDS = 0.25   # timings arriving closer than this belong to the same button press
MATCH_TOLERANCE = 0.25     # two presses match when every timing is within 25%
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')   # console control codes in the output
MAX_TIMING_US = 20000      # longer marks/spaces are idle gaps between presses (frame separators)

# The keys the TCs already use (same names as LG_KEYS / ROKU_KEYS / SAMSUNG_KEYS in ir_keys.py).
KEYMAP = [
    ('Power', 'Power'), ('Home', 'Home'), ('Back', 'Back / Return'), ('OK', 'OK / Enter'),
    ('Up', 'Up'), ('Down', 'Down'), ('Left', 'Left'), ('Right', 'Right'),
    ('ChUp', 'CH +'), ('ChDown', 'CH -'), ('Input', 'Input (source)'),
]


def keys_to_capture(names):
    """KEYMAP, or only the named keys (python ir_capture.py LG Input); unknown names are captured as given."""
    if not names:
        return KEYMAP
    known = {key.lower(): (key, label) for key, label in KEYMAP}
    return [known.get(name.lower(), (name, name)) for name in names]


def existing_keys(brand):
    """The <BRAND>_KEYS already in ir_keys.py, so a capture adds to them instead of replacing them."""
    try:
        import ir_keys
        return dict(getattr(ir_keys, f"{brand.upper()}_KEYS", {}))
    except ImportError:
        return {}


class FlircReceiver:
    """Runs `flirc_util device_log -i -p` and turns its output into frames of IR timings."""

    def __init__(self, raw_log_path):
        self.frames = queue.Queue()
        self.activity = None        # {'at': perf_counter, 'pulses': n, 'kind': 'key' | 'noise'}
        self.status = 'connecting'  # 'listening' | 'connecting' | an error message shown on screen
        self.raw_log = open(raw_log_path, 'a', encoding='utf-8')
        self.process = None
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Keep `flirc_util device_log` running in a pseudo console (piped output would be held back in a
        buffer, so presses would not show up); if it fails or the device drops, report it and restart."""
        while not self.stopped:
            try:
                self.process = PtyProcess.spawn([FLIRC_UTIL_PATH, 'device_log', '-i', '-p'])
            except Exception as error:
                self.status = f"cannot start flirc_util: {error}"
                time.sleep(2)
                continue
            self._read()
            if self.stopped:
                return
            if self.status == 'listening':
                self.status = 'Flirc stopped responding'
            time.sleep(1.0)

    def _chunks(self):
        """Raw output pieces from the console, via a thread (reads block)."""
        pieces = queue.Queue()

        def pump():
            try:
                while True:
                    data = self.process.read(4096)
                    if data:
                        pieces.put(data)
            except Exception:
                pieces.put(None)

        threading.Thread(target=pump, daemon=True).start()
        return pieces

    def _read(self):
        """The IR debug log prints '+mark -space ...' in microseconds. A press is complete when a long idle
        value appears (e.g. -14139000, printed when the next press starts) or the output goes quiet."""
        pieces = self._chunks()
        text, pending, last_at = '', [], time.perf_counter()
        while not self.stopped:
            try:
                data = pieces.get(timeout=0.05)
            except queue.Empty:
                data = ''
            if data is None:                       # the process ended
                break
            if data:
                text += ANSI_ESCAPE.sub('', data)
                last_at = time.perf_counter()
            lines = text.split('\n')
            quiet = time.perf_counter() - last_at > FRAME_GAP_SECONDS
            text = '' if quiet else lines.pop()    # keep an unfinished line unless the output went quiet
            for line in lines:
                pending = self._handle_line(line, pending)
            if quiet and pending:
                self._flush(pending)
                pending = []
        if pending:
            self._flush(pending)

    def _handle_line(self, line, pending):
        if not line.strip():
            return pending
        self.raw_log.write(line.rstrip('\r') + '\n')
        self.raw_log.flush()
        lowered = line.lower()
        if 'persistance mode' in lowered or 'persistence mode' in lowered:
            self.status = 'listening'
        elif 'disconnected' in lowered or 'error' in lowered or line.lstrip().startswith('[E]'):
            self.status = 'Flirc not responding (' + line.strip()[:60] + ')'
            return pending
        for token in re.findall(r'[+-]\d+', line):
            value = abs(int(token))
            if value > MAX_TIMING_US:              # idle gap: the previous press is complete
                self._flush(pending)
                pending = []
            else:
                pending.append(value)
        return pending

    def _flush(self, timings):
        if not timings:
            return
        # Noise bursts are short; a real key frame has a header and many bits.
        kind = 'key' if len(timings) >= MIN_PULSES else 'noise'
        self.activity = {'at': time.perf_counter(), 'pulses': len(timings), 'kind': kind}
        if kind == 'key':
            self.frames.put(list(timings))

    def drain(self):
        while not self.frames.empty():
            self.frames.get_nowait()

    def next_frame(self, timeout=0.0):
        """A key frame if one arrived (repeat frames of the same press are dropped), else None."""
        try:
            frame = self.frames.get(timeout=timeout) if timeout else self.frames.get_nowait()
        except queue.Empty:
            return None
        time.sleep(FRAME_GAP_SECONDS)
        self.drain()
        return frame

    def close(self):
        self.stopped = True
        if self.process is not None:
            try:
                self.process.terminate(force=True)
            except Exception:
                pass
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


WINDOW = "IR capture"
GREEN, ORANGE, WHITE, GRAY, YELLOW, RED = (80, 220, 80), (0, 165, 255), (240, 240, 240), (150, 150, 150), (0, 230, 255), (80, 80, 255)


def draw_screen(brand, index, label, step, message, message_color, activity, keys, skipped, status='listening'):
    canvas = np.full((600, 1000, 3), 32, np.uint8)
    now = time.perf_counter()
    if activity and now - activity['at'] < 0.6:                 # flash the border on any IR signal
        color = GREEN if activity['kind'] == 'key' else ORANGE
        cv2.rectangle(canvas, (0, 0), (999, 599), color, 18)
        text = (f"IR received: {activity['pulses']} pulses" if activity['kind'] == 'key'
                else f"IR weak / noise: {activity['pulses']} pulses - point the remote at the Flirc")
        cv2.putText(canvas, text, (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if status == 'listening':
        cv2.putText(canvas, "Flirc: listening", (720, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
    else:                                                       # the receiver is not working: say so loudly
        cv2.rectangle(canvas, (0, 245), (999, 395), (40, 40, 140), -1)
        cv2.putText(canvas, "Flirc is not receiving - reconnecting...", (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 3, cv2.LINE_AA)
        cv2.putText(canvas, "If this stays: unplug and re-plug the Flirc USB, close the Flirc app.", (40, 345),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, status[:80], (40, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{brand} remote  -  key {index}/{len(KEYMAP)}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, GRAY, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Press:  {label}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, WHITE, 4, cv2.LINE_AA)
    cv2.putText(canvas, step, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, YELLOW, 2, cv2.LINE_AA)
    if message:
        cv2.putText(canvas, message, (40, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, message_color, 2, cv2.LINE_AA)
    for i, (key, _) in enumerate(KEYMAP):                       # progress of all keys
        x, y = 40 + (i % 5) * 190, 430 + (i // 5) * 50
        if key in keys:
            mark, color = '[x]', GREEN
        elif key in skipped:
            mark, color = '[-]', GRAY
        elif i == index - 1:
            mark, color = '[>]', YELLOW
        else:
            mark, color = '[ ]', GRAY
        cv2.putText(canvas, f"{mark} {key}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "S = skip   R = redo this key   B = previous key   Q = finish", (40, 580),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, canvas)


def main():
    brand = sys.argv[1] if len(sys.argv) > 1 else input("Remote brand (e.g. Hisense): ").strip() or 'Hisense'
    global KEYMAP
    KEYMAP = keys_to_capture(sys.argv[2:])          # the screen and the loop walk through these keys
    json_path = os.path.join(SCRIPT_DIR, f"ir_keys_{brand.lower()}.json")
    keys = existing_keys(brand)
    if os.path.exists(json_path):
        keys.update(json.load(open(json_path, encoding='utf-8')))
    if keys:
        print(f"{brand}: {len(keys)} keys already saved ({', '.join(keys)}); they are kept.")
    skipped = set()
    receiver = FlircReceiver(os.path.join(SCRIPT_DIR, f"ir_capture_{brand.lower()}_raw.log"))
    time.sleep(1.0)
    receiver.drain()
    print(f"Capturing the {brand} remote - follow the '{WINDOW}' window.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    index, first = 0, None
    message, message_color = '', WHITE
    try:
        while index < len(KEYMAP):
            key, label = KEYMAP[index]
            if first is None:
                step = "Press it once" + (" (already captured: press to replace, S keeps it)" if key in keys else "")
            else:
                step = "Got it - press the SAME button again to confirm"
            draw_screen(brand, index + 1, label, step, message, message_color, receiver.activity, keys, skipped,
                        receiver.status)
            pressed = cv2.waitKey(30) & 0xFF
            if pressed in (ord('q'), ord('Q'), 27):
                break
            if pressed in (ord('s'), ord('S')):
                if key not in keys:
                    skipped.add(key)
                index, first, message = index + 1, None, f"{key} skipped"
                message_color = GRAY
                continue
            if pressed in (ord('r'), ord('R')):
                first, message = None, f"Redo {key}: press it once"
                message_color = WHITE
                continue
            if pressed in (ord('b'), ord('B')) and index > 0:
                index, first, message = index - 1, None, ''
                continue
            frame = receiver.next_frame(timeout=0.0)
            if frame is None:
                continue
            winsound.Beep(1200, 80)
            if first is None:
                first, message, message_color = frame, f"1st press OK ({len(frame)} pulses)", GREEN
                continue
            if not frames_match(first, frame):
                first, message, message_color = frame, "The two presses differ - press it again", RED
                continue
            keys[key] = ','.join(str(t) for t in first)
            skipped.discard(key)
            json.dump(keys, open(json_path, 'w', encoding='utf-8'), indent=1)
            code = decode_nec(first)
            message = f"Saved {key}" + (f"  (code {code})" if code else f"  ({len(first)} pulses)")
            message_color, first = GREEN, None
            winsound.Beep(1600, 120)
            winsound.Beep(2200, 160)
            index += 1                      # straight to the next key; the 'Saved' message stays on screen
        draw_screen(brand, min(index + 1, len(KEYMAP)), "done", f"{len(keys)} keys captured - closing",
                    '', WHITE, None, keys, skipped)
        cv2.waitKey(1200)
    finally:
        receiver.close()
        cv2.destroyAllWindows()
    write_ir_keys_py(brand, keys)
    print(f"{len(keys)} keys saved to {json_path} and ir_keys.py ({brand.upper()}_KEYS): {', '.join(keys)}")


if __name__ == "__main__":
    main()
