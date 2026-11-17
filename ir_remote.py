"""IR control of TVs without the LG RS-232 line (Roku, Samsung, ...) through the Flirc USB IR dongle.

The keysets are the raw timings captured with ir_capture.py (ir_keys.py). The suite names its keys the
LG RS-232 way (DpadUp, P_ON, ...); KEY_ALIASES maps them to the captured IR names. Keys a keyset does not
have (number keys, LiveTV, colour keys, ...) are reported and skipped.
"""
import subprocess
import threading
import time

from ir_keys import HISENSEVIDAA_KEYS, LG_KEYS, ROKU_KEYS, SAMSUNG_KEYS

FLIRC_UTIL_PATH = r"C:\Program Files (x86)\Flirc\flirc_util.exe"
FLIRC_IDLE_WAKE_SECONDS = 20    # the Flirc sleeps when idle: before a send after this long, wake it first

KEYSETS = {
    'Roku': ROKU_KEYS,
    'Samsung': SAMSUNG_KEYS,
    'LG': LG_KEYS,
    'HisenseVidaa': HISENSEVIDAA_KEYS,
}

# suite key name (LG RS-232) -> IR key name; IR power is a toggle, so ON and OFF both send Power
KEY_ALIASES = {
    'DpadUp': 'Up', 'DpadDn': 'Down', 'DpadLt': 'Left', 'DpadRt': 'Right',
    'P_ON': 'Power', 'P_OFF': 'Power', 'P_TOG': 'Power',
}
FALLBACKS = {'Exit': 'Back'}    # used when the keyset has no key of that name


class IrRemote:
    def __init__(self, brand):
        if brand not in KEYSETS:
            raise ValueError(f"Unsupported IR keyset: {brand} (known: {', '.join(KEYSETS)})")
        self.brand = brand
        self.keys = KEYSETS[brand]
        self.last_at = 0.0
        self.lock = threading.Lock()
        self.missing_reported = set()

    def ir_name(self, key):
        """The IR key for a suite key name, or None when this keyset cannot send it."""
        name = KEY_ALIASES.get(key, key)
        if name in self.keys:
            return name
        fallback = FALLBACKS.get(key)
        return fallback if fallback in self.keys else None

    def missing(self, keys):
        return [key for key in keys if self.ir_name(key) is None]

    def wake(self):
        """Talk to the Flirc without sending IR so it is awake for the next send."""
        try:
            subprocess.run([FLIRC_UTIL_PATH, 'version'], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"[IR] Flirc wake-up failed: {error}")
        self.last_at = time.perf_counter()

    def send(self, key):
        """Send one key; returns perf_counter() right before it goes out, or None when it cannot be sent."""
        name = self.ir_name(key)
        if name is None:
            if key not in self.missing_reported:
                self.missing_reported.add(key)
                print(f"⚠ [IR] {self.brand} keyset has no '{key}' key: skipped "
                      f"(capture it with ir_capture.py to use it)")
            return None
        with self.lock:
            if time.perf_counter() - self.last_at >= FLIRC_IDLE_WAKE_SECONDS:
                self.wake()
            sent_at = time.perf_counter()
            try:
                subprocess.run([FLIRC_UTIL_PATH, 'sendir', f"--csv={self.keys[name]}", '--repeat=1'],
                               check=True, capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.SubprocessError) as error:
                print(f"❌ [IR] Sending '{name}' failed: {error}")
                return None
            self.last_at = time.perf_counter()
        return sent_at
