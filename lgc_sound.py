"""Sounds that are heard on a Bluetooth speaker / soundbar (e.g. the HT-ST5000 used as the PC output).

winsound.Beep tones of 0.08-0.45 s are lost on a Bluetooth link that has gone quiet (it drops the first
fraction of a second while it wakes up). Every tone is played as a short WAV through the default output,
with ~0.4 s of silence in front when nothing was played in the last 2 s.

install() (run on import) replaces winsound.Beep in this process, so every script and module that calls
winsound.Beep (suite_progress, lgc_similarity, Omedia_TC02 functions) goes through play().
"""
import io
import math
import struct
import time
import wave
import winsound

RATE = 22050
VOLUME = 0.6
WAKE_SECONDS = 0.4             # silence in front of a sound after a quiet period
QUIET_AFTER = 2.0              # seconds without sound after which the link is woken again
_ORIGINAL_BEEP = winsound.Beep
_last_end = [0.0]


def _wav(notes, lead_seconds):
    frames = bytearray()
    frames += b'\x00\x00' * int(RATE * lead_seconds)
    for frequency, ms in notes:
        count = int(RATE * ms / 1000)
        fade = max(1, min(int(RATE * 0.005), count // 4))      # 5 ms in / out: no clicks
        for i in range(count):
            envelope = min(1.0, i / fade, (count - 1 - i) / fade)
            value = int(32767 * VOLUME * envelope * math.sin(2 * math.pi * frequency * i / RATE))
            frames += struct.pack('<h', value)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(bytes(frames))
    return buffer.getvalue()


def play(*notes):
    """Play (frequency Hz, ms) notes one after another (blocking, like winsound.Beep)."""
    lead = WAKE_SECONDS if time.perf_counter() - _last_end[0] > QUIET_AFTER else 0.0
    try:
        winsound.PlaySound(_wav(notes, lead), winsound.SND_MEMORY | winsound.SND_NODEFAULT)
    except Exception:
        for frequency, ms in notes:                            # no sound device for PlaySound
            try:
                _ORIGINAL_BEEP(frequency, ms)
            except Exception:
                pass
    _last_end[0] = time.perf_counter()


def beep(frequency, ms):
    """winsound.Beep replacement."""
    play((frequency, ms))


def install():
    if winsound.Beep is not beep:
        winsound.Beep = beep


install()
