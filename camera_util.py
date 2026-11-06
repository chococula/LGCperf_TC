"""Open the USB camera that really delivers pictures.

A camera can open but deliver nothing: DirectShow then returns a black frame once per second (seen with the
LifeCam after a USB re-plug), and its index can move (0 <-> 1) with the USB port. open_camera() tries the
configured index first, then the others, and keeps the first one that streams real frames.
"""
import time

import cv2

PROBE_FRAMES = 8
MAX_FRAME_SECONDS = 0.5       # a streaming camera delivers a frame in ~33 ms; a stuck one takes 1 s
MIN_FRAME_STD = 2.0           # an all-black (or all-one-value) frame is not a picture


def _streams(camera):
    good = 0
    for _ in range(PROBE_FRAMES):
        started = time.perf_counter()
        ok, frame = camera.read()
        if ok and time.perf_counter() - started < MAX_FRAME_SECONDS and frame.std() >= MIN_FRAME_STD:
            good += 1
    return good >= PROBE_FRAMES // 2


def open_camera(preferred=1, candidates=4):
    """Return an opened cv2.VideoCapture that streams real frames; exits with a clear message if none does."""
    order = [preferred] + [index for index in range(candidates) if index != preferred]
    for index in order:
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if _streams(camera):
                note = "" if index == preferred else f" (index {preferred} gave no picture)"
                print(f"[CAMERA] Initialized (index: {index}){note}")
                return camera
            print(f"[CAMERA] Index {index} opens but sends no picture (black or stuck).")
        camera.release()
    raise SystemExit("[CAMERA] No camera sends a picture. Close other apps using the camera (Teams, Zoom, Camera), "
                     "re-plug the camera USB, then run again.")
