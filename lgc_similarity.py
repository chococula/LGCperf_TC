"""End-of-launch detection by similarity for the LGC TCs (the Omedia_TC01 method).

The end screen (LG Channels home) is captured once per SoC as a reference, together with a box around the
TV screen in the camera view. Each run records:
  - Home_Time  (Response_Time_ms): the TV screen matches the reference (TM_CCOEFF_NORMED >= threshold)
  - Settle_Time: the screen looks like the reference and has stopped moving for SETTLE_SECONDS
with a 2 s pre-roll before the key, every frame saved with its timer and similarity, RESULT_HOME.jpg,
RESULT_SETTLED.jpg and run.mp4 / run.gif (same files as Omedia_TC01).
"""
import os
import queue
import threading
import time

import cv2
import numpy as np
import pandas as pd

from config_store import load_saved_configuration, save_configuration
from run_video import make_run_videos

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = "LGC_similarity_config.json"     # {SoC: {'screen': [x, y, w, h], 'threshold': 0.7}}
WINDOW = "LGC similarity"
PRE_ROLL_SECONDS = 2.0
CHECK_SECONDS = 3.0             # live check of the reference right after it is captured
SCREEN_MARGIN = 12              # the screen area is searched this many pixels around (small camera shifts)
DEFAULT_THRESHOLD = 0.7
SETTLE_SECONDS = 1.0            # still this long = movement stopped; the time is the first still frame
SETTLE_STILL_DIFF = 2.0         # frame-to-frame change below this = still (camera noise is ~0.5)
SETTLE_CHANGED = 10.0           # ... and the screen must differ this much from before the key
SETTLE_LIKE_REFERENCE = 0.9     # ... and look like the reference (>= 90% of the threshold)
SETTLE_EXTRA_SECONDS = 8.0      # after the similarity hit, keep recording up to this long for the settle


def reference_file(soc):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in soc)
    return os.path.join(SCRIPT_DIR, f"LGC_Home_reference_{safe}.jpg")


def _settings():
    return load_saved_configuration(SETTINGS_FILE, {'models': {}})


def _gray(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def similarity(frame, reference, screen):
    """Template match of the TV screen area of the reference inside the same area (+ margin) of the frame."""
    x, y, w, h = screen
    gray = _gray(frame)
    m = SCREEN_MARGIN
    area = gray[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
    return float(np.max(cv2.matchTemplate(area, reference[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)))


def _small(frame, screen):
    x, y, w, h = screen
    return cv2.resize(_gray(frame)[y:y + h, x:x + w], (max(8, w // 4), max(8, h // 4)),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _overlay(frame, lines):
    image = frame.copy()
    box_h = 18 + 34 * len(lines)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 640), box_h), (40, 40, 40), -1)
    for i, (text, color, scale) in enumerate(lines):
        cv2.putText(image, text, (8, 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return image


def _read(cap):
    for _ in range(5):
        ok, frame = cap.read()
        if ok:
            return frame
    raise RuntimeError("Camera frame read failed.")


# ---------- one-time preparation per SoC ----------

def prepare(cap, soc, open_end_screen):
    """Make sure the reference and the TV screen area exist for this SoC; returns (reference, screen, threshold).

    open_end_screen(): sends the keys that open the end screen (e.g. Home, Right, OK -> LG Channels).
    """
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    path = reference_file(soc)
    reference = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if reference is not None:
        cv2.imshow("Saved LG Channels reference", reference)
        cv2.waitKey(1)
        reuse = input(f"Reuse the saved LG Channels reference for {soc}? (Y/n): ").strip().lower()
        cv2.destroyWindow("Saved LG Channels reference")
        if reuse not in ('', 'y', 'yes'):
            reference = None
    if reference is None:
        print("\n[SIMILARITY] Opening LG Channels to capture its home screen as the reference...")
        open_end_screen()
        print("[SIMILARITY] When the LG Channels home has fully loaded, press ENTER in the camera window.")
        while True:
            frame = _read(cap)
            cv2.imshow(WINDOW, _overlay(frame, [("ENTER when LG Channels home has fully loaded",
                                                 (0, 255, 255), 0.8)]))
            if cv2.waitKey(30) & 0xFF in (10, 13):
                break
        samples, end = [], time.perf_counter() + 2.0
        while time.perf_counter() < end:
            samples.append(_gray(_read(cap)))
        reference = np.median(np.stack(samples), axis=0).astype(np.uint8)
        cv2.imwrite(path, reference)
        print(f"[SIMILARITY] Reference saved: {path}")
        model.pop('screen', None)                  # a new reference gets a new screen area
    if not model.get('screen'):
        frame = _read(cap)
        while True:
            print("[SIMILARITY] Drag a box around the TV screen only (not the wall or stand), then press ENTER.")
            x, y, w, h = (int(v) for v in cv2.selectROI("Select the TV screen, then ENTER", frame,
                                                        showCrosshair=False))
            cv2.destroyWindow("Select the TV screen, then ENTER")
            if w * h > 0.85 * frame.shape[0] * frame.shape[1]:
                print("[SIMILARITY] That covers almost the whole camera view; box the TV screen only.")
            elif w >= 60 and h >= 40:
                break
            else:
                print("[SIMILARITY] The box is too small.")
        model['screen'] = [x, y, w, h]
    model.setdefault('threshold', DEFAULT_THRESHOLD)
    save_configuration(SETTINGS_FILE, settings)
    return reference, model['screen'], _check(cap, reference, model, settings)


def _check(cap, reference, model, settings):
    """The TV shows the end screen now: the live similarity must reach the threshold, else offer to lower it."""
    while True:
        threshold, scores, end = model['threshold'], [], time.perf_counter() + CHECK_SECONDS
        while time.perf_counter() < end:
            frame = _read(cap)
            score = similarity(frame, reference, model['screen'])
            scores.append(score)
            color = (0, 255, 0) if score >= threshold else (0, 0, 255)
            cv2.imshow(WINDOW, _overlay(frame, [("CHECK: live LG Channels home vs reference", (255, 255, 255), 0.8),
                                                (f"Similarity {score:.3f} / {threshold:.3f}", color, 0.8)]))
            cv2.waitKey(1)
        lowest = min(scores)
        print(f"[SIMILARITY] Live vs reference: lowest {lowest:.3f}, threshold {threshold:.3f}")
        if lowest >= threshold:
            return threshold
        suggested = max(0.3, round(lowest - 0.05, 2))
        answer = input(f"  Is LG Channels home on screen? L = lower the threshold to {suggested:.2f}, "
                       "C = continue, Enter = check again: ").strip().lower()
        if answer == 'l':
            model['threshold'] = suggested
            save_configuration(SETTINGS_FILE, settings)
        elif answer == 'c':
            return threshold


# ---------- measurement ----------

class _Grabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.items, self.running = cap, queue.Queue(), True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.items.put((time.perf_counter(), frame))


class _Writer(threading.Thread):
    def __init__(self, folder):
        super().__init__(daemon=True)
        self.folder, self.items, self.saved = folder, queue.Queue(), 0
        os.makedirs(folder, exist_ok=True)
        self.start()

    def run(self):
        while True:
            item = self.items.get()
            if item is None:
                return
            if cv2.imwrite(os.path.join(self.folder, item[0]), item[1]):
                self.saved += 1

    def close(self):
        self.items.put(None)
        self.join()


def measure(cap, press_key, soc, out_dir, timeout, run_label="RUN"):
    """Pre-roll, press_key() (returns the perf_counter of the send; the timer starts there), then wait for the
    similarity hit and the settle. Returns {'detected', 'response_time_ms', 'settle_time_ms', 'frame_count',
    'dir', 'similarity_threshold'}."""
    model = _settings()['models'].get(soc) or {}
    reference = cv2.imread(reference_file(soc), cv2.IMREAD_GRAYSCALE)
    if reference is None or not model.get('screen'):
        raise RuntimeError(f"No LG Channels reference / screen area for {soc}; run the preparation first.")
    screen, threshold = model['screen'], model.get('threshold', DEFAULT_THRESHOLD)
    writer = _Writer(os.path.join(out_dir, "frames"))
    grabber = _Grabber(cap)
    grabber.start()
    try:
        time.sleep(PRE_ROLL_SECONDS)
        pre = []
        while not grabber.items.empty():
            pre.append(grabber.items.get())
        key_at = press_key()
        before = _small(pre[-1][1], screen) if pre else None
        for i, (t, frame) in enumerate(f for f in pre if f[0] >= key_at - PRE_ROLL_SECONDS):
            ms = (t - key_at) * 1000.0
            writer.items.put((f"frame_p{i + 1:04d}_{ms:09.1f}ms_sim{similarity(frame, reference, screen):.3f}.jpg",
                              _overlay(frame, [(f"{run_label}  BEFORE KEY  {ms / 1000:.3f} s", (255, 255, 255), 0.9)])))
        home_ms = settle_ms = None
        armed, previous, still_from, count = False, None, None, 0
        while True:
            try:
                t, frame = grabber.items.get(timeout=1.0)
            except queue.Empty:
                continue
            ms = (t - key_at) * 1000.0
            if ms < 0:
                continue
            count += 1
            score = similarity(frame, reference, screen)
            small = _small(frame, screen)
            if before is None:
                before = small
            moved = float(np.mean(np.abs(small - previous))) if previous is not None else 99.0
            previous = small
            changed = float(np.mean(np.abs(small - before))) >= SETTLE_CHANGED
            armed = armed or score < threshold
            new_home = home_ms is None and armed and score >= threshold
            if new_home:
                home_ms = ms
                cv2.imwrite(os.path.join(out_dir, "RESULT_HOME.jpg"), _overlay(frame, [
                    (f"{run_label}  HOME (SIMILARITY)  {ms / 1000:.3f} s", (0, 255, 0), 0.9),
                    (f"Similarity {score:.3f} / {threshold:.3f}", (0, 255, 0), 0.8)]))
                print(f"[{run_label}] LG Channels home matched at {ms:.2f} ms (similarity {score:.3f}).")
            if settle_ms is None:
                if changed and moved < SETTLE_STILL_DIFF and score >= SETTLE_LIKE_REFERENCE * threshold:
                    still_from = still_from or (ms, frame.copy())
                    if ms - still_from[0] >= SETTLE_SECONDS * 1000.0:
                        settle_ms = still_from[0]
                        cv2.imwrite(os.path.join(out_dir, "RESULT_SETTLED.jpg"), _overlay(still_from[1], [
                            (f"{run_label}  MOVEMENT STOPPED  {settle_ms / 1000:.3f} s", (0, 255, 0), 0.9)]))
                        print(f"[{run_label}] Movement stopped at {settle_ms:.2f} ms.")
                else:
                    still_from = None
            color = (0, 255, 0) if score >= threshold else (0, 255, 255)
            lines = [(f"{run_label}  #{count}  TIMER {ms / 1000:.3f} s", color, 0.9),
                     (f"Similarity {score:.3f} / {threshold:.3f}   Movement {min(moved, 99):.1f}", color, 0.7)]
            if home_ms is not None:
                lines.append((f"HOME at {home_ms / 1000:.3f} s", (0, 255, 0), 0.7))
            if settle_ms is not None:
                lines.append((f"MOVEMENT STOPPED at {settle_ms / 1000:.3f} s", (0, 255, 0), 0.7))
            shown = _overlay(frame, lines)
            writer.items.put((f"frame_{count:05d}_{ms:09.1f}ms_sim{score:.3f}.jpg", shown))
            cv2.imshow(WINDOW, shown)
            cv2.waitKey(1)
            if home_ms is not None and settle_ms is not None and ms >= max(home_ms, settle_ms) + 500:
                break
            if home_ms is not None and ms - home_ms > SETTLE_EXTRA_SECONDS * 1000.0:
                print(f"[{run_label}] Still moving {SETTLE_EXTRA_SECONDS:g} s after the match; no settle time.")
                break
            if ms > timeout * 1000.0:
                print(f"[{run_label}] LG Channels home not matched within {timeout:g} s.")
                break
    finally:
        grabber.running = False
        grabber.join(timeout=2)
        writer.close()
    ends = [v for v in (home_ms, settle_ms) if v is not None]
    try:
        make_run_videos(out_dir, threshold=threshold, run_label=run_label, end_ms=max(ends) if ends else None)
    except Exception as error:                     # the measurement stands without the video
        print(f"[{run_label}] Video not made: {error}")
    return {
        'detected': home_ms is not None,
        'response_time_ms': home_ms if home_ms is not None else ms,
        'settle_time_ms': settle_ms,
        'frame_count': count,
        'dir': out_dir,
        'similarity_threshold': threshold,
    }


def append_row(csv_path, row):
    """Append one row; when it brings new columns, rewrite the file so every column lines up."""
    new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if not set(new.columns) <= set(old.columns):
            pd.concat([old, new.astype(str)], ignore_index=True).to_csv(csv_path, index=False)
            return
        new = new.reindex(columns=old.columns)
    new.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
