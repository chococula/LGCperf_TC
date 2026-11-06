"""End detection for the LGC TCs, the same methods as the Omedia TCs (detect() picks one per key):

- similarity (screens, e.g. LG Channels home, Movies & TV): the end screen is saved once per SoC as a
  reference (target 'Home', 'MovTV', ...) with a box around the TV screen. Each run records
    Response_Time_ms: the TV screen matches the reference (TM_CCOEFF_NORMED >= threshold)
    Settle_Time_ms:   the screen looks like the reference and has stopped moving for SETTLE_SECONDS
  A target with no reference yet is measured by the settle once and its end screen is offered as reference.
- settle (scrolls): the screen changed and then stood still for SETTLE_SECONDS.
- channel (CH+/CH-, channel number): Omedia_TC02 landing - after the key the whole screen goes black,
  then the picture lands in an area that is empty while loading (set once per SoC by one unmeasured change).
Each run keeps a 2 s pre-roll, every frame with its timer, RESULT_*.jpg and run.mp4 / run.gif.
"""
import os
import queue
import threading
import time

import cv2
import numpy as np
import pandas as pd
import winsound

from config_store import load_saved_configuration, save_configuration
from run_video import make_run_videos

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = "LGC_similarity_config.json"     # {SoC: {'screen': [x, y, w, h], 'threshold': 0.7}}
WINDOW = "LGC similarity"
STATUS = lambda: []            # extra overlay lines (FullSuite: TC i/N, step j/M)
PRE_ROLL_SECONDS = 2.0
CHECK_SECONDS = 3.0             # live check of the reference right after it is captured
SCREEN_MARGIN = 12              # the screen area is searched this many pixels around (small camera shifts)
DEFAULT_THRESHOLD = 0.7
SETTLE_SECONDS = 1.0            # still this long = movement stopped; the time is the first still frame
SETTLE_STILL_DIFF = 2.0         # frame-to-frame change below this = still (camera noise is ~0.5)
SETTLE_CHANGED = 10.0           # ... and the screen must differ this much from before the key
SETTLE_LIKE_REFERENCE = 0.9     # ... and look like the reference (>= 90% of the threshold)
SETTLE_EXTRA_SECONDS = 8.0      # after the similarity hit, keep recording up to this long for the settle
SETTLE_HOLD_SECONDS = 3.0       # settle only: the screen must stay still this long before the run ends


def reference_file(soc, target='Home'):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in soc)
    return os.path.join(SCRIPT_DIR, f"LGC_{target}_reference_{safe}.jpg")


def _settings():
    return load_saved_configuration(SETTINGS_FILE, {'models': {}})


def _threshold(model, target):
    if target == 'Home' and 'threshold' in model:          # the first (LG Channels home) setting
        return model['threshold']
    return model.get('thresholds', {}).get(target, DEFAULT_THRESHOLD)


def ensure_screen(cap, soc):
    """The box around the TV screen in the camera view for this SoC (asked once)."""
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    if not model.get('screen'):
        model['screen'] = _select(cap, "Select the TV screen, then ENTER",
                                  "Drag a box around the TV screen only (not the wall or stand), then press ENTER.",
                                  whole_screen=True)
        save_configuration(SETTINGS_FILE, settings)
    return model['screen']


def _select(cap, window, message, whole_screen=False, frame=None):
    frame = _read(cap) if frame is None else frame
    attention()
    while True:
        print(f"[SIMILARITY] {message}")
        x, y, w, h = (int(v) for v in cv2.selectROI(window, frame, showCrosshair=False))
        cv2.destroyWindow(window)
        if whole_screen and w * h > 0.85 * frame.shape[0] * frame.shape[1]:
            print("[SIMILARITY] That covers almost the whole camera view; box the TV screen only.")
        elif w >= (60 if whole_screen else 20) and h >= (40 if whole_screen else 20):
            return [x, y, w, h]
        else:
            print("[SIMILARITY] The box is too small.")


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


def attention():
    """The user has to do something now (Enter, click, drag)."""
    for frequency, ms in ((1000, 120), (1400, 120), (1000, 120), (1400, 160)):
        winsound.Beep(frequency, ms)


def _overlay(frame, lines):
    lines = list(lines) + list(STATUS())
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
        attention()
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
        model['screen'] = _select(cap, "Select the TV screen, then ENTER",
                                  "Drag a box around the TV screen only (not the wall or stand), then press ENTER.",
                                  whole_screen=True)
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


def measure(cap, press_key, soc, out_dir, timeout, run_label="RUN", target='Home'):
    """Pre-roll, press_key() (returns the perf_counter of the send; the timer starts there), then wait for the
    similarity hit and the settle. target=None: settle only (scrolls). A target with no reference yet is
    measured by the settle and its end screen is offered as the reference.
    Returns {'detected', 'response_time_ms', 'settle_time_ms', 'frame_count', 'dir', 'similarity_threshold'}."""
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    screen = ensure_screen(cap, soc)
    model = _settings()['models'][soc]
    reference = cv2.imread(reference_file(soc, target), cv2.IMREAD_GRAYSCALE) if target else None
    threshold = _threshold(model, target) if target else None
    learning = target is not None and reference is None
    if learning:
        print(f"[{run_label}] No {target} reference for {soc} yet: this run ends at the settle, and its end "
              "screen is offered as the reference.")
    writer = _Writer(os.path.join(out_dir, "frames"))
    grabber = _Grabber(cap)
    grabber.start()
    settled_frame = None
    try:
        time.sleep(PRE_ROLL_SECONDS)
        pre = []
        while not grabber.items.empty():
            pre.append(grabber.items.get())
        key_at = press_key()
        before = _small(pre[-1][1], screen) if pre else None
        score_of = (lambda f: similarity(f, reference, screen)) if reference is not None else (lambda f: 0.0)
        for i, (t, frame) in enumerate(f for f in pre if f[0] >= key_at - PRE_ROLL_SECONDS):
            ms = (t - key_at) * 1000.0
            writer.items.put((f"frame_p{i + 1:04d}_{ms:09.1f}ms_sim{score_of(frame):.3f}.jpg",
                              _overlay(frame, [(f"{run_label}  BEFORE KEY  {ms / 1000:.3f} s", (255, 255, 255), 0.9)])))
        home_ms = settle_ms = None
        armed, previous, still_from, count, ms = False, None, None, 0, 0.0
        while True:
            try:
                t, frame = grabber.items.get(timeout=1.0)
            except queue.Empty:
                continue
            ms = (t - key_at) * 1000.0
            if ms < 0:
                continue
            count += 1
            score = score_of(frame)
            small = _small(frame, screen)
            if before is None:
                before = small
            moved = float(np.mean(np.abs(small - previous))) if previous is not None else 99.0
            previous = small
            changed = float(np.mean(np.abs(small - before))) >= SETTLE_CHANGED
            if reference is not None:
                armed = armed or score < threshold
                if home_ms is None and armed and score >= threshold:
                    home_ms = ms
                    cv2.imwrite(os.path.join(out_dir, "RESULT_HOME.jpg"), _overlay(frame, [
                        (f"{run_label}  {target.upper()} (SIMILARITY)  {ms / 1000:.3f} s", (0, 255, 0), 0.9),
                        (f"Similarity {score:.3f} / {threshold:.3f}", (0, 255, 0), 0.8)]))
                    print(f"[{run_label}] {target} matched at {ms:.2f} ms (similarity {score:.3f}).")
                    winsound.Beep(1500, 250)
                    winsound.Beep(2500, 350)
            if reference is None:
                # Settle only: the start of the LAST still stretch, so a still loading screen that is
                # followed by the real page does not count. Recording ends after SETTLE_HOLD_SECONDS still.
                if moved >= SETTLE_STILL_DIFF:
                    still_from, settle_ms = None, None
                elif changed:
                    still_from = still_from or (ms, frame.copy())
                    if settle_ms is None and ms - still_from[0] >= SETTLE_SECONDS * 1000.0:
                        settle_ms, settled_frame = still_from
                        cv2.imwrite(os.path.join(out_dir, "RESULT_SETTLED.jpg"), _overlay(settled_frame, [
                            (f"{run_label}  MOVEMENT STOPPED  {settle_ms / 1000:.3f} s", (0, 255, 0), 0.9)]))
            elif settle_ms is None:
                if changed and moved < SETTLE_STILL_DIFF and score >= SETTLE_LIKE_REFERENCE * threshold:
                    still_from = still_from or (ms, frame.copy())
                    if ms - still_from[0] >= SETTLE_SECONDS * 1000.0:
                        settle_ms, settled_frame = still_from
                        cv2.imwrite(os.path.join(out_dir, "RESULT_SETTLED.jpg"), _overlay(settled_frame, [
                            (f"{run_label}  MOVEMENT STOPPED  {settle_ms / 1000:.3f} s", (0, 255, 0), 0.9)]))
                        print(f"[{run_label}] Movement stopped at {settle_ms:.2f} ms.")
                else:
                    still_from = None
            hit = home_ms is not None if reference is not None else settle_ms is not None
            color = (0, 255, 0) if hit else (0, 255, 255)
            info = f"Similarity {score:.3f} / {threshold:.3f}   " if reference is not None else ""
            lines = [(f"{run_label}  #{count}  TIMER {ms / 1000:.3f} s", color, 0.9),
                     (f"{info}Movement {min(moved, 99):.1f}", color, 0.7)]
            if home_ms is not None:
                lines.append((f"{target.upper()} at {home_ms / 1000:.3f} s", (0, 255, 0), 0.7))
            if settle_ms is not None:
                lines.append((f"MOVEMENT STOPPED at {settle_ms / 1000:.3f} s", (0, 255, 0), 0.7))
            shown = _overlay(frame, lines)
            writer.items.put((f"frame_{count:05d}_{ms:09.1f}ms_sim{score:.3f}.jpg", shown))
            cv2.imshow(WINDOW, shown)
            cv2.waitKey(1)
            if reference is None and settle_ms is not None and ms >= settle_ms + SETTLE_HOLD_SECONDS * 1000.0:
                print(f"[{run_label}] Movement stopped at {settle_ms:.2f} ms (still {SETTLE_HOLD_SECONDS:g} s).")
                break
            if home_ms is not None and settle_ms is not None and ms >= max(home_ms, settle_ms) + 500:
                break
            if home_ms is not None and ms - home_ms > SETTLE_EXTRA_SECONDS * 1000.0:
                print(f"[{run_label}] Still moving {SETTLE_EXTRA_SECONDS:g} s after the match; no settle time.")
                break
            if ms > timeout * 1000.0:
                print(f"[{run_label}] End not detected within {timeout:g} s.")
                break
    finally:
        grabber.running = False
        grabber.join(timeout=2)
        writer.close()
    ends = [v for v in (home_ms, settle_ms) if v is not None]
    if reference is None and settle_ms is not None:
        winsound.Beep(1500, 250)
        winsound.Beep(2500, 350)
    elif not ends:
        winsound.Beep(400, 700)                     # nothing detected
    try:
        make_run_videos(out_dir, threshold=threshold or 1.0, run_label=run_label,
                        end_ms=max(ends) if ends else None)
    except Exception as error:                     # the measurement stands without the video
        print(f"[{run_label}] Video not made: {error}")
    if learning and settled_frame is not None:
        _offer_reference(settled_frame, soc, target)
    primary = home_ms if reference is not None else settle_ms
    return {
        'detected': primary is not None,
        'response_time_ms': primary if primary is not None else ms,
        'settle_time_ms': settle_ms,
        'frame_count': count,
        'dir': out_dir,
        'similarity_threshold': threshold,
        'method': 'similarity' if reference is not None else 'settle',
    }


def _offer_reference(frame, soc, target):
    attention()
    cv2.imshow(f"{target} reference?", frame)
    cv2.waitKey(1)
    answer = input(f"Save this end screen as the {target} reference for {soc}? (Y/n): ").strip().lower()
    cv2.destroyWindow(f"{target} reference?")
    if answer in ('', 'y', 'yes'):
        cv2.imwrite(reference_file(soc, target), _gray(frame))
        print(f"[SIMILARITY] {target} reference saved; the next runs end by similarity.")


def ensure_landing(cap, soc, change_once):
    """The area that is empty (black) while a new channel loads, set once per SoC by one unmeasured change."""
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    if model.get('landing'):
        return model['landing']
    print("\n[CHANNEL] One unmeasured channel change to choose the landing area.")
    change_once()
    print("[CHANNEL] Press ENTER in the camera window while the screen is black / loading.")
    attention()
    while True:
        frame = _read(cap)
        cv2.imshow(WINDOW, _overlay(frame, [("ENTER while the screen is black / loading", (0, 255, 255), 0.8)]))
        if cv2.waitKey(1) & 0xFF in (10, 13):
            break
    model['landing'] = _select(cap, "Select the landing area, then ENTER",
                               "Drag over an area that is EMPTY (black) now and filled when the picture lands.",
                               frame=frame)
    save_configuration(SETTINGS_FILE, settings)
    print("[CHANNEL] Waiting 8 s for the channel to play...")
    end = time.perf_counter() + 8
    while time.perf_counter() < end:
        cv2.imshow(WINDOW, _overlay(_read(cap), [("Waiting for the channel to play", (255, 255, 255), 0.8)]))
        cv2.waitKey(1)
    return model['landing']


def has_landing(soc):
    return bool((_settings()['models'].get(soc) or {}).get('landing'))


ROUTE_KEYS = {'Up': 'DpadUp', 'Down': 'DpadDn', 'Left': 'DpadLt', 'Right': 'DpadRt'}


def plan_route_from_home(send, cap, soc, name, what, fallback):
    """At the start of a TC: the route from Home to `what` for this SoC, asked before the long run begins.

    A saved route can be kept (Enter). Otherwise Home is opened and the user clicks the highlighted item and
    the target (Omedia_TC02.select_plan; S keeps `fallback`). Only the route is saved - nothing is opened;
    Exit closes Home again. navigate_from_home() later replays the saved route + OK.
    """
    settings = _settings()
    routes = settings['models'].setdefault(soc, {}).setdefault('routes', {})
    saved = routes.get(name)
    if saved is not None:
        attention()
        keep = input(f"[NAV] Home -> {what} for {soc}: {', '.join(saved) or 'no move'} + OK. "
                     "Keep it? (Y/n): ").strip().lower()
        if keep in ('', 'y', 'yes'):
            return saved
    import Omedia_TC02 as planner
    send('Home', 3)
    attention()
    print(f"\n[NAV] Click the highlighted item on Home, then {what}; arrow keys adjust, ENTER saves "
          "(S = default keys).")
    chosen = planner.select_plan(cap, 1, what, None)
    route = list(fallback) if chosen is None else list(chosen[0])
    routes[name] = route
    save_configuration(SETTINGS_FILE, settings)
    print(f"[NAV] Saved for {soc}: Home -> {', '.join(route) or 'no move'} + OK")
    send('Exit', 2)
    return route


def navigate_from_home(send, cap, soc, name, what, fallback):
    """With Home on screen: move to an item and press OK, like the Omedia TCs.

    The first time on a SoC the user clicks the highlighted item and the target on the camera view
    (Omedia_TC02.select_plan: the planned keys are drawn, arrow keys adjust, Enter goes, S keeps fallback);
    the route is saved per SoC and replayed without clicks afterwards. send(key, delay) sends one LGC key.
    """
    settings = _settings()
    routes = settings['models'].setdefault(soc, {}).setdefault('routes', {})
    route = routes.get(name)
    if route is None:
        import Omedia_TC02 as planner
        print(f"\n[NAV] First time on {soc}: click the highlighted item, then {what}; ENTER sends the keys + OK.")
        attention()
        chosen = planner.select_plan(cap, 1, what)
        if chosen is None:
            route = list(fallback)
            print(f"[NAV] Skipped: using the default keys {route}.")
        else:
            route = list(chosen[0])
            routes[name] = route
            save_configuration(SETTINGS_FILE, settings)
            print(f"[NAV] Route to {what} saved for {soc}: {route or 'no move'}.")
    print(f"[NAV] Home -> {what}: {', '.join(route) or 'no move'} + OK")
    for key in route:
        send(ROUTE_KEYS[key], 1)
    send('OK', 2)


def measure_channel(cap, press_key, change_once, soc, out_dir, timeout, run_idx, run_label="RUN"):
    """Channel change like Omedia_TC02: after the key the whole TV screen goes black, then the picture lands
    in the landing area (its pattern changes). Returns the LGC result dict."""
    import Omedia_TC02 as channel            # the landing detection proven on Hisense and LG
    screen = ensure_screen(cap, soc)
    landing = ensure_landing(cap, soc, change_once)
    capture = channel.CameraFrameCapture(cap)
    capture.start()
    try:
        pre_frames, _, _ = channel.capture_pre_roll(capture, landing)
        capture.clear_buffer()
        key_at = press_key()
        result = channel.measure_landing(capture, landing, run_idx, out_dir, timeout, key_at, pre_frames,
                                         require_black=True, screen_roi=screen)
    finally:
        capture.stop()
    detected = result['status'] == 'PASS'
    try:
        make_run_videos(out_dir, run_label=run_label, end_ms=result['response_time_ms'] if detected else None)
    except Exception as error:
        print(f"[{run_label}] Video not made: {error}")
    return {
        'detected': detected,
        'response_time_ms': result['response_time_ms'],
        'frame_count': result['frame_count'],
        'dir': out_dir,
        'method': 'channel',
    }


CHANNEL_KEYS = ('ChUp', 'ChDown')
SCROLL_KEYS = ('DpadUp', 'DpadDn', 'DpadLt', 'DpadRt')


def detect(cap, press_key, resend_key, soc, run_idx, out_dir, timeout, trigger_key, mode=None, target=None,
           run_label=None):
    """Pick the method: mode 'channel' | 'settle' | 'similarity' (default from the key: CH+/CH- -> channel,
    arrows -> settle, others -> similarity with target, default the LG Channels home)."""
    run_label = run_label or f"RUN {run_idx}"
    mode = mode or ('channel' if trigger_key in CHANNEL_KEYS else 'settle' if trigger_key in SCROLL_KEYS
                    else 'similarity')
    if mode == 'channel':
        result = measure_channel(cap, press_key, resend_key, soc, out_dir, timeout, run_idx, run_label)
    else:
        result = measure(cap, press_key, soc, out_dir, timeout, run_label,
                         target=None if mode == 'settle' else (target or 'Home'))
    result['run'] = run_idx
    return result

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
