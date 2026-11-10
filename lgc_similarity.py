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
import hangul_keys  # noqa: F401  keys typed in Korean mode count as the Latin key (y for ㅛ)
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
    """Once per SoC, with one unmeasured channel change: does the screen go black (then the landing area is
    chosen on the darkest frame) or does the old picture freeze and the new one cut in (no area needed)?"""
    import Omedia_TC02 as channel
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    transition = model.get('transition')
    if transition == 'freeze' or (transition == 'black' and model.get('landing')):
        return model.get('landing')
    screen = ensure_screen(cap, soc)
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    print("\n[CHANNEL] One unmeasured channel change to see how this TV changes channels...")
    change_once()
    start_at, first_levels = time.perf_counter(), []
    darkest, darkest_level, black = None, float('inf'), False
    while time.perf_counter() - start_at < 6.0:
        frame = _read(cap)
        level = channel.screen_brightness(frame, screen)
        if time.perf_counter() - start_at < 0.25:
            first_levels.append(level)
        before = max(first_levels) if first_levels else level
        if level < darkest_level:
            darkest, darkest_level = frame, level
        if (level < channel.BLACK_SCREEN and level < before * channel.BLACK_RATIO
                and level < before - channel.BLACK_DROP):
            black = True
        cv2.imshow(WINDOW, _overlay(frame, [(f"Channel change: {'BLACK screen seen' if black else 'watching'}",
                                             (0, 255, 255), 0.8)]))
        cv2.waitKey(1)
    model['transition'] = 'black' if black else 'freeze'
    if not black:
        print(f"[CHANNEL] No black screen on {soc}: the landing is the cut after the old picture freezes.")
    elif not model.get('landing'):
        model['landing'] = _select(cap, "Select the landing area, then ENTER",
                                   "Drag over an area that was EMPTY (black) here and is filled when the picture lands.",
                                   frame=darkest)
    else:
        print(f"[CHANNEL] Black screen on {soc}: using the saved landing area.")
    save_configuration(SETTINGS_FILE, settings)
    return model.get('landing')


def has_landing(soc):
    """True when the channel-change type (and for a black screen, the landing area) is saved for the SoC."""
    model = _settings()['models'].get(soc) or {}
    return model.get('transition') == 'freeze' or (model.get('transition') == 'black' and bool(model.get('landing')))


ROUTE_KEYS = {'Up': 'DpadUp', 'Down': 'DpadDn', 'Left': 'DpadLt', 'Right': 'DpadRt'}


NAV_WINDOW = "LGC navigator"
MOVE_DIFF = 6.0                 # a tile whose focus comes or goes changes at least this much (gray, mean)
MOVE_KEY_GAP = 0.8              # seconds between the arrow keys of a route


def _close(window):
    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass                      # never opened


def _clicks_of(chosen):
    """[highlighted item, target] from select_plan, or None (a typed route without the click)."""
    return [chosen[1], chosen[2]] if chosen[1] is not None and chosen[2] is not None else None


def _route_points(soc, name):
    """(route, clicks) saved for a route: clicks = [highlighted item, target] on the camera view, or None."""
    model = _settings()['models'].get(soc) or {}
    clicks = (model.get('route_clicks') or {}).get(name)
    return (model.get('routes') or {}).get(name), ([tuple(c) for c in clicks] if clicks else None)


def _save_route(soc, name, route, clicks):
    settings = _settings()
    model = settings['models'].setdefault(soc, {})
    model.setdefault('routes', {})[name] = list(route)
    if clicks:
        model.setdefault('route_clicks', {})[name] = [[int(v) for v in c] for c in clicks]
    save_configuration(SETTINGS_FILE, settings)


def _route_counts(route):
    rows = route.count('Down') - route.count('Up')
    cols = route.count('Right') - route.count('Left')
    return rows, cols


def _navigator_image(frame, route, clicks, lines):
    """The camera view with the planned path drawn (like the Omedia navigator)."""
    import Omedia_TC02 as planner
    image = frame.copy()
    if clicks:
        tile = planner.tile_box_at(frame, clicks[0])
        _, _, steps = planner.plan_moves(frame, clicks[0], clicks[1], tile)
        rows, cols = _route_counts(route)
        planner.draw_plan(image, clicks[0], rows, cols, steps)
    return _overlay(image, lines)


NAV_CHECK_SECONDS = 10         # during the run the navigator waits this long for the user, then goes


def _show_navigator(cap, route, clicks, what, seconds=NAV_CHECK_SECONDS):
    """Show the planned path for the user to check; returns 'go' (ENTER or time up), 'manual' (M) or
    'replan' (R)."""
    attention()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        left = max(0.0, end - time.perf_counter())
        frame = _read(cap)
        cv2.imshow(NAV_WINDOW, _navigator_image(frame, route, clicks, [
            (f"To {what}: {', '.join(route) or 'no move'} + OK", (0, 255, 255), 0.7),
            (f"Going in {left:.0f} s   ENTER = go now", (255, 255, 255), 0.7),
            ("M = manual (L R U D)   R = click again", (255, 255, 255), 0.7)]))
        code = cv2.waitKey(30) & 0xFF
        if code in (10, 13):
            return 'go'
        if code in (ord('m'), ord('M')):
            return 'manual'
        if code in (ord('r'), ord('R')):
            return 'replan'
    return 'go'


def replay_route(send, soc, name, fallback, delay=1):
    """Send the saved route keys (no check, no OK): opens the way to a later route's start screen."""
    route = (_settings()['models'].get(soc) or {}).get('routes', {}).get(name) or list(fallback)
    for key in route:
        send(ROUTE_KEYS[key], delay)
    return route


def _tile_change(before, after, box):
    """Change of a tile and its edge (the focus is often a border or a zoom around the tile)."""
    x, y, w, h = box
    mx, my = max(6, w // 5), max(6, h // 5)
    x, y, w, h = x - mx, y - my, w + 2 * mx, h + 2 * my
    hh, ww = before.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(ww, x + w), min(hh, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    a = cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def _move_and_check(send, cap, route, clicks, what):
    """Send the route keys while the camera is shown, then check the focus left the highlighted item and
    reached the target (both tiles changed). Returns (ok, start_change, target_change); ok is None when
    there is nothing to compare (no saved clicks or no move)."""
    import Omedia_TC02 as planner
    before = _read(cap)
    boxes = [planner.tile_box_at(before, c) for c in clicks] if clicks else None
    for index, key in enumerate(route, 1):
        send(ROUTE_KEYS[key], 0)
        end = time.perf_counter() + MOVE_KEY_GAP
        while time.perf_counter() < end:
            cv2.imshow(NAV_WINDOW, _overlay(_read(cap), [(f"{what}: {key} ({index}/{len(route)})",
                                                          (0, 255, 255), 0.7)]))
            cv2.waitKey(1)
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end:
        after = _read(cap)
        cv2.waitKey(1)
    if not boxes or not route:
        return None, 0.0, 0.0
    start_change, target_change = (_tile_change(before, after, box) for box in boxes)
    ok = start_change >= MOVE_DIFF and target_change >= MOVE_DIFF
    image = after.copy()
    for (x, y, w, h), changed in zip(boxes, (start_change, target_change)):
        color = (0, 255, 0) if changed >= MOVE_DIFF else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
    cv2.imshow(NAV_WINDOW, _overlay(image, [
        (f"Focus on {what}: {'OK' if ok else 'NOT SEEN'}", (0, 255, 0) if ok else (0, 0, 255), 0.8),
        (f"change: highlighted {start_change:.0f}, target {target_change:.0f} (>= {MOVE_DIFF:.0f})",
         (255, 255, 255), 0.6)]))
    cv2.waitKey(1500)
    print(f"[NAV] Move check {what}: {'OK' if ok else 'focus NOT seen on the target'} "
          f"(highlighted {start_change:.1f}, target {target_change:.1f})")
    return ok, start_change, target_change


def _ask_on_screen(cap, lines, keys):
    """Show the camera with lines until one of `keys` (chars, ENTER = '\r') is pressed; returns it."""
    attention()
    while True:
        cv2.imshow(NAV_WINDOW, _overlay(_read(cap), lines))
        code = cv2.waitKey(30) & 0xFF
        char = '\r' if code in (10, 13) else chr(code).lower() if code < 128 else ''
        if char in keys:
            return char


TEACH_KEYS = {'l': 'Left', 'r': 'Right', 'u': 'Up', 'd': 'Down'}
OPPOSITE = {'Left': 'Right', 'Right': 'Left', 'Up': 'Down', 'Down': 'Up'}


def _simplify(route):
    """Drop key pairs that cancel out (Left then Right ...)."""
    out = []
    for key in route:
        if out and OPPOSITE[out[-1]] == key:
            out.pop()
        else:
            out.append(key)
    return out


def _teach_route(send, cap, what):
    """Manual: every L / R / U / D (or arrow key) the user presses is sent to the TV right away, so the
    focus can be watched on the camera. Backspace sends the opposite key (undo). ENTER = the focus is on
    `what`: returns the pressed keys (the TV is already there). ESC = cancel (None)."""
    import Omedia_TC02 as planner                  # only its arrow-key codes
    attention()
    print(f"\n[NAV] MANUAL to {what}: press L R U D (or arrow keys) in the camera window - each key is sent "
          "to the TV now. Backspace = undo, ENTER = the focus is on the target, ESC = cancel.")
    taught = []
    while True:
        shown = _simplify(taught)
        cv2.imshow(NAV_WINDOW, _overlay(_read(cap), [
            (f"MANUAL to {what}: {', '.join(shown) or '-'}", (0, 255, 255), 0.7),
            ("L R U D / arrows = send now   Backspace = undo", (255, 255, 255), 0.6),
            ("ENTER = focus is on the target   ESC = cancel", (255, 255, 255), 0.6)]))
        code = cv2.waitKeyEx(30)
        if code == -1:
            continue
        char = chr(code & 0xFF).lower() if 0 <= code < 256 else ''
        key = planner.ARROW_KEYS.get(code) or TEACH_KEYS.get(char)
        if key:
            send(ROUTE_KEYS[key], 0)
            taught.append(key)
            print(f"[NAV]   sent {key}  -> {', '.join(_simplify(taught))}")
        elif code in (8,) and taught:
            undo = OPPOSITE[taught[-1]]
            send(ROUTE_KEYS[undo], 0)
            taught.append(undo)
            print(f"[NAV]   undo: sent {undo}  -> {', '.join(_simplify(taught))}")
        elif code in (10, 13):
            _close(NAV_WINDOW)
            return taught
        elif code == 27:
            _close(NAV_WINDOW)
            return None


def _select_plan(cap, what, saved=None):
    """LGC navigator (like Omedia_TC02.select_plan, with manual next to start over): click the highlighted
    item, then the target; the planned keys are drawn. Arrow keys adjust, R = start over, M = manual (the
    user presses L R U D and each key is sent), ENTER = go, Q = stop.
    Returns (route, focus_point, target_point), or None for manual."""
    import Omedia_TC02 as planner                     # drawing / tile helpers only
    window = f"Select {what}"
    clicks = list(saved or [])
    plan = {'rows': 0, 'cols': 0, 'steps': None, 'tile': None}

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) >= 2:
                clicks.clear()
                plan['steps'] = None
            clicks.append((x, y))

    print(f"[NAV] Click (1) the item highlighted now, then (2) {what}. Arrow keys adjust, R = start over, "
          "M = manual (L R U D, each key sent), ENTER = go, Q = stop.")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        frame = _read(cap)
        preview = frame.copy()
        keys_line = ("R = start over   M = MANUAL (L R U D)   Q = stop", (255, 255, 255), 0.7)
        if len(clicks) == 0:
            lines = [("1) Click the item that is HIGHLIGHTED now", (0, 255, 0), 0.8), keys_line]
        elif len(clicks) == 1:
            cv2.circle(preview, clicks[0], 10, (255, 255, 255), 3)
            lines = [(f"2) Click {what}", (0, 255, 0), 0.8), keys_line]
        else:
            if plan['steps'] is None:
                plan['tile'] = planner.tile_box_at(frame, clicks[0])
                plan['rows'], plan['cols'], plan['steps'] = planner.plan_moves(frame, clicks[0], clicks[1],
                                                                               plan['tile'])
            x, y, w, h = plan['tile']
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 255), 1)
            planner.draw_plan(preview, clicks[0], plan['rows'], plan['cols'], plan['steps'])
            route = planner.route_from_plan(plan['rows'], plan['cols'])
            lines = [(f"Planned: {planner.describe_route(route)} + OK", (0, 255, 255), 0.8),
                     ("Arrows adjust   ENTER = go", (255, 255, 255), 0.7), keys_line]
        cv2.imshow(window, _overlay(preview, lines))
        code = cv2.waitKeyEx(30)
        if code in planner.ARROW_KEYS and len(clicks) == 2:
            key = planner.ARROW_KEYS[code]
            plan['rows'] += {'Down': 1, 'Up': -1}.get(key, 0)
            plan['cols'] += {'Right': 1, 'Left': -1}.get(key, 0)
        elif code in (10, 13) and len(clicks) == 2:
            _close(window)
            route = planner.route_from_plan(plan['rows'], plan['cols'])
            print(f"[NAV] Planned keys to {what}: {planner.describe_route(route)} + OK")
            return route, clicks[0], clicks[1]
        elif code in (ord('r'), ord('R')):
            clicks.clear()
            plan['steps'] = None
        elif code in (ord('m'), ord('M'), ord('s'), ord('S')):
            _close(window)
            return None
        elif code in (ord('q'), ord('Q')):
            _close(window)
            raise KeyboardInterrupt(f"Selection of {what} canceled.")


def _open_start(send, open_start):
    """Open the screen a route starts on: open_start() when given, else Home."""
    if open_start is None:
        send('Home', 3)
    else:
        open_start()


def plan_route_from_home(send, cap, soc, name, what, fallback, open_start=None, start_name='Home'):
    """At the start of a TC (the user is there): open the start screen, show the navigator with the saved
    path drawn (or click the highlighted item and the target), move there and check the focus arrived.

    S in the navigator, or M after the move = manual: the user presses L R U D, each key goes to the TV at
    once, ENTER takes the pressed keys as the route. open_start(): opens the screen the route starts on
    (default: Home), start_name names it. Nothing is opened (no OK): Exit closes the start screen.
    navigate_from_home() later shows the same path, moves and checks again, then OK.
    """
    import Omedia_TC02 as planner
    saved_route, saved_clicks = _route_points(soc, name)
    while True:
        _open_start(send, open_start)
        attention()
        print(f"\n[NAV] {start_name} -> {what}: click the highlighted item, then {what}; arrow keys adjust, "
              "ENTER = move there. M = MANUAL (you press L R U D, each key is sent to the TV).")
        chosen = _select_plan(cap, what, saved_clicks)
        if chosen is None:
            taught = _teach_route(send, cap, what)
            if taught is None:
                send('Exit', 2)
                continue
            route, clicks = _simplify(taught), None
            break
        route, clicks = list(chosen[0]), _clicks_of(chosen)
        ok, _, _ = _move_and_check(send, cap, route, clicks, what)
        verdict = 'focus seen on the target' if ok else 'focus NOT seen on the target' if ok is False else 'look'
        answer = _ask_on_screen(cap, [(f"{start_name} -> {what}: {', '.join(route) or 'no move'} ({verdict})",
                                       (0, 255, 0) if ok else (0, 255, 255), 0.7),
                                      ("Focus on the target?  ENTER = yes, save", (255, 255, 255), 0.7),
                                      (f"R = click again   M = manual from {start_name} (L R U D)",
                                       (255, 255, 255), 0.7)],
                                ('\r', 'r', 'm'))
        if answer == '\r':
            break
        if answer == 'm':
            print(f"[NAV] Manual: opening {start_name} again; the route is only the keys you press now.")
            send('Exit', 2)
            _open_start(send, open_start)
            taught = _teach_route(send, cap, what)
            if taught is not None:
                route, clicks = _simplify(taught), None
                break
        saved_clicks = None
        send('Exit', 2)
    _close(NAV_WINDOW)
    _save_route(soc, name, route, clicks)
    print(f"[NAV] Saved for {soc}: {start_name} -> {', '.join(route) or 'no move'} + OK")
    send('Exit', 2)
    return route


def navigate_from_home(send, cap, soc, name, what, fallback, open_start=None):
    """With the start screen (Home, LG Channels) on: show the navigator with the planned path, send the
    route, check the focus reached the target, then OK.

    No saved route: the user clicks the highlighted item and the target (M = manual).
    The focus not seen on the target: beep and ask - ENTER = OK anyway, R = click again from where the focus
    is now, M = manual: the start screen is opened again (open_start(), else Home) and the keys the user
    presses (each sent at once) are the new route, saved; Q = stop. send(key, delay) sends one LGC key.
    """
    import Omedia_TC02 as planner
    route, clicks = _route_points(soc, name)
    if route is None:
        print(f"\n[NAV] First time on {soc}: click the highlighted item, then {what}; ENTER moves there. "
              "M = MANUAL (you press L R U D).")
        attention()
        chosen = _select_plan(cap, what)
        if chosen is None:
            taught = _teach_route(send, cap, what)
            if taught is None:
                raise KeyboardInterrupt(f"Navigation to {what} canceled.")
            route = _simplify(taught)
            _save_route(soc, name, route, None)
            print(f"[NAV] Route to {what} saved for {soc}: {route or 'no move'}.")
            send('OK', 2)
            return
        route, clicks = list(chosen[0]), _clicks_of(chosen)
        _save_route(soc, name, route, clicks)
        print(f"[NAV] Route to {what} saved for {soc}: {route or 'no move'}.")
    else:
        action = _show_navigator(cap, route, clicks, what)
        if action == 'manual':
            taught = _teach_route(send, cap, what)
            if taught is not None:
                route = _simplify(taught)
                _save_route(soc, name, route, None)
                print(f"[NAV] Route to {what} saved for {soc}: {', '.join(route) or 'no move'}.")
                _close(NAV_WINDOW)
                send('OK', 2)
                return
        elif action == 'replan':
            chosen = _select_plan(cap, what)
            if chosen is None:                                  # M: manual from the start screen (on now)
                taught = _teach_route(send, cap, what)
                if taught is not None:
                    route = _simplify(taught)
                    _save_route(soc, name, route, None)
                    print(f"[NAV] Route to {what} saved for {soc}: {', '.join(route) or 'no move'}.")
                    _close(NAV_WINDOW)
                    send('OK', 2)
                    return
            else:
                route, clicks = list(chosen[0]), _clicks_of(chosen)
                _save_route(soc, name, route, clicks)
                print(f"[NAV] Route to {what} saved for {soc}: {', '.join(route) or 'no move'}.")
    print(f"[NAV] -> {what}: {', '.join(route) or 'no move'} + OK")
    ok, _, _ = _move_and_check(send, cap, route, clicks, what)
    while ok is False:
        print(f"[NAV] The focus was not seen on {what}.")
        answer = _ask_on_screen(cap, [(f"Focus NOT seen on {what}", (0, 0, 255), 0.8),
                                      ("ENTER = OK anyway   R = click again", (255, 255, 255), 0.7),
                                      ("M = manual from the start screen (L R U D)   Q = stop", (255, 255, 255), 0.7)],
                                ('\r', 'r', 'm', 'q'))
        if answer == 'q':
            raise KeyboardInterrupt(f"Navigation to {what} stopped by the user.")
        if answer == '\r':
            break
        if answer == 'r':
            chosen = _select_plan(cap, what)                    # from where the focus is now
            if chosen is not None:
                ok, _, _ = _move_and_check(send, cap, list(chosen[0]), _clicks_of(chosen), what)
                continue
            answer = 'm'                                        # M in the navigator
        if answer == 'm':
            print("[NAV] Manual: opening the start screen again; the route is only the keys you press now.")
            send('Exit', 2)
            _open_start(send, open_start)
            taught = _teach_route(send, cap, what)
            if taught is None:
                continue
            route = _simplify(taught)
            _save_route(soc, name, route, None)
            print(f"[NAV] Route to {what} saved for {soc}: {', '.join(route) or 'no move'}.")
            break
    _close(NAV_WINDOW)
    send('OK', 2)


def _measure_freeze_cut(capture, screen, run_idx, out_dir, timeout, key_at, pre_frames):
    """Time from the key until the new channel cuts in after the old picture froze (FreezeCut)."""
    import Omedia_TC02 as channel
    x, y, w, h = screen
    writer = channel.FrameWriter(os.path.join(out_dir, "frames"))
    for index, (captured_at, frame, _) in enumerate(pre_frames, 1):
        ms = (captured_at - key_at) * 1000.0
        writer.save(f"frame_p{index:04d}_{ms:09.1f}ms_sim0.000.jpg",
                    channel.draw_overlay(frame.copy(), [(f"RUN {run_idx}  BEFORE KEY  {ms / 1000.0:.3f} s",
                                                         (255, 255, 255), 1.4)]))
    detector, landed_ms, stop_at, count, last_ms = FreezeCut(), None, None, 0, 0.0
    print(f"[RUN {run_idx}] Waiting for the old picture to freeze and the new channel to cut in...")
    try:
        while time.perf_counter() - key_at <= timeout:
            captured = capture.get_frame()
            if captured is None:
                continue
            captured_at, frame = captured
            if captured_at < key_at:
                continue
            count += 1
            last_ms = ms = (captured_at - key_at) * 1000.0
            if landed_ms is None:
                landed_ms = detector.feed(ms, _screen_small(frame, screen))
                if landed_ms is not None:
                    stop_at = ms + channel.STREAM_EXTRA_SECONDS * 1000.0
                    print(f"[RUN {run_idx}] Landed at {landed_ms:.2f} ms.")
                    winsound.Beep(1500, 250)
                    winsound.Beep(2500, 350)
            color = (0, 255, 0) if landed_ms is not None else (0, 255, 255)
            lines = [(f"RUN {run_idx}  #{count}  TIMER {ms / 1000.0:.3f} s", color, 1.4),
                     (f"Change {detector.step:.1f}  [{'landed' if landed_ms is not None else detector.phase}]",
                      color, 1.0)]
            if landed_ms is not None:
                lines.append((f"LANDED at {landed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            image = frame.copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1)
            writer.save(f"frame_{count:05d}_{ms:09.1f}ms_sim{min(detector.step / CUT_STEP, 9.999):.3f}.jpg",
                        channel.draw_overlay(image.copy(), lines))
            cv2.imshow(channel.CAPTURE_WINDOW, channel.draw_overlay(image, lines + list(STATUS())))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")
            if stop_at is not None and ms >= stop_at:
                break
    finally:
        writer.close()
    if landed_ms is None:
        print(f"[RUN {run_idx}] No freeze -> cut within {timeout:g} s (did the channel change?).")
        winsound.Beep(400, 700)
    return {'status': 'PASS' if landed_ms is not None else 'FAIL',
            'response_time_ms': landed_ms if landed_ms is not None else last_ms, 'frame_count': count}


def measure_channel(cap, press_key, change_once, soc, out_dir, timeout, run_idx, run_label="RUN"):
    """Channel change like Omedia_TC02: after the key the whole TV screen goes black, then the picture lands
    in the landing area (its pattern changes). Returns the LGC result dict."""
    import Omedia_TC02 as channel            # the landing detection proven on Hisense and LG
    screen = ensure_screen(cap, soc)
    landing = ensure_landing(cap, soc, change_once)
    freeze = (_settings()['models'].get(soc) or {}).get('transition') == 'freeze'
    capture = channel.CameraFrameCapture(cap)
    capture.start()
    try:
        pre_frames, _, _ = channel.capture_pre_roll(capture, screen if freeze else landing)
        capture.clear_buffer()
        key_at = press_key()
        if freeze:
            result = _measure_freeze_cut(capture, screen, run_idx, out_dir, timeout, key_at, pre_frames)
        else:
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
        'method': 'channel-freeze' if freeze else 'channel',
    }


# Channel change without a black screen (LG K25Lp, LGC 4.1.14): the old channel keeps playing ~0.8 s, then the
# TV freezes its last picture (frame-to-frame change ~0.2-0.5) for ~0.8 s, then the new channel cuts in
# (one frame 15-48, or a 2-3 frame crossfade). A cut of the old programme itself is not preceded by a freeze.
FREEZE_SECONDS = 0.4            # still at least this long right before the cut
FREEZE_STEP = 0.7               # frame-to-frame change of a still (frozen) picture
FREEZE_FRACTION = 0.75          # share of still frames in that window (the channel banner animates in it)
CUT_STEP = 10.0                 # frame-to-frame change of the cut to the new channel
CUT_ONSET_STEP = 2.0            # the landing time is the first frame of the cut that moved this much
CUT_PERSIST_SECONDS = 0.5       # the new picture must stay different from the frozen one this long
CUT_MIN_REACTION_MS = 300


def _screen_small(frame, screen):
    x, y, w, h = screen
    gray = cv2.GaussianBlur(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (9, 9), 0)
    return cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


class FreezeCut:
    """Feed (elapsed_ms, screen image from _screen_small); returns the landing time once confirmed."""

    def __init__(self):
        self.history = []                 # (elapsed_ms, step, image)
        self.candidate = None             # (onset_ms, frozen image)
        self.step = 0.0
        self.phase = 'playing'

    def feed(self, elapsed_ms, image):
        previous = self.history[-1][2] if self.history else None
        self.step = float(np.mean(np.abs(image - previous))) if previous is not None else 0.0
        self.history.append((elapsed_ms, self.step, image))
        self.history = [h for h in self.history if h[0] >= elapsed_ms - 3000]
        recent = [st for t, st, _ in self.history[:-1] if t >= elapsed_ms - FREEZE_SECONDS * 1000]
        if self.candidate is None:
            still = bool(recent) and sum(st <= FREEZE_STEP for st in recent) >= FREEZE_FRACTION * len(recent)
            self.phase = 'frozen' if still else 'playing'
            if elapsed_ms >= CUT_MIN_REACTION_MS and self.step >= CUT_STEP:
                i = len(self.history) - 1          # back to the first frame of the cut (crossfade)
                while i > 0 and len(self.history) - i < 4 and self.history[i - 1][1] >= CUT_ONSET_STEP:
                    i -= 1
                onset = self.history[i][0]
                before = [st for t, st, _ in self.history[:i] if onset - FREEZE_SECONDS * 1000 <= t < onset]
                if (i > 0 and len(before) >= 5
                        and sum(st <= FREEZE_STEP for st in before) >= FREEZE_FRACTION * len(before)):
                    self.candidate = (onset, self.history[i - 1][2])
                    self.phase = 'cut'
            return None
        if float(np.mean(np.abs(image - self.candidate[1]))) < CUT_STEP * 0.6:
            self.candidate, self.phase = None, 'playing'          # a flash (banner), not the new channel
            return None
        if elapsed_ms - self.candidate[0] >= CUT_PERSIST_SECONDS * 1000:
            return self.candidate[0]
        return None


# ---------- black area: the picture must fill an area that is black while the new channel loads ----------
# (Native -> Pluto: black screen + channel banner / Pluto logo first; a cut to it is not the new picture)
AREA_SETUP_SECONDS = 8.0        # frames recorded after the unmeasured change to pick the black screen from
AREA_DARK_MARGIN = 15           # the area counts as black up to its level on the chosen frame + this
# The black screen is not steady: the camera exposure lifts it in steps (~+10 at a time, 26 -> 74 in 1.5 s)
# and the menu / banner appear. The picture arriving is a sudden change of the area: against the frame
# AREA_STEP_MS earlier, the pattern (mean removed) moves >= AREA_FILL_STEP or the level >= AREA_FILL_RISE.
AREA_STEP_MS = 200
AREA_FILL_STEP = 15
AREA_FILL_RISE = 30
AREA_ONSET_STEP = 8             # the landing time is the first frame that moved this far from that frame
AREA_PERSIST_SECONDS = 0.5      # ... and the area stays that different this long
AREA_BLACK_FRAMES = 3           # black = this many frames in a row at the black level (not a frame of the fade)


def _area_gray(frame, box):
    x, y, w, h = box
    gray = cv2.GaussianBlur(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (9, 9), 0)
    return cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)


def _pattern(area, reference):
    """Change of the picture in the area with each one's mean brightness removed."""
    return float(np.mean(np.abs((area - area.mean()) - (reference - reference.mean()))))


def has_black_area(soc, name):
    return bool(((_settings()['models'].get(soc) or {}).get('black_areas') or {}).get(name))


def ensure_black_area(cap, soc, name, what, change_once, back):
    """Once per SoC and test: one unmeasured change (change_once), the frames of the next 8 s are kept and the
    user picks the black / logo screen (arrows step, starts at the darkest), then drags an area that is
    BLACK there (not the logo or banner) and filled by the picture when the channel plays. back() returns
    to the start channel. Saved as models[soc]['black_areas'][name] = {'box', 'dark'}."""
    settings = _settings()
    areas = settings['models'].setdefault(soc, {}).setdefault('black_areas', {})
    if areas.get(name):
        return areas[name]
    screen = ensure_screen(cap, soc)
    print(f"\n[AREA] {what}: one unmeasured change to choose the black area.")
    change_once()
    started, frames = time.perf_counter(), []
    while time.perf_counter() - started < AREA_SETUP_SECONDS:
        frame = _read(cap)
        frames.append((time.perf_counter() - started, frame))
        cv2.imshow(WINDOW, _overlay(frame, [("Recording the channel change...", (0, 255, 255), 0.8)]))
        cv2.waitKey(1)
    x, y, w, h = screen
    levels = [float(cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).mean()) for _, f in frames]
    index = int(np.argmin(levels))
    attention()
    print("[AREA] Left / Right = step through the frames, ENTER = the frame with the black / logo screen.")
    import Omedia_TC02 as planner                      # arrow-key codes only
    window = "Pick the black / logo screen"
    while True:
        t, frame = frames[index]
        cv2.imshow(window, _overlay(frame, [
            (f"Frame {index + 1}/{len(frames)}  +{t:.2f} s after the key", (0, 255, 255), 0.7),
            ("Left / Right = step   ENTER = this black / logo screen", (255, 255, 255), 0.6)]))
        code = cv2.waitKeyEx(30)
        key = planner.ARROW_KEYS.get(code)
        if key == 'Left':
            index = max(0, index - 1)
        elif key == 'Right':
            index = min(len(frames) - 1, index + 1)
        elif code in (10, 13):
            break
    _close(window)
    box = _select(cap, "Select the black area, then ENTER",
                  "Drag over an area that is BLACK here (not the logo or banner) and filled when the picture plays.",
                  frame=frames[index][1])
    dark = float(_area_gray(frames[index][1], box).mean())
    settings = _settings()
    settings['models'].setdefault(soc, {}).setdefault('black_areas', {})[name] = {'box': box, 'dark': dark}
    save_configuration(SETTINGS_FILE, settings)
    print(f"[AREA] Saved for {soc} / {name}: {box} (black level {dark:.0f}).")
    back()
    return settings['models'][soc]['black_areas'][name]


def _measure_black_area_frames(capture, area, run_idx, out_dir, timeout, key_at, pre_frames):
    """Time from the key until the picture fills the black area (after the area went black)."""
    import Omedia_TC02 as channel
    box = area['box']
    x, y, w, h = box
    dark_limit = area['dark'] + AREA_DARK_MARGIN
    writer = channel.FrameWriter(os.path.join(out_dir, "frames"))
    for index, (captured_at, frame, _) in enumerate(pre_frames, 1):
        ms = (captured_at - key_at) * 1000.0
        writer.save(f"frame_p{index:04d}_{ms:09.1f}ms_sim0.000.jpg",
                    channel.draw_overlay(frame.copy(), [(f"RUN {run_idx}  BEFORE KEY  {ms / 1000.0:.3f} s",
                                                         (255, 255, 255), 1.4)]))
    phase, history, candidate, dark_run = 'waiting for black', [], None, 0
    landed_ms, stop_at, count, last_ms, change = None, None, 0, 0.0, 0.0
    print(f"[RUN {run_idx}] Waiting for the area to go black, then fill with the picture...")
    try:
        while time.perf_counter() - key_at <= timeout:
            captured = capture.get_frame()
            if captured is None:
                continue
            captured_at, frame = captured
            if captured_at < key_at:
                continue
            count += 1
            last_ms = ms = (captured_at - key_at) * 1000.0
            current = _area_gray(frame, box)
            level = float(current.mean())
            if landed_ms is None:
                if phase == 'waiting for black':
                    dark_run = dark_run + 1 if ms >= CUT_MIN_REACTION_MS and level <= dark_limit else 0
                    if dark_run >= AREA_BLACK_FRAMES:
                        phase, history = 'black', [(ms, current)]
                else:
                    def moved(a, ref):
                        return max(_pattern(a, ref), abs(float(a.mean()) - float(ref.mean())) * AREA_FILL_STEP / AREA_FILL_RISE)
                    history = (history + [(ms, current)])[-60:]
                    earlier = [(t, a) for t, a in history if t <= ms - AREA_STEP_MS]
                    if candidate is None:
                        change = moved(current, earlier[-1][1]) if earlier else 0.0
                        if change >= AREA_FILL_STEP:
                            ref_t, ref = earlier[-1]
                            onset = next((t for t, a in history if t > ref_t and moved(a, ref) >= AREA_ONSET_STEP), ms)
                            candidate, phase = (onset, ref), 'filling'
                    else:
                        change = moved(current, candidate[1])
                        if change < AREA_FILL_STEP * 0.6:
                            candidate, phase = None, 'black'    # went back (a flash)
                        elif ms - candidate[0] >= AREA_PERSIST_SECONDS * 1000.0:
                            landed_ms = candidate[0]
                            stop_at = ms + channel.STREAM_EXTRA_SECONDS * 1000.0
                            print(f"[RUN {run_idx}] Landed at {landed_ms:.2f} ms.")
                            winsound.Beep(1500, 250)
                            winsound.Beep(2500, 350)
            color = (0, 255, 0) if landed_ms is not None else (0, 255, 255)
            lines = [(f"RUN {run_idx}  #{count}  TIMER {ms / 1000.0:.3f} s", color, 1.4),
                     (f"Area {level:.0f} (black <= {dark_limit:.0f})  change {change:.0f} / {AREA_FILL_STEP}  "
                      f"[{'landed' if landed_ms is not None else phase}]", color, 1.0)]
            if landed_ms is not None:
                lines.append((f"LANDED at {landed_ms / 1000.0:.3f} s", (0, 255, 0), 1.0))
            image = frame.copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            writer.save(f"frame_{count:05d}_{ms:09.1f}ms_sim{min(change / AREA_FILL_STEP, 9.999):.3f}.jpg",
                        channel.draw_overlay(image.copy(), lines))
            cv2.imshow(channel.CAPTURE_WINDOW, channel.draw_overlay(image, lines + list(STATUS())))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Measurement canceled by user.")
            if stop_at is not None and ms >= stop_at:
                break
    finally:
        writer.close()
    if landed_ms is None:
        print(f"[RUN {run_idx}] No picture in the black area within {timeout:g} s"
              + (" (the area never went black)." if phase == 'waiting for black' else "."))
        winsound.Beep(400, 700)
    return {'status': 'PASS' if landed_ms is not None else 'FAIL',
            'response_time_ms': landed_ms if landed_ms is not None else last_ms, 'frame_count': count}


def measure_black_area(cap, press_key, soc, name, out_dir, timeout, run_idx, run_label="RUN"):
    """The landing = the picture filling the saved black area of test `name` (ensure_black_area first)."""
    import Omedia_TC02 as channel
    area = ((_settings()['models'].get(soc) or {}).get('black_areas') or {}).get(name)
    if not area:
        raise RuntimeError(f"No black area saved for {soc} / {name} (ensure_black_area first).")
    capture = channel.CameraFrameCapture(cap)
    capture.start()
    try:
        pre_frames, _, _ = channel.capture_pre_roll(capture, area['box'])
        capture.clear_buffer()
        key_at = press_key()
        result = _measure_black_area_frames(capture, area, run_idx, out_dir, timeout, key_at, pre_frames)
    finally:
        capture.stop()
    detected = result['status'] == 'PASS'
    try:
        make_run_videos(out_dir, run_label=run_label, end_ms=result['response_time_ms'] if detected else None)
    except Exception as error:
        print(f"[{run_label}] Video not made: {error}")
    return {'detected': detected, 'response_time_ms': result['response_time_ms'],
            'frame_count': result['frame_count'], 'dir': out_dir, 'method': 'black-area'}


CHANNEL_KEYS = ('ChUp', 'ChDown')
SCROLL_KEYS = ('DpadUp', 'DpadDn', 'DpadLt', 'DpadRt')


def detect(cap, press_key, resend_key, soc, run_idx, out_dir, timeout, trigger_key, mode=None, target=None,
           run_label=None):
    """Pick the method: mode 'channel' | 'area' | 'settle' | 'similarity' (default from the key: CH+/CH- ->
    channel, arrows -> settle, others -> similarity with target, default the LG Channels home).
    'area': the picture fills the black area saved for test `target` (ensure_black_area)."""
    run_label = run_label or f"RUN {run_idx}"
    mode = mode or ('channel' if trigger_key in CHANNEL_KEYS else 'settle' if trigger_key in SCROLL_KEYS
                    else 'similarity')
    if mode == 'area':
        result = measure_black_area(cap, press_key, soc, target, out_dir, timeout, run_idx, run_label)
    elif mode == 'channel':
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
