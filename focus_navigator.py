"""Camera-guided D-pad navigation: move the TV's focus highlight onto a target the user clicked.

How it works
- Areas that change on their own (preview videos, animated banners) are measured first with no
  key pressed and ignored afterwards.
- After each D-pad key, the screen is compared with the screen before the key. Only tile-sized
  changes count as focus candidates (a whole banner or background changing is ignored). When the
  old and new focus tiles touch, their merged change is split at the old tile's edge.
- Once the focus position is known, the new focus is the candidate that lies in the key's
  direction from the old focus, with the least sideways offset.
- Finding the focus the first time: keys toward the target are tried first; a key that changed
  the screen without revealing the focus is undone with the opposite key.
- The clicked target stays put unless the page scrolls; then it is re-located with a small image
  template searched in the same row band.
- Keys are chosen greedily: fix the vertical offset first (rows), then the horizontal one.
- Every step can be saved as an annotated image (debug_dir) to see what the camera decided.

Everything works on a reduced grayscale copy of the camera frame (ANALYSIS_WIDTH pixels wide).
"""
import os
import time

import cv2
import numpy as np

ANALYSIS_WIDTH = 320
DIFF_THRESHOLD = 28            # gray-level change that counts as "changed"
MIN_REGION_FRACTION = 0.002    # ignore changed regions smaller than this share of the frame
MAX_FOCUS_FRACTION = 0.20      # a changed region bigger than this is background, not the focus
SCROLL_FRACTION = 0.35         # more than this share changed with no focus move = the page scrolled
TEMPLATE_HALF_SIZE = 18        # target template is (2 * this) px square at ANALYSIS_WIDTH
TEMPLATE_MIN_SCORE = 0.45
ARRIVE_FRACTION = 0.07         # focus within 7% of the frame width of the target = arrived
MIN_STEP = 3                   # px (analysis scale) a focus must move along the key direction
MAX_STEPS = 30

MOVES = {'Up': (0, -1), 'Down': (0, 1), 'Left': (-1, 0), 'Right': (1, 0)}
OPPOSITE = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}


def prepare(frame):
    """Reduce a BGR camera frame to the small blurred grayscale image used for analysis."""
    scale = ANALYSIS_WIDTH / frame.shape[1]
    small = cv2.resize(frame, (ANALYSIS_WIDTH, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def to_analysis(point, frame_shape):
    scale = ANALYSIS_WIDTH / frame_shape[1]
    return point[0] * scale, point[1] * scale


def to_frame(point, frame_shape):
    scale = frame_shape[1] / ANALYSIS_WIDTH
    return int(point[0] * scale), int(point[1] * scale)


def change_mask(before, after, ignore=None):
    diff = cv2.absdiff(before, after)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if ignore is not None:
        mask[ignore > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Small closing only: separate tiles (and a banner next to a tile) must stay separate regions.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def row_band(box, shape):
    """Rows covered by the focus tile (with some margin): a Left/Right move stays in this band."""
    x, y, w, h = box
    return max(0, y - h // 2), min(shape[0], y + h + h // 2)


def changed_regions(before, after, ignore=None, band=None):
    """Bounding boxes (x, y, w, h) of regions that changed, largest first, plus the changed fraction.

    `band` = (top, bottom) limits the regions to those rows (the fraction still counts the whole screen).
    """
    mask = change_mask(before, after, ignore)
    fraction = float(np.count_nonzero(mask)) / mask.size
    if band is not None:
        mask = mask.copy()
        mask[:band[0], :] = 0
        mask[band[1]:, :] = 0
        # Rows changed across nearly the whole width belong to a banner/background, not a tile.
        mask[(mask > 0).mean(axis=1) > 0.8, :] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_REGION_FRACTION * mask.size
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes, fraction


def animation_mask(stills):
    """Areas that change between stills taken with no key pressed (videos, animated banners)."""
    mask = None
    for a, b in zip(stills, stills[1:]):
        current = change_mask(a, b)
        mask = current if mask is None else cv2.bitwise_or(mask, current)
    if mask is None:
        return None
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))


def center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def contains(box, point, margin=0):
    x, y, w, h = box
    return x - margin <= point[0] <= x + w + margin and y - margin <= point[1] <= y + h + margin


def beyond(box, previous_box, key):
    """Part of `box` past the previous focus tile in the key's direction (adjacent tiles merge into one region)."""
    x, y, w, h = box
    px, py, pw, ph = previous_box
    if key == 'Right':
        x0 = max(x, px + pw)
        return (x0, y, x + w - x0, h) if x + w - x0 > MIN_STEP else None
    if key == 'Left':
        x1 = min(x + w, px)
        return (x, y, x1 - x, h) if x1 - x > MIN_STEP else None
    if key == 'Down':
        y0 = max(y, py + ph)
        return (x, y0, w, y + h - y0) if y + h - y0 > MIN_STEP else None
    y1 = min(y + h, py)
    return (x, y, w, y1 - y) if y1 - y > MIN_STEP else None


def focus_candidates(boxes, image_area):
    return [b for b in boxes if b[2] * b[3] <= MAX_FOCUS_FRACTION * image_area]


def pick_new_focus(key, boxes, image_area, previous=None, previous_box=None):
    """Choose the new focus (center, box) among changed regions, or None."""
    dx, dy = MOVES[key]
    if previous is not None and previous_box is not None:
        # Split regions that merged the old and new focus tiles before judging their size.
        split = []
        for box in boxes:
            if contains(box, previous, margin=2):
                part = beyond(box, previous_box, key)
                if part is not None:
                    split.append(part)
            else:
                split.append(box)
        boxes = split
    candidates = focus_candidates(boxes, image_area)
    if not candidates:
        return None
    if previous is not None:
        best, best_score = None, None
        for box in candidates:
            if contains(box, previous, margin=2):
                continue                      # that is the tile the focus just left
            cx, cy = center(box)
            vx, vy = cx - previous[0], cy - previous[1]
            along = vx * dx + vy * dy
            sideways = abs(vx * dy - vy * dx)
            if along < MIN_STEP:
                continue
            score = sideways * 2.0 + along * 0.5   # nearest tile in that direction, little sideways drift
            if best_score is None or score < best_score:
                best, best_score = box, score
        return (center(best), best) if best is not None else None
    if len(candidates) >= 2:
        best = max(candidates[:2], key=lambda b: center(b)[0] * dx + center(b)[1] * dy)
        return center(best), best
    x, y, w, h = candidates[0]
    if dx:
        half = (x + w // 2, y, w - w // 2, h) if dx > 0 else (x, y, w // 2, h)
    else:
        half = (x, y + h // 2, w, h - h // 2) if dy > 0 else (x, y, w, h // 2)
    return center(half), half


class TargetTracker:
    """Keeps the clicked target located on screen with a small template."""

    def __init__(self, image, point):
        self.point = (float(point[0]), float(point[1]))
        self.template = self._crop(image, self.point)

    @staticmethod
    def _crop(image, point):
        h, w = image.shape
        x, y = int(point[0]), int(point[1])
        s = TEMPLATE_HALF_SIZE
        x0, y0 = max(0, x - s), max(0, y - s)
        x1, y1 = min(w, x + s), min(h, y + s)
        return image[y0:y1, x0:x1].copy(), (x - x0, y - y0)

    def in_place(self, image):
        """True when the target still looks the same at its current spot (nothing scrolled under it)."""
        template, (ox, oy) = self.template
        th, tw = template.shape
        h, w = image.shape
        x, y = int(self.point[0]) - ox, int(self.point[1]) - oy
        if th < 8 or tw < 8 or x < 0 or y < 0 or x + tw > w or y + th > h:
            return False
        return cv2.matchTemplate(image[y:y + th, x:x + tw], template, cv2.TM_CCOEFF_NORMED)[0][0] >= 0.7

    def relocate(self, image):
        """Find the target again after a scroll, searching the same row band around the last spot."""
        template, (ox, oy) = self.template
        th, tw = template.shape
        if th < 8 or tw < 8:
            return self.point
        h, w = image.shape
        # Still in place? Then nothing scrolled under the target.
        x, y = int(self.point[0]) - ox, int(self.point[1]) - oy
        if 0 <= x and 0 <= y and x + tw <= w and y + th <= h:
            in_place = cv2.matchTemplate(image[y:y + th, x:x + tw], template, cv2.TM_CCOEFF_NORMED)[0][0]
            if in_place >= 0.7:
                return self.point
        band = TEMPLATE_HALF_SIZE * 2
        y0 = max(0, int(self.point[1]) - oy - band)
        y1 = min(h, int(self.point[1]) - oy + th + band)
        region = image[y0:y1, :]
        if region.shape[0] < th or region.shape[1] < tw:
            return self.point
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score >= TEMPLATE_MIN_SCORE:
            self.point = (location[0] + ox, y0 + location[1] + oy)
        return self.point


def keys_toward(focus, target):
    """Keys that reduce the offset to the target: rows first, then columns."""
    tolerance = ARRIVE_FRACTION * ANALYSIS_WIDTH
    dx = target[0] - focus[0]
    dy = target[1] - focus[1]
    keys = []
    if abs(dy) > tolerance:
        keys.append('Down' if dy > 0 else 'Up')
    if abs(dx) > tolerance:
        keys.append('Right' if dx > 0 else 'Left')
    return keys


def choose_key(focus, target, blocked):
    """Greedy key choice. Returns None when the focus is on the target."""
    options = keys_toward(focus, target)
    if not options:
        return None
    for key in options:
        if key not in blocked:
            return key
    return options[0]


class StepRecorder:
    """Saves an annotated image per navigation step when a debug folder is given."""

    def __init__(self, debug_dir):
        self.debug_dir = debug_dir
        self.count = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def save(self, frame, label, boxes=(), focus_box=None, target=None, note=''):
        if not self.debug_dir:
            return
        self.count += 1
        image = frame.copy()
        scale = frame.shape[1] / ANALYSIS_WIDTH
        for x, y, w, h in boxes:
            cv2.rectangle(image, (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale)),
                          (160, 160, 160), 1)
        if focus_box is not None:
            x, y, w, h = focus_box
            cv2.rectangle(image, (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale)),
                          (0, 255, 255), 3)
        if target is not None:
            cv2.drawMarker(image, to_frame(target, frame.shape), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(image, f"{self.count:02d} {label} {note}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.debug_dir, f"step_{self.count:02d}_{label}.jpg"), image)


def navigate_to_target(press_key, grab_still, target_point_frame, log=print, max_steps=MAX_STEPS,
                       debug_dir=None):
    """Drive the focus onto the target.

    press_key(key)  sends one D-pad key.
    grab_still()    returns a settled BGR frame after the screen has finished animating.
    target_point_frame is the clicked point in camera-frame pixels.
    Returns (route, final_frame, focus_box_in_frame_px) where route is the list of keys that
    actually stayed in effect; raises RuntimeError if the target cannot be reached.
    """
    recorder = StepRecorder(debug_dir)
    frame = grab_still()
    shape = frame.shape
    current = prepare(frame)
    area = current.size
    ignore = animation_mask([current, prepare(grab_still()), prepare(grab_still())])
    if ignore is not None and np.count_nonzero(ignore) > 0.5 * area:
        ignore = None   # almost everything is moving; masking would hide the focus too
    tracker = TargetTracker(current, to_analysis(target_point_frame, shape))
    recorder.save(frame, "start", target=tracker.point)
    route = []

    def step(key, band=None):
        nonlocal frame, current
        press_key(key)
        frame = grab_still()
        after = prepare(frame)
        boxes, fraction = changed_regions(current, after, ignore, band)
        current = after
        return boxes, fraction

    # 1) Find where the focus is: try keys toward the target first, undoing keys that changed the
    #    screen without showing a focus move.
    focus = focus_box = None
    screen_center = (ANALYSIS_WIDTH / 2.0, current.shape[0] / 2.0)
    first_keys = keys_toward(screen_center, tracker.point) or ['Down']
    for key in first_keys + [k for k in ('Down', 'Right', 'Up', 'Left') if k not in first_keys]:
        boxes, fraction = step(key)
        found = pick_new_focus(key, boxes, area)
        if found is not None:
            focus, focus_box = found
            route.append(key)
            recorder.save(frame, key, boxes, focus_box, tracker.point, "focus found")
            log(f"[NAV] Focus found at {to_frame(focus, shape)} after {key}")
            break
        if fraction > MIN_REGION_FRACTION:
            recorder.save(frame, key, boxes, None, tracker.point, f"no focus ({fraction:.0%} changed) - undo")
            log(f"[NAV] {key}: focus not visible ({fraction:.0%} changed); undoing with {OPPOSITE[key]}")
            step(OPPOSITE[key])
        else:
            recorder.save(frame, key, boxes, None, tracker.point, "no change")
            log(f"[NAV] {key}: nothing changed")
    if focus is None:
        raise RuntimeError("Could not see the focus highlight move. Check that the screen is showing "
                           f"and the camera sees the whole TV. Step images: {debug_dir or '(not saved)'}")

    # 2) Walk to the target.
    blocked = set()
    for _ in range(max_steps):
        key = choose_key(focus, tracker.point, blocked)
        if key is None:
            log(f"[NAV] Reached target with route {route}")
            recorder.save(frame, "arrived", (), focus_box, tracker.point, ' '.join(route))
            return route, frame, _box_to_frame(focus_box, shape)
        band = row_band(focus_box, current.shape) if key in ('Left', 'Right') else None
        boxes, fraction = step(key, band)
        # Look for the focus move first: a banner or background that changes with the focus also makes
        # a large change, and must not be mistaken for a scroll.
        found = pick_new_focus(key, boxes, area, previous=focus, previous_box=focus_box)
        if found is None and fraction >= SCROLL_FRACTION and not tracker.in_place(current):
            route.append(key)
            blocked.discard(OPPOSITE[key])
            tracker.relocate(current)
            recorder.save(frame, key, boxes, focus_box, tracker.point, f"scrolled {fraction:.0%}")
            log(f"[NAV] {key}: screen scrolled ({fraction:.0%} changed); keeping focus position")
            continue
        if found is None:
            blocked.add(key)
            if fraction <= MIN_REGION_FRACTION:
                recorder.save(frame, key, boxes, focus_box, tracker.point, "did not move - blocked")
                log(f"[NAV] {key}: focus did not move; treating {key} as blocked here")
            else:
                # Something changed but no tile moved the right way: undo to stay on a known spot.
                recorder.save(frame, key, boxes, focus_box, tracker.point, "unclear - undo")
                log(f"[NAV] {key}: focus move unclear ({fraction:.0%} changed); undoing")
                step(OPPOSITE[key])
            if len(blocked) == 4:
                break
            continue
        route.append(key)
        blocked.discard(OPPOSITE[key])
        focus, focus_box = found
        recorder.save(frame, key, boxes, focus_box, tracker.point)
        log(f"[NAV] {key} -> focus {to_frame(focus, shape)}, target {to_frame(tracker.point, shape)}")

    raise RuntimeError(f"Could not reach the target within {max_steps} key presses (route so far: {route}). "
                       f"Step images: {debug_dir or '(not saved)'}")


def _box_to_frame(box, shape):
    if box is None:
        return None
    scale = shape[1] / ANALYSIS_WIDTH
    x, y, w, h = box
    return int(x * scale), int(y * scale), int(w * scale), int(h * scale)


def settle_and_grab(read_frame, settle_seconds=0.6, sample_seconds=0.2):
    """Wait for UI animation to finish, then return the median of a few frames (removes noise)."""
    end_settle = time.perf_counter() + settle_seconds
    while time.perf_counter() < end_settle:
        read_frame()
    samples = []
    end_sample = time.perf_counter() + sample_seconds
    while time.perf_counter() < end_sample or not samples:
        frame = read_frame()
        if frame is not None:
            samples.append(frame)
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
