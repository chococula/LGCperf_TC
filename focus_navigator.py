"""Camera-guided D-pad navigation: move the TV's focus highlight onto a target the user clicked.

How it works
- After each D-pad key, the screen is compared with the screen before the key. The focus highlight
  leaves one tile and lands on another, so the difference shows two regions; the one lying in the
  direction of the key press is the new focus position.
- The clicked target stays put unless the page scrolls; then it is re-located with a small image
  template, searched near its last position (the focus highlight changes a tile's look, so a
  free search could jump to a neighbour).
- Keys are chosen greedily: fix the vertical offset first (rows), then the horizontal one.

Everything works on a reduced grayscale copy of the camera frame (ANALYSIS_WIDTH pixels wide).
"""
import time

import cv2
import numpy as np

ANALYSIS_WIDTH = 320
DIFF_THRESHOLD = 28            # gray-level change that counts as "changed"
MIN_REGION_FRACTION = 0.002    # ignore changed regions smaller than this share of the frame
SCROLL_FRACTION = 0.35         # more than this share changed = the page scrolled
TEMPLATE_HALF_SIZE = 18        # target template is (2 * this) px square at ANALYSIS_WIDTH
TEMPLATE_MIN_SCORE = 0.45
ARRIVE_FRACTION = 0.07         # focus within 7% of the frame width of the target = arrived
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


def changed_regions(before, after):
    """Bounding boxes (x, y, w, h) of regions that changed, largest first, plus the changed fraction."""
    diff = cv2.absdiff(before, after)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fraction = float(np.count_nonzero(mask)) / mask.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_REGION_FRACTION * mask.size
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes, fraction


def center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def new_focus_after(key, before, after):
    """Where the focus went after pressing `key`, as (center, box), or None if nothing moved.

    The two largest changed regions are the old and new focus; the new one lies further in the
    key's direction. A single merged region is split and its leading half is used.
    """
    boxes, fraction = changed_regions(before, after)
    if not boxes:
        return None, fraction
    dx, dy = MOVES[key]
    if len(boxes) >= 2:
        candidates = boxes[:2]
        best = max(candidates, key=lambda b: center(b)[0] * dx + center(b)[1] * dy)
        return (center(best), best), fraction
    x, y, w, h = boxes[0]
    if dx:
        half = (x + w // 2, y, w - w // 2, h) if dx > 0 else (x, y, w // 2, h)
    else:
        half = (x, y + h // 2, w, h - h // 2) if dy > 0 else (x, y, w, h // 2)
    return (center(half), half), fraction


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

    def relocate(self, image):
        """Find the target again after a scroll, searching the same row band around the last spot."""
        template, (ox, oy) = self.template
        th, tw = template.shape
        if th < 8 or tw < 8:
            return self.point
        h, w = image.shape
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


def choose_key(focus, target, blocked):
    """Greedy key choice: rows first, then columns. Returns None when the focus is on the target."""
    tolerance = ARRIVE_FRACTION * ANALYSIS_WIDTH
    dx = target[0] - focus[0]
    dy = target[1] - focus[1]
    options = []
    if abs(dy) > tolerance:
        options.append('Down' if dy > 0 else 'Up')
    if abs(dx) > tolerance:
        options.append('Right' if dx > 0 else 'Left')
    if not options:
        return None
    for key in options:
        if key not in blocked:
            return key
    return options[0]


def navigate_to_target(press_key, grab_still, target_point_frame, log=print, max_steps=MAX_STEPS):
    """Drive the focus onto the target.

    press_key(key)  sends one D-pad key.
    grab_still()    returns a settled BGR frame after the screen has finished animating.
    target_point_frame is the clicked point in camera-frame pixels.
    Returns (route, final_frame, focus_box_in_frame_px) where route is the list of keys sent;
    raises RuntimeError if the target cannot be reached.
    """
    frame = grab_still()
    shape = frame.shape
    current = prepare(frame)
    tracker = TargetTracker(current, to_analysis(target_point_frame, shape))
    route = []
    focus = None
    focus_box = None
    blocked = set()

    # The first move only reveals where the focus is; pick the direction toward the target from
    # the screen centre as a reasonable guess.
    guess_center = (ANALYSIS_WIDTH / 2.0, current.shape[0] / 2.0)
    first_key = choose_key(guess_center, tracker.point, blocked) or 'Right'
    keys_to_try = [first_key] + [k for k in ('Right', 'Down', 'Left', 'Up') if k != first_key]

    for key in keys_to_try:
        press_key(key)
        route.append(key)
        frame = grab_still()
        after = prepare(frame)
        found, fraction = new_focus_after(key, current, after)
        current = after
        if fraction >= SCROLL_FRACTION:
            tracker.relocate(current)
        if found is not None and fraction < SCROLL_FRACTION:
            focus, focus_box = found
            log(f"[NAV] Focus found at {to_frame(focus, shape)} after {key}")
            break
        log(f"[NAV] {key}: focus not visible (changed {fraction:.0%}); trying another direction")
    if focus is None:
        raise RuntimeError("Could not see the focus highlight move. Check that the Home screen is showing.")

    for _ in range(max_steps):
        target = tracker.point
        key = choose_key(focus, target, blocked)
        if key is None:
            log(f"[NAV] Reached target with route {route}")
            return route, frame, _box_to_frame(focus_box, shape)
        press_key(key)
        route.append(key)
        frame = grab_still()
        after = prepare(frame)
        found, fraction = new_focus_after(key, current, after)
        current = after
        if found is None:
            log(f"[NAV] {key}: focus did not move; treating {key} as blocked here")
            route.pop()
            blocked.add(key)
            if len(blocked) == 4:
                break
            continue
        blocked.discard(OPPOSITE[key])
        if fraction >= SCROLL_FRACTION:
            # The page scrolled: the focus usually stays where it was on screen while content moves.
            log(f"[NAV] {key}: screen scrolled ({fraction:.0%} changed); keeping focus position")
            tracker.relocate(current)
        else:
            focus, focus_box = found
        log(f"[NAV] {key} -> focus {to_frame(focus, shape)}, target {to_frame(tracker.point, shape)}")

    raise RuntimeError(f"Could not reach the target within {max_steps} key presses (route so far: {route}).")


def _box_to_frame(box, shape):
    if box is None:
        return None
    scale = shape[1] / ANALYSIS_WIDTH
    x, y, w, h = box
    return int(x * scale), int(y * scale), int(w * scale), int(h * scale)


def region_similarity(frame_a, frame_b, point_frame, half_width_fraction=0.12):
    """Similarity of the area around point_frame in two frames (used to confirm the focus is right)."""
    a, b = prepare(frame_a), prepare(frame_b)
    x, y = to_analysis(point_frame, frame_a.shape)
    half = int(half_width_fraction * ANALYSIS_WIDTH)
    h, w = a.shape
    x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
    x1, y1 = min(w, int(x) + half), min(h, int(y) + half)
    patch_a, patch_b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    if patch_a.size == 0 or patch_a.std() < 1 or patch_b.std() < 1:
        return 0.0
    return float(cv2.matchTemplate(patch_b, patch_a, cv2.TM_CCOEFF_NORMED)[0][0])


def settle_and_grab(read_frame, settle_seconds=1.2, sample_seconds=0.4):
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
