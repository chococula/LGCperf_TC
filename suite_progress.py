"""Where the FullSuite is: TC i/N, run, step j/M with countdowns, and sounds at the moments that matter.

Before each TC run its steps (timed waits, AC power cycles, measurements) are counted by a dry run of the
TC function with every TV, camera and plug action replaced; during the real run each such action calls
step(), so the screen and the console show "TC 2/4 TC03 | Step 5/21 NBC 11-1 | 02:31 left".
"""
import time
import winsound

import cv2
import numpy as np

WINDOW = "LGC FullSuite"
# Quick flow check: the same keys and screens with short waits, to verify navigation before a real run.
QUICK = {'on': False, 'wait': 10, 'boot': 60}
BOOT_WORDS = ('power on', 'power stabilization', 'boot')    # the TV must finish booting: keep up to 60 s


def ask_quick():
    answer = input("Quick flow check with short waits (check navigation / screens, results not valid)? (y/N): ")
    QUICK['on'] = answer.strip().lower() in ('y', 'yes')
    if QUICK['on']:
        print(f"[QUICK] Waits cut to {QUICK['wait']} s (boot waits after power on to {QUICK['boot']} s); "
              "results go to a separate *_QUICK.csv.")
    return QUICK['on']


def quick_seconds(seconds, name):
    """The wait to use: unchanged normally, shortened in the quick flow check."""
    if not QUICK['on']:
        return seconds
    limit = QUICK['boot'] if any(word in name.lower() for word in BOOT_WORDS) else QUICK['wait']
    return min(seconds, limit)
state = {'tc_index': 0, 'tc_total': 0, 'tc_label': '', 'run_idx': 1, 'runs': 1,
         'steps': [], 'step': 0, 'step_name': '', 'camera': None, 'phase': ''}


# ---------- sounds ----------

def _tones(*notes):
    for frequency, ms in notes:
        winsound.Beep(frequency, ms)


def sound(event):
    """tc_start, countdown_start, tick (last 3 s), countdown_end, ac_off, ac_on, attention, tc_done, all_done."""
    _tones(*{
        'tc_start': ((660, 120), (880, 120), (1100, 160)),
        'countdown_start': ((1000, 150),),
        'tick': ((900, 80),),
        'countdown_end': ((1500, 250),),
        'ac_off': ((500, 450),),
        'ac_on': ((1200, 450),),
        'attention': ((1000, 120), (1400, 120), (1000, 120), (1400, 160)),
        'tc_done': ((1200, 150), (1500, 250)),
        'all_done': ((1047, 150), (1319, 150), (1568, 150), (2093, 400)),
    }[event])


# ---------- plan and position ----------

def begin_tc(index, total, label, runs, camera=None):
    state.update(tc_index=index, tc_total=total, tc_label=label, runs=runs, run_idx=1, steps=[], step=0,
                 step_name='')
    if camera is not None:
        state['camera'] = camera
    print(f"\n{'#' * 60}\n  TC {index}/{total}: {label}  ({runs} run{'s' if runs > 1 else ''})\n{'#' * 60}")
    sound('tc_start')


def begin_run(run_idx, steps, phase=''):
    """steps: [(kind, name, seconds)] from the dry run of this TC run (phase 'Setup' for the setup)."""
    state.update(run_idx=run_idx, steps=list(steps), step=0, step_name='', phase=phase)
    total = sum(quick_seconds(seconds, name) for _, name, seconds in steps)
    print(f"[PLAN] {state['tc_label']} {phase or f'run {run_idx}'}: {len(steps)} steps, about {_mmss(total)} of timed waits")


def step(kind, name):
    """The next step starts (a timed wait, an AC power cycle or a measurement)."""
    state['step'] += 1
    state['step_name'] = name
    if state['step'] > len(state['steps']):            # an extra step the dry run did not see (one-time setup)
        state['steps'].append((kind, name, 0))
    print(f"\n[{header()}] {name}")


def header():
    s = state
    run = f" {s['phase']}" if s['phase'] else f" run {s['run_idx']}/{s['runs']}" if s['runs'] > 1 else ""
    return f"{'[QUICK] ' if QUICK['on'] else ''}TC {s['tc_index']}/{s['tc_total']} {s['tc_label']}{run} | Step {s['step']}/{len(s['steps'])}"


def remaining_after_current():
    return sum(quick_seconds(seconds, name) for _, name, seconds in state['steps'][state['step']:])


def status_lines():
    """Lines for the measurement windows (lgc_similarity draws them under its own)."""
    if not state['tc_label']:
        return []
    return [(f"{header()}: {state['step_name']}", (255, 255, 255), 0.55)]


def _mmss(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------- countdown ----------

def countdown(seconds, name, as_step=True):
    """A timed wait (one step unless as_step=False): camera view with TC / step / time left, console line,
    ticks in the last 3 s."""
    if as_step:
        step('wait', name)
        sound('countdown_start')
    seconds = quick_seconds(seconds, name)
    camera = state['camera']
    end = time.perf_counter() + seconds
    last_tick = None
    while True:
        left = end - time.perf_counter()
        if left <= 0:
            break
        whole = int(left) + 1                          # rounded up: never 00:00 while time is left
        rest = remaining_after_current() + whole
        if whole != last_tick:
            last_tick = whole
            line = f"[{header()}] {name}: {_mmss(whole)} left | TC run ~{_mmss(rest)} left"
            print(f"\r{line:<110}", end="", flush=True)
            if whole <= 3:
                sound('tick')
        frame = None
        if camera is not None:
            ok, frame = camera.read()
            frame = frame if ok else None
        if frame is None:
            frame = np.full((480, 640, 3), 30, np.uint8)
        top = (f"TC {state['tc_index']}/{state['tc_total']} {state['tc_label']}"
               + (f" {state['phase']}" if state['phase'] else
                  f" run {state['run_idx']}/{state['runs']}" if state['runs'] > 1 else "")
               + f"  |  Step {state['step']}/{len(state['steps'])}")
        cv2.imshow(WINDOW, _panel(frame, [
            (top, (0, 255, 255), 0.55),
            (f"{name}:  {_mmss(whole)} left   (TC run ~{_mmss(rest)})", (0, 255, 0), 0.55),
        ]))
        cv2.waitKey(1)
        time.sleep(max(0.0, min(0.1, end - time.perf_counter())))
    print(f"\r[{header()}] {name}: done{' ' * 60}")
    if as_step:
        sound('countdown_end')


def _panel(frame, lines):
    image = frame.copy()
    height = 10 + 24 * len(lines)                  # a thin bar: the TV stays visible
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (25, 25, 25), -1)
    for i, (text, color, scale) in enumerate(lines):
        cv2.putText(image, text, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return image


# ---------- dry run ----------

def plan_steps(module, run_fn, args, extra_patches=()):
    """Run run_fn(*args) with every action replaced and return its steps [(kind, name, seconds)].

    module: the FullSuite module (its globals are swapped and restored). extra_patches: (object, name, fn).
    """
    steps = []

    def record(kind, name, seconds=0):
        steps.append((kind, name, seconds))

    def fake_measure(*a, trigger_key='OK', **k):
        key = a[6] if len(a) > 6 else trigger_key
        record('measure', f"Measure ({key})")
        return None

    patches = [
        (module, 'send_key', lambda *a, **k: None),
        (module, 'send_shell_command_with_debug', lambda *a, **k: None),
        (module, 'wait_with_countdown_noKeyInput', lambda seconds, description="Timer": record('wait', description, seconds)),
        (module, 'run_ac_power_cycle', lambda ip, off: record('ac', f"AC power OFF {off} s -> ON", off)),
        (module, 'perform_motion_detection', fake_measure),
        (module, 'measure_lgc_home', lambda *a, **k: (record('measure', "Measure (OK)"), None)[1]),
        (module, 'wait_for_app_ready', lambda *a, **k: True),
        (module, 'wait_for_screen_stable', lambda *a, **k: True),
        (module, 'save_result', lambda *a, **k: None),
        (module, 'make_dir', lambda *a, **k: _scratch()),
        (module, 'prepare_channel_landing', lambda *a, **k: None),
    ] + list(extra_patches)
    saved = []
    for obj, name, fn in patches:
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, fn)
    real_sleep = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        run_fn(*args)
    except Exception as error:                      # an unexpected path: show what was counted
        print(f"[PLAN] Step count may be short ({type(error).__name__}: {error})")
    finally:
        time.sleep = real_sleep
        for obj, name, fn in saved:
            setattr(obj, name, fn)
    return steps


def _scratch():
    import tempfile
    return tempfile.mkdtemp(prefix="lgc_plan_")


# ---------- channel check: after Num keys + OK, did the TV really change channel? ----------

DIGIT_KEYS = {**{f'Num_0{d}': str(d) for d in range(10)}, 'DASH': '-'}
CHANGE_DIFF = 12.0            # mean change of the screen (1/8 size, gray) that means a new picture / black
CHANGE_SECONDS = 8.0
_entry = {'digits': '', 'before': None, 'busy': False}


def _screen_small():
    camera = state['camera']
    if camera is None:
        return None
    frame = None
    for _ in range(3):                         # drop buffered frames
        ok, read = camera.read()
        frame = read if ok else frame
    if frame is None:
        return None
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame[h // 10:h * 9 // 10, w // 10:w * 9 // 10], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (gray.shape[1] // 8, gray.shape[0] // 8)).astype(np.float32), frame


def before_key(key):
    """Call before a key is sent (not while a measurement is running)."""
    if _entry['busy'] or state.get('measuring'):
        return
    if key in DIGIT_KEYS:
        if not _entry['digits']:
            _entry['before'] = _screen_small()
        _entry['digits'] += DIGIT_KEYS[key]
    elif key != 'OK':
        _entry['digits'] = ''


def after_key(key, send):
    """Call after a key is sent; send(key, delay) sends one key (used to send the channel again)."""
    if key != 'OK' or not _entry['digits'] or _entry['busy'] or state.get('measuring'):
        return
    channel, before = _entry['digits'], _entry['before']
    _entry['digits'], _entry['before'] = '', None
    _entry['busy'] = True
    try:
        _verify_channel(channel, before, send)
    finally:
        _entry['busy'] = False


def _changed(before, seconds=CHANGE_SECONDS):
    if before is None:
        return None
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        now = _screen_small()
        if now is not None and float(np.mean(np.abs(now[0] - before[0]))) >= CHANGE_DIFF:
            return True
        time.sleep(0.2)
    return False


def _send_channel(channel, send):
    reverse = {v: k for k, v in DIGIT_KEYS.items()}
    for char in channel:
        send(reverse[char], 1)
    send('OK', 2)


def _verify_channel(channel, before, send):
    changed = _changed(before)
    if changed is False:
        print(f"\n[CHANNEL CHECK] Channel {channel}: the screen did not change - sending it again.")
        sound('attention')
        before = _screen_small()
        _send_channel(channel, send)
        changed = _changed(before)
        if changed is False:
            print(f"[CHANNEL CHECK] Channel {channel}: still no change (already on it, or the keys were not "
                  "received). Check the TV.")
    elif changed:
        print(f"[CHANNEL CHECK] Channel {channel}: the screen changed.")
    if not QUICK['on'] or state['camera'] is None:
        return
    sound('attention')                         # quick flow check: the user confirms the channel
    print(f"[CHANNEL CHECK] Is the TV on channel {channel}?  ENTER = yes, R = send it again, Q = stop")
    while True:
        now = _screen_small()
        if now is not None:
            cv2.imshow(WINDOW, _panel(now[1], [(f"Is the TV on channel {channel}?", (0, 255, 255), 0.6),
                                               ("ENTER = yes   R = send again   Q = stop", (255, 255, 255), 0.55)]))
        key = cv2.waitKey(50) & 0xFF
        if key in (10, 13):
            print(f"[CHANNEL CHECK] Channel {channel}: confirmed.")
            return
        if key in (ord('r'), ord('R')):
            _send_channel(channel, send)
        if key in (ord('q'), ord('Q')):
            raise KeyboardInterrupt(f"Stopped at the channel {channel} check.")
