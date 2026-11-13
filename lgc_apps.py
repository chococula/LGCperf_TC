"""Apps on the Home bar for the Real World TCs (TC10 / TC11 / TC12): routes from Home by the navigator
(lgc_similarity: planned path, focus check, manual L R U D) and the check, before the run, that every app
the TC uses is installed and signed in.

Routes are saved per SoC in LGC_similarity_config.json (models[soc]['routes'][name]); the defaults are the
old fixed keys (Home bar: LG Channels, YouTube, Netflix, Amazon; Live TV PIP below).
"""
import time

import cv2

import lgc_similarity

# app -> (route name, default keys from Home, needs sign-in)
APPS = {
    'LG Channels': ('tc05_lgc_app', ['Right'], False),
    'YouTube': ('app_youtube', ['Right'] * 2, True),
    'Netflix': ('app_netflix', ['Right'] * 3, True),
    'Amazon': ('app_amazon', ['Right'] * 4, True),
    'Live TV PIP': ('tc03_pip_live_tv', ['Down'], False),
}
APP_LOAD_SECONDS = 20          # the app's first screen is shown this long before the user is asked

# what each Real World TC uses
TC_APPS = {
    'TC10': ['Amazon', 'Netflix', 'YouTube', 'LG Channels'],
    'TC11': ['Netflix', 'YouTube', 'Live TV PIP', 'LG Channels'],
    'TC12': ['Netflix', 'YouTube', 'Live TV PIP', 'LG Channels'],
}


def open_app(send, cap, soc, app):
    """With Home on screen: the navigator route to the app (focus checked), then OK."""
    name, fallback, _ = APPS[app]
    lgc_similarity.navigate_from_home(send, cap, soc, name, f"{app} on Home", fallback)


def focus_app(send, cap, soc, app):
    """With Home on screen: the navigator route to the app (focus checked), no OK (the caller presses it)."""
    name, fallback, _ = APPS[app]
    lgc_similarity.navigate_from_home(send, cap, soc, name, f"{app} on Home", fallback, press_ok=False)


def _show(cap, seconds, lines):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        left = end - time.perf_counter()
        frame = lgc_similarity._read(cap)
        cv2.imshow(lgc_similarity.NAV_WINDOW, lgc_similarity._overlay(
            frame, list(lines) + [(f"{left:.0f} s   (ENTER = ask now)", (255, 255, 255), 0.6)]))
        if cv2.waitKey(30) & 0xFF in (10, 13):
            return


def check_apps(send, cap, soc, apps):
    """Before the run (the user is at the TV): for every app, confirm its navigator route from Home (planned
    path drawn, the keys are sent and the focus is checked); apps that need an account are then opened and
    the user confirms on the camera view that they are installed and signed in (R = open again after
    fixing it on the TV, S = skip the check, Q = stop). Returns {app: 'ok' | 'skipped' | 'route'}."""
    status = {}
    for app in apps:
        name, fallback, needs_sign_in = APPS[app]
        print(f"\n[APPS] {app}: route from Home")
        lgc_similarity.plan_route_from_home(send, cap, soc, name, f"{app} on Home", fallback)
        if not needs_sign_in:
            status[app] = 'route'
            continue
        while True:
            print(f"[APPS] {app}: opening it to check it is installed and signed in")
            send('Home', 3)
            lgc_similarity.replay_route(send, soc, name, fallback)
            send('OK', 2)
            _show(cap, APP_LOAD_SECONDS, [(f"{app}: loading...", (0, 255, 255), 0.7)])
            answer = lgc_similarity._ask_on_screen(cap, [
                (f"{app}: installed and signed in (no login / install screen)?", (0, 255, 255), 0.7),
                ("ENTER = yes   R = open again (fix it on the TV first)", (255, 255, 255), 0.6),
                ("S = skip this check   Q = stop", (255, 255, 255), 0.6)], ('\r', 'r', 's', 'q'))
            if answer == 'q':
                raise KeyboardInterrupt(f"Stopped at the {app} check.")
            if answer == '\r':
                status[app] = 'ok'
                print(f"[APPS] {app}: installed and signed in (confirmed).")
                break
            if answer == 's':
                status[app] = 'skipped'
                print(f"[APPS] {app}: check skipped.")
                break
        send('Home', 3)
    lgc_similarity._close(lgc_similarity.NAV_WINDOW)
    send('Exit', 2)
    print("[APPS] " + ", ".join(f"{a}: {s}" for a, s in status.items()))
    return status


# ---------- inside an app: the steps from the app's first screen to playback ----------
STEP_LOAD_SECONDS = 6          # after each in-app OK, the next screen loads this long
CONFIRMED_IN_APP = set()       # apps whose in-app steps the user confirmed in this run: replayed, not asked
PLAY_CHECK_SECONDS = 3         # normal run: the picture must move within this window after the last step
PLAY_MOTION = 2.0              # mean frame-to-frame change of the TV screen that counts as playing


def _app_steps(soc):
    return lgc_similarity._settings()['models'].get(soc, {}).get('app_steps', {})


def _save_app_steps(soc, app, steps):
    settings = lgc_similarity._settings()
    settings['models'].setdefault(soc, {}).setdefault('app_steps', {})[app] = steps
    lgc_similarity.save_configuration(lgc_similarity.SETTINGS_FILE, settings)


def _reopen(send, cap, soc, app):
    name, fallback, _ = APPS[app]
    send('Home', 3)
    lgc_similarity.replay_route(send, soc, name, fallback)
    send('OK', 2)
    _show(cap, APP_LOAD_SECONDS, [(f"{app}: loading...", (0, 255, 255), 0.7)])


def _send_route(send, route):
    for key in route:
        send(lgc_similarity.ROUTE_KEYS[key], 0.8)


def _stop_navigating(cap, app, steps):
    """After every in-app step: 'y' = playing, stop navigating; 'n' = one more step; 'r' = redo the app."""
    return lgc_similarity._ask_on_screen(cap, [
        (f"{app}: stop navigating? (playing normally after {len(steps)} step(s))", (0, 255, 255), 0.7),
        ("Y = yes, stop (save, go on)   N = no, next navigation step", (255, 255, 255), 0.6),
        ("R = start the app again and redo every step", (255, 255, 255), 0.6)], ('y', 'n', 'r'))


def teach_in_app(send, cap, soc, app):
    """With the app's first screen on: one navigation step after another (navigator: click the highlighted
    item and the next target, or M = manual L R U D; the keys + OK are sent and the focus is checked), and
    after every step the question 'stop navigating?' until Y (playing normally): then the steps are saved.
    N = the next step; R = start the app again and redo; P in the navigator = playing already, no more step."""
    saved = _app_steps(soc).get(app) or []
    while True:
        steps = []
        answer = None
        while True:
            what = f"{app} step {len(steps) + 1}"
            previous = saved[len(steps)] if len(steps) < len(saved) else None
            saved_hint = (f" (saved: {', '.join(previous['route']) or 'OK only'})" if previous else "")
            print(f"\n[APP] {what}: click the highlighted item, then the next target; M = manual; "
                  f"P = playing normally{saved_hint}")
            lgc_similarity.attention()
            clicks = [tuple(c) for c in previous['clicks']] if previous and previous.get('clicks') else None
            chosen = lgc_similarity._select_plan(cap, what, clicks, done_text="playing normally (done)")
            if chosen == 'done':
                answer = 'y'
                break
            if chosen is None:                                   # manual: each key is sent at once
                taught = lgc_similarity._teach_route(send, cap, what)
                if taught is None:
                    continue
                route, clicks = lgc_similarity._simplify(taught), None
            else:
                route, clicks = list(chosen[0]), lgc_similarity._clicks_of(chosen)
                ok, _, _ = lgc_similarity._move_and_check(send, cap, route, clicks, what)
                if ok is False:
                    answer = lgc_similarity._ask_on_screen(cap, [
                        (f"{what}: focus NOT seen on the target", (0, 0, 255), 0.7),
                        ("ENTER = OK anyway   M = fix with L R U D from here", (255, 255, 255), 0.6)],
                        ('\r', 'm'))
                    if answer == 'm':
                        extra = lgc_similarity._teach_route(send, cap, what) or []
                        route, clicks = lgc_similarity._simplify(route + extra), None
            send('OK', 2)
            steps.append({'route': route, 'clicks': [[int(v) for v in c] for c in clicks] if clicks else None})
            print(f"[APP] {what}: {', '.join(route) or 'no move'} + OK")
            _show(cap, STEP_LOAD_SECONDS, [(f"{what} sent, loading...", (0, 255, 255), 0.7)])
            answer = _stop_navigating(cap, app, steps)          # asked after every step until Y
            if answer in ('y', 'r'):
                break
        if answer == 'y':
            _save_app_steps(soc, app, steps)
            CONFIRMED_IN_APP.add((soc, app))
            print(f"[APP] {app}: {len(steps)} step(s) saved for {soc}: "
                  + " | ".join(f"{', '.join(st['route']) or 'no move'} + OK" for st in steps))
            lgc_similarity._close(lgc_similarity.NAV_WINDOW)
            return steps
        saved = steps
        _reopen(send, cap, soc, app)


def _playing(cap, seconds=PLAY_CHECK_SECONDS):
    """Whether the TV picture moves (video playing) within `seconds`."""
    previous, moves = None, []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame = lgc_similarity._read(cap)
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 45)).astype('float32')
        if previous is not None:
            moves.append(float(abs(small - previous).mean()))
        previous = small
        cv2.waitKey(1)
    return bool(moves) and max(moves) >= PLAY_MOTION


def play_in_app(send, cap, soc, app, legacy):
    """With the app's first screen on, start the playback.
    Quick flow check, or no steps saved yet for this app: navigation steps are asked one by one until the
    user answers Y to 'stop navigating?' (teach_in_app), then saved.
    Normal run with saved steps: they are sent (each route + OK, then STEP_LOAD_SECONDS) and a moving
    picture is checked (a warning only). legacy() (the old OK, OK) is kept for callers but no longer used."""
    import suite_progress
    steps = _app_steps(soc).get(app)
    confirmed = (soc, app) in CONFIRMED_IN_APP
    if confirmed:
        print(f"[APP] {app}: steps confirmed earlier in this run, replaying them.")
    if (suite_progress.QUICK['on'] and not confirmed) or steps is None:
        if steps is None and not suite_progress.QUICK['on']:
            print(f"[APP] {app}: no saved in-app steps for {soc} yet: asking them now.")
        teach_in_app(send, cap, soc, app)
        return
    for index, st in enumerate(steps, 1):
        print(f"[APP] {app} step {index}/{len(steps)}: {', '.join(st['route']) or 'no move'} + OK")
        _send_route(send, st['route'])
        send('OK', 2)
        _show(cap, STEP_LOAD_SECONDS, [(f"{app} step {index}/{len(steps)}", (0, 255, 255), 0.7)])
    lgc_similarity._close(lgc_similarity.NAV_WINDOW)
    if not _playing(cap):
        print(f"[APP] {app}: the picture does not move - is it playing? (run the quick flow check to redo "
              "the steps)")
        suite_progress.sound('attention')


def apps_for(labels):
    """The apps the selected Real World TCs use, in a fixed order, each once."""
    wanted = {app for label in labels for app in TC_APPS.get(label, [])}
    return [app for app in APPS if app in wanted]
