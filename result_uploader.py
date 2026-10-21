import json
import os
import threading
import urllib.request


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_CONFIG_FILE = os.path.join(SCRIPT_DIR, "result_upload_config.json")
PENDING_FILE = os.path.join(SCRIPT_DIR, "result_upload_pending.jsonl")

_lock = threading.Lock()
_threads = []


def _load_upload_config():
    try:
        with open(UPLOAD_CONFIG_FILE, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or not config.get('url'):
        return None
    return config


def _post(config, row):
    payload = json.dumps({'token': config.get('token', ''), 'row': row}).encode('utf-8')
    request = urllib.request.Request(
        config['url'],
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode('utf-8'))
    if not body.get('ok'):
        raise RuntimeError(body.get('error', 'unknown error'))


def _take_pending():
    if not os.path.isfile(PENDING_FILE):
        return []
    with open(PENDING_FILE, 'r', encoding='utf-8') as pending_file:
        rows = [json.loads(line) for line in pending_file if line.strip()]
    os.remove(PENDING_FILE)
    return rows


def _save_pending(rows):
    with open(PENDING_FILE, 'a', encoding='utf-8') as pending_file:
        for row in rows:
            pending_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _upload_worker(config, row):
    with _lock:
        rows = _take_pending() + [row]
        failed = []
        for index, pending_row in enumerate(rows):
            try:
                _post(config, pending_row)
            except Exception as error:
                print(f"[UPLOAD] Failed ({error}); {len(rows) - index} row(s) kept for retry.")
                failed = rows[index:]
                break
        else:
            print(f"[UPLOAD] Uploaded {len(rows)} row(s) to the result dashboard.")
        if failed:
            _save_pending(failed)


def upload_result(row):
    """Send one result row to the dashboard in the background. Failed rows are retried next time."""
    config = _load_upload_config()
    if config is None:
        return
    thread = threading.Thread(target=_upload_worker, args=(config, dict(row)), daemon=True)
    thread.start()
    _threads.append(thread)


def wait_for_uploads(timeout=60):
    for thread in _threads:
        thread.join(timeout=timeout)
    _threads.clear()
