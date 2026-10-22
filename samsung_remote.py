"""Samsung Smart TV remote over the network (the same API the SmartThings remote uses).

Keys sent this way reach apps such as Samsung TV Plus like the Bluetooth remote's keys, so CH+/CH-
work where the IR codes are ignored. The first connection shows an "Allow" prompt on the TV; the token
is then kept in samsung_tv_token.txt (not in git).
"""
import os
import time

from samsungtvws import SamsungTVWS

TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samsung_tv_token.txt")
CHANNEL_KEYS = {'Up': 'KEY_CHUP', 'Down': 'KEY_CHDOWN'}


class NetworkRemote:
    def __init__(self, host, name='LGCperf', timeout=60):
        self.host = host
        self.tv = SamsungTVWS(host=host, port=8002, token_file=TOKEN_FILE, name=name, timeout=timeout,
                              key_press_delay=0)
        self.opened = False

    def info(self):
        """Model and power state from the TV's REST endpoint."""
        device = self.tv.rest_device_info().get('device', {})
        return device.get('modelName', '?'), device.get('name', '?'), device.get('PowerState', '?')

    def open(self):
        if not self.opened:
            self.tv.open()
            self.opened = True

    def send(self, key):
        """Send one key; returns perf_counter() taken right before it goes out (the timer start)."""
        self.open()
        sent_at = time.perf_counter()
        try:
            self.tv.send_key(key, key_press_delay=0)
        except Exception:
            # The socket may have closed while idle: reconnect once and resend.
            self.close()
            self.open()
            sent_at = time.perf_counter()
            self.tv.send_key(key, key_press_delay=0)
        return sent_at

    def close(self):
        if self.opened:
            try:
                self.tv.close()
            finally:
                self.opened = False
