"""Keys typed while the keyboard is in Korean (Hangul) mode count as the Latin key under them (2-beolsik):
'ㅛ' -> 'y', 'ㅜ' -> 'n', '요' -> 'dy', 'ㅃ' -> 'Q'.

install() (run on import):
- console input(): the answer is converted (the console is another process, its IME cannot be turned off);
- OpenCV windows: the IME is turned off for this process (Windows), so keys arrive as Latin even in Hangul
  mode; cv2.waitKey / cv2.waitKeyEx also convert a Hangul key code if one still arrives.
"""
import builtins
import sys

import cv2

KEY_OF = {
    'ㅂ': 'q', 'ㅈ': 'w', 'ㄷ': 'e', 'ㄱ': 'r', 'ㅅ': 't', 'ㅛ': 'y', 'ㅕ': 'u', 'ㅑ': 'i', 'ㅐ': 'o', 'ㅔ': 'p',
    'ㅁ': 'a', 'ㄴ': 's', 'ㅇ': 'd', 'ㄹ': 'f', 'ㅎ': 'g', 'ㅗ': 'h', 'ㅓ': 'j', 'ㅏ': 'k', 'ㅣ': 'l',
    'ㅋ': 'z', 'ㅌ': 'x', 'ㅊ': 'c', 'ㅍ': 'v', 'ㅠ': 'b', 'ㅜ': 'n', 'ㅡ': 'm',
    # with Shift
    'ㅃ': 'Q', 'ㅉ': 'W', 'ㄸ': 'E', 'ㄲ': 'R', 'ㅆ': 'T', 'ㅒ': 'O', 'ㅖ': 'P',
    # two keys
    'ㅘ': 'hk', 'ㅙ': 'ho', 'ㅚ': 'hl', 'ㅝ': 'nj', 'ㅞ': 'np', 'ㅟ': 'nl', 'ㅢ': 'ml',
    'ㄳ': 'rt', 'ㄵ': 'sw', 'ㄶ': 'sg', 'ㄺ': 'fr', 'ㄻ': 'fa', 'ㄼ': 'fq', 'ㄽ': 'ft', 'ㄾ': 'fx',
    'ㄿ': 'fv', 'ㅀ': 'fg', 'ㅄ': 'qt',
}
INITIALS = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
VOWELS = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
FINALS = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
          'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def to_latin(text):
    """Hangul (letters or composed syllables) -> the keys typed for it; everything else unchanged."""
    out = []
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:                       # a composed syllable: initial + vowel (+ final)
            index = code - 0xAC00
            parts = (INITIALS[index // 588], VOWELS[index % 588 // 28], FINALS[index % 28])
            out.append(''.join(KEY_OF.get(part, part) for part in parts))
        else:
            out.append(KEY_OF.get(char, char))
    return ''.join(out)


def _latin_code(code):
    """A key code from cv2.waitKeyEx: a Hangul letter -> the code of its (first) Latin key."""
    if code is None or code < 0x80:
        return code
    char = chr(code & 0xFFFF) if (code & 0xFFFF) else ''
    latin = to_latin(char)
    return ord(latin[0]) if latin and latin != char else code


def _disable_ime():
    """Turn the IME off for every window of this process (OpenCV windows then get Latin keys)."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        return bool(ctypes.windll.imm32.ImmDisableIME(ctypes.c_uint32(0xFFFFFFFF)))
    except Exception:
        return False


def install():
    if getattr(builtins.input, '_hangul_keys', False):
        return
    original_input = builtins.input

    def input_latin(prompt=''):
        return to_latin(original_input(prompt))
    input_latin._hangul_keys = True
    builtins.input = input_latin

    original_wait_ex = cv2.waitKeyEx

    def wait_key_ex(delay=0):
        return _latin_code(original_wait_ex(delay))

    def wait_key(delay=0):
        code = wait_key_ex(delay)
        return -1 if code == -1 else code & 0xFF

    cv2.waitKeyEx = wait_key_ex
    cv2.waitKey = wait_key
    _disable_ime()


install()
