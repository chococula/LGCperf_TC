"""LG Channels lineup from the channel server (the URL used by chPerfLog.sh): which channel numbers are
Native or Pluto, and which pairs are one ChUp / ChDown apart. TC05 NtN / NtP / PtP take their channels
from it and show the check before they run.

The server answers JSON or XML (<ArrayList><item><channelNumber>...); both are read. The last good lineup
per country is kept in LGC_lineup_<country>.json and used when the server cannot be reached.
"""
import json
import os
import urllib.request
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINEUP_URL = "https://kic.cmpiss.lgtvcommon.com/api/service/external/channels?country={country}&status=P"
PLUTO = "Pluto TV"
LIST_LIMIT = 15                 # pairs listed (more can be typed by channel number)
_LOADED = {}                    # country -> (items, source), fetched once per run of the script
_COUNTRY = []


def _cache_file(country):
    return os.path.join(SCRIPT_DIR, f"LGC_lineup_{country}.json")


def parse(body):
    """[{'number', 'name', 'provider'}] from the server's JSON or XML."""
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else body
    text = text.strip()
    items = []
    if text.startswith('<'):
        for item in ET.fromstring(text).iter('item'):
            items.append({'number': (item.findtext('channelNumber') or '').strip(),
                          'name': (item.findtext('channelName') or '').strip(),
                          'provider': (item.findtext('providerName') or '').strip()})
    else:
        data = json.loads(text)
        if isinstance(data, dict):                         # {"...": [ ... ]}
            data = next((v for v in data.values() if isinstance(v, list)), [])
        for item in data:
            items.append({'number': str(item.get('channelNumber', '')).strip(),
                          'name': str(item.get('channelName', '')).strip(),
                          'provider': str(item.get('providerName', '')).strip()})
    return [i for i in items if i['number']]


def fetch(country='US', timeout=15):
    """The lineup from the server (saved as the cache), else the cached one. Returns (items, source)."""
    if country in _LOADED:
        return _LOADED[country]
    url = LINEUP_URL.format(country=country)
    result = ([], 'none')
    try:
        request = urllib.request.Request(url, headers={'Accept': 'application/json, application/xml;q=0.9'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            items = parse(response.read())
        if items:
            with open(_cache_file(country), 'w', encoding='utf-8') as cache:
                json.dump(items, cache, ensure_ascii=False, indent=1)
            result = (items, 'server')
    except Exception as error:
        print(f"[LINEUP] Server not reached ({error}); trying the saved lineup.")
    if not result[0]:
        try:
            with open(_cache_file(country), encoding='utf-8') as cache:
                result = (json.load(cache), 'saved file')
        except (OSError, ValueError):
            pass
    _LOADED[country] = result
    return result


def _number_key(number):
    try:
        return float(number)
    except ValueError:
        return None


def channels(items):
    """One entry per channel number in ChUp order: {'number', 'kind', 'name', 'provider'}; kind 'Pluto',
    'Native' or 'Mixed' (several regional channels share a number; both kinds on one number = 'Mixed')."""
    by_number = {}
    for item in items:
        if _number_key(item['number']) is None:
            continue
        by_number.setdefault(item['number'], []).append(item)
    result = []
    for number in sorted(by_number, key=_number_key):
        entries = by_number[number]
        pluto = [e['provider'] == PLUTO for e in entries]
        kind = 'Pluto' if all(pluto) else 'Native' if not any(pluto) else 'Mixed'
        names = sorted({e['name'] for e in entries})
        result.append({'number': number, 'kind': kind,
                       'name': names[0] + (f" (+{len(names) - 1})" if len(names) > 1 else ""),
                       'provider': '/'.join(sorted({e['provider'] for e in entries}))})
    return result


def _pair(lineup, src, key, dst):
    return {'from': src['number'], 'key': key, 'to': dst['number'],
            'from_name': src['name'], 'from_provider': src['provider'], 'from_kind': src['kind'],
            'to_name': dst['name'], 'to_provider': dst['provider'], 'to_kind': dst['kind']}


def pairs(items, from_kind, to_kind, prefer=()):
    """Every pair one ChUp / ChDown apart going from a `from_kind` to a `to_kind` channel ('Native' /
    'Pluto'); pairs whose names contain a `prefer` word first, then ChUp pairs, then by number."""
    lineup = channels(items)
    found = []
    for low, high in zip(lineup, lineup[1:]):
        if low['kind'] == from_kind and high['kind'] == to_kind:
            found.append(_pair(lineup, low, 'ChUp', high))
        if high['kind'] == from_kind and low['kind'] == to_kind:
            found.append(_pair(lineup, high, 'ChDown', low))

    def rank(p):
        names = f"{p['from_name']} {p['to_name']}".lower()
        return (-sum(w.lower() in names for w in prefer), p['key'] != 'ChUp', _number_key(p['from']))
    found.sort(key=rank)
    return found


def native_to_pluto(items):
    return pairs(items, 'Native', 'Pluto')


def describe(pair):
    return (f"{pair['from']} {pair.get('from_name', '')} [{pair.get('from_provider', '')}] "
            f"--{pair['key']}-->  {pair['to']} {pair.get('to_name', '')} [{pair.get('to_provider', '')}]")


def check(items, pair, from_kind, to_kind):
    """[(ok, text)]: the From / To kinds and that the key from From lands on To in the lineup."""
    lineup = channels(items)
    index = {c['number']: i for i, c in enumerate(lineup)}
    lines = []
    for role, number, want in (('From', pair['from'], from_kind), ('To', pair['to'], to_kind)):
        if number not in index:
            lines.append((False, f"{role} {number}: not in the lineup"))
            continue
        c = lineup[index[number]]
        lines.append((c['kind'] == want, f"{role} {number} {c['name']} [{c['provider']}]: {c['kind']}"
                                         + ("" if c['kind'] == want else f" (should be {want})")))
    if pair['from'] in index:
        i = index[pair['from']] + (1 if pair['key'] == 'ChUp' else -1)
        lands = lineup[i]['number'] if 0 <= i < len(lineup) else None
        lines.append((lands == pair['to'], f"{pair['key']} from {pair['from']} lands on {lands} in the lineup"
                                           + ("" if lands == pair['to'] else f" (not {pair['to']})")))
    return lines


def show(label, pair, from_kind, to_kind, country=None):
    """Print the channels a TC uses and the lineup check; returns True when every check passed."""
    country = country or (_COUNTRY[0] if _COUNTRY else 'US')
    items, source = fetch(country)
    known = {c['number']: c for c in channels(items)} if items else {}
    pair = dict(pair)
    for role in ('from', 'to'):                        # names from the lineup (also for typed channels)
        c = known.get(pair[role])
        if c and not pair.get(f'{role}_name'):
            pair[f'{role}_name'], pair[f'{role}_provider'] = c['name'], c['provider']
    print(f"\n+{'-' * 78}")
    print(f"| {label} channels ({from_kind} -> {to_kind}), {country} lineup from the {source}")
    print(f"|   measured: {describe(pair)}")
    ok_all = True
    if not items:
        print("|   (no lineup: not checked)")
        ok_all = False
    for ok, text in (check(items, pair, from_kind, to_kind) if items else []):
        ok_all &= ok
        print(f"|   {'OK ' if ok else 'XX '} {text}")
    print(f"+{'-' * 78}")
    return ok_all


def choose_pair(ask, label, from_kind, to_kind, prefer=()):
    """Ask the lineup country (once), list the `from_kind` -> `to_kind` pairs one key apart, let the user
    pick (list number or From channel number), show the check and confirm. ask = the script's input()."""
    if not _COUNTRY:
        _COUNTRY.append((ask("Channel lineup country (default: US): ").strip() or "US").upper())
    country = _COUNTRY[0]
    items, source = fetch(country)
    found = pairs(items, from_kind, to_kind, prefer) if items else []
    while True:
        if found:
            print(f"\n[LINEUP] {label}: {from_kind} -> {to_kind} with one key ({country} lineup, {source}, "
                  f"{len(found)} pairs)")
            for index, pair in enumerate(found[:LIST_LIMIT], 1):
                print(f"  {index:2d}) {describe(pair)}")
            if len(found) > LIST_LIMIT:
                print(f"      ... {len(found) - LIST_LIMIT} more: type the From channel number")
            answer = ask(f"{label} pair: list number or From channel (default: 1): ").strip() or "1"
            chosen = next((p for p in found if p['from'] == answer), None)
            if chosen is None and answer.isdigit() and 1 <= int(answer) <= min(len(found), LIST_LIMIT):
                chosen = found[int(answer) - 1]
            if chosen is None:
                print(f"[LINEUP] '{answer}' is not a {from_kind} channel next to a {to_kind} channel.")
                continue
        else:
            print(f"[LINEUP] No {country} lineup ({source}): enter the {label} channels yourself.")
            src = ask(f"{label} From channel ({from_kind}) (default: 121): ").strip() or "121"
            key = ('ChDown' if ask(f"{label} key, U = ChUp / D = ChDown (default: U): ").strip().lower()
                   in ('d', 'down', 'chdown') else 'ChUp')
            dst = ask(f"{label} To channel ({to_kind}) (default: 122): ").strip() or "122"
            chosen = {'from': src, 'key': key, 'to': dst}
        show(label, chosen, from_kind, to_kind, country)
        if ask(f"Use these channels for {label}? (Y/n): ").strip().lower() in ('', 'y', 'yes'):
            return chosen
