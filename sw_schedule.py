"""EiBi shortwave broadcast schedule lookup.

周波数 + UTC時刻 から「放送局名・言語・放送先」を引くための短波番組表。
データは EiBi (Eike Bierwirth) www.eibispace.de のスケジュールを使用。
ライセンス: 無償で利用・複製・配布・サードパーティソフトへの同梱が自由
(README.TXT "Conditions of use" 参照)。

- 初回はネットから取得し %APPDATA%/KomorebiSDR/ にキャッシュ (14日で更新)
- オフライン時はキャッシュのみ使用。無ければ空 (局名は Unknown のまま)
"""

import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone

from config import config_dir

SCHEDULE_PAGE = "http://www.eibispace.de/"
CACHE_NAME = "eibi_schedule.csv"
CACHE_MAX_AGE = 14 * 24 * 3600
DEFAULT_CSV = "http://www.eibispace.de/dx/sked-a26.csv"

# 言語コードの表示名 (EiBi独自の略号。主要なもののみ)
LANG_NAMES = {
    "M": "中国語", "E": "English", "S": "Español", "R": "Русский", "F": "Français",
    "K": "한국어", "VN": "Tiếng Việt", "J": "日本語", "A": "العربية", "D": "Deutsch",
    "P": "Português", "I": "Italiano", "HA": "Hausa", "HI": "Hindi", "MO": "Mongolian",
    "FS": "فارسی", "CA": "Cantonese", "RO": "Română", "AM": "አማርኛ", "PS": "پښتو",
    "TB": "Tibetan", "NO": "Norsk", "BR": "Burmese", "SW": "Kiswahili",
    "ID": "Indonesia", "TH": "ไทย", "TR": "Türkçe", "UR": "Urdu", "BE": "Bengali",
    "F,E": "Multi", "UI": "", "DR": "", "-CW": "", "-HF": "", "-TS": "", "-TY": "",
    # ISOコード互換 (テスト/外部データ用)
    "ja": "日本語", "zh": "中国語", "en": "English", "ko": "한국어", "de": "Deutsch",
    "fr": "Français", "es": "Español", "pt": "Português", "ru": "Русский",
}

_schedule = []
_lock = threading.Lock()
_load_attempted = 0.0


def cache_path() -> str:
    return os.path.join(config_dir(), CACHE_NAME)


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "KomorebiSDR/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def find_current_csv(timeout: float = 15.0) -> str | None:
    """トップページから現行シーズンの sked-*.csv リンクを抽出"""
    try:
        html = _http_get(SCHEDULE_PAGE, timeout).decode("latin-1", "ignore")
        m = re.findall(r"dx/sked-([ab])(\d+)\.csv", html)
        if m:
            # 文字列ソートでは a26 と b25 を誤順序にする (b>a でb25が勝ち)。
            # 季節の時系列は b25 < a26 < b26 < a27 なので、
            # aN = 2N, bN = 2N+1 で数値化して最大を選ぶ。
            def key(pr):
                letter, num = pr
                return 2 * int(num) + (1 if letter == "b" else 0)
            best = max(m, key=key)
            return SCHEDULE_PAGE + f"sked-{best[0]}{best[1]}.csv"
    except Exception:
        pass
    return None


def download_schedule(timeout: float = 20.0) -> bool:
    """現行スケジュールを取得してキャッシュへ保存 (アトミック)"""
    url = find_current_csv(timeout) or DEFAULT_CSV
    data = _http_get(url, timeout)
    if len(data) < 5000:
        return False
    path = cache_path()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def _hhmm_to_min(value: str) -> int:
    v = int(value)
    return (v // 100) * 60 + (v % 100)


def _parse_time(txt: str):
    """'0900-1000' のようなHHMM表記を分に変換 (3桁はHMM扱い)"""
    m = re.match(r"(\d{3,4})-(\d{3,4})", txt.strip())
    if not m:
        return None, None
    return _hhmm_to_min(m.group(1)), _hhmm_to_min(m.group(2))


def parse_schedule(text: str) -> list:
    """EiBi CSV (セミコロン区切り) をエントリのリストへ変換"""
    lines = text.replace("\r", "").split("\n")
    entries = []
    for line in lines[1:]:
        parts = line.split(";")
        if len(parts) < 5:
            continue
        try:
            khz = float(parts[0])
        except ValueError:
            continue
        start, stop = _parse_time(parts[1])
        if start is None:
            continue
        entries.append({
            "freq_hz": int(round(khz * 1000.0)),
            "start_min": start,
            "stop_min": stop,
            "days": parts[2].strip(),
            "itu": parts[3].strip(),
            "station": parts[4].strip(),
            "language": parts[5].strip() if len(parts) > 5 else "",
            "target": parts[6].strip() if len(parts) > 6 else "",
            "remarks": parts[7].strip() if len(parts) > 7 else "",
        })
    return entries


def load_schedule(force: bool = False):
    """スケジュールをメモリへロード (初回/期限切れはネット取得)"""
    global _schedule, _load_attempted
    with _lock:
        if _schedule and not force:
            return _schedule
        now = time.time()
        if not force and now - _load_attempted < 300.0:
            return _schedule
        _load_attempted = now
        path = cache_path()
        fresh = os.path.exists(path) and (now - os.path.getmtime(path)) < CACHE_MAX_AGE
        if force or not fresh:
            try:
                download_schedule()
            except Exception:
                pass  # オフラインでもキャッシュがあれば使う
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="latin-1", errors="ignore") as f:
                    _schedule = parse_schedule(f.read())
            except Exception:
                _schedule = []
        return _schedule


def _day_active(days: str, weekday: int) -> bool:
    """Days欄の判定 (空=毎日, 数字列=曜日, それ以外の書式は許容)"""
    days = days.strip()
    if not days:
        return True
    if days.isdigit():
        return str(weekday) in days
    return True


def lookup(freq_hz: int, when_utc: datetime = None, tolerance_hz: int = 2000):
    """周波数・時刻に一致する放送を返す (見つからなければ None)"""
    if not _schedule:
        return None
    when_utc = when_utc or datetime.now(timezone.utc)
    minute = when_utc.hour * 60 + when_utc.minute
    weekday = when_utc.weekday() + 1  # 1=月曜
    best = None
    for e in _schedule:
        if abs(e["freq_hz"] - freq_hz) > tolerance_hz:
            continue
        if not _day_active(e["days"], weekday):
            continue
        s, t = e["start_min"], e["stop_min"]
        inside = (s <= minute < t) if t > s else (minute >= s or minute < t)
        if not inside:
            continue
        dur = (t - s) if t > s else (1440 - s + t)
        has_name = 1 if e["station"] else 0
        # 名前・言語があり、時間幅が短い(具体的な)放送を優先
        score = (has_name, 1 if e["language"] else 0, -dur)
        if best is None or score > best[0]:
            best = (score, e)
    if best is None:
        return None
    e = best[1]
    lang = LANG_NAMES.get(e["language"], e["language"])
    label = e["station"] or "?"
    if lang:
        label += f" ({lang})"
    return {
        "name": label,
        "station": e["station"],
        "language": lang,
        "target": e["target"],
        "itu": e["itu"],
        "time": f"{e['start_min']//60:02d}{e['start_min']%60:02d}-"
                f"{e['stop_min']//60:02d}{e['stop_min']%60:02d}",
        "remarks": e["remarks"],
    }
