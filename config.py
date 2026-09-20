"""
Application configuration & region profiles.

- 地域プロファイル (FM帯域・ステップ・ディエンファシス時定数・AMステップ)
- 設定ファイルの読み書き (%APPDATA%/AntigravitySDR/config.json)
- OSロケールからの国・言語の自動判定
"""

import json
import locale
import os


APP_NAME = "AntigravitySDR"


def config_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    path = os.path.join(base, APP_NAME)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        path = os.path.dirname(os.path.abspath(__file__))
    return path


CONFIG_PATH = os.path.join(config_dir(), "config.json")
LOG_PATH = os.path.join(config_dir(), "app.log")


# ----------------------------------------------------------------------
# 地域プロファイル
#   fm_start/fm_end : スキャン対象のFM帯域 (MHz)
#   fm_step_mhz     : 周波数グリッド (MHz)
#   deemphasis_us   : FMプレエンファシス時定数 (日本/欧州=50, 米国/韓国=75)
#   mw_step_khz     : 中波の周波数ステップ (日本/欧州=9, 米国=10)
# ----------------------------------------------------------------------
REGIONS = {
    "JP": {
        "label": "Japan",
        "fm_start": 76.0, "fm_end": 95.0, "fm_step_mhz": 0.1,
        "deemphasis_us": 50.0, "mw_step_khz": 9,
        "default_freq_hz": 83200000, "default_mode": "WFM",
    },
    "US": {
        "label": "North America",
        "fm_start": 87.5, "fm_end": 108.0, "fm_step_mhz": 0.2,
        "deemphasis_us": 75.0, "mw_step_khz": 10,
        "default_freq_hz": 100100000, "default_mode": "WFM",
    },
    "CCIR": {
        "label": "Europe / International",
        "fm_start": 87.5, "fm_end": 108.0, "fm_step_mhz": 0.1,
        "deemphasis_us": 50.0, "mw_step_khz": 9,
        "default_freq_hz": 98000000, "default_mode": "WFM",
    },
    "OIRT": {
        "label": "Eastern Europe (OIRT)",
        "fm_start": 65.8, "fm_end": 74.0, "fm_step_mhz": 0.03,
        "deemphasis_us": 50.0, "mw_step_khz": 9,
        "default_freq_hz": 70000000, "default_mode": "WFM",
    },
}

# 国コード -> 地域プロファイル
_COUNTRY_REGION = {
    "JP": "JP",
    "US": "US", "CA": "US", "MX": "US", "KR": "US",
    "RU": "OIRT", "BY": "OIRT", "UA": "OIRT",
}


def _system_locale() -> str:
    """OSロケール名を 'ja-JP' / 'ja_JP' のような形式で取得"""
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(85)
            if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, 85):
                return buf.value  # 例: "ja-JP"
        except Exception:
            pass
    try:
        name = locale.getlocale()[0]
        if name:
            return name
    except Exception:
        pass
    return os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")


def detect_country() -> str:
    """OSロケールから国コードを推定 (例: ja-JP / en_US -> JP / US)"""
    loc = _system_locale().replace("-", "_")
    parts = [p for p in loc.split(".")[0].split("_") if p]
    if len(parts) >= 2:
        # 末尾が国/地域コード (ja_JP -> JP)
        for part in reversed(parts[1:]):
            if len(part) == 2 and part.isalpha():
                return part.upper()
    # 言語のみ (ja) の場合は国不明
    return ""


def detect_language() -> str:
    """OSロケールから表示言語を推定 (ja / en)"""
    loc = _system_locale().lower()
    return "ja" if loc.startswith("ja") else "en"


def region_profile(country: str) -> dict:
    region = _COUNTRY_REGION.get((country or "").upper(), "CCIR")
    profile = dict(REGIONS[region])
    profile["region"] = region
    profile["country"] = (country or "??").upper()
    profile["fm_start_hz"] = int(profile["fm_start"] * 1e6)
    profile["fm_end_hz"] = int(profile["fm_end"] * 1e6)
    return profile


# ----------------------------------------------------------------------
# 設定ファイル
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    "country": None,          # None = OSロケールから自動判定
    "language": None,         # None = OSロケールから自動判定 ("ja" / "en")
    "volume": 0.7,
    "presets_fm": [],         # [{"name": str, "freq_hz": int}, ...]
    "presets_am": [],
    "presets_region": None,   # プリセットを生成した地域 (地域変更で無効化)
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for k, v in loaded.items():
                cfg[k] = v
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return cfg


def save_config(cfg: dict):
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass
