"""
Application configuration & region profiles.

- 地域プロファイル (FM帯域・ステップ・ディエンファシス時定数・AMステップ)
- 設定ファイルの読み書き (%APPDATA%/KomorebiSDR/config.json)
- OSロケールからの国・言語の自動判定
"""

import json
import locale
import os
import shutil
import threading


APP_NAME = "KomorebiSDR"
_CONFIG_LOCK = threading.Lock()


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
# 短波 (HF) 放送バンド (kHz)
#   RTL-SDRのダイレクトサンプリングで受信可能な ~14.4MHz 以下を対象。
#   それ以上のバンドはアップコンバータが必要。
# ----------------------------------------------------------------------
SHORTWAVE_BANDS = [
    ("120m", 2300, 2495),
    ("90m", 3200, 3400),
    ("75m", 3900, 4000),
    ("60m", 4750, 5060),
    ("49m", 5900, 6200),
    ("41m", 7200, 7450),
    ("31m", 9400, 9900),
    ("25m", 11600, 12100),
    ("22m", 13570, 13870),
]

SW_MAX_HZ = 14400000  # ダイレクトサンプリングの実用上限


# ----------------------------------------------------------------------
# アマチュア無線 HF バンド (kHz。ダイレクトサンプリング上限以下のみ)
#   15m/10m・VHF/UHFは別経路のため対象外。モードは10MHz境でLSB/USB。
# ----------------------------------------------------------------------
HAM_BANDS = [
    ("80m", 3500, 3570),
    ("40m", 7000, 7200),
    ("20m", 14000, 14350),
]


def ham_band_mode(freq_hz: int) -> str:
    """アマチュア無線の慣例モード (10MHz未満LSB、以上USB)。CWはSSBで検出後に切替"""
    return "LSB" if freq_hz < 10000000 else "USB"


def shortwave_band_name(freq_hz: int) -> str:
    """周波数から短波放送バンド名 (49m等) を返す。中波は MW。"""
    for name, lo_khz, hi_khz, *_rest in list(SHORTWAVE_BANDS) + list(MW_BANDS):
        if lo_khz * 1000 <= freq_hz <= hi_khz * 1000:
            return name
    return "SW"


def shortwave_scan_centers(rate_hz: float, bands=None, usable_max_hz: int = SW_MAX_HZ,
                           overlap_hz: int = 100000) -> list:
    """短波バンドを覆うセンタ周波数リストを生成 (各帯域は窓幅で確実にカバー)。
    bands要素は (name, lo_khz, hi_khz[, grid_hz])。4要素目のグリッドは
    scan_band_hf側で使用し、ここでは無視する。"""
    bands = bands if bands is not None else SHORTWAVE_BANDS
    centers = []
    half = rate_hz / 2.0 - overlap_hz
    if half <= 0:
        return centers
    step = 2.0 * half
    for _name, lo_khz, hi_khz, *_rest in bands:
        lo = lo_khz * 1000
        hi = min(hi_khz * 1000, usable_max_hz)
        if hi <= lo:
            continue
        fc = lo + half
        if fc > hi - half:
            # 帯域幅が窓幅未満: 下限を外す中心ではなく帯域中央1点でカバー
            centers.append(int(round((lo + hi) / 2.0)))
            continue
        while True:
            centers.append(int(round(min(fc, hi - half))))
            if fc + half >= hi:
                break
            fc += step
    return sorted(set(centers))


# ----------------------------------------------------------------------
# 中波 (MW) バンド (kHz)。短波スキャンと同一経路で走査する。
# 4要素目はグリッド (9kHz)。NHK第1 594 / 第2 693kHzを含む。
# ----------------------------------------------------------------------
MW_BANDS = [
    ("MW", 531, 1602, 9000),
]


# ----------------------------------------------------------------------
# アマチュア無線 VHF/UHF バンド (Hz)。通常チューナー経路で走査する。
# ----------------------------------------------------------------------
HAM_VHF_BANDS = [
    ("2m", 144000000, 146000000, "NFM"),
    ("70cm", 430000000, 440000000, "NFM"),
]


# ----------------------------------------------------------------------
# 設定ファイル
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    "country": None,          # None = OSロケールから自動判定
    "language": None,         # None = OSロケールから自動判定 ("ja" / "en")
    "volume": 0.7,
    "stereo": True,           # FMステレオ復調
    "stereo_nr": True,        # ステレオノイズリダクション (弱電界ヒス対策)
    "presets_fm": [],         # [{"name": str, "freq_hz": int}, ...]
    "presets_am": [],
    "presets_region": None,   # プリセットを生成した地域 (地域変更で無効化)
    "ppm": None,              # ドングルPPM較正値 (None = 未較正)。PpmCalibratorが自動更新
    # 黒魔法三点セット (弱電界検出補助)。master既定OFF: 全機能が既存経路のみで動作。
    # 各機能は独立にON/OFFでき、音声を壊しうるRMT/SRは既定で無効または最小強度。
    "black_magic": {
        "enabled": False,
        "cyclostationary": {
            "enabled": True,       # master ON時のみ有効 (PLL置換ではなく助言)
            "pilot_frequency_hz": 19000.0,
            "min_confidence": 0.55,
            "smoothing_seconds": 0.25,
        },
        "rmt_denoiser": {
            "enabled": False,      # 既定無効 (音声変形リスクのため)
            "max_strength": 0.65,
            "max_rank": 8,
            "max_matrix_size": 256,
            "cpu_budget_percent": 20.0,
        },
        "stochastic_resonance": {
            "enabled": False,      # 既定無効 (検出補助専用)
            "detector_only": True,  # Falseは拒否される (音声経路保護)
            "sigma_ratio_min": 0.01,
            "sigma_ratio_max": 0.10,
            "trials": 4,
            "min_snr_db": -5.0,
            "max_snr_db": 12.0,
        },
        "adaptive_notch": {
            "enabled": False,      # 既定無効 (ハムのない環境では素通し)
            "base_hz": 0.0,        # 0=50/60Hz自動選択、50.0/60.0で固定
            "max_harmonic": 5,
            "line_on_db": 8.0,
        },
        "squelch_assist": {
            "enabled": False,      # 既定無効 (cyclo→スケルチ統合)
            "open_conf": 0.75,     # これ超で開
            "close_conf": 0.55,    # これ割れ＋S低で閉 (ヒステリシス)
            "close_smeter_db": -25.0,
            "open_smeter_db": -40.0,
            "min_close_blocks": 20,  # 一度閉じたら最低保持 (呼吸防止)
        },
        "seeking": {
            "enabled": False,      # 既定無効 (Q最適収束。RMT capの微調整のみ)
        },
    },
}


def _clean_preset_list(v, default_mode: str) -> list:
    """プリセット配列を検証・正規化 (破損エントリは除去)。"""
    valid_modes = {"WFM", "AM", "NFM", "USB", "LSB", "CW"}
    out = []
    if not isinstance(v, list):
        return out
    for p in v:
        if not isinstance(p, dict):
            continue
        f = p.get("freq_hz")
        if isinstance(f, bool) or not isinstance(f, (int, float)):
            continue
        fi = int(f)
        if not (100000 <= fi <= 1750000000):
            continue
        m = p.get("mode", default_mode)
        if not isinstance(m, str) or m not in valid_modes:
            m = default_mode
        out.append({"name": str(p.get("name", "?")), "freq_hz": fi,
                    "mode": m})
    return out


def _clean_black_magic(v) -> dict:
    """black_magic設定を検証・正規化 (破損値は既定へ戻す。未知キーは捨てる)。"""
    import copy
    default = copy.deepcopy(DEFAULT_CONFIG["black_magic"])
    if not isinstance(v, dict):
        return default
    out = copy.deepcopy(default)
    m = v.get("enabled", False)
    if isinstance(m, bool):
        out["enabled"] = m

    def _num(d, key, lo, hi, integer=False):
        try:
            x = d.get(key, None)
            if isinstance(x, bool):
                return
            if isinstance(x, (int, float)):
                v = min(max(float(x), lo), hi)
                out_sub[key] = int(round(v)) if integer else v
        except (TypeError, ValueError):
            pass

    c = v.get("cyclostationary")
    if isinstance(c, dict):
        out_sub = out["cyclostationary"]
        if isinstance(c.get("enabled"), bool):
            out_sub["enabled"] = c["enabled"]
        _num(c, "pilot_frequency_hz", 1000.0, 100000.0)
        _num(c, "min_confidence", 0.0, 1.0)
        _num(c, "smoothing_seconds", 0.01, 5.0)
    r = v.get("rmt_denoiser")
    if isinstance(r, dict):
        out_sub = out["rmt_denoiser"]
        if isinstance(r.get("enabled"), bool):
            out_sub["enabled"] = r["enabled"]
        _num(r, "max_strength", 0.0, 1.0)
        _num(r, "max_rank", 1, 64, integer=True)
        _num(r, "max_matrix_size", 16, 4096, integer=True)
        _num(r, "cpu_budget_percent", 0.0, 100.0)
    s = v.get("stochastic_resonance")
    if isinstance(s, dict):
        out_sub = out["stochastic_resonance"]
        if isinstance(s.get("enabled"), bool):
            out_sub["enabled"] = s["enabled"]
        # detector_only=Falseは受け付けない (音声経路保護のため常にTrue)
        _num(s, "sigma_ratio_min", 0.0, 1.0)
        _num(s, "sigma_ratio_max", 0.0, 1.0)
        _num(s, "trials", 2, 16, integer=True)
        _num(s, "min_snr_db", -40.0, 40.0)
        _num(s, "max_snr_db", -40.0, 40.0)
    if out["stochastic_resonance"]["sigma_ratio_min"] > \
            out["stochastic_resonance"]["sigma_ratio_max"]:
        out["stochastic_resonance"]["sigma_ratio_min"] = \
            out["stochastic_resonance"]["sigma_ratio_max"]
    a = v.get("adaptive_notch")
    if isinstance(a, dict):
        out_sub = out["adaptive_notch"]
        if isinstance(a.get("enabled"), bool):
            out_sub["enabled"] = a["enabled"]
        _num(a, "base_hz", 0.0, 100.0)
        _num(a, "max_harmonic", 1, 9, integer=True)
        _num(a, "line_on_db", 0.0, 40.0)
    q = v.get("squelch_assist")
    if isinstance(q, dict):
        out_sub = out["squelch_assist"]
        if isinstance(q.get("enabled"), bool):
            out_sub["enabled"] = q["enabled"]
        _num(q, "open_conf", 0.0, 1.0)
        _num(q, "close_conf", 0.0, 1.0)
        _num(q, "close_smeter_db", -120.0, 0.0)
        _num(q, "open_smeter_db", -120.0, 0.0)
        _num(q, "min_close_blocks", 0, 200, integer=True)
    sk = v.get("seeking")
    if isinstance(sk, dict):
        if isinstance(sk.get("enabled"), bool):
            out["seeking"]["enabled"] = sk["enabled"]
    return out


def load_config() -> dict:
    # 浅コピーだとプリセットリストがDEFAULT_CONFIGの参照を共有し、
    # 呼出側のリスト直接変更で既定値が汚れるため深く複製する
    cfg = {k: (list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_CONFIG.items()}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for k, v in loaded.items():
                if k not in DEFAULT_CONFIG:
                    continue  # 未知キーは残留させない
                if k in ("presets_fm", "presets_am"):
                    cfg[k] = _clean_preset_list(
                        v, "WFM" if k == "presets_fm" else "AM")
                elif k == "volume":
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float)):
                        cfg[k] = max(0.0, min(1.0, float(v)))
                elif k in ("stereo", "stereo_nr"):
                    if isinstance(v, bool):
                        cfg[k] = v
                elif k in ("country", "language", "presets_region"):
                    if v is None or isinstance(v, str):
                        cfg[k] = v
                elif k == "ppm":
                    if v is None:
                        cfg[k] = None
                    elif isinstance(v, bool):
                        pass
                    elif isinstance(v, (int, float)) and -200.0 <= float(v) <= 200.0:
                        cfg[k] = int(v)
                elif k == "black_magic":
                    cfg[k] = _clean_black_magic(v)
                # 型不一致は既定値を維持 (破損値でのクラッシュ防止)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return cfg


def save_config(cfg: dict):
    with _CONFIG_LOCK:
        tmp = f"{CONFIG_PATH}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            os.replace(tmp, CONFIG_PATH)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
