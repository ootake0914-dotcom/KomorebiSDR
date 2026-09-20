"""
Internationalization (日本語 / English).

OSロケールから自動判定し、`t(key, **fmt)` で翻訳文字列を取得する。
言語は `set_language("ja"|"en")` で変更可能。
"""

_lang = "en"

STRINGS = {
    "ja": {
        "spectrum": "スペクトラム",
        "waterfall": "ウォーターフォール",
        "waveform": "音声波形",
        "status": "ステータス",
        "tuning": "選局",
        "gain_audio": "ゲイン / 音声",
        "mode": "モード",
        "freq": "周波数",
        "station_unknown": "不明な局",
        "region": "地域",
        "detected": "検出局",
        "rx_status": "受信状態",
        "stereo": "ステレオ",
        "mono": "モノラル",
        "blend": "ブレンド",
        # ステータスメッセージ
        "idle": "待機中 (Auto Seek / 全帯域スキャン可能)",
        "scan_button": "FMスキャン",
        "scan_sw_button": "短波スキャン",
        "scanning": "帯域スキャン中 ({start}〜{end}MHz)...",
        "scan_done": "スキャン完了: {n}局検出！",
        "sw_scanning": "短波スキャン中 (2〜14MHz)...",
        "sw_scan_done": "短波スキャン完了: {n}局 (最強 {freq}MHz {snr}dB)",
        "sw_scan_none": "短波局は見つかりませんでした (アンテナ/伝搬を確認)",
        "stereo_on_msg": "ステレオ受信: ON",
        "mono_on_msg": "モノラル受信: ON",
        "nr": "NR",
        "nr_on_msg": "ステレオノイズリダクション: ON",
        "nr_off_msg": "ステレオノイズリダクション: OFF",
        "bfo_msg": "BFO微調整: {hz}Hz (SSB/CW)",
        "receiving": "受信中: {freq}MHz",
        "tuned": "同調: {freq}MHz, SNR:{snr}dB",
        "auto_tuned": "自動選局: {freq}MHz, SNR:{snr}dB",
        "first_seek_scan": "初回シーク準備: 帯域スキャン中...",
        "preset_updated": "プリセットを更新しました ({n}局)",
        # ゲイン
        "gain_fixed": "固定 {db}dB",
        "gain_converged": "収束 {db}dB",
        "gain_searching": "{ctrl}: 探索中",
        "gain_manual": "手動 {db}dB",
        "gain_locked_msg": "ゲインを {db}dB に完全固定（決め打ち）しました",
        "gain_research_msg": "自動ゲイン最適化（再探索）を開始しました",
        "manual_gain_set": "手動ゲイン: {db}dB に固定しました",
        "auto_gain_resumed": "自動ゲイン最適化を再開しました",
        "lock_fixed": "固定",
        "lock_converged": "収束",
        "lock_searching": "探索中",
        # トグル
        "dx_on": "DX超高感度モード: ON",
        "dx_off": "DX超高感度モード: OFF",
        "afc_on": "AFC自動周波数追従: ON",
        "afc_off": "AFC自動周波数追従: OFF",
        # エラー
        "no_device_title": "RTL-SDR が見つかりません",
        "no_device_body": [
            "USBドングルを接続し、WinUSB/ドライバを設定してください。",
            "1. RTL-SDR を USB に接続",
            "2. Zadig (zadig.akeo.ie) で 'Bulk-In, Interface (Interface 0)' に",
            "   WinUSB ドライバをインストール",
            "3. 本アプリを再起動",
            "",
            "詳細は README を参照してください。",
        ],
        "device_error_title": "ハードウェア初期化エラー",
        "quit_hint": "何かキーを押すか、ウィンドウを閉じると終了します",
    },
    "en": {
        "spectrum": "SPECTRUM",
        "waterfall": "WATERFALL",
        "waveform": "AUDIO WAVEFORM",
        "status": "STATUS",
        "tuning": "TUNING",
        "gain_audio": "GAIN / AUDIO",
        "mode": "MODE",
        "freq": "FREQUENCY",
        "station_unknown": "Unknown Station",
        "region": "Region",
        "detected": "Stations",
        "rx_status": "Status",
        "stereo": "STEREO",
        "mono": "MONO",
        "blend": "BLEND",
        "idle": "Idle (Auto Seek / full-band scan available)",
        "scan_button": "FM scan",
        "scan_sw_button": "SW scan",
        "scanning": "Scanning ({start}–{end} MHz)...",
        "scan_done": "Scan complete: {n} stations found",
        "sw_scanning": "Scanning shortwave (2–14 MHz)...",
        "sw_scan_done": "SW scan done: {n} stations (best {freq} MHz {snr} dB)",
        "sw_scan_none": "No shortwave stations found (check antenna/propagation)",
        "stereo_on_msg": "Stereo reception: ON",
        "mono_on_msg": "Mono reception: ON",
        "nr": "NR",
        "nr_on_msg": "Stereo noise reduction: ON",
        "nr_off_msg": "Stereo noise reduction: OFF",
        "bfo_msg": "BFO fine tune: {hz} Hz (SSB/CW)",
        "receiving": "Receiving: {freq} MHz",
        "tuned": "Tuned: {freq} MHz, SNR:{snr} dB",
        "auto_tuned": "Auto-tuned: {freq} MHz, SNR:{snr} dB",
        "first_seek_scan": "Preparing first seek: scanning band...",
        "preset_updated": "Presets updated ({n} stations)",
        "gain_fixed": "Locked {db}dB",
        "gain_converged": "Converged {db}dB",
        "gain_searching": "{ctrl}: Searching",
        "gain_manual": "Manual {db}dB",
        "gain_locked_msg": "Gain locked at {db} dB",
        "gain_research_msg": "Auto gain re-search started",
        "manual_gain_set": "Manual gain set to {db} dB",
        "auto_gain_resumed": "Auto gain optimization resumed",
        "lock_fixed": "Locked",
        "lock_converged": "Converged",
        "lock_searching": "Searching",
        "dx_on": "DX high-sensitivity mode: ON",
        "dx_off": "DX high-sensitivity mode: OFF",
        "afc_on": "AFC tracking: ON",
        "afc_off": "AFC tracking: OFF",
        "no_device_title": "RTL-SDR not found",
        "no_device_body": [
            "Connect your USB dongle and install the WinUSB driver.",
            "1. Plug in the RTL-SDR",
            "2. Run Zadig (zadig.akeo.ie) and install the WinUSB driver for",
            "   'Bulk-In, Interface (Interface 0)'",
            "3. Restart this application",
            "",
            "See README for details.",
        ],
        "device_error_title": "Hardware initialization error",
        "quit_hint": "Press any key or close the window to exit",
    },
}


def set_language(lang: str):
    global _lang
    if lang in STRINGS:
        _lang = lang
    elif lang and lang.lower().startswith("ja"):
        _lang = "ja"
    else:
        _lang = "en"


def get_language() -> str:
    return _lang


def t(key: str, **fmt) -> str:
    table = STRINGS.get(_lang, STRINGS["en"])
    text = table.get(key)
    if text is None:
        text = STRINGS["en"].get(key, key)
    if isinstance(text, str) and fmt:
        try:
            return text.format(**fmt)
        except Exception:
            return text
    return text
