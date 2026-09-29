"""Antenna-aware profiler (Phase 1: learn + advise, no control changes).

帯域ごとの床・SNRをEMA学習し、アンテナ助言を出す純粋ロジック。
DSP制御には触れない (Hyper既存適応と干渉させない) ため安全。
"""

import time


def band_key(freq_hz: int) -> str:
    """周波数→帯域キー (表示・学習単位)"""
    try:
        f = int(freq_hz)
    except Exception:
        return "UNK"
    if f < 3000000:
        return "MW"
    if f < 30000000:
        return "HF"
    if f < 108000000:
        return "FM"
    if f < 300000000:
        return "VHF"
    return "UHF"


class AntennaProfiler:
    """帯域別に床/SNRを学習し、助言文字列を返す (スレッド安全・純粋計算)"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.bands: dict[str, dict] = {}

    def update(self, band: str, snr_db: float, floor_db: float) -> None:
        """1Hz程度の間引き呼出しを想定。非有限は無視する"""
        try:
            s = float(snr_db)
            fl = float(floor_db)
        except (TypeError, ValueError):
            return
        import math
        if not (math.isfinite(s) and math.isfinite(fl)):
            return
        st = self.bands.get(band)
        if st is None:
            self.bands[band] = {"snr": s, "floor": fl, "n": 1,
                                "t": time.time()}
            return
        a = self.alpha
        st["snr"] += a * (s - st["snr"])
        st["floor"] += a * (fl - st["floor"])
        st["n"] += 1
        st["t"] = time.time()

    def advice(self, band: str) -> str:
        """帯域への助言 (未学習時は空文字)"""
        st = self.bands.get(band)
        if st is None or st["n"] < 10:
            return ""
        snr, fl = st["snr"], st["floor"]
        if snr < 3.0 and fl > -10.0:
            return f"{band}: 床高・SNR低 (ノイズ源確認/ATU推奨)"
        if snr < 8.0:
            return f"{band}: 弱電界寄り (共振化で改善余地)"
        if snr >= 15.0:
            return f"{band}: 好調"
        return f"{band}: 通常"

    def bias_hz(self, band: str) -> float:
        """帯域別カットオフバイアス (第2段・自動選択用)。
        学習不足時は0.0 (従来動作と同一)。弱電界帯のみ最大-1500Hzへ
        狭め寄せし、到達点を変えず収束を速める。強電界帯は0のまま。"""
        st = self.bands.get(band)
        if st is None or st["n"] < 20:
            return 0.0
        snr = st["snr"]
        if snr >= 8.0:
            return 0.0
        import math
        if not math.isfinite(snr):
            return 0.0
        return -1500.0 * min(1.0, max(0.0, (8.0 - snr) / 8.0))

    def summary(self) -> dict:
        return {b: {"snr_db": round(v["snr"], 1),
                    "floor_db": round(v["floor"], 1),
                    "n": v["n"], "advice": self.advice(b)}
                for b, v in self.bands.items()}
