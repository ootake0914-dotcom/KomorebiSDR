"""
Dongle PPM auto-calibrator (PPM auto-calibration using broadcast carriers).

放送局の搬送波を絶対周波数基準として、RTL-SDRドングル水晶の個体誤差 (PPM)
をバックグラウンドで推定・適用する。NHK-FM等の放送搬送波は高精度のため、
複数局の中央値で局側誤差を相殺し、ドングル誤差だけを抽出できる。

測定原理 (実測で符号確認済み):
  受信残差誤差 err_hz = -afc_offset_hz (AFC定常時)
  ppm_sample = err_hz * 1e6 / freq_hz
"""

import statistics


class PpmCalibrator:
    """PPM推定器 (状態保持・ハードウェア非依存。ドライバ操作は呼出側で行う)"""

    MIN_SAMPLES = 3          # 推定に必要な最低局数
    MIN_SPREAD_HZ = 2000000  # 局周波数の最低分散 (単一局の局誤差に引っ張られないため)
    MAX_PPM = 150.0          # 適用上限 (これ超えは測定異常とみなす)
    OUTLIER_PPM = 15.0       # 中央値からの外れ値棄却幅

    def __init__(self):
        # {丸め周波数Hz: 最新残差Hz} (同一局の再訪は上書きで1票のまま)
        self._samples: dict[int, float] = {}

    @staticmethod
    def err_to_ppm(err_hz: float, freq_hz: float) -> float:
        return float(err_hz) * 1e6 / float(freq_hz)

    def collect(self, freq_hz: int, afc_offset_hz: float) -> int:
        """1サンプル (選局周波数, AFC定常オフセット) を記録。戻り値は有効局数。"""
        try:
            f = int(freq_hz)
            err = -float(afc_offset_hz)
        except (TypeError, ValueError):
            return len(self._samples)
        if f <= 0 or not abs(err) < 20000.0:
            return len(self._samples)
        # 同一局 (10kHz丸め) は最新値で上書き
        self._samples[(f // 10000) * 10000] = err
        return len(self._samples)

    def estimate(self):
        """中央値PPMを推定する。戻り値 (ppm_or_None, n_stations, confident)。"""
        n = len(self._samples)
        if n < self.MIN_SAMPLES:
            return None, n, False
        freqs = sorted(self._samples)
        if freqs[-1] - freqs[0] < self.MIN_SPREAD_HZ:
            return None, n, False
        ppms = sorted(self.err_to_ppm(e, f) for f, e in self._samples.items())
        med = float(statistics.median(ppms))
        # 外れ値 (局側の大ズレ) を棄却して再中央値
        kept = [p for p in ppms if abs(p - med) <= self.OUTLIER_PPM]
        if len(kept) < self.MIN_SAMPLES:
            return None, n, False
        ppm = float(statistics.median(kept))
        if abs(ppm) > self.MAX_PPM:
            return None, n, False
        return ppm, n, True

    def clear(self):
        self._samples.clear()


UNKNOWN_STATION = "Unknown FM Station"


def pick_ppm_stations(stations, n: int = 5, min_snr: float = 8.0) -> list:
    """PPM較正巡回に使う強局候補を選ぶ純粋関数。
    FM帯のみ・SNR順。公称周波数が正確な既知局を優先し、足りなければ
    未知局の強力なもの (グリッド±12kHz以内のため外れ値棄却が効く) で補う。"""
    fm = []
    try:
        for s in (stations or []):
            if not isinstance(s, dict):
                continue
            f = s.get("freq_hz")
            snr = s.get("snr_db")
            if isinstance(f, bool) or not isinstance(f, (int, float)):
                continue
            if f < 24000000:
                continue
            try:
                snr_f = float(snr)
            except (TypeError, ValueError):
                continue
            if snr_f < min_snr:
                continue
            fm.append(s)
    except Exception:
        return []
    known = [s for s in fm if s.get("name") != UNKNOWN_STATION]
    unknown = [s for s in fm if s.get("name") == UNKNOWN_STATION]
    known.sort(key=lambda s: float(s.get("snr_db", 0.0)), reverse=True)
    unknown.sort(key=lambda s: float(s.get("snr_db", 0.0)), reverse=True)
    return (known + unknown)[:max(0, int(n))]
