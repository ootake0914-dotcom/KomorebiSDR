"""RT締切プロファイル (HexaMIDI rt_profile 方式)

リアルタイムDSPブロックの処理時間を「事前確保したリング」へ記録し、
p50/p95/p99 と締切(budget)超過回数を低オーバーヘッドで監視する。
- 追加メモリ確保なし (起動時に固定長numpy配列を確保)
- 分位点は np.partition (O(n)) で算出し、通常のsortを避ける
- EWMA平均は追加領域なしで更新
"""

import numpy as np


class RtProfile:
    def __init__(self, budget_ms: float, size: int = 256):
        self.budget_ms = float(budget_ms)
        self._buf = np.zeros(int(size), dtype=np.float32)
        self._idx = 0
        self._n = 0
        self.blocks = 0
        self.misses = 0
        self._consec = 0
        self.max_consecutive_misses = 0
        self.last_ms = 0.0
        self.avg_ms = 0.0

    def add(self, ms: float):
        try:
            ms = float(ms)
        except Exception:
            return
        if not np.isfinite(ms):
            return
        self.last_ms = ms
        self._buf[self._idx] = ms
        self._idx += 1
        if self._idx >= len(self._buf):
            self._idx = 0
        if self._n < len(self._buf):
            self._n += 1
        self.blocks += 1
        self.avg_ms += (ms - self.avg_ms) * 0.05

        if ms > self.budget_ms:
            self.misses += 1
            self._consec += 1
            if self._consec > self.max_consecutive_misses:
                self.max_consecutive_misses = self._consec
        else:
            self._consec = 0

    def _pct(self, q: float) -> float:
        if self._n == 0:
            return 0.0
        k = int(q * (self._n - 1))
        return float(np.partition(self._buf[: self._n], k)[k])

    def percentiles(self):
        return self._pct(0.50), self._pct(0.95), self._pct(0.99)

    @property
    def headroom(self) -> float:
        p95 = self._pct(0.95)
        return self.budget_ms / p95 if p95 > 1e-6 else 0.0

    def summary(self) -> str:
        p50, _p95, p99 = self.percentiles()
        text = f"DSP {p50:.1f}/{p99:.1f}ms {self.headroom:.1f}x"
        if self.misses:
            text += f" MISS{self.misses}"
        return text
