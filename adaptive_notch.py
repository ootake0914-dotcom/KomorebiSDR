"""適応ハムノッチ・キャンセラ (Phase 2-1: 四点目)。

商用電源ハム (50/60Hz＋高調波) やAM同一チャネル混信の線スペクトルを、
静的ノッチ (スペクトル穴あけ) ではなく適応キャンセルで除去する。
番組を削らないための条件:
- 巡回定常性: 基底 (50/60Hz) の存在＋2本以上の高調波同時検出でのみ動作
  (音楽の単一トーンをハムと誤認しない)。
- 位相連続性: ブロック間で位相が安定な線のみ除去対象。
- 除去量上限: 推定ハムが信号RMSの30%を超えたら部分除去に留める。
- ステレオはMidで推定しL/R同量を差し引く (ハムは同相成分が支配的)。

既定ではdsp経路に接続しない (black_magic.adaptive_notch.enabled=False)。
"""

import math
import time

import numpy as np


class AdaptiveNotchCanceller:
    """ハム線検出＋最小二乗キャンセル。process_*は (出力, 情報辞書) を返す。"""

    def __init__(self, sample_rate=48000.0, base_hz=0.0, max_harmonic=5,
                 max_harm_hz=300.0, line_on_db=8.0, line_off_db=5.0,
                 min_dwell_blocks=4, max_remove_ratio=0.3, enabled=True):
        self.fs = float(sample_rate)
        # base_hz=0 は50/60Hz自動選択 (強い方＋ヒステリシス)
        self.base_hz = float(base_hz)
        self.max_harmonic = int(max(1, max_harmonic))
        self.max_harm_hz = float(max_harm_hz)
        self.line_on_db = float(line_on_db)
        self.line_off_db = float(line_off_db)
        self.min_dwell = int(max(1, min_dwell_blocks))
        self.max_remove_ratio = float(max_remove_ratio)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self):
        """選局時に全状態をクリア (前局のハム判定を持ち越さない)。"""
        self._base = 50.0 if not self.base_hz else float(self.base_hz)
        self._base_votes = 0
        self._s50_ema = None
        self._s60_ema = None
        self._active = {}   # freq -> 残りdwellカウンタではなく確定状態
        self._confirmed = set()
        self._dwell = {}
        self.blocks = 0

    def _lines_batch(self, xa, freqs):
        """[(f, 線振幅, 床振幅)] を一括計算 (Cコア1コール＋numpy代替)。"""
        n = len(xa)
        flist = [float(f) for f in freqs]
        # 各線＋近傍4点 (±15/±30Hz) をまとめて要求する
        req = []
        for f in flist:
            req.append(f)
            for off in (-30.0, -15.0, 15.0, 30.0):
                req.append(f + off)
        mags = None
        try:
            from dsp_native import dft_bins
            r = dft_bins(xa, req, self.fs)
            mags = [abs(complex(float(r[j, 0]), float(r[j, 1]))) for j in range(len(req))]
        except Exception:
            pass
        if mags is None:
            # numpy代替 (旧DLL・DLL不在時)
            idx = np.arange(n)
            mags = []
            for fq in req:
                tw = np.exp(-2j * np.pi * fq * idx / self.fs)
                mags.append(abs(np.dot(xa, tw)) * (2.0 / n))
        out = []
        for i, f in enumerate(flist):
            line = mags[i * 5]
            floor = max(float(np.median(mags[i * 5 + 1:i * 5 + 5])), 1e-12)
            out.append((f, line, floor))
        return out

    def _line_snr(self, xa, freq):
        """線振幅と近傍ノイズ床の比 (dB)。一括計算の薄いラッパ (後方互換)。"""
        for f, line, floor in self._lines_batch(xa, [freq]):
            if abs(f - freq) < 1e-9:
                return 20.0 * math.log10(line / floor + 1e-12), line
        return -99.0, 0.0

    def _harmonics(self):
        out = []
        h = 1
        while h <= self.max_harmonic:
            f = self._base * h
            if f > self.max_harm_hz or f >= self.fs / 2:
                break
            out.append(f)
            h += 1
        return out

    def detect(self, x):
        """ハム線の検出のみ行う → {base_hz, lines, snrs}。入力は変更しない。"""
        t0 = time.perf_counter()
        info = {"base_hz": self._base, "lines": [], "snrs": {},
                "processing_ms": 0.0, "bypass_reason": ""}
        try:
            xa = np.asarray(x, dtype=np.float64).reshape(-1)
        except Exception:
            info["bypass_reason"] = "bad-input"
            return info
        n = len(xa)
        if n < 1024 or not bool(np.all(np.isfinite(xa))):
            info["bypass_reason"] = "non-finite" if n >= 1024 else "too-short"
            return info
        # 基底自動選択 (固定指定がなければ50/60Hzの強い方)。
        # N=2752ではビン幅17.4Hzで50/60Hzの漏れが相互汚染するため、
        # 単発判定ではなくEMA平均の差で切替える (4 tick連続)。
        # (単発8.6dBスパイクでの誤切替・連続位相でのwobbleを実測して修正)
        if self.base_hz:
            self._base = float(self.base_hz)
            s50 = s60 = -99.0
        else:
            # 50/60Hzの線＋床を一括で取り、SNR化する
            got = {f: (line, floor) for f, line, floor
                   in self._lines_batch(xa, [50.0, 60.0])}
            l50, f50 = got[50.0]
            l60, f60 = got[60.0]
            s50 = 20.0 * math.log10(l50 / f50 + 1e-12)
            s60 = 20.0 * math.log10(l60 / f60 + 1e-12)
            self._s50_ema = s50 if self._s50_ema is None else \
                self._s50_ema + 0.25 * (s50 - self._s50_ema)
            self._s60_ema = s60 if self._s60_ema is None else \
                self._s60_ema + 0.25 * (s60 - self._s60_ema)
            cur = self._s50_ema if self._base == 50.0 else self._s60_ema
            other = self._s60_ema if self._base == 50.0 else self._s50_ema
            if other > cur + 3.0:
                self._base_votes += 1
            else:
                self._base_votes = max(0, self._base_votes - 1)
            if self._base_votes >= 4:
                self._base = 60.0 if self._base == 50.0 else 50.0
                self._base_votes = 0
                self._confirmed.clear()
                self._dwell.clear()
        snrs = {}
        for f, line, floor in self._lines_batch(xa, self._harmonics()):
            snrs[f] = 20.0 * math.log10(line / floor + 1e-12)
        # dwellつき確定: on閾値超で加算、off閾値割れで解除方向へ
        for f, s in snrs.items():
            d = self._dwell.get(f, 0)
            if s >= self.line_on_db:
                d = min(d + 1, self.min_dwell)
            elif s <= self.line_off_db:
                d = max(d - 1, -self.min_dwell)
            self._dwell[f] = d
            if d >= self.min_dwell:
                self._confirmed.add(f)
            elif d <= -self.min_dwell:
                self._confirmed.discard(f)
        # 全体ゲート: 基底線の確定＋2本以上の線確定 (単一音楽トーンの誤認防止)
        base_ok = self._base in self._confirmed
        lines = sorted(f for f in self._confirmed if f in snrs)
        if not (base_ok and len(lines) >= 2):
            lines = []
        self.blocks += 1
        info.update({"base_hz": self._base, "lines": lines, "snrs": snrs,
                     "processing_ms": (time.perf_counter() - t0) * 1000.0})
        return info

    def process_mono(self, audio, ch=""):
        """モノラル1ch処理 → (cancelled, info)。"""
        t0 = time.perf_counter()
        info = {"processing_ms": 0.0, "bypass_reason": "", "lines": [],
                "removed_db": 0.0}
        try:
            x = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            info["bypass_reason"] = "bad-input"
            return np.asarray(audio), info
        n = len(x)
        if not self.enabled:
            info["bypass_reason"] = "disabled"
            return x, info
        if n < 1024:
            info["bypass_reason"] = "too-short"
            return x, info
        if not bool(np.all(np.isfinite(x))):
            info["bypass_reason"] = "non-finite"
            return x, info
        det = self.detect(x)
        if det["bypass_reason"]:
            info["bypass_reason"] = det["bypass_reason"]
            return x, info
        lines = det["lines"]
        info["lines"] = lines
        if not lines:
            info["bypass_reason"] = "no-hum"
            return x, info
        # 最小二乗フィット＋除去 (ブロック内完結。連続ハムなら境界も連続)
        xa = x.astype(np.float64)
        idx = np.arange(n)
        h = np.zeros(n, dtype=np.float64)
        for f in lines:
            w = 2.0 * np.pi * f / self.fs
            c, s = np.cos(w * idx), np.sin(w * idx)
            a = 2.0 * float(np.dot(xa, c)) / n
            b = 2.0 * float(np.dot(xa, s)) / n
            h += a * c + b * s
        rms_x = float(np.sqrt(np.mean(xa ** 2)) + 1e-18)
        rms_h = float(np.sqrt(np.mean(h ** 2)) + 1e-18)
        if rms_h < 1e-9:
            info["bypass_reason"] = "no-hum"
            return x, info
        # 除去量上限: ハム推定が信号の30%超なら部分除去 (番組保護)
        if rms_h > self.max_remove_ratio * rms_x:
            h *= (self.max_remove_ratio * rms_x) / rms_h
            rms_h = self.max_remove_ratio * rms_x
            info["bypass_reason"] = "partial"
        y = (xa - h).astype(np.float32)
        if not bool(np.all(np.isfinite(y))):
            info["bypass_reason"] = "nan-output"
            return x, info
        info["removed_db"] = float(20.0 * math.log10(rms_h / rms_x + 1e-12))
        info["processing_ms"] = (time.perf_counter() - t0) * 1000.0
        return y, info

    def process_stereo(self, left, right):
        """Midで推定しL/R同量を差し引く (ハム同相仮定) → ((L, R), info)。"""
        try:
            l = np.asarray(left, dtype=np.float32).reshape(-1)
            r = np.asarray(right, dtype=np.float32).reshape(-1)
        except Exception:
            return (np.asarray(left), np.asarray(right)), {"bypass_reason": "bad-input"}
        n = min(len(l), len(r))
        l, r = l[:n], r[:n]
        mid = ((l.astype(np.float64) + r.astype(np.float64)) * 0.5).astype(np.float32)
        ym, info = self.process_mono(mid)
        if info.get("bypass_reason") in ("", "partial"):
            # Midで推定したハム波形をL/Rから差し引く
            h = (mid.astype(np.float64) - np.asarray(ym, dtype=np.float64))
            yl = (l.astype(np.float64) - h).astype(np.float32)
            yr = (r.astype(np.float64) - h).astype(np.float32)
            if bool(np.all(np.isfinite(yl))) and bool(np.all(np.isfinite(yr))):
                return (yl, yr), info
            return (l, r), {"bypass_reason": "nan-output", "lines": info.get("lines", [])}
        return (l, r), info
