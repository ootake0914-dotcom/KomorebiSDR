"""確率共鳴による弱信号検出補助 (C)。

注意: 確率共鳴は真のSNRを改善しない。非線形な検出器 (スケルチ・
パイロット判定) の「中間領域での不安定性」を、微小ノイズを加えた
複数試行の一致率で補助する用途に限定する。

- メイン音声・IQ経路には一切ノイズを加えない (detector_only=True固定。
  Falseの指定は拒否する)。
- 検出用副経路のみ: 入力ブロックへ微小ノイズを加えた複数試行で
  基礎検出器を実行し、一致率をconfidenceに変換する。
- 有効条件: SNR範囲内・基礎confidence中間領域・クリップなし・
  有限入力・候補存在。それ以外は無効＋理由を返す。
- ノイズ量を増やして誤検出が増える場合は採用しない (reject判定)。

基礎検出器は呼び出し側が渡す関数 bool fn(x_noisy) で、入力配列は
複写して使う (呼び出し側の配列を書き換えない)。
"""

import math
import time

import numpy as np


class StochasticResonanceDetector:
    """検出専用の確率共鳴アシスタント。音声入出力は持たない。"""

    def __init__(self, detector_only=True, sigma_ratio_min=0.01,
                 sigma_ratio_max=0.10, trials=4, min_snr_db=-5.0,
                 max_snr_db=12.0, mid_lo=0.3, mid_hi=0.7, seed=12345):
        if not detector_only:
            raise ValueError("detector_only=Falseは禁止 (音声経路へのノイズ付加になる)")
        self.detector_only = True
        self.sigma_min = float(sigma_ratio_min)
        self.sigma_max = float(sigma_ratio_max)
        self.trials = int(max(2, trials))
        self.min_snr_db = float(min_snr_db)
        self.max_snr_db = float(max_snr_db)
        self.mid_lo = float(mid_lo)
        self.mid_hi = float(mid_hi)
        self.seed = int(seed)
        self.reset_stats()

    def reset_stats(self):
        """選局時にカウンタをクリア (前局の検出率を持ち越さない)。"""
        self.normal_hits = 0
        self.sr_hits = 0
        self.normal_fa = 0
        self.sr_fa = 0
        self.assessed = 0
        self.adopted = 0
        self.rejected = 0
        self._rng = np.random.default_rng(self.seed)

    def reset(self):
        self.reset_stats()

    def _sigma_ratio(self, snr_db):
        """弱いほど大きく (上限つき)。SNR範囲外は0 (無効扱いと等価)。"""
        if not math.isfinite(snr_db):
            return 0.0
        if snr_db < self.min_snr_db or snr_db > self.max_snr_db:
            return 0.0
        t = (self.max_snr_db - snr_db) / max(self.max_snr_db - self.min_snr_db, 1e-9)
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def assess(self, x, base_detect_fn, noise_floor, snr_db,
               base_confidence, clip=False, candidate_present=True,
               has_signal_truth=None):
        """副経路評価 → 結果辞書。xは読み取り専用 (複写して使う)。

        base_detect_fn: noisy配列→bool(検出)の基礎検出器。
        has_signal_truth: ベンチ用の真値 (Noneなら誤検出評価をしない)。
        """
        t0 = time.perf_counter()
        out = {"enabled": False, "reason": "", "sr_confidence": 0.0,
               "agreement": 0.0, "sigma_ratio": 0.0, "adopt": False,
               "processing_ms": 0.0}
        try:
            xa = np.array(np.asarray(x, dtype=np.float64).reshape(-1),
                          dtype=np.float64, copy=True)
        except Exception:
            out["reason"] = "bad-input"
            return out
        n = len(xa)
        if n < 16 or not bool(np.all(np.isfinite(xa))):
            out["reason"] = "non-finite" if n >= 16 else "too-short"
            return out
        try:
            floor = float(noise_floor)
            snr = float(snr_db)
            bconf = float(base_confidence)
        except (TypeError, ValueError):
            out["reason"] = "bad-metrics"
            return out
        if not (math.isfinite(floor) and math.isfinite(snr) and math.isfinite(bconf)):
            out["reason"] = "bad-metrics"
            return out
        if bool(clip):
            out["reason"] = "clipped"
            return out
        if not bool(candidate_present):
            out["reason"] = "no-candidate"
            return out
        # 基礎検出が確定済み (高すぎ/低すぎ) なら共鳴の出番なし
        if bconf >= self.mid_hi:
            out["reason"] = "confident-present"
            out["sr_confidence"] = bconf
            return out
        if bconf <= self.mid_lo:
            out["reason"] = "confident-absent"
            out["sr_confidence"] = bconf
            return out
        ratio = self._sigma_ratio(snr)
        if ratio <= 0.0:
            out["reason"] = "snr-out-of-range"
            return out
        # 飽和防止: 付加ノイズが入力ピークの1/4を超えないよう制限
        peak = float(np.max(np.abs(xa))) + 1e-18
        sigma = min(floor * ratio, 0.25 * peak)
        if not math.isfinite(sigma) or sigma <= 0.0:
            out["reason"] = "bad-sigma"
            return out

        hits = 0
        for _ in range(self.trials):
            noisy = xa + sigma * self._rng.standard_normal(n)
            try:
                if bool(base_detect_fn(noisy)):
                    hits += 1
            except Exception:
                pass
        agreement = hits / self.trials
        # 一致率をconfidence化: 全一致/全不一致は確定、中間は基礎値を維持
        if hits == self.trials:
            sr_conf = min(0.95, bconf + 0.2)
        elif hits == 0:
            sr_conf = max(0.05, bconf - 0.2)
        else:
            sr_conf = bconf
        out.update({"enabled": True, "reason": "",
                    "sr_confidence": float(sr_conf),
                    "agreement": float(agreement),
                    "sigma_ratio": float(sigma / (floor + 1e-18))})
        self.assessed += 1
        # ベンチ評価: 真値があれば通常/共鳴の命中・誤検出を記録し、
        # 共鳴が誤検出を増やす条件では採用しない (reject)
        if has_signal_truth is not None:
            truth = bool(has_signal_truth)
            try:
                normal = bool(base_detect_fn(xa))
            except Exception:
                normal = False
            if truth:
                self.normal_hits += int(normal)
                self.sr_hits += int(hits > self.trials // 2)
            else:
                self.normal_fa += int(normal)
                self.sr_fa += int(hits > self.trials // 2)
            fa_worse = (self.sr_fa > self.normal_fa
                        and self.assessed >= 4)
            if fa_worse:
                out["adopt"] = False
                out["reason"] = "fa-worse-reject"
                self.rejected += 1
            else:
                out["adopt"] = True
                self.adopted += 1
        else:
            out["adopt"] = True
            self.adopted += 1
        out["processing_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    def stats(self):
        """通常/共鳴の検出率比較用カウンタ。"""
        return {"assessed": self.assessed, "adopted": self.adopted,
                "rejected": self.rejected,
                "normal_hits": self.normal_hits, "sr_hits": self.sr_hits,
                "normal_fa": self.normal_fa, "sr_fa": self.sr_fa}
