"""巡回定常性を利用したFMパイロット検出・ロック補助 (A)。

19kHzステレオパイロットの周期構造を、軽量なGoertzel検出＋ブロック間
位相連続性評価で検出する。巨大な二次統計量は作らず、リアルタイム動作を優先。

- 入力: FM復調後のMPX実信号 (例: 288kHzのdemod_scaled)。
- 出力: pilot_power/noise_power/snr (dB)、pilot_present、confidence(0..1)、
  estimated_phase、estimated_frequency_offset。
- PLLを置き換えない。既存PLLへの介入は「blend上昇レート制限」の助言のみで、
  本モジュール自体は検出値を返すだけ (副作用なし)。

NOTE: 旧adaptive_rfのCyclostationary系は効果未検証のため削除済み。
本モジュールはGoertzel＋ヒステリシス＋最小継続時間の別設計で、既定では
dsp経路に接続しない (black_magic.cyclostationary.enabled=False)。
"""

import math
import time

import numpy as np


class CyclostationaryPilotDetector:
    """19kHzパイロットのGoertzel検出＋位相連続性評価器。"""

    def __init__(self, sample_rate=288000.0, target_hz=19000.0,
                 min_confidence=0.55, smoothing_seconds=0.25,
                 snr_on_db=6.0, snr_off_db=3.0, min_dwell_blocks=3,
                 narrow_db=3.0, abs_floor_dbfs=-80.0):
        self.fs = float(sample_rate)
        self.target_hz = float(target_hz)
        self.min_confidence = float(min_confidence)
        self.smoothing_seconds = float(max(1e-3, smoothing_seconds))
        self.snr_on_db = float(snr_on_db)
        self.snr_off_db = float(snr_off_db)
        self.min_dwell_blocks = int(max(1, min_dwell_blocks))
        self.narrow_db = float(narrow_db)
        self.abs_floor = float(10.0 ** (abs_floor_dbfs / 20.0))
        self.reset()

    def reset(self):
        """選局時に全状態をクリア (前局の位相・confidenceを持ち越さない)。"""
        self.confidence = 0.0
        self.pilot_present = False
        self._dwell = 0
        self._prev_phase = None
        self._phasor_sum = 0.0 + 0.0j
        self._phasor_n = 0
        self.coherence = 0.0
        self.blocks = 0
        # 周波数追従: 中心ビンのオフセット (-2..+2ビン)。隣接ビン支配が
        # 3ブロック連続したら1ビンずつ移動する (単発スパイクでは動かない)
        self._koff = 0
        self._up_n = 0
        self._dn_n = 0

    def _goertzel(self, x, k):
        """単一DFTビンの複素振幅 (正規化: 正弦振幅=|X|*2/N)。
        Cコア (sdr_dft_bins) 優先、なければnumpy。後方互換のため残すが、
        update()は一括計算を使う (9回の個別呼出しは無駄なため)。"""
        try:
            from dsp_native import dft_bins
            n = len(x)
            r = dft_bins(x, [float(k) * self.fs / max(n, 1)], self.fs)
            return complex(float(r[0, 0]), float(r[0, 1]))
        except Exception:
            return 0.0 + 0.0j

    def _bins_batch(self, xa, ks):
        """複数ビンを一括計算 → {k: complex}。Cコア1コールで済ませる。"""
        n = len(xa)
        ks = [int(k) for k in ks]
        try:
            from dsp_native import dft_bins
            freqs = [kk * self.fs / n for kk in ks]
            r = dft_bins(xa, freqs, self.fs)
            return {kk: complex(float(r[j, 0]), float(r[j, 1]))
                    for j, kk in enumerate(ks)}
        except Exception:
            return {kk: self._goertzel(xa, kk) for kk in ks}

    def update(self, x):
        """1ブロックを評価し、検出辞書を返す (入力配列は変更しない)。"""
        t0 = time.perf_counter()
        out = {
            "pilot_power_db": -120.0, "noise_power_db": -120.0,
            "pilot_snr_db": -99.0, "pilot_present": False,
            "confidence": float(self.confidence),
            "estimated_phase": 0.0, "estimated_frequency_offset": 0.0,
            "processing_ms": 0.0, "bypass_reason": "",
        }
        try:
            xa = np.asarray(x, dtype=np.float64).reshape(-1)
        except Exception:
            out["bypass_reason"] = "bad-input"
            return out
        n = len(xa)
        if n < 64 or not bool(np.all(np.isfinite(xa))):
            # 非有限入力はPLL状態汚染防止のため検出失敗扱い (confidenceは減衰)
            self.confidence *= 0.5
            out["confidence"] = float(self.confidence)
            out["pilot_present"] = False
            out["bypass_reason"] = "non-finite" if n >= 64 else "too-short"
            out["processing_ms"] = (time.perf_counter() - t0) * 1000.0
            return out

        k0 = int(round(self.target_hz * n / self.fs))
        k0 = min(max(k0, 4), n - 5)
        # 追従中心ビン (オフセット時は隣接ビンへ寄る)
        kc = min(max(k0 + self._koff, 2), n - 3)
        # 全ビンを一括計算 (Cコア1コール。個別9回より高速)
        nbs = [min(max(kc + int(round(off_hz * n / self.fs)), 1), n - 2)
               for off_hz in (1000.0, -1000.0, 1500.0, -1500.0, 2500.0, -2500.0)]
        B = self._bins_batch(xa, [kc - 1, kc, kc + 1] + nbs)
        cm = B[kc - 1]
        c0 = B[kc]
        cp = B[kc + 1]
        # ノイズ床は離調6ビン (±1/±1.5/±2.5kHz相当) の中央値
        # (平均では単一ビンのたまたまの盛り上がりでSNRが跳ねて誤検出するため。
        # ±1kHzは矩形窓の19kHzサイドローブが無視できる距離)
        nb = [abs(B[kb]) for kb in nbs]
        noise_amp = max(float(np.median(nb)), 1e-12)
        # 周波数追従: 隣接ビン支配が3ブロック連続したら中心を1ビン移動
        # (範囲±2ビン。単発スパイクでは動かず、真のオフセットに追従する)
        a_m, a_0, a_p = abs(cm), abs(c0), abs(cp)
        if a_p > 2.0 * max(a_0, a_m) and a_p >= self.abs_floor:
            self._up_n += 1
            self._dn_n = 0
        elif a_m > 2.0 * max(a_0, a_p) and a_m >= self.abs_floor:
            self._dn_n += 1
            self._up_n = 0
        else:
            self._up_n = 0
            self._dn_n = 0
        if self._up_n >= 3 and self._koff < 2:
            self._koff += 1
            self._up_n = 0
        elif self._dn_n >= 3 and self._koff > -2:
            self._koff -= 1
            self._dn_n = 0

        pilot_amp = abs(c0)
        amp_m = abs(cm)
        amp_p = abs(cp)
        # 放物線補間で真のピーク振幅・位置を推定
        # (オフセット時は主ビンから漏れるため、補間ピークを振幅とする。
        # 狭帯域判定は min(隣接) との比で取り、広帯域ノイズの盛り上がりを弾く)
        den = amp_m - 2.0 * pilot_amp + amp_p
        if abs(den) > 1e-15:
            shift = 0.5 * (amp_m - amp_p) / den
            shift = float(min(max(shift, -1.0), 1.0))
        else:
            shift = 0.0
        peak_amp = max(pilot_amp - 0.25 * (amp_m - amp_p) * shift, 1e-12)
        narrow_db = 20.0 * math.log10(peak_amp / (min(amp_m, amp_p) + 1e-12))
        pilot_db = 20.0 * math.log10(peak_amp + 1e-12)
        noise_db = 20.0 * math.log10(noise_amp)
        snr_db = 20.0 * math.log10(peak_amp / noise_amp + 1e-12)

        # 放物線補間のビンずれ→周波数オフセット推定
        # (追従中心からのずれ＋補間シフト。範囲: 中心±1.5ビン幅)
        bin_hz = self.fs / n
        freq_offset = (float(kc - k0) + shift) * bin_hz
        phase = math.atan2(c0.imag, c0.real)
        # 位相連続性: 連続音ならブロック間の位相進みは理論値
        # (2π f N/fs) に一致する。残差の単位フェーザ平均をとり、
        # 1=連続性が高い、0=ランダム (ノイズ) とする。
        if self._prev_phase is not None:
            expected_adv = 2.0 * math.pi * self.target_hz * n / self.fs
            resid = (phase - self._prev_phase - expected_adv + math.pi) % (2.0 * math.pi) - math.pi
            self._phasor_sum += complex(math.cos(resid), math.sin(resid))
            self._phasor_n += 1
            if self._phasor_n > 8:
                # 古い履歴の影響を指数忘却 (8ブロック窓相当)
                self._phasor_sum *= 0.875
                self._phasor_n = 8
            self.coherence = abs(self._phasor_sum) / max(1, self._phasor_n)
        self._prev_phase = phase

        # confidence: SNRロジスティック × 位相コヒーレンス、EMA平滑
        mid = 0.5 * (self.snr_on_db + self.snr_off_db)
        raw = 1.0 / (1.0 + math.exp(-(snr_db - mid) / 1.5))
        if pilot_amp < self.abs_floor:
            raw *= pilot_amp / self.abs_floor
        raw *= 0.5 + 0.5 * float(min(max(self.coherence, 0.0), 1.0))
        dt = n / self.fs
        alpha = 1.0 - math.exp(-dt / self.smoothing_seconds)
        # 狭帯域でない強入力 (広帯域ノイズの盛り上がり) は信頼度を下げる。
        # NOTE: 平滑narrow化は試したが効果なしで立上りが2ブロック遅くなる
        # だけだったため不採用 (10dB帯の脱落はnarrowではなくSNRテールが原因)。
        # 単発スパイク耐性はdwell＋confidenceゲートで確保する。
        narrow_use = narrow_db
        if narrow_use < self.narrow_db:
            raw *= max(0.0, narrow_use / self.narrow_db)
        self.confidence += alpha * (raw - self.confidence)
        self.blocks += 1

        # ヒステリシス＋最小継続時間つきpresent判定 (narrowは瞬時値で評価)
        want = (snr_db >= self.snr_on_db and self.confidence >= self.min_confidence
                and narrow_use >= self.narrow_db and pilot_amp >= self.abs_floor)
        must_off = (snr_db <= self.snr_off_db
                    or self.confidence < self.min_confidence - 0.15)
        self._dwell += 1
        if want and not self.pilot_present and self._dwell >= self.min_dwell_blocks:
            self.pilot_present = True
            self._dwell = 0
        elif must_off and self.pilot_present and self._dwell >= self.min_dwell_blocks:
            self.pilot_present = False
            self._dwell = 0

        out.update({
            "pilot_power_db": float(pilot_db), "noise_power_db": float(noise_db),
            "pilot_snr_db": float(snr_db), "pilot_present": bool(self.pilot_present),
            "confidence": float(self.confidence),
            "estimated_phase": float(phase),
            "estimated_frequency_offset": float(freq_offset),
            "processing_ms": (time.perf_counter() - t0) * 1000.0,
        })
        return out
