"""FRESH (Frequency-Shift) サイクロ定常フィルタ (Phase P2 試作)。

アナログFMのMPXに含まれる19kHzパイロットにより、複素ベースバンドは
α=±19kHzにスペクトル相関 (サイクロ定常性) を持つ。定常雑音には相関が
ないため、周波数シフト枝 X(f-α) を参照とするLCMV結合

    Y(f) = X(f) - (C(f)/Ss(f)) X(f-α)

で雑音だけを部分相殺する。C=スペクトル相関、Ss=シフト枝PSD。
コヒーレンスが低い (パイロット不明瞭・強信号・無信号) 場合は完全バイパス
するため、透明性は構造的に保証される。既定OFF (bm_fresh_enabled=False)。

【試行結果 (不採用)】合成stereo FMでAB: SNR10で音声SNR 29.3→19.8dB、
SNR3で25.1→16.6dBと約-9dB悪化。原因: 広帯域FMの側波帯全体が19kHz
シフト間で自己相関 (信号が参照枝から線形予測可能) のため、LCMV減算が
雑音ではなく信号を消す。ブラインドLCMV-FRESHは既知余剰帯域を持つ
デジタル線形変調向けで、アナログ広帯域FMには不適。本ファイルは
研究記録として残す (パイプライン未配線・既定OFFのため動作影響ゼロ)。
"""

import math
import time

import numpy as np


class SafeFreshFilter:
    """LCMV-FRESH＋安全層。processは (出力, 情報辞書) を返す。"""

    def __init__(self, sample_rate=288000.0, alpha_hz=19000.0,
                 smooth_hz=2000.0, max_sub_gain=0.5, max_bin_change=0.5,
                 coh_on=0.05, sig_bw_hz=60000.0, enabled=False):
        self.fs = float(sample_rate)
        self.alpha = float(alpha_hz)
        self.smooth_hz = float(smooth_hz)
        self.max_sub_gain = float(max_sub_gain)
        self.max_bin_change = float(max_bin_change)
        self.coh_on = float(coh_on)
        self.sig_bw = float(sig_bw_hz)
        self.enabled = bool(enabled)
        self.blocks = 0
        self.bypassed = 0
        self.last_coherence = 0.0

    def reset(self):
        self.blocks = 0
        self.bypassed = 0
        self.last_coherence = 0.0

    @staticmethod
    def _boxcar(v, m):
        if m <= 1:
            return v
        c = np.concatenate(([0.0], np.cumsum(v)))
        out = (c[m:] - c[:-m]) / float(m)
        pad_l = (len(v) - len(out)) // 2
        return np.concatenate((np.full(pad_l, out[0]), out,
                               np.full(len(v) - len(out) - pad_l, out[-1])))

    def process(self, iq, s_meter_dbfs=-40.0, cyclo_conf=0.0, clip=False):
        """複素IFブロック処理 → (iq_out, info)。入力は変更しない。"""
        t0 = time.perf_counter()
        info = {"processing_ms": 0.0, "bypass_reason": "", "coherence": 0.0,
                "active_ratio": 0.0}
        try:
            x = np.asarray(iq, dtype=np.complex128).reshape(-1)
        except Exception:
            info["bypass_reason"] = "bad-input"
            return np.asarray(iq), info
        n = len(x)
        if not self.enabled:
            info["bypass_reason"] = "disabled"
            return np.asarray(iq), info
        if n < 4096 or not bool(np.all(np.isfinite(x))):
            info["bypass_reason"] = "too-short" if n < 4096 else "non-finite"
            return np.asarray(iq), info
        if bool(clip):
            info["bypass_reason"] = "adc-clip"
            return np.asarray(iq), info
        try:
            s_db = float(s_meter_dbfs)
        except (TypeError, ValueError):
            s_db = -40.0
        # 弱電界ゲート: 強信号では出番なし (副作用ゼロを構造保証)
        if not math.isfinite(s_db) or s_db > -28.0:
            info["bypass_reason"] = "strong-signal"
            self.bypassed += 1
            return np.asarray(iq), info
        try:
            cc = float(cyclo_conf)
        except (TypeError, ValueError):
            cc = 0.0
        # パイロット不明瞭でも強すぎても出番なし (中庸の弱電界のみ)
        if not (0.25 <= cc <= 0.95):
            info["bypass_reason"] = "cyclo-gate"
            self.bypassed += 1
            return np.asarray(iq), info

        nfft = 1
        while nfft < n:
            nfft *= 2
        X = np.fft.fft(x, n=nfft)
        ka = int(round(self.alpha * nfft / self.fs))
        if ka <= 0 or ka >= nfft // 2:
            info["bypass_reason"] = "bad-alpha"
            return np.asarray(iq), info
        Xs = np.roll(X, ka)  # X(f-α)
        m = max(1, int(round(self.smooth_hz * nfft / self.fs)))
        S = self._boxcar(np.abs(X) ** 2, m) + 1e-24
        Ss = self._boxcar(np.abs(Xs) ** 2, m) + 1e-24
        C = self._boxcar(X * np.conj(Xs), m)
        rho2 = (np.abs(C) ** 2) / (S * Ss)
        freqs = np.fft.fftfreq(nfft, 1.0 / self.fs)
        band = np.abs(freqs) <= self.sig_bw
        coh = float(np.mean(rho2[band])) if np.any(band) else 0.0
        self.last_coherence = coh
        info["coherence"] = coh
        if coh < self.coh_on:
            info["bypass_reason"] = "low-coherence"
            self.bypassed += 1
            return np.asarray(iq), info
        # LCMV減算枝 (ゲイン上限付き)
        w2 = C / Ss
        over = np.abs(w2) > self.max_sub_gain
        w2[over] *= self.max_sub_gain / np.maximum(np.abs(w2[over]), 1e-18)
        Y = X - w2 * Xs
        # ビン毎の変化量上限 (過剰処理の禁止)
        denom = np.abs(X) + 1e-18
        chg = np.abs(Y - X) / denom
        lim = chg > self.max_bin_change
        Y[lim] = X[lim] + (Y[lim] - X[lim]) * (
            self.max_bin_change / np.maximum(chg[lim], 1e-18))
        y = np.fft.ifft(Y)[:n]
        if not bool(np.all(np.isfinite(y))):
            info["bypass_reason"] = "nan-output"
            return np.asarray(iq), info
        info["active_ratio"] = float(np.mean(chg[band] > 0.01)) if np.any(band) else 0.0
        self.blocks += 1
        info["processing_ms"] = (time.perf_counter() - t0) * 1000.0
        return y.astype(np.complex64), info
