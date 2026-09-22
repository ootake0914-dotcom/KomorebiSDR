"""RMT/Hankel最適収縮ノイズ低減の安全層 (B)。

既存 adaptive_audio.RmtHankelDenoiser (監査済み: L=24の24x24固有値分解、
強信号バイパス、Overlap-Lookahead境界連続) をコアとして借用し、
以下の安全層だけを追加する。コアの数学的内容は変更しない。

- 最大行列・最大ランクの上限 (Lは sqrt(max_matrix_size)とmax_rankで制限)
- CPU予算超過時の自動バイパス＋処理時間計測
- 出力ブレンドによる連続強度制御 (コアのON/OFF切除を滑らかにする)
- SNR帯別の適応強度マップ (設定可能)
- ステレオはMid/Side分離し、Mid通常・Side低強度 (L/R独立処理の禁止)
- RMS差・高域損失の監視と過剰時の自動減衰、NaN/Inf即時バイパス
- 無音時の増幅防止

既定ではdsp経路に接続しない (black_magic.rmt_denoiser.enabled=False)。
"""

import math
import time

import numpy as np

from adaptive_audio import RmtHankelDenoiser

# SNR帯→強度の既定マップ (上限・下限・帯内線形補間。設定で上書き可能)
# bands: [(snr_db境界, strength), ...] 降順。snr>30: 0.05、20-30: 0.05-0.40、
# 10-20: 0.40-0.65、<10: 0.65-0.85 (いずれもmax_strengthで頭打ち)。
# 値は合成2トーン掃引 (Phase 1-1) で検証済み: 高域損失<=1dBの範囲で
# 改善最大となる強度を各帯で選ぶと (20→0.40、10→0.65) が最適だった。
# なお合成定常トーンでは+10〜+30dB出るが、実番組では+2dB級 (bench条件7参照)。
DEFAULT_STRENGTH_BANDS = (
    (30.0, 0.05),
    (20.0, 0.40),
    (10.0, 0.65),
    (-99.0, 0.85),
)


def strength_for_snr(snr_db, bands=DEFAULT_STRENGTH_BANDS, max_strength=0.65):
    """SNR(dB)→目標強度 (0..1) の区分線形マップ。"""
    try:
        s = float(snr_db)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(s):
        return 0.0
    bs = list(bands)
    if s >= bs[0][0]:
        v = bs[0][1]
    else:
        v = bs[-1][1]
        for (hi_db, hi_v), (lo_db, lo_v) in zip(bs, bs[1:]):
            if lo_db <= s < hi_db:
                t = (s - lo_db) / max(hi_db - lo_db, 1e-9)
                v = lo_v + t * (hi_v - lo_v)
                break
    return float(min(max(v, 0.0), max_strength))


class SafeRmtDenoiser:
    """RMTコア＋安全層。process_*は (出力, 情報辞書) を返す。"""

    def __init__(self, sample_rate=48000.0, max_strength=0.65, max_rank=8,
                 max_matrix_size=256, cpu_budget_percent=20.0,
                 side_ratio=0.4, strength_bands=None, hf_cutoff_hz=10000.0,
                 enabled=True):
        self.fs = float(sample_rate)
        self.max_strength = float(max_strength)
        self.max_rank = int(max(1, max_rank))
        self.cpu_budget = float(cpu_budget_percent)
        self.side_ratio = float(min(max(side_ratio, 0.0), 1.0))
        self.bands = strength_bands or DEFAULT_STRENGTH_BANDS
        self.hf_cutoff = float(hf_cutoff_hz)
        self.enabled = bool(enabled)
        # 行列上限: L^2 <= max_matrix_size。
        # max_rankは保持ランクの上限として強度スケーリングで適用する
        # (process_mono内で retained>max_rank 時に target を絞る)。
        # L自体をmax_rank以下に潰すと周波数分解能が落ちるため行列上限のみで決める。
        L = min(24, int(math.isqrt(max(16, int(max_matrix_size)))))
        L = max(4, L)
        self.embed_dim = L
        self.core = RmtHankelDenoiser(sample_rate=self.fs, embed_dim=L)
        self._eff_strength = 0.0
        self._hf_lp = 0.0
        self._a_hf = float(1.0 - math.exp(-2.0 * math.pi * self.hf_cutoff / self.fs))
        self.blocks = 0
        self.bypassed = 0
        # 遅延整合バッファ (ch毎): コア出力dはlookahead分だけ入力より
        # 進んでいるため、ブレンド相手のxを同量遅延させて位相整合する。
        # しないと1kHzで112°ずれたdとのブレンドが打消し合って番組を削る
        # (合成2トーンで番組-2.4dBの劣化を実測)。
        self._delay = {}

    def reset(self):
        """選局時に全状態をクリア (前局のカーネル・強度を持ち越さない)。"""
        self.core.reset()
        self._eff_strength = 0.0
        self._hf_lp = 0.0
        self.blocks = 0
        self.bypassed = 0
        self._delay.clear()

    def _hf_energy_db(self, x, y):
        """高域エネルギーの入出力比 (dB。負=損失)。rfftで10kHz以上を比較する。"""
        try:
            xa = np.asarray(x, dtype=np.float64).reshape(-1)
            ya = np.asarray(y, dtype=np.float64).reshape(-1)
            n = len(xa)
            if len(ya) != n or n < 64:
                return 0.0
            freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
            m = freqs >= self.hf_cutoff
            if not bool(np.any(m)):
                return 0.0
            ex = float(np.sum(np.abs(np.fft.rfft(xa))[m]) ** 2) + 1e-18
            ey = float(np.sum(np.abs(np.fft.rfft(ya))[m]) ** 2) + 1e-18
            return float(10.0 * math.log10(ey / ex))
        except Exception:
            return 0.0

    def process_mono(self, audio, s_meter_dbfs=-40.0, snr_db=None, ch=""):
        """モノラル1ch処理 → (denoised, info)。入力は変更しない。"""
        t0 = time.perf_counter()
        info = {"processing_ms": 0.0, "bypass_reason": "",
                "estimated_noise_power": 0.0, "retained_rank": 0,
                "eff_strength": 0.0, "rms_diff_db": 0.0, "hf_loss_db": 0.0}
        try:
            x = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            info["bypass_reason"] = "bad-input"
            return np.asarray(audio), info
        n = len(x)
        if not self.enabled:
            info["bypass_reason"] = "disabled"
            return x, info
        if n < self.embed_dim * 4:
            info["bypass_reason"] = "too-short"
            return x, info
        if not bool(np.all(np.isfinite(x))):
            info["bypass_reason"] = "non-finite"
            self._eff_strength *= 0.5
            return x, info
        # 遅延整合は全経路で一定にする: d (lookahead分だけ進む) との
        # ブレンド相手だけでなく、バイパス出力も同量遅延させる。
        # バイパス毎に遅延が入抜すると15サンプルのタイムジャンプ
        # (1kHzで0.45FSの段差クリック) が生じることを実測して修正。
        # 0.3msの固定遅延。ブロック境界はtailで連続。
        D = int(min(max(getattr(self.core, "half_taps", 0), 0), n - 1))
        if D > 0:
            tail = self._delay.get(ch)
            if tail is None or len(tail) != D:
                tail = np.zeros(D, dtype=np.float32)
            x_ext = np.concatenate((tail, x))
            self._delay[ch] = x_ext[-D:].copy()
            xd = x_ext[:n]
        else:
            xd = x
        rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-18)
        if rms_in < 1e-6:
            # 無音は増幅も処理もしない
            info["bypass_reason"] = "silent"
            return xd, info
        try:
            s_db = float(s_meter_dbfs)
        except (TypeError, ValueError):
            s_db = -40.0
        if not math.isfinite(s_db):
            s_db = -40.0
        if s_db > -28.0:
            # 強信号はコア同様に完全バイパス (過剰処理の禁止)
            info["bypass_reason"] = "strong-signal"
            self._eff_strength = 0.0
            return xd, info
        if self.cpu_budget <= 0.0:
            info["bypass_reason"] = "cpu-budget"
            self.bypassed += 1
            return xd, info

        target = strength_for_snr(snr_db if snr_db is not None else s_db + 28.0,
                                  self.bands, self.max_strength)
        # ランク超過時は強度を滑らかに絞る (上限のハード切断を避ける)
        retained = int(getattr(self.core, "last_retained_rank", 0))
        if retained > self.max_rank:
            target *= self.max_rank / max(retained, 1)
        # 強度は急変させずEMAで追従 (音楽的アーティファクト防止)
        self._eff_strength += 0.3 * (target - self._eff_strength)
        s = float(self._eff_strength)
        info["eff_strength"] = s
        if s < 0.01:
            info["bypass_reason"] = "strength-min"
            return xd, info

        try:
            d = np.asarray(self.core.process(x, ch=ch,
                                             s_meter_dbfs=s_db),
                           dtype=np.float32).reshape(-1)
        except Exception:
            info["bypass_reason"] = "core-error"
            return xd, info
        if len(d) != n or not bool(np.all(np.isfinite(d))):
            info["bypass_reason"] = "nan-output"
            return xd, info
        y = ((1.0 - s) * xd + s * d).astype(np.float32)

        # 監視: RMS差と高域損失。過剰なら強度を自動で半減させる
        rms_out = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-18)
        info["rms_diff_db"] = float(20.0 * math.log10(rms_out / rms_in))
        if rms_out > rms_in * 1.12:
            # 勝手な増幅は禁止: 入力へ戻す
            y = xd.copy()
            info["bypass_reason"] = "auto-level"
            self._eff_strength *= 0.5
            return y, info
        hf = self._hf_energy_db(xd, y)
        info["hf_loss_db"] = float(hf)
        if hf < -6.0 or info["rms_diff_db"] < -3.0:
            self._eff_strength *= 0.5

        info["estimated_noise_power"] = float(getattr(self.core, "last_sigma2", 0.0))
        info["retained_rank"] = int(getattr(self.core, "last_retained_rank", 0))
        self.blocks += 1
        el_ms = (time.perf_counter() - t0) * 1000.0
        info["processing_ms"] = float(el_ms)
        # CPU予算: ブロック時間に対する割合で判定
        budget_ms = max(self.cpu_budget / 100.0 * (n / self.fs) * 1000.0, 0.0)
        if el_ms > budget_ms and budget_ms > 0.0:
            info["bypass_reason"] = "cpu-budget"
            self.bypassed += 1
            return xd, info
        return y, info

    def process_stereo(self, left, right, s_meter_dbfs=-40.0, snr_db=None):
        """ステレオ処理 (Mid通常・Side低強度) → ((L, R), info)。"""
        try:
            l = np.asarray(left, dtype=np.float32).reshape(-1)
            r = np.asarray(right, dtype=np.float32).reshape(-1)
        except Exception:
            return (np.asarray(left), np.asarray(right)), {"bypass_reason": "bad-input"}
        n = min(len(l), len(r))
        l, r = l[:n], r[:n]
        mid = ((l.astype(np.float64) + r.astype(np.float64)) * 0.5).astype(np.float32)
        side = ((l.astype(np.float64) - r.astype(np.float64)) * 0.5).astype(np.float32)
        ym, im = self.process_mono(mid, s_meter_dbfs, snr_db, ch="bm_m")
        saved_eff = self._eff_strength
        # Sideは低強度: 一時的に目標を絞って処理し、強度状態はMid基準に戻す
        self._eff_strength = saved_eff * self.side_ratio
        ys, iss = self.process_mono(side, s_meter_dbfs, snr_db, ch="bm_s")
        self._eff_strength = saved_eff
        yl = (ym.astype(np.float64) + ys.astype(np.float64)).astype(np.float32)
        yr = (ym.astype(np.float64) - ys.astype(np.float64)).astype(np.float32)
        info = {"mid": im, "side": iss,
                "processing_ms": float(im.get("processing_ms", 0.0)
                                       + iss.get("processing_ms", 0.0)),
                "bypass_reason": str(im.get("bypass_reason", ""))}
        return (yl, yr), info
