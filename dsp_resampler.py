"""
Adaptive drift resampler (extracted from dsp.py).

RTL-SDRとDAC間の独立クロック偏差補正リサンプラの正準の保持場所。
`dsp.py` は後方互換のため同名を再エクスポートする。
"""

import numpy as np


class AdaptiveDriftResampler:
    """
    RTL-SDRとDAC間の独立クロック偏差（ドリフト）を微小に伸縮補正し、
    サンプル欠落・重複・アンダーランを抑える適応型分数リサンプラ。
    ジッターバッファ（約400ms）と不感帯により通常時は0ppm・ビットパーフェクト通過。
    """

    def __init__(self, target_chunks: float = 8.0, max_ppm: float = 2000.0):
        self.target_chunks = target_chunks
        # 不感帯は目標水位から派生 (target-3.5〜+2.5)。既定8.0では
        # 4.5〜10.5チャンク (約230ms〜525ms) となり、preroll 5chを帯域内に含む。
        self.deadband_low = target_chunks - 3.5
        self.deadband_high = target_chunks + 2.5
        # ±2000ppm (0.2%, ≈3.5cent) までは聴感上ほぼ知覚不能。
        # クロック偏差に加え、GUI描画(GIL)による微小な処理落ちも吸収し、
        # バッファ枯渇(音飛び)を未然に防ぐ。
        self.max_ratio_offset = max_ppm * 1e-6
        self.integral_error = 0.0
        self.current_ratio = 1.0
        self.phase = 0.0
        self.last_sample = 0.0
        self.prev_sample = 0.0
        self.drift_ppm = 0.0
        # ポリフェーズ・窓関数sinc補間テーブル (8タップ/128位相)。
        # 旧Catmull-Rom(4点3次)はドリフト補正作動中にHFが落ちていた
        # (実測: ±50ppmで 14k -1.14dB / 15k -1.44dB。補正のON/OFFで
        # HFレベルが呼吸し得る)。窓sincは同一遅延系でHFまで平坦。
        self._taps = 12
        self._phases = 128
        self._beta = 8.0
        _k = np.arange(-5, 7, dtype=np.float64)  # -5..6
        _mu = np.arange(self._phases, dtype=np.float64) / self._phases
        _kk = _k[None, :] - _mu[:, None]
        # c: sinc遮断。β: Kaiser窓。12タップβ8/c0.98が最良バランス
        # (数値評価: 1kHz円軌道env誤差5.8e-5 / 15k droop -0.04dB。
        # 8タップβ5はHF -0.13dBだが1k env 4.5e-4で位相同期テスト不合格)。
        _c = 0.98
        _M = self._taps / 2.0
        _w = np.i0(self._beta * np.sqrt(
            np.maximum(0.0, 1.0 - (_kk / _M) ** 2))) / np.i0(self._beta)
        _h = _c * np.sinc(_c * _kk) * _w
        _h /= _h.sum(axis=1, keepdims=True)
        self._poly = _h
        self._hist = None  # 直近4サンプル (補間の前置ヒストリ)

    def update_feedback(self, current_chunks: float, dt: float = 0.05):
        """
        オーディオバッファの残存チャンク数に応じた不感帯付き高精度PI制御。
        通常時（4.5〜10.5チャンク）は 0ppm・ビットパーフェクト通過。
        """
        # 不感帯（Deadband）判定: 健全領域
        if self.deadband_low <= current_chunks <= self.deadband_high:
            self.integral_error = 0.0
            # 無視できる補正量ならビットパーフェクト（1.0）へ戻す。
            # 一方、有意な補正は保持する。毎回1.0に戻すと恒常的な微小不足で
            # バッファが枯渇→アンダーランを繰り返すため。
            if abs(self.current_ratio - 1.0) < 200e-6:
                self.current_ratio = 1.0
                self.drift_ppm = 0.0
            return

        if current_chunks < self.deadband_low:
            error = current_chunks - self.deadband_low  # 負値（バッファ減）
        else:
            error = current_chunks - self.deadband_high  # 正値（バッファ増）

        # 積分器の更新（アンチワインドアップ付き）
        self.integral_error = float(np.clip(self.integral_error + error * dt, -3.0, 3.0))

        # PI制御ゲイン: 実際にバッファ水位を制御できる強さに再調整。
        # (旧値 kp=3e-5 は補正能力が実質ゼロで、枯渇を放置していた)
        kp = 8e-4
        ki = 5e-5
        adj = kp * error + ki * self.integral_error
        adj = float(np.clip(adj, -self.max_ratio_offset, self.max_ratio_offset))

        # 0.5ppm未満の微小な揺らぎはゼロ（ビットパーフェクト）に丸める
        if abs(adj) < 0.5e-6:
            adj = 0.0

        self.current_ratio = 1.0 + adj
        self.drift_ppm = adj * 1e6

    def _take_hist(self, audio: np.ndarray) -> np.ndarray:
        """直近6サンプルを補間前置ヒストリとして保持 (チャネル形状保存)。"""
        a = np.asarray(audio, dtype=np.float64)
        n = len(a)
        if n >= 6:
            return a[-6:].copy()
        pad = np.zeros((6 - n,) + a.shape[1:], dtype=np.float64)
        return np.concatenate((pad, a), axis=0)

    def _poly_interp(self, ext: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """12タップ/128位相ポリフェーズ窓sinc補間 (extは前置6+本体+後置6)。"""
        ei = indices + 6.0
        i0 = np.floor(ei).astype(np.int64)
        frac = ei - i0
        p = np.clip((frac * self._phases).astype(np.int64),
                    0, self._phases - 1)
        base = i0 - 5
        np.clip(base, 0, len(ext) - self._taps, out=base)
        out = np.zeros((len(indices),) + ext.shape[1:], dtype=np.float64)
        for t in range(self._taps):
            w = self._poly[p, t].reshape((-1,) + (1,) * (ext.ndim - 1))
            out += w * ext[base + t]
        return out

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        is_stereo = (audio.ndim == 2)
        n_in = len(audio)
        ratio = self.current_ratio
        new_hist = self._take_hist(audio)

        if is_stereo:
            channels = audio.shape[1]
            if not isinstance(self.last_sample, np.ndarray) or len(self.last_sample) != channels:
                self.prev_sample = np.zeros(channels, dtype=np.float64)
                self.last_sample = np.zeros(channels, dtype=np.float64)

            if abs(ratio - 1.0) < 1e-6:
                d = int(np.floor(self.phase + 0.5))
                if d <= 0:
                    self.prev_sample = audio[-2].astype(np.float64) if n_in >= 2 else self.last_sample.copy()
                    self.last_sample = audio[-1].astype(np.float64)
                    self._hist = new_hist
                    return audio
                d = min(d, n_in)
                out = np.empty((n_in - d, channels), dtype=np.float32)
                if d == 1:
                    # d==1は先頭1サンプルを捨てるだけ (d>1の一般経路 out=audio[d:] と
                    # 同じ意味)。旧実装は last_sample を先頭に重複挿入し末尾を1つ
                    # 落としていたため、サンプル重複＋欠落 (クリック) が生じていた。
                    out[:] = audio[1:]
                else:
                    ext = np.concatenate(([self.prev_sample, self.last_sample], audio), axis=0)
                    out[:] = ext[n_in + 2 - len(out):n_in + 2]
                self.phase -= d
                self.prev_sample = audio[-2].astype(np.float64) if n_in >= 2 else self.last_sample.copy()
                self.last_sample = audio[-1].astype(np.float64)
                self._hist = new_hist
                return out

            hist = self._hist
            if hist is None or hist.shape[1:] != (channels,):
                hist = np.zeros((6, channels), dtype=np.float64)
            last = np.repeat(audio[-1][None, :], 6, axis=0).astype(np.float64)
            ext_audio = np.concatenate(
                (hist, np.asarray(audio, dtype=np.float64), last), axis=0)
            indices = np.arange(self.phase, n_in, ratio)
            if len(indices) == 0:
                self.phase -= n_in
                self.prev_sample = audio[-2].astype(np.float64) if n_in >= 2 else self.last_sample.copy()
                self.last_sample = audio[-1].astype(np.float64)
                self._hist = new_hist
                return np.zeros((0, channels), dtype=np.float32)

            out = self._poly_interp(ext_audio, indices)
            last_idx = indices[-1] + ratio
            self.phase = float(last_idx - n_in)
            self.prev_sample = audio[-2].astype(np.float64) if n_in >= 2 else self.last_sample.copy()
            self.last_sample = audio[-1].astype(np.float64)
            self._hist = new_hist
            return out.astype(np.float32)

        # モノラル (1次元配列)
        if isinstance(self.last_sample, np.ndarray):
            self.prev_sample = 0.0
            self.last_sample = 0.0

        if abs(ratio - 1.0) < 1e-6:
            # 整数遅延バイパス: ドリフトなし時は補間フィルタを掛けない。
            # 旧実装は残留phaseで恒常的に線形補間し高域を削っていた。
            # 位相残差は整数部のみ消費し、小数部は非可聴のまま保持する。
            d = int(np.floor(self.phase + 0.5))
            if d <= 0:
                self.prev_sample = float(audio[-2]) if n_in >= 2 else self.last_sample
                self.last_sample = float(audio[-1])
                self._hist = new_hist
                return audio
            d = min(d, n_in)
            out = np.empty(n_in - d, dtype=np.float32)
            if d == 1:
                # d==1は先頭1サンプルを捨てるだけ (d>1の一般経路 out=audio[d:] と
                # 同じ意味)。旧実装は last_sample を重複挿入しサンプル欠落を生んでいた。
                out[:] = audio[1:]
            else:
                ext = np.concatenate(([self.prev_sample, self.last_sample], audio))
                out[:] = ext[n_in + 2 - len(out):n_in + 2]
            self.phase -= d
            self.prev_sample = float(audio[-2]) if n_in >= 2 else self.last_sample
            self.last_sample = float(audio[-1])
            self._hist = new_hist
            return out

        hist = self._hist
        if hist is None or hist.ndim != 1 or len(hist) != 6:
            hist = np.zeros(6, dtype=np.float64)
        ext_audio = np.concatenate(
            (hist, np.asarray(audio, dtype=np.float64),
             np.repeat(float(audio[-1]), 6)))
        indices = np.arange(self.phase, n_in, ratio)
        if len(indices) == 0:
            self.phase -= n_in
            self.prev_sample = float(audio[-2]) if n_in >= 2 else self.last_sample
            self.last_sample = float(audio[-1])
            self._hist = new_hist
            return np.zeros(0, dtype=np.float32)

        out = self._poly_interp(ext_audio, indices)
        last_idx = indices[-1] + ratio
        self.phase = float(last_idx - n_in)
        self.prev_sample = float(audio[-2]) if n_in >= 2 else self.last_sample
        self.last_sample = float(audio[-1])
        self._hist = new_hist

        return out.astype(np.float32)
