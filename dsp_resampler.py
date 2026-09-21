"""
Adaptive drift resampler (extracted from dsp.py).

RTL-SDRとDAC間の独立クロック偏差補正リサンプラの正準の保持場所。
`dsp.py` は後方互換のため同名を再エクスポートする。
"""

import numpy as np


class AdaptiveDriftResampler:
    """
    RTL-SDRとDAC間の独立クロック偏差（ドリフト）を微小に伸縮補正し、
    サンプル欠落・重複・アンダーランを完全根絶する適応型分数リサンプラ。
    深層ジッターバッファ（約400ms）と広域不感帯により通常時は0ppm・完全ビットパーフェクト通過。
    """

    def __init__(self, target_chunks: float = 8.0, max_ppm: float = 2000.0):
        self.target_chunks = target_chunks
        self.deadband_low = 5.5   # 5.5〜10.5チャンク（約275ms〜525ms）の間は不感帯
        self.deadband_high = 10.5
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

    def update_feedback(self, current_chunks: float, dt: float = 0.05):
        """
        オーディオバッファの残存チャンク数に応じた不感帯付き高精度PI制御。
        通常時（5.5〜10.5チャンク）は 0ppm・完全ビットパーフェクト通過。
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

        # 0.5ppm未満の微小な揺らぎは完全ゼロ（ビットパーフェクト）に丸める
        if abs(adj) < 0.5e-6:
            adj = 0.0

        self.current_ratio = 1.0 + adj
        self.drift_ppm = adj * 1e6

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        is_stereo = (audio.ndim == 2)
        n_in = len(audio)
        ratio = self.current_ratio

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
                return out

            ext_audio = np.concatenate(([self.prev_sample, self.last_sample], audio,
                                        [audio[-1], audio[-1]]), axis=0).astype(np.float64)
            indices = np.arange(self.phase, n_in, ratio)
            if len(indices) == 0:
                self.phase -= n_in
                self.prev_sample = audio[-2].astype(np.float64) if n_in >= 2 else self.last_sample.copy()
                self.last_sample = audio[-1].astype(np.float64)
                return np.zeros((0, channels), dtype=np.float32)

            # Catmull-Rom 4点3次補間 (ステレオ2chを一括ブロードキャスト計算)
            ei = indices + 2.0
            i0 = np.floor(ei).astype(np.int64)
            f = (ei - i0)[:, np.newaxis].astype(np.float64)
            i0 = np.clip(i0, 1, n_in + 1)
            p0 = ext_audio[i0 - 1]
            p1 = ext_audio[i0]
            p2 = ext_audio[i0 + 1]
            p3 = ext_audio[i0 + 2]
            out = (p1 + 0.5 * f * (p2 - p0
                   + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                   + f * (3.0 * (p1 - p2) + p3 - p0))))

            last_idx = indices[-1] + ratio
            self.phase = float(last_idx - n_in)
            self.prev_sample = audio[-2].astype(np.float64) if n_in >= 2 else self.last_sample.copy()
            self.last_sample = audio[-1].astype(np.float64)
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
            return out

        # 境界連続性のために前2サンプルと末尾を付加
        ext_audio = np.concatenate(([self.prev_sample, self.last_sample], audio,
                                    [audio[-1], audio[-1]]))
        ext_audio = ext_audio.astype(np.float64)

        indices = np.arange(self.phase, n_in, ratio)
        if len(indices) == 0:
            self.phase -= n_in
            self.prev_sample = float(audio[-2]) if n_in >= 2 else self.last_sample
            self.last_sample = float(audio[-1])
            return np.zeros(0, dtype=np.float32)

        # Catmull-Rom 4点3次補間 (線形補間の高域ロールオフ/imagingを排除)
        ei = indices + 2.0
        i0 = np.floor(ei).astype(np.int64)
        f = (ei - i0).astype(np.float64)
        i0 = np.clip(i0, 1, n_in + 1)
        p0 = ext_audio[i0 - 1]
        p1 = ext_audio[i0]
        p2 = ext_audio[i0 + 1]
        p3 = ext_audio[i0 + 2]
        out = (p1 + 0.5 * f * (p2 - p0
               + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
               + f * (3.0 * (p1 - p2) + p3 - p0))))

        last_idx = indices[-1] + ratio
        self.phase = float(last_idx - n_in)
        self.prev_sample = float(audio[-2]) if n_in >= 2 else self.last_sample
        self.last_sample = float(audio[-1])

        return out.astype(np.float32)
