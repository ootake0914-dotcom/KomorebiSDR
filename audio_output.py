"""
Audio Output Module using sounddevice - Hi-Fi Edition.
低遅延・高品位なオーディオ再生エンジン。
- スレッドセーフなジッター吸収キュー
- ソフトニー・ダイナミクスコントロール（歪み防止 & 音量安定化）
- ボリュームフェード & アンダーラン保護
"""

import threading
import time
import queue
import numpy as np
import sounddevice as sd


class AudioOutput:
    """sounddevice を用いた高品位オーディオ再生マネージャ"""

    def __init__(self, sample_rate: int = 48000, blocksize: int = 1024):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.volume = 0.7  # デフォルト音量 70%
        self.is_muted = False

        # 音声サンプル用キューと残存バッファ
        # (深めにして、万一の処理遅延でも古い音声の切り捨て=早回しを起こさない)
        # 内部表現は常にステレオ (N, 2)。モノラル入力はL/Rへ複製する。
        self.audio_queue = queue.Queue(maxsize=200)
        self.remainder = np.empty((0, 2), dtype=np.float32)
        self.stream = None
        self.is_running = False
        self.is_prerolled = False
        self.preroll_threshold = 8  # 深層ジッターバッファ: 約400ms (8チャンク) 蓄積して完全安定再生
        self.last_out_samples = np.zeros(2, dtype=np.float32)

        # リアルタイムコールバック内の動的確保を排除 (uac2: pre-allocated pool / single-copy)
        self._empty = np.empty((0, 2), dtype=np.float32)
        self._scratch = np.zeros((max(int(blocksize), 8192), 2), dtype=np.float32)
        self._absbuf = np.empty_like(self._scratch)
        self._mask = np.empty(self._scratch.shape, dtype=bool)

        # 長時間安定稼働モニタリング統計
        self.underrun_count = 0
        self.overflow_count = 0
        self.total_callbacks = 0
        self.total_frames_played = 0
        self.callback_us_max = 0.0

    def _audio_callback(self, outdata, frames, time_info, status):
        """深層ジッターバッファによる完全安定オーディオ再生コールバック。

        リアルタイムパスでは新規メモリ確保を行わず、事前確保した _scratch へ
        キュー内容を直接コピーする (uac2 の pre-allocated pool / single-copy 方式)。
        """
        t0 = time.perf_counter()
        frames = int(frames)
        if frames > len(self._scratch):
            # 想定外の大きなブロック要求時のみ拡張 (通常は発生しない)
            self._scratch = np.zeros((frames, 2), dtype=np.float32)
            self._absbuf = np.empty_like(self._scratch)
            self._mask = np.empty(self._scratch.shape, dtype=bool)

        # プレロール判定: バッファが深層クッション(約400ms)まで蓄積されるまで待機
        if not self.is_prerolled:
            if self.audio_queue.qsize() >= self.preroll_threshold:
                self.is_prerolled = True
            else:
                outdata.fill(0.0)
                self.last_out_samples[:] = 0.0
                return

        scratch = self._scratch
        n = 0

        # 1. 前回のコールバックで余ったサンプルを先頭から消費
        if len(self.remainder) > 0:
            take = min(len(self.remainder), frames)
            scratch[:take] = self.remainder[:take]
            n = take
            self.remainder = self.remainder[take:] if take < len(self.remainder) else self._empty

        # 2. 不足分をキューから時系列順にコピー (余りはremainderへ、コピーなしのビュー)
        while n < frames:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            take = min(len(chunk), frames - n)
            scratch[n:n + take] = chunk[:take]
            n += take
            if take < len(chunk):
                self.remainder = chunk[take:]

        self.total_callbacks += 1
        self.total_frames_played += frames

        # アンダーフロー補完 (直前サンプルから0Vへコサイン減衰、バッファへ直接書込)
        if n < frames:
            self.underrun_count += 1
            missing = frames - n
            fade_len = min(32, missing)
            if n > 0:
                last = scratch[n - 1]
            else:
                last = self.last_out_samples
            if fade_len > 0 and np.any(np.abs(last) > 1e-4):
                fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_len, dtype=np.float32)))
                scratch[n:n + fade_len] = last[None, :] * fade[:, None]
                scratch[n + fade_len:frames] = 0.0
            else:
                scratch[n:frames] = 0.0

            # キューが完全に空なら再プレロール(4チャンク)して小刻みなバタつきを防止
            if self.audio_queue.empty() and len(self.remainder) == 0:
                self.is_prerolled = False
                self.preroll_threshold = 4

        data = scratch[:frames]

        # ミュート処理
        if self.is_muted:
            outdata[:, 0] = 0.0
            outdata[:, 1] = 0.0
            us = (time.perf_counter() - t0) * 1e6
            if us > self.callback_us_max:
                self.callback_us_max = us
            return

        # 音量スケーリング (in-place、確保なし)
        np.multiply(data, self.volume, out=data)

        # ソフトリミッター (0.85超のみ圧縮。通常はmaskが全てFalseで何もしない)
        threshold = 0.85
        np.absolute(data, out=self._absbuf[:frames])
        over = np.greater(self._absbuf[:frames], threshold, out=self._mask[:frames])
        if over.any():
            idx = over
            sign = np.sign(data[idx])
            mag = self._absbuf[:frames][idx]
            compressed = threshold + (1.0 - threshold) * np.tanh(
                (mag - threshold) / (1.0 - threshold))
            data[idx] = sign * compressed

        # ステレオ出力 (outdata は C-contiguous な (frames,2))
        outdata[:, 0] = data[:, 0]
        outdata[:, 1] = data[:, 1]
        if n > 0:
            self.last_out_samples[:] = data[n - 1]

        us = (time.perf_counter() - t0) * 1e6
        if us > self.callback_us_max:
            self.callback_us_max = us

    def start(self):
        if self.is_running:
            return
        self.is_prerolled = False
        self.remainder = np.empty((0, 2), dtype=np.float32)
        self.last_out_samples = np.zeros(2, dtype=np.float32)
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=2,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.is_running = True
        except Exception:
            # オーディオデバイスが無い環境でも受信自体は継続する (無音)
            self.stream = None
            self.is_running = False

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        self.is_prerolled = False
        self.remainder = np.empty((0, 2), dtype=np.float32)
        self.last_out_samples = np.zeros(2, dtype=np.float32)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def put_audio(self, samples: np.ndarray):
        if len(samples) == 0:
            return
        if self.stream is None:
            # オーディオ出力なし: キューへ溜め込まず破棄
            return
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            # モノラル -> ステレオ複製
            arr = np.stack((arr, arr), axis=1)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            flat = arr.ravel()
            arr = np.stack((flat, flat), axis=1)
        if self.audio_queue.full():
            self.overflow_count += 1
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.audio_queue.put_nowait(arr)
        except queue.Full:
            self.overflow_count += 1

    def get_queue_size(self) -> int:
        """キュー内の現在の残存チャンク数 (目標: 3〜5チャンク)"""
        return self.audio_queue.qsize() + (1 if len(self.remainder) > 0 else 0)

    def get_buffer_fill_ratio(self) -> float:
        """バッファキュー充填率 (0.0〜1.0)"""
        return float(self.get_queue_size() / max(1, self.audio_queue.maxsize))

    def get_stats(self) -> dict:
        """長時間稼働の安定性統計を取得"""
        return {
            "fill_ratio": self.get_buffer_fill_ratio(),
            "underrun_count": self.underrun_count,
            "overflow_count": self.overflow_count,
            "total_callbacks": self.total_callbacks,
            "total_frames_played": self.total_frames_played,
        }

    def set_volume(self, vol: float):
        self.volume = max(0.0, min(1.0, vol))

    def set_mute(self, muted: bool):
        self.is_muted = muted
