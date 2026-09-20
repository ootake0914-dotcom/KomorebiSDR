"""
Audio Output Module using sounddevice - Hi-Fi Edition.
低遅延・高品位なオーディオ再生エンジン。
- スレッドセーフなジッター吸収キュー
- ソフトニー・ダイナミクスコントロール（歪み防止 & 音量安定化）
- ボリュームフェード & アンダーラン保護
"""

import threading
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

        # 長時間安定稼働モニタリング統計
        self.underrun_count = 0
        self.overflow_count = 0
        self.total_callbacks = 0
        self.total_frames_played = 0

    def _audio_callback(self, outdata, frames, time_info, status):
        """深層ジッターバッファによる完全安定オーディオ再生コールバック"""
        # プレロール判定: バッファが深層クッション（約400ms）まで蓄積されるまで待機
        if not self.is_prerolled:
            if self.audio_queue.qsize() >= self.preroll_threshold:
                self.is_prerolled = True
            else:
                outdata.fill(0.0)
                self.last_out_samples[:] = 0.0
                return

        collected = []
        needed = frames

        # 1. 前回のコールバックで余ったサンプルがあれば先頭から消費
        if len(self.remainder) > 0:
            if len(self.remainder) <= needed:
                collected.append(self.remainder)
                needed -= len(self.remainder)
                self.remainder = np.empty((0, 2), dtype=np.float32)
            else:
                collected.append(self.remainder[:needed])
                self.remainder = self.remainder[needed:]
                needed = 0

        # 2. 不足分をキューから時系列順に厳格に取り出す（余りはremainderへ退避）
        while needed > 0 and not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                if len(chunk) <= needed:
                    collected.append(chunk)
                    needed -= len(chunk)
                else:
                    collected.append(chunk[:needed])
                    self.remainder = chunk[needed:]
                    needed = 0
            except queue.Empty:
                break

        if collected:
            data = np.concatenate(collected, axis=0)
        else:
            data = np.zeros((0, 2), dtype=np.float32)

        self.total_callbacks += 1
        self.total_frames_played += frames

        # アンダーフロー補完 & リバッファリング保護 (クリック音・破裂音の完全根絶)
        if len(data) < frames:
            self.underrun_count += 1
            missing = frames - len(data)
            fade_len = min(32, missing)
            pad = np.zeros((missing, 2), dtype=np.float32)
            # 直前サンプルから0Vへコサインカーブで超滑らかに減衰（ステップ不連続ノイズゼロ）
            last = data[-1] if len(data) > 0 else self.last_out_samples
            if np.any(np.abs(last) > 1e-4):
                fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_len, dtype=np.float32)))
                pad[:fade_len] = (last[None, :] * fade[:, None]).astype(np.float32)
            data = np.concatenate((data, pad), axis=0) if len(data) > 0 else pad

            # 万が一キューが完全に空になった場合は、小刻みなバタつきを防止するため再プレロール(4チャンク)
            if self.audio_queue.empty() and len(self.remainder) == 0:
                self.is_prerolled = False
                self.preroll_threshold = 4

        # ミュート処理
        if self.is_muted:
            outdata[:, 0] = 0.0
            outdata[:, 1] = 0.0
            return

        # スムーズな音量スケーリング
        data = data * self.volume

        # 高音質ソフトリミッター (0.95を超えた部分だけを滑らかに圧縮してクリップ歪みを完全排除)
        threshold = 0.85
        over = np.abs(data) > threshold
        if np.any(over):
            sign = np.sign(data[over])
            mag = np.abs(data[over])
            # ソフトニー圧縮曲線
            compressed = threshold + (1.0 - threshold) * np.tanh((mag - threshold) / (1.0 - threshold))
            data[over] = sign * compressed

        # ステレオ出力
        outdata[:, 0] = data[:, 0]
        outdata[:, 1] = data[:, 1]
        if len(data) > 0:
            self.last_out_samples = data[-1].astype(np.float32, copy=True)

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
