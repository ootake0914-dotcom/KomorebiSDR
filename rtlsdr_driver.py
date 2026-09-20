"""
RTL-SDR C library wrapper using ctypes.
RTL-SDRドングルの制御と生IQサンプル取得を行う低レイヤドライバモジュール。
"""

import ctypes
from ctypes import byref, c_int, c_uint32, c_void_p, POINTER, create_string_buffer
import os
import sys
import threading
import numpy as np


class RtlSdrDriver:
    """RTL-SDR DLL ラッパークラス"""

    def __init__(self, dll_path: str = None):
        if dll_path is None:
            # カレントディレクトリまたはスクリプト配置ディレクトリから探す
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cand_paths = [
                os.path.join(base_dir, "rtlsdr.dll"),
                "rtlsdr.dll"
            ]
            for p in cand_paths:
                if os.path.exists(p):
                    dll_path = p
                    break
            if dll_path is None:
                dll_path = "rtlsdr.dll"

        # DLL読み込み
        try:
            self._dll = ctypes.CDLL(dll_path)
        except Exception as e:
            raise RuntimeError(f"rtlsdr.dll のロードに失敗しました: {dll_path} ({e})")

        self._setup_function_signatures()
        self.dev = c_void_p(0)
        self.is_open = False
        self.sample_rate = 2048000
        self.center_freq = 80000000

        # 非同期ストリームのライフサイクル管理 (同期読み込みとの競合・ハング防止)
        self._async_lock = threading.Lock()
        self._async_active = threading.Event()
        self._async_stop = threading.Event()
        self._c_callback = None

    def _setup_function_signatures(self):
        """C言語APIの関数プロトタイプを定義"""
        self._dll.rtlsdr_get_device_count.restype = c_int

        self._dll.rtlsdr_get_device_name.argtypes = [c_uint32]
        self._dll.rtlsdr_get_device_name.restype = ctypes.c_char_p

        self._dll.rtlsdr_open.argtypes = [POINTER(c_void_p), c_uint32]
        self._dll.rtlsdr_open.restype = c_int

        self._dll.rtlsdr_close.argtypes = [c_void_p]
        self._dll.rtlsdr_close.restype = c_int

        self._dll.rtlsdr_set_center_freq.argtypes = [c_void_p, c_uint32]
        self._dll.rtlsdr_set_center_freq.restype = c_int

        self._dll.rtlsdr_get_center_freq.argtypes = [c_void_p]
        self._dll.rtlsdr_get_center_freq.restype = c_uint32

        self._dll.rtlsdr_set_sample_rate.argtypes = [c_void_p, c_uint32]
        self._dll.rtlsdr_set_sample_rate.restype = c_int

        self._dll.rtlsdr_get_sample_rate.argtypes = [c_void_p]
        self._dll.rtlsdr_get_sample_rate.restype = c_uint32

        self._dll.rtlsdr_set_tuner_gain_mode.argtypes = [c_void_p, c_int]
        self._dll.rtlsdr_set_tuner_gain_mode.restype = c_int

        self._dll.rtlsdr_get_tuner_gains.argtypes = [c_void_p, POINTER(c_int)]
        self._dll.rtlsdr_get_tuner_gains.restype = c_int

        self._dll.rtlsdr_set_tuner_gain.argtypes = [c_void_p, c_int]
        self._dll.rtlsdr_set_tuner_gain.restype = c_int

        self._dll.rtlsdr_set_direct_sampling.argtypes = [c_void_p, c_int]
        self._dll.rtlsdr_set_direct_sampling.restype = c_int

        self._dll.rtlsdr_reset_buffer.argtypes = [c_void_p]
        self._dll.rtlsdr_reset_buffer.restype = c_int

        self._dll.rtlsdr_read_sync.argtypes = [
            c_void_p,
            c_void_p,
            c_int,
            POINTER(c_int),
        ]
        self._dll.rtlsdr_read_sync.restype = c_int

        # OSS (rtl_fm / pyrtlsdr) 標準の非同期ストリーミングAPI
        self.ASYNC_CB_TYPE = ctypes.CFUNCTYPE(None, POINTER(ctypes.c_ubyte), c_uint32, c_void_p)
        self._dll.rtlsdr_read_async.argtypes = [
            c_void_p,
            self.ASYNC_CB_TYPE,
            c_void_p,
            c_uint32,
            c_uint32,
        ]
        self._dll.rtlsdr_read_async.restype = c_int

        self._dll.rtlsdr_cancel_async.argtypes = [c_void_p]
        self._dll.rtlsdr_cancel_async.restype = c_int

    def get_device_count(self) -> int:
        return self._dll.rtlsdr_get_device_count()

    def get_device_name(self, index: int = 0) -> str:
        name = self._dll.rtlsdr_get_device_name(index)
        return name.decode("utf-8", errors="ignore") if name else "Unknown"

    def open(self, index: int = 0):
        if self.is_open:
            return
        res = self._dll.rtlsdr_open(byref(self.dev), index)
        if res != 0 or not self.dev:
            raise RuntimeError(f"RTL-SDRデバイス (index {index}) のオープンに失敗しました (code: {res})")
        self.is_open = True
        self.reset_buffer()

    def close(self):
        if self._async_active.is_set():
            self.cancel_async()
        if self.is_open and self.dev:
            self._dll.rtlsdr_close(self.dev)
            self.dev = c_void_p(0)
            self.is_open = False

    def reset_buffer(self):
        if self.is_open:
            self._dll.rtlsdr_reset_buffer(self.dev)

    def set_sample_rate(self, rate_hz: int):
        if not self.is_open:
            return
        res = self._dll.rtlsdr_set_sample_rate(self.dev, rate_hz)
        if res != 0:
            raise RuntimeError(f"サンプルレート {rate_hz} Hz の設定に失敗しました")
        self.sample_rate = rate_hz

    def get_sample_rate(self) -> int:
        if not self.is_open:
            return self.sample_rate
        return self._dll.rtlsdr_get_sample_rate(self.dev)

    def set_center_freq(self, freq_hz: int):
        if not self.is_open:
            return
        res = self._dll.rtlsdr_set_center_freq(self.dev, freq_hz)
        if res != 0:
            raise RuntimeError(f"中心周波数 {freq_hz} Hz の設定に失敗しました")
        self.center_freq = freq_hz

    def get_center_freq(self) -> int:
        if not self.is_open:
            return self.center_freq
        return self._dll.rtlsdr_get_center_freq(self.dev)

    def set_gain_mode(self, manual: bool):
        if not self.is_open:
            return
        mode = 1 if manual else 0
        self._dll.rtlsdr_set_tuner_gain_mode(self.dev, mode)

    def get_gains(self) -> list[float]:
        """利用可能なゲイン値リスト(dB単位)を取得"""
        if not self.is_open:
            return []
        num_gains = self._dll.rtlsdr_get_tuner_gains(self.dev, None)
        if num_gains <= 0:
            return []
        buf = (c_int * num_gains)()
        self._dll.rtlsdr_get_tuner_gains(self.dev, buf)
        # 1/10 dB 単位で返ってくるので float(dB) に変換
        return [val / 10.0 for val in buf]

    def set_gain(self, gain_db: float):
        """ゲインを設定 (dB)"""
        if not self.is_open:
            return
        gain_tenths = int(round(gain_db * 10))
        self._dll.rtlsdr_set_tuner_gain(self.dev, gain_tenths)

    def set_direct_sampling(self, mode: int):
        """
        ダイレクトサンプリングモード設定
        0: 無効 (通常チューナー経由)
        1: I-ADC直接接続
        2: Q-ADC直接接続 (多くのRTL-SDR Blog V3/V4のHFモードはQ branch = 2)
        """
        if not self.is_open:
            return
        self._dll.rtlsdr_set_direct_sampling(self.dev, mode)

    def read_sync(self, num_bytes: int = 131072) -> np.ndarray:
        """
        同期読み込みにより符号なし8bit整数の配列 (I, Q, I, Q, ...) を取得
        """
        if not self.is_open:
            return np.empty(0, dtype=np.uint8)

        buf = create_string_buffer(num_bytes)
        n_read = c_int(0)
        res = self._dll.rtlsdr_read_sync(self.dev, buf, num_bytes, byref(n_read))
        if res != 0:
            raise RuntimeError(f"IQデータの読み込みエラーが発生しました (code: {res})")

        # 高速にNumPy配列化 (frombuffer)
        raw_bytes = np.frombuffer(buf.raw[: n_read.value], dtype=np.uint8)
        return raw_bytes

    def read_async(self, callback_fn, num_buffers: int = 16, buffer_len: int = 132096):
        """
        OSS (rtl_fm) と同一のカーネルレベル非同期ストリーミング受信。
        librtlsdr内部で複数のUSBバルク転送バッファを循環させ、パケットロスを完全根絶。
        バッファ長132096はUSB 2.0パケット(512B)と復調単位(48B)の両方の完全倍数(1536B * 86)。
        :param callback_fn: 新しい生IQデータ (np.ndarray uint8) を受け取るPythonコールバック
        """
        if not self.is_open:
            return

        def _internal_cb(buf_ptr, length, ctx):
            # 高速にNumPy配列化してコールバック呼び出し
            # frombuffer / ctypes.string_at
            raw_bytes = np.ctypeslib.as_array(buf_ptr, shape=(length,))
            callback_fn(raw_bytes.copy())

        # cancel_asyncとの競合を排除:
        # 停止要求後に開始されたread_asyncはCループへ入らず即座に復帰する。
        with self._async_lock:
            if self._async_stop.is_set():
                return
            self._async_active.set()

        # C関数ポインタの生成（GC回収防止のためインスタンスに保持）
        self._c_callback = self.ASYNC_CB_TYPE(_internal_cb)
        try:
            self._dll.rtlsdr_read_async(
                self.dev,
                self._c_callback,
                None,
                c_uint32(num_buffers),
                c_uint32(buffer_len),
            )
        finally:
            self._async_active.clear()

    def cancel_async(self, timeout: float = 2.0):
        """
        非同期受信ループを安全に停止し、Cループが実際に抜けるまで待機する。
        これにより、停止直後にread_syncを呼んでも競合・ハングしないことが保証される。
        """
        active = False
        with self._async_lock:
            self._async_stop.set()
            if self._async_active.is_set() and self.is_open and self.dev:
                active = True
                self._dll.rtlsdr_cancel_async(self.dev)
        if active:
            self._async_active.wait(timeout=timeout)

    def resume_async(self):
        """cancel_async後、再度read_asyncを受け付ける状態に戻す"""
        with self._async_lock:
            self._async_stop.clear()

    def is_async_active(self) -> bool:
        return self._async_active.is_set()
