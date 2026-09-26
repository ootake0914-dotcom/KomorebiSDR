"""
RTL-SDR C library wrapper using ctypes.
RTL-SDRドングルの制御と生IQサンプル取得を行う低レイヤドライバモジュール。
"""

import ctypes
import ctypes.util
from ctypes import byref, c_int, c_uint32, c_void_p, POINTER, create_string_buffer
import os
import sys
import threading
import time
import numpy as np


class RtlSdrDriver:
    """RTL-SDR DLL/SO ラッパークラス"""

    def __init__(self, dll_path: str = None):
        if dll_path is None:
            # カレントディレクトリ、スクリプト配置ディレクトリ、またはOS標準ライブラリパスから探す
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cand_paths = [
                os.path.join(base_dir, "rtlsdr.dll"),
                "rtlsdr.dll",
                os.path.join(base_dir, "librtlsdr.so"),
                os.path.join(base_dir, "librtlsdr.so.0"),
                "librtlsdr.so",
                "librtlsdr.so.0",
                os.path.join(base_dir, "librtlsdr.dylib"),
                "librtlsdr.dylib",
            ]
            system_lib = ctypes.util.find_library("rtlsdr")
            if system_lib:
                cand_paths.insert(0, system_lib)
            for p in cand_paths:
                if os.path.exists(p):
                    dll_path = p
                    break
            if dll_path is None:
                dll_path = system_lib or ("rtlsdr.dll" if sys.platform.startswith("win") else "librtlsdr.so")

        # DLL/SO読み込み
        try:
            self._dll = ctypes.CDLL(dll_path)
        except Exception as e:
            raise RuntimeError(f"RTL-SDR ライブラリ ({dll_path}) のロードに失敗しました: {e}")

        self._setup_function_signatures()
        self.dev = c_void_p(0)
        self.is_open = False
        self.sample_rate = 2048000
        self.center_freq = 80000000
        # PPM較正値 (ドングル水晶の個体誤差。HW APIがあればドングル側、
        # 無ければset_center_freq時のソフトウェア周波数オフセットで補正)
        self.ppm = 0
        self._ppm_hw = False

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

        # PPM周波数較正 (古いDLLに無い場合はHW経路を使わずSWフォールバック)
        self._has_ppm_api = all(hasattr(self._dll, n) for n in
                                ("rtlsdr_set_freq_correction", "rtlsdr_get_freq_correction"))
        if self._has_ppm_api:
            self._dll.rtlsdr_set_freq_correction.argtypes = [c_void_p, c_int]
            self._dll.rtlsdr_set_freq_correction.restype = c_int
            self._dll.rtlsdr_get_freq_correction.argtypes = [c_void_p]
            self._dll.rtlsdr_get_freq_correction.restype = c_int

    def get_device_count(self) -> int:
        return self._dll.rtlsdr_get_device_count()

    def get_device_name(self, index: int = 0) -> str:
        name = self._dll.rtlsdr_get_device_name(index)
        return name.decode("utf-8", errors="ignore") if name else "Unknown"

    def open(self, index: int = 0):
        if self.is_open:
            # closeガードでskipされた死ハンドルの場合は再オープンを試みる
            if getattr(self, "_close_pending", False) and not self._async_active.is_set():
                try:
                    self.dev = c_void_p(0)
                    self.is_open = False
                    self._close_pending = False
                except Exception:
                    pass
            else:
                return
        res = self._dll.rtlsdr_open(byref(self.dev), index)
        if res != 0 or not self.dev:
            raise RuntimeError(f"RTL-SDRデバイス (index {index}) のオープンに失敗しました (code: {res})")
        self.is_open = True
        # 前デバイス/close時の設定キャッシュを無効化 (新デバイスには未設定状態から適用する)
        self._direct_sampling_mode = None
        # デバイスリセットでPPM補正は消えるため、保持値を再適用する
        if self.ppm != 0:
            try:
                self._apply_ppm_hw(self.ppm)
            except Exception:
                pass
        self.reset_buffer()

    def close(self):
        if self._async_active.is_set():
            if not self.cancel_async():
                # Cループ残留中のcloseはuse-after-close(crash)のため実行しない。
                # ハンドルはリークするが、クラッシュより安全。復帰は次回open時。
                print("[WARN] async loop still active; skip rtlsdr_close "
                      "(handle kept, will retry on next open)", file=sys.stderr)
                self._close_pending = True
                return
        if self.is_open and self.dev:
            self._dll.rtlsdr_close(self.dev)
            self.dev = c_void_p(0)
            self.is_open = False
            self._close_pending = False
            self._direct_sampling_mode = None

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

    def compensated_freq(self, freq_hz: int) -> int:
        """SWフォールバック時の同調周波数 (HW補正中は素通し)。テスト容易性のため分離。"""
        if self._ppm_hw or self.ppm == 0:
            return int(freq_hz)
        return int(round(freq_hz * (1.0 - self.ppm / 1e6)))

    # 同調許容範囲 (プリセット検証と一致。範囲外はc_uint32ラップで
    # GHz誤同調になるため入口で弾く)
    FREQ_MIN_HZ = 100000
    FREQ_MAX_HZ = 1750000000

    def set_center_freq(self, freq_hz: int):
        if not self.is_open:
            return
        f = int(freq_hz)
        if not (self.FREQ_MIN_HZ <= f <= self.FREQ_MAX_HZ):
            raise ValueError(f"中心周波数 {f} Hz が範囲外です "
                             f"({self.FREQ_MIN_HZ}〜{self.FREQ_MAX_HZ} Hz)")
        tune_hz = self.compensated_freq(f)
        res = self._dll.rtlsdr_set_center_freq(self.dev, tune_hz)
        if res != 0:
            raise RuntimeError(f"中心周波数 {freq_hz} Hz の設定に失敗しました")
        self.center_freq = int(freq_hz)

    def _apply_ppm_hw(self, ppm: int) -> bool:
        """HWのPPM補正を試みる。成功=True。open前やAPI欠落時はFalse。"""
        if not self._has_ppm_api or not self.is_open or not self.dev:
            self._ppm_hw = False
            return False
        res = self._dll.rtlsdr_set_freq_correction(self.dev, int(ppm))
        self._ppm_hw = (res == 0)
        return self._ppm_hw

    def set_ppm_correction(self, ppm: int) -> bool:
        """ドングルPPM較正値を設定する。戻り値True=HW補正、False=SWフォールバック。
        未open時は保持のみ行い、open時に再適用する。"""
        self.ppm = int(ppm)
        if not self.is_open:
            self._ppm_hw = False
            return False
        try:
            return self._apply_ppm_hw(self.ppm)
        except Exception:
            self._ppm_hw = False
            return False

    def get_ppm_correction(self) -> int:
        """現在のPPM較正値 (保持値。HW読戻しではない)"""
        return int(self.ppm)

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
        """ゲインを設定 (dB)。HWが拒否したら例外 (沈黙成功させない)"""
        if not self.is_open:
            return
        gain_tenths = int(round(gain_db * 10))
        res = self._dll.rtlsdr_set_tuner_gain(self.dev, gain_tenths)
        if res != 0:
            raise RuntimeError(f"ゲイン {gain_db} dB の設定に失敗しました (code: {res})")

    def set_direct_sampling(self, mode: int):
        """
        ダイレクトサンプリングモード設定
        0: 無効 (通常チューナー経由)
        1: I-ADC直接接続
        2: Q-ADC直接接続 (多くのRTL-SDR Blog V3/V4のHFモードはQ branch = 2)
        """
        if not self.is_open:
            return
        # 同一モードの重複呼び出しをスキップ (チューナーの不要な再起動・PLLロック外れ防止)
        if getattr(self, "_direct_sampling_mode", None) == mode:
            return
        self._direct_sampling_mode = mode
        self._dll.rtlsdr_set_direct_sampling(self.dev, mode)
        # 通常モード(0)への復帰時はチューナーPLLのセトリング時間を確保
        if mode == 0:
            time.sleep(0.02)

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
        librtlsdr内部で複数のUSBバルク転送バッファを循環させ、パケットロスを抑える。
        バッファ長132096はUSB 2.0パケット(512B)と復調単位(48B)の両方の倍数(1536B * 86)。
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
        # cancel_asyncとの競合: _async_active.set() 後に cancel が来た場合、
        # _async_stop を再確認して C ループを開始しない (開始後のcancel漏れを防ぐ)
        with self._async_lock:
            if self._async_stop.is_set():
                self._async_active.clear()
                return
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

    def cancel_async(self, timeout: float = 2.0) -> bool:
        """
        非同期受信ループを安全に停止し、Cループが実際に抜けるまで待機する。
        これにより、停止直後にread_syncを呼んでも競合・ハングしないことが保証される。
        戻り値: ループが抜けた(True)/タイムアウトで残留(False)。
        注意: threading.Event.waitはフラグset時に即復帰するため、
        クリア待ちには使えない。ポーリングで実際に抜けを確認する。
        """
        active = False
        with self._async_lock:
            self._async_stop.set()
            if self._async_active.is_set() and self.is_open and self.dev:
                active = True
                self._dll.rtlsdr_cancel_async(self.dev)
        if active:
            deadline = time.monotonic() + timeout
            while self._async_active.is_set():
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)
        return True

    def resume_async(self):
        """cancel_async後、再度read_asyncを受け付ける状態に戻す"""
        with self._async_lock:
            self._async_stop.clear()

    def is_async_active(self) -> bool:
        return self._async_active.is_set()
