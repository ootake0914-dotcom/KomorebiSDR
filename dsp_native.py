"""
Native C core boundary (extracted from dsp.py).

`dsp.py` から分離したネイティブCコア (`sdr_core.dll`) のロード・境界モジュール。
正準の保持場所はこのファイル。`dsp.py` は後方互換のため同名を再エクスポートする。

公開シンボル:
- _load_native_core(), _NATIVE, NATIVE_CORE_ENABLED
- NATIVE_AM_SYNC / NATIVE_PLL3 / NATIVE_FIR / NATIVE_POLY / NATIVE_PLLFM / NATIVE_CMA
- enable_fast_fpu(), _fptr()
"""

import ctypes
import os


def _load_native_core():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cand_names = ["sdr_core.dll", "sdr_core.so", "sdr_core.dylib"]
    target_path = None
    for name in cand_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            target_path = p
            break
    if not target_path:
        return None
    try:
        lib = ctypes.CDLL(target_path)
        f = ctypes.c_float
        pf = ctypes.POINTER(f)
        lib.sdr_version.restype = ctypes.c_int
        lib.sdr_bilinear_deemphasis.argtypes = [pf, pf, ctypes.c_int, f, f, f, pf]
        lib.sdr_bilinear_deemphasis.restype = None
        lib.sdr_one_pole_highpass.argtypes = [pf, pf, ctypes.c_int, f, pf]
        lib.sdr_one_pole_highpass.restype = None
        lib.sdr_fm_demod.argtypes = [pf, pf, ctypes.c_int, pf]
        lib.sdr_fm_demod.restype = None
        lib.sdr_mix_freq.argtypes = [pf, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        lib.sdr_mix_freq.restype = None
        lib.sdr_suppress_clicks.argtypes = [pf, ctypes.c_int, f]
        lib.sdr_suppress_clicks.restype = ctypes.c_int
        lib.sdr_stereo_pll.argtypes = [pf, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_double,
                                       ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_double, pf, pf, pf]
        lib.sdr_stereo_pll.restype = None
        # 追加関数は無くてもネイティブコア全体を無効化しない (旧DLLとの後方互換)
        global NATIVE_AM_SYNC, NATIVE_PLL3, NATIVE_FIR, NATIVE_POLY, NATIVE_PLLFM, NATIVE_CMA
        if hasattr(lib, "sdr_stereo_pll3"):
            lib.sdr_stereo_pll3.argtypes = [pf, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_double, pf, pf, pf, pf, pf]
            lib.sdr_stereo_pll3.restype = None
            NATIVE_PLL3 = True
        if hasattr(lib, "sdr_am_sync"):
            lib.sdr_am_sync.argtypes = [pf, pf, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_double, ctypes.c_double, ctypes.c_double, pf]
            lib.sdr_am_sync.restype = None
            NATIVE_AM_SYNC = True
        if hasattr(lib, "sdr_fir_real"):
            lib.sdr_fir_real.argtypes = [pf, pf, pf, ctypes.c_int, ctypes.c_int]
            lib.sdr_fir_real.restype = None
            NATIVE_FIR = True
        if hasattr(lib, "sdr_polyphase_decim"):
            lib.sdr_polyphase_decim.argtypes = [pf, pf, pf, ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int]
            lib.sdr_polyphase_decim.restype = None
            NATIVE_POLY = True
        global NATIVE_PLLFM
        if hasattr(lib, "sdr_pll_fm_demod"):
            lib.sdr_pll_fm_demod.argtypes = [pf, pf, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_double),
                                             ctypes.c_double, ctypes.c_double]
            lib.sdr_pll_fm_demod.restype = None
            NATIVE_PLLFM = True
        global NATIVE_CMA
        if hasattr(lib, "sdr_cma_equalize"):
            lib.sdr_cma_equalize.argtypes = [pf, pf, ctypes.c_int, pf,
                                             ctypes.c_int, ctypes.c_float]
            lib.sdr_cma_equalize.restype = None
            NATIVE_CMA = True
        if hasattr(lib, "sdr_fast_fpu"):
            try:
                lib.sdr_fast_fpu()  # FTZ/DAZ有効化 (denormalジッタ対策)
            except Exception:
                pass
        if lib.sdr_version() < 1:
            return None
        return lib
    except Exception:
        return None


NATIVE_AM_SYNC = False
NATIVE_PLL3 = False
NATIVE_FIR = False
NATIVE_POLY = False
NATIVE_PLLFM = False
NATIVE_CMA = False
_NATIVE = _load_native_core()
NATIVE_CORE_ENABLED = _NATIVE is not None


def enable_fast_fpu():
    """カレントスレッドでFTZ/DAZを有効化 (非正規化数denormalによるCPUスパイク防止)"""
    if _NATIVE is not None and hasattr(_NATIVE, "sdr_fast_fpu"):
        try:
            _NATIVE.sdr_fast_fpu()
        except Exception:
            pass


def _fptr(arr):
    import numpy as np
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
