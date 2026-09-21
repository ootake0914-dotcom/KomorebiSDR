"""Native C core vs pure-Python equivalence test (no hardware required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import dsp
from dsp import SdrDspPipeline


def maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> int:
    if not dsp.NATIVE_CORE_ENABLED:
        print("SKIP: sdr_core.dll not found (pure Python fallback active)")
        return 0

    rng = np.random.default_rng(42)
    ok = True
    native_lib = dsp._NATIVE

    x = (rng.standard_normal(2752) * 0.1).astype(np.float32)
    for name, method in (
        ("deemphasis", "_apply_bilinear_deemphasis"),
        ("dc_highpass", "_apply_dc_highpass"),
        ("voice_highpass", "_apply_voice_highpass"),
    ):
        d_native = SdrDspPipeline(1152000, 48000)
        y_native = getattr(d_native, method)(x)
        dsp._NATIVE = None
        d_py = SdrDspPipeline(1152000, 48000)
        y_py = getattr(d_py, method)(x)
        dsp._NATIVE = native_lib
        err = maxdiff(y_native, y_py)
        good = err < 1e-5
        print(f"[{'OK' if good else 'FAIL'}] {name}: max diff {err:.2e}")
        ok &= good

    iq = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64) * 0.1
    d_native = SdrDspPipeline(1152000, 48000)
    d_native.offset_freq = 150000.0
    m_native = d_native.mix_frequency(iq.copy(), mode="WFM")
    dsp._NATIVE = None
    d_py = SdrDspPipeline(1152000, 48000)
    d_py.offset_freq = 150000.0
    m_py = d_py.mix_frequency(iq.copy(), mode="WFM")
    dsp._NATIVE = native_lib
    err = maxdiff(m_native, m_py)
    print(f"[{'OK' if err < 1e-5 else 'FAIL'}] mix_frequency: max diff {err:.2e}")
    ok &= err < 1e-5

    # フルパイプライン (4ブロック。クリーンFMトーンでPLL/差分両経路を等価確認。
    # ノイズ入力ではPLLと差分法が原理的に異なる出力になるため対象外)
    N = (132096 // 2) * 4
    t = np.arange(N) / 1152000.0
    mpx = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    mpx = mpx / (np.max(np.abs(mpx)) + 1e-9)
    ph = 2 * np.pi * 30000.0 * np.cumsum(mpx) / 1152000.0
    iq = 0.6 * np.exp(1j * ph)
    raw = np.empty(2 * N, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    d_native = SdrDspPipeline(1152000, 48000)
    d_native.set_offset_freq(150000.0)
    d_native.set_stereo_enabled(False)
    d_native.rds_enabled = False
    d_native.afc_enabled = False
    # PLLはtrig実装差で長時間軌道が発散するため (各実装は自己無矛盾)、
    # 複数ブロック比較は差分法に固定して厳密等価を確認する。
    # PLL自体の等価性は下記の単ブロック試験で確認する。
    d_native.fm_pll_enabled = False
    # EKF/リーマン等のPython専用適応は
    # ネイティブ等価の対象外のため無効化 (有効だと原理的に差分が出る)
    for _attr in ("ekf_enabled", "cognitive_enabled", "riemann_always"):
        try:
            setattr(d_native, _attr, False)
        except Exception:
            pass
    try:
        for _obj in ("riemann_demodulator",
                      "ultra_squelch", "cognitive_eq"):
            _o = getattr(d_native, _obj, None)
            if _o is not None and hasattr(_o, "enabled"):
                _o.enabled = False
    except Exception:
        pass
    out_native = np.concatenate([d_native.process(raw[k * 132096:(k + 1) * 132096], mode="WFM")[0]
                                 for k in range(4)])
    dsp._NATIVE = None
    d_py = SdrDspPipeline(1152000, 48000)
    d_py.set_offset_freq(150000.0)
    d_py.set_stereo_enabled(False)
    d_py.rds_enabled = False
    d_py.afc_enabled = False
    d_py.fm_pll_enabled = False
    for _attr in ("ekf_enabled", "cognitive_enabled", "riemann_always"):
        try:
            setattr(d_py, _attr, False)
        except Exception:
            pass
    try:
        for _obj in ("riemann_demodulator",
                      "ultra_squelch", "cognitive_eq"):
            _o = getattr(d_py, _obj, None)
            if _o is not None and hasattr(_o, "enabled"):
                _o.enabled = False
    except Exception:
        pass
    out_py = np.concatenate([d_py.process(raw[k * 132096:(k + 1) * 132096], mode="WFM")[0]
                             for k in range(4)])
    dsp._NATIVE = native_lib
    err = maxdiff(out_native, out_py)
    print(f"[{'OK' if err < 1e-3 else 'FAIL'}] full pipeline: max diff {err:.2e}")
    ok &= err < 1e-3

    # PLL復調の等価性 (クリーンFMトーン。ノイズ入力ではPLLと差分法が
    # 原理的に異なるため、PLL同士の比較のみ有効)
    n = 16512
    tt = np.arange(n) / 288000.0
    m2 = 0.5 * np.sin(2 * np.pi * 1000.0 * tt)
    ph2 = 2 * np.pi * 75000.0 * np.cumsum(m2) / 288000.0
    iqt = np.exp(1j * ph2).astype(np.complex64)
    d_n = SdrDspPipeline(1152000, 48000)
    d_n.set_stereo_enabled(False)
    d_n.rds_enabled = False
    y_n = d_n.demodulate_wfm(iqt)
    dsp._NATIVE = None
    d_p = SdrDspPipeline(1152000, 48000)
    d_p.set_stereo_enabled(False)
    d_p.rds_enabled = False
    y_p = d_p.demodulate_wfm(iqt)
    dsp._NATIVE = native_lib
    err = maxdiff(y_n, y_p)
    print(f"[{'OK' if err < 1e-4 else 'FAIL'}] pll demod: max diff {err:.2e}")
    ok &= err < 1e-4

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
