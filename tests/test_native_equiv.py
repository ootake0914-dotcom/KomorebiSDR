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

    # フルパイプライン (4ブロック)
    raw = rng.integers(0, 256, size=132096 * 4, dtype=np.uint8)
    d_native = SdrDspPipeline(1152000, 48000)
    d_native.set_offset_freq(150000.0)
    out_native = np.concatenate([d_native.process(raw[k * 132096:(k + 1) * 132096], mode="WFM")[0]
                                 for k in range(4)])
    dsp._NATIVE = None
    d_py = SdrDspPipeline(1152000, 48000)
    d_py.set_offset_freq(150000.0)
    out_py = np.concatenate([d_py.process(raw[k * 132096:(k + 1) * 132096], mode="WFM")[0]
                             for k in range(4)])
    dsp._NATIVE = native_lib
    err = maxdiff(out_native, out_py)
    print(f"[{'OK' if err < 1e-3 else 'FAIL'}] full pipeline: max diff {err:.2e}")
    ok &= err < 1e-3

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
