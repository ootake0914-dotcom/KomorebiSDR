"""Non-cognitive default high-cut test (no hardware required).

FM doc section 3: with filter_mode="clean" as default, the non-cognitive
path applied fir_audio_clean (-6dB @ 8.5kHz, -50dB @ 10kHz), wiping
everything above 9kHz whenever no controller was driving the pipeline.
Default must be "wide" so a 12kHz tone survives within -6dB
(deemphasis-corrected).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

SR = 48000
TAU_DEEMPH = 50e-6


def _deemph_db(f_hz: float) -> float:
    w = 2.0 * np.pi * f_hz * TAU_DEEMPH
    return -10.0 * np.log10(1.0 + w * w)


def _tone_out_pow(dsp, freq_hz: float) -> float:
    n = SR  # 1s; measure steady-state tail (FIR + DC-HPF transient excluded)
    t = np.arange(n) / SR
    x = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    y = dsp._post_process_wfm(x.copy(), "", skip_mono_nr=True)
    tail = np.asarray(y[-SR // 2:], dtype=np.float64)
    return float(np.mean(tail * tail))


def test_default_is_wide():
    dsp = SdrDspPipeline(1152000, SR)
    assert not dsp.cognitive_enabled
    assert dsp.filter_mode == "wide", \
        f"non-cognitive default must be wide (got {dsp.filter_mode})"
    print("[OK] default filter_mode is wide")


def test_12k_tone_preserved():
    dsp = SdrDspPipeline(1152000, SR)
    p12 = _tone_out_pow(dsp, 12000.0)
    # fresh pipeline per tone: histories must not leak across measurements
    dsp2 = SdrDspPipeline(1152000, SR)
    p1 = _tone_out_pow(dsp2, 1000.0)
    meas_db = 10.0 * np.log10((p12 + 1e-18) / (p1 + 1e-18))
    expect_db = _deemph_db(12000.0) - _deemph_db(1000.0)
    residual_db = meas_db - expect_db
    print(f"[*] 12k/1k = {meas_db:+.1f}dB "
          f"(deemph expect {expect_db:+.1f}dB, filter residual {residual_db:+.1f}dB)")
    assert residual_db > -6.0, \
        f"default high-cut kills 12kHz ({residual_db:+.1f}dB)"
    print("[OK] 12kHz tone preserved under default settings")


def main() -> int:
    try:
        test_default_is_wide()
        test_12k_tone_preserved()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL WIDE-DEFAULT TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
