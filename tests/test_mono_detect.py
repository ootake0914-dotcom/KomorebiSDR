"""Mono-program detector test (no hardware required).

_update_stereo_nr's mono branch must fire on true-mono (leakage-like Side)
and stay off on level-panned legitimate stereo, even though M/S
correlation alone cannot tell them apart (Cov(M,S)=(Var(L)-Var(R))/4).
The S/M energy gate provides the distinction.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

SR = 48000
BLK = 2752
NB = 60


def _run(diff_fn, mono_fn):
    dsp = SdrDspPipeline(1152000, SR)
    for _ in range(NB):
        dsp._update_stereo_nr(diff_fn(), mono_fn())
    return dsp


def _tone(f, n, amp=1.0, seed=0):
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float64)


def test_panned_stereo_not_mono():
    """6dBパン振りの正当ステレオ (ρ=+0.6) をモノラル化しないこと"""
    n = BLK
    L = _tone(1000.0, n)
    R = _tone(5000.0, n, amp=0.5)
    mono = (L + R) / 2.0
    diff = (L - R) / 2.0
    dsp = _run(lambda: diff, lambda: mono)
    print(f"[*] panned stereo: rho={dsp._nr_mono_rho:+.3f} "
          f"sm={dsp._nr_sm_db:+.1f}dB mono_w={dsp._nr_mono_w:.3f}")
    assert dsp._nr_mono_rho > 0.4, "test setup broken (rho should be high)"
    assert dsp._nr_mono_w < 0.3, \
        f"panned stereo mis-detected as mono (mono_w={dsp._nr_mono_w:.3f})"
    print("[OK] panned stereo protected")


def test_true_mono_detected():
    """真モノラル (漏れ-30dB＋微小ヒス) は検出すること"""
    rng = np.random.default_rng(3)
    n = BLK
    m = _tone(1000.0, n, amp=0.7)
    leak = 0.03 * m + 0.005 * rng.standard_normal(n)
    dsp = _run(lambda: leak, lambda: m)
    print(f"[*] true mono: rho={dsp._nr_mono_rho:+.3f} "
          f"sm={dsp._nr_sm_db:+.1f}dB mono_w={dsp._nr_mono_w:.3f}")
    assert dsp._nr_mono_w > 0.7, \
        f"true mono missed (mono_w={dsp._nr_mono_w:.3f})"
    print("[OK] true mono detected")


def test_centered_stereo_untouched():
    """等レベル中央ステレオは従来通り無反応のこと"""
    n = BLK
    L = _tone(1000.0, n)
    R = _tone(5000.0, n)
    mono = (L + R) / 2.0
    diff = (L - R) / 2.0
    dsp = _run(lambda: diff, lambda: mono)
    print(f"[*] centered stereo: rho={dsp._nr_mono_rho:+.3f} "
          f"mono_w={dsp._nr_mono_w:.3f}")
    assert dsp._nr_mono_w < 0.3
    print("[OK] centered stereo untouched")


def main() -> int:
    try:
        test_panned_stereo_not_mono()
        test_true_mono_detected()
        test_centered_stereo_untouched()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL MONO-DETECT TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
