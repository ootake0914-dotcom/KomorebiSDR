"""Lookahead limiter native equivalence test (no hardware required).

Verifies audio_output._lookahead_limit (native sdr_lookahead_limiter when
available, Python fallback otherwise):
- cold path (no peak over threshold): bit-exact vs pure Python
- hot path (over-threshold transient): bit-exact vs pure Python
- block-split vs bulk: identical (delay-line continuity)
- hot-path CPU budget: native must stay well under the Python cost
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from audio_output import AudioOutput


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


def _fresh_pair():
    return AudioOutput(), AudioOutput()


def test_cold_exact():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2752 * 3, 2)) * 0.1).astype(np.float32)
    a_py, a_nat = _fresh_pair()
    y_py = a_py._lookahead_limit_py(x.copy())
    y_nat = a_nat._lookahead_limit(x.copy())
    assert maxdiff(y_py, y_nat) == 0.0
    assert a_py._lim_env == a_nat._lim_env
    assert maxdiff(a_py._lim_delay, a_nat._lim_delay) == 0.0
    print("[OK] cold path bit-exact")


def test_hot_exact():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((2752 * 3, 2)) * 0.4).astype(np.float32)
    x[3000, 0] = 1.5
    x[5000, 1] = -1.2
    a_py, a_nat = _fresh_pair()
    y_py = a_py._lookahead_limit_py(x.copy())
    y_nat = a_nat._lookahead_limit(x.copy())
    assert maxdiff(y_py, y_nat) == 0.0, maxdiff(y_py, y_nat)
    assert a_py._lim_env == a_nat._lim_env
    assert maxdiff(a_py._lim_delay, a_nat._lim_delay) == 0.0
    # brickwall: limited peak must respect the threshold
    assert float(np.max(np.abs(y_nat))) <= 0.98 + 1e-6
    print("[OK] hot path bit-exact (brickwall holds)")


def test_split_bulk():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((2752 * 3, 2)) * 0.5).astype(np.float32)
    x[100, 1] = 1.8
    a_bulk, a_split = _fresh_pair()
    y_bulk = a_bulk._lookahead_limit(x.copy())
    parts = [x[i:i + 2752] for i in range(0, len(x), 2752)]
    y_split = np.concatenate([a_split._lookahead_limit(p.copy())
                              for p in parts])
    assert maxdiff(y_bulk, y_split) == 0.0
    print("[OK] split/bulk identical")


def test_hot_budget():
    try:
        import dsp_native as _dn
        if _dn._NATIVE is None or not _dn.NATIVE_LOOKAHEAD:
            print("SKIP: native lookahead unavailable (fallback path)")
            return
    except ImportError:
        print("SKIP: dsp_native unavailable")
        return
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((2752, 2)) * 0.5).astype(np.float32)
    x[1000, 0] = 1.5
    ao = AudioOutput()
    for _ in range(5):
        ao._lookahead_limit(x.copy())
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        ao._lookahead_limit(x.copy())
    ms = (time.perf_counter() - t0) / reps * 1000.0
    print(f"[*] native hot {ms:.3f} ms/block")
    assert ms < 1.0, f"hot path too slow: {ms:.2f} ms"
    print("[OK] hot path budget")


def main() -> int:
    try:
        test_cold_exact()
        test_hot_exact()
        test_split_bulk()
        test_hot_budget()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL LOOKAHEAD LIMITER TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
