"""ABハーネスの単体テスト (合成データのみ・実機不要)。"""

import os
import sys

import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import ab_benchmark as ab


def test_load_npy_raw():
    raw = (np.arange(132096 * 2) % 256).astype(np.uint8)
    p = os.path.join(tempfile.gettempdir(), "ab_test_raw.npy")
    np.save(p, raw)
    try:
        out = ab.load_iq(p)
        assert out.dtype == np.uint8 and np.array_equal(out, raw)
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_load_cs16():
    p = os.path.join(tempfile.gettempdir(), "ab_test.cs16")
    iq = (np.arange(1000) - 500).astype(np.int16)
    iq.tofile(p)
    try:
        out = ab.load_iq(p)
        assert out.dtype == np.uint8 and len(out) == 1000
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_chatter_and_pct():
    assert ab.chatter([0.1, 0.9, 0.1, 0.9]) == 3
    assert ab.chatter([0.1, 0.1]) == 0
    assert ab.pct([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert ab.pct([], 99) == 0.0


def test_band_energy_diff():
    sr = 48000
    t = np.arange(sr) / sr
    a = np.sin(2 * np.pi * 12000.0 * t)
    b = 0.5 * np.sin(2 * np.pi * 12000.0 * t)
    ea = ab.band_energy(a, 10000, 15000, sr)
    eb = ab.band_energy(b, 10000, 15000, sr)
    assert abs(10 * np.log10(eb / ea) + 6.02) < 0.5


def main() -> int:
    try:
        test_load_npy_raw()
        print("[*] npy読込 OK")
        test_load_cs16()
        print("[*] cs16読込 OK")
        test_chatter_and_pct()
        print("[*] 指標関数 OK")
        test_band_energy_diff()
        print("[*] 高域差 OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL AB HARNESS TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
