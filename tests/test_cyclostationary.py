"""A: 巡回定常性パイロット検出器のテスト (合成信号のみ・実機不要)。"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyclostationary_detector import CyclostationaryPilotDetector

FS = 288000.0
N = 66048  # 実ブロック相当 (66048/288k = 0.229s)


def _tone(freq, amp, n=N, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (amp * np.sin(2.0 * np.pi * freq * t)
            + 0.001 * rng.standard_normal(n)).astype(np.float64)


def test_strong_pilot_detected():
    det = CyclostationaryPilotDetector(sample_rate=FS)
    out = None
    for i in range(10):
        out = det.update(_tone(19000.0, 0.1, seed=i))
    assert out["pilot_present"] is True
    assert out["confidence"] >= 0.55
    assert out["pilot_snr_db"] >= 6.0


def test_noise_only_no_false_detection():
    det = CyclostationaryPilotDetector(sample_rate=FS)
    rng = np.random.default_rng(7)
    # 単一ブロックのSNRスパイクでconfidenceが一瞬上がることはあるため、
    # 機能要件は「presentが立たない」こと (ヒステリシス＋最小継続＋
    # min_confidenceゲートで誤検出を阻止) と、confidenceがpresent閾値未満。
    for _ in range(30):
        out = det.update(0.01 * rng.standard_normal(N))
        assert out["pilot_present"] is False
        assert out["confidence"] < 0.55


def test_frequency_offset_tracked():
    det = CyclostationaryPilotDetector(sample_rate=FS)
    out = None
    for i in range(12):
        out = det.update(_tone(19003.0, 0.1, seed=100 + i))
    # 3Hzオフセットでも検出を維持し、推定誤差はビン幅(4.36Hz)以内
    assert out["pilot_present"] is True
    assert abs(out["estimated_frequency_offset"] - 3.0) < 4.36


def test_phase_continuous_across_blocks():
    det = CyclostationaryPilotDetector(sample_rate=FS)
    # 連続正弦を4分割して投入: 位相差は理論値 2π f N'/fs (mod 2π) に一致
    nsub = N // 4
    full = _tone(19000.0, 0.1, n=N, seed=3)
    phases = []
    for i in range(4):
        out = det.update(full[i * nsub:(i + 1) * nsub])
        phases.append(out["estimated_phase"])
    expected = (2.0 * math.pi * 19000.0 * nsub / FS) % (2.0 * math.pi)
    for a, b in zip(phases, phases[1:]):
        d = (b - a) % (2.0 * math.pi)
        assert abs((d - expected + math.pi) % (2.0 * math.pi) - math.pi) < 0.15
    assert det.coherence > 0.9


def test_confidence_smoothed():
    det = CyclostationaryPilotDetector(sample_rate=FS)
    first = det.update(_tone(19000.0, 0.1, seed=0))
    last = None
    for i in range(1, 10):
        last = det.update(_tone(19000.0, 0.1, seed=i))
    # EMA平滑: 初ブロック < 収束後 (急変しない)
    assert first["confidence"] <= last["confidence"]
    assert last["confidence"] > 0.5


def test_nonfinite_safe():
    det = CyclostationaryPilotDetector(sample_rate=FS)
    bad = np.full(N, np.nan)
    out = det.update(bad)
    assert out["pilot_present"] is False
    assert out["bypass_reason"] == "non-finite"


def main() -> int:
    try:
        test_strong_pilot_detected()
        print("[*] 強パイロット検出 OK")
        test_noise_only_no_false_detection()
        print("[*] ノイズのみ誤検出なし OK")
        test_frequency_offset_tracked()
        print("[*] 周波数オフセット追従 OK")
        test_phase_continuous_across_blocks()
        print("[*] ブロック境界位相連続 OK")
        test_confidence_smoothed()
        print("[*] confidence平滑化 OK")
        test_nonfinite_safe()
        print("[*] 非有限入力安全 OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL CYCLOSTATIONARY TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
