"""Ultrasonic squelch auto-threshold tests (no hardware required).

- Threshold follows the noise floor (floor + margin, bounded)
- Open/mute decisions still correct with auto threshold on
- Manual mode (auto off) keeps the fixed threshold behavior
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_rf import UltrasonicSquelchTracker

SR = 288000.0


def make_noise(n, level=0.5, seed=11):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float32) * level)


def make_carrier(n, level=0.3):
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 1000.0 * t) * level).astype(np.float32)
    rng = np.random.default_rng(12)
    return (tone + rng.standard_normal(n).astype(np.float32) * 0.005)


def test_threshold_follows_floor():
    print("===== test_threshold_follows_floor =====")
    sq = UltrasonicSquelchTracker(sample_rate=SR)
    assert sq.auto_threshold
    # 高ノイズを3秒: フロアはゆっくり上がり、閾値は floor+6 に張り付く
    for _ in range(int(3.0 * SR / 12000)):
        sq.process(make_noise(12000))
    print(f"[*] noisy: floor={sq.floor_db:.1f}dB thr={sq.threshold_effective:.1f}dB "
          f"open={sq.is_open}")
    assert not sq.is_open, "noise must mute"
    # 閾値 = clip(floor+6): 上限-25dBで頭打ちになる (何でもミュート防止)
    want = min(-25.0, max(-50.0, sq.floor_db + 6.0))
    assert abs(sq.threshold_effective - want) < 1e-6
    assert sq.floor_db > -20.0, "floor should rise toward noise"
    # 静かなキャリアを5秒: フロアが下がり、閾値が追従して開く
    for _ in range(int(5.0 * SR / 12000)):
        sq.process(make_carrier(12000))
    print(f"[*] carrier: floor={sq.floor_db:.1f}dB thr={sq.threshold_effective:.1f}dB "
          f"open={sq.is_open}")
    assert sq.is_open, "carrier must open"
    assert sq.threshold_effective < -30.0, "threshold should drop in quiet"
    assert -50.0 <= sq.threshold_effective <= -25.0, "threshold must stay bounded"
    print("[OK] threshold follows floor")


def test_manual_mode_unchanged():
    print("===== test_manual_mode_unchanged =====")
    sq = UltrasonicSquelchTracker(sample_rate=SR, noise_threshold_db=-38.0)
    sq.auto_threshold = False
    for _ in range(int(2.0 * SR / 12000)):
        sq.process(make_noise(12000, level=0.8))
    assert sq.threshold_effective == -38.0, "manual mode must keep fixed threshold"
    assert not sq.is_open
    for _ in range(int(2.0 * SR / 12000)):
        sq.process(make_carrier(12000))
    assert sq.is_open
    print("[OK] manual mode unchanged")


def test_reset_restores():
    print("===== test_reset_restores =====")
    sq = UltrasonicSquelchTracker(sample_rate=SR)
    for _ in range(int(2.0 * SR / 12000)):
        sq.process(make_noise(12000))
    sq.reset()
    assert sq.floor_db == -20.0
    assert sq.threshold_effective == sq.threshold_db
    assert sq.is_open
    print("[OK] reset restores")


def main() -> int:
    try:
        test_threshold_follows_floor()
        test_manual_mode_unchanged()
        test_reset_restores()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL SQUELCH AUTO TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
