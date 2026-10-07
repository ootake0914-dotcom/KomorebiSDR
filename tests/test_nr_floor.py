"""NR hiss-estimator denominator guard test (no hardware required).

_update_stereo_nr's ratio floor/mf explodes when the 300-3kHz program band
is empty (HF-only content: high woodwinds, cymbals, applause), driving
nr_gain to 0 and wiping Side highs on clean signals. The guard freezes
estimation (holding last state) instead of computing garbage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

SR = 48000
BLK = 2752


def test_hf_only_clean():
    """高域のみのクリーン信号でNRが全閉しないこと (回帰: hiss +48dB)"""
    dsp = SdrDspPipeline(1152000, SR)
    n = BLK
    t = np.arange(n) / SR
    mono = (0.001 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    diff = (0.5 * np.sin(2 * np.pi * 6000.0 * t)).astype(np.float32)
    for _ in range(30):
        dsp._update_stereo_nr(diff, mono)
    print(f"[*] HF-only: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"gain={dsp.stereo_nr_gain:.3f} cut={dsp.stereo_cut_hz:.0f}Hz")
    assert dsp.stereo_hiss_db < 8.0, \
        f"non-physical hiss estimate ({dsp.stereo_hiss_db:+.1f}dB)"
    assert dsp.stereo_nr_gain > 0.9, "clean HF wiped by NR"
    assert dsp.stereo_cut_hz > 12000.0, "cut collapsed on clean HF"
    print("[OK] HF-only clean preserved")


def test_legit_hiss_still_caught():
    """通常のヒス (強い番組＋高域ノイズ) は従来通りNRが効くこと"""
    rng = np.random.default_rng(5)
    dsp = SdrDspPipeline(1152000, SR)
    n = BLK
    t = np.arange(n) / SR
    mono = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(60):
        hiss = (rng.standard_normal(n) * 0.15).astype(np.float32)
        dsp._update_stereo_nr((mono * 0.1 + hiss).astype(np.float32), mono)
    print(f"[*] hissy: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"wiener_w={dsp._nr_s_w:.3f} (blend gain={dsp.stereo_nr_gain:.3f})")
    # ヒス推定はWiener系 (-46〜-26dB) で駆動する。ブレンド系 (-18〜-4dB) は
    # 極端な弱電界専用のため、このレベルでは動かなくて正しい。
    assert dsp._nr_s_w > 0.5, "Wiener path no longer sees real hiss"
    print("[OK] legitimate hiss still suppressed")


def test_silence_bounded():
    """無音で発散・クラッシュしないこと"""
    dsp = SdrDspPipeline(1152000, SR)
    z = np.zeros(BLK, dtype=np.float32)
    for _ in range(10):
        dsp._update_stereo_nr(z, z)
    assert np.isfinite(dsp.stereo_hiss_db)
    assert 0.0 <= dsp.stereo_nr_gain <= 1.0
    print("[OK] silence bounded")


def main() -> int:
    try:
        test_hf_only_clean()
        test_legit_hiss_still_caught()
        test_silence_bounded()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL NR-FLOOR TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
