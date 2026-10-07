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


def test_weak_field_hiss_not_frozen():
    """弱電界の真性ヒスは緩和ガードで凍結されずNRが効くこと

    緩和ガード (mf<floor*10) は強電界 (gate_hi以上) 限定。
    弱電界では真性ヒスがあり得るため素通しし、Wienerが立たねばならない。
    """
    rng = np.random.default_rng(7)
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 15.0  # 弱電界: 割引なし
    n = BLK
    t = np.arange(n) / SR
    mono = (0.05 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(60):
        hiss = (rng.standard_normal(n) * 0.15).astype(np.float32)
        dsp._update_stereo_nr((mono * 0.1 + hiss).astype(np.float32), mono)
    print(f"[*] weak-field hissy: wiener_w={dsp._nr_s_w:.3f}")
    assert dsp._nr_s_w > 0.5, "weak-field hiss missed (over-frozen)"
    print("[OK] weak-field hiss still suppressed")


def test_hf_program_strong_field():
    """実素材の高域寄り番組 (3.5kHz以上) でSideが削られないこと。

    追加改善案§3の被害ケース (mf/floor=2.40、旧ガード素通しで
    nr_gain=0.089/cut=5514Hzまで崩壊)。K=15ガード＋強電界条件で
    凍結され、gain/cut共に保全される。Wiener側 (slow) も初回
    プライムで-60に倒すため、Side抑圧は残らない。
    """
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 52.0  # C/N=40dB強電界の実測値
    n = BLK
    t = np.arange(n) / SR
    prog_hf = (0.25 * np.sin(2 * np.pi * 3500.0 * t)
               + 0.25 * np.sin(2 * np.pi * 4000.0 * t)
               + 0.20 * np.sin(2 * np.pi * 6000.0 * t)
               + 0.15 * np.sin(2 * np.pi * 8000.0 * t)).astype(np.float32)
    body_mf = (0.003 * np.sin(2 * np.pi * 500.0 * t)
               + 0.003 * np.sin(2 * np.pi * 1500.0 * t)
               + 0.002 * np.sin(2 * np.pi * 2500.0 * t)).astype(np.float32)
    mono = (prog_hf * 0.5 + body_mf).astype(np.float32)
    diff = prog_hf.astype(np.float32)
    for _ in range(100):
        dsp._update_stereo_nr(diff, mono)
    print(f"[*] HF-program: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"gain={dsp.stereo_nr_gain:.3f} cut_eff={dsp._nr_cut_eff:.0f}Hz "
          f"wiener_w={dsp._nr_s_w:.3f}")
    assert dsp.stereo_nr_gain > 0.8, "strong-field HF program wiped Side"
    assert dsp._nr_cut_eff > 10000.0, "cut collapsed on HF program"
    assert dsp._nr_s_w < 0.1, "Wiener still suppresses HF program"
    print("[OK] HF program in strong field preserved")


def test_normal_cn10_unchanged():
    """通常番組C/N10ではガードが発動せず従来通りNRが効くこと。

    追加改善案§3の正常側ケース (mf/floor=33)。K=15でも余裕で
    素通しし、変更前と同一の振る舞い (gain=0.980) になることの確認。
    """
    rng = np.random.default_rng(11)
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 23.0  # C/N=10dBの推定値
    n = BLK
    t = np.arange(n) / SR
    mono = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(100):
        hiss = (rng.standard_normal(n) * 0.5 / (10 ** (10.0 / 20.0))).astype(np.float32)
        dsp._update_stereo_nr((mono * 0.1 + hiss).astype(np.float32), mono)
    print(f"[*] CN10 normal: gain={dsp.stereo_nr_gain:.3f} "
          f"hiss={dsp.stereo_hiss_db:+.1f}dB")
    assert 0.95 <= dsp.stereo_nr_gain <= 1.0, \
        f"CN10 behavior changed ({dsp.stereo_nr_gain:.3f} vs 0.980)"
    print("[OK] CN10 normal program unchanged")


def main() -> int:
    try:
        test_hf_only_clean()
        test_legit_hiss_still_caught()
        test_silence_bounded()
        test_weak_field_hiss_not_frozen()
        test_hf_program_strong_field()
        test_normal_cn10_unchanged()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL NR-FLOOR TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
