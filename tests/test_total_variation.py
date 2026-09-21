"""
Unit tests for TotalVariationDenoiser (Condat 1D Total Variation Denoising).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_audio import TotalVariationDenoiser


def test_edge_preservation_and_flat_smoothing():
    """急峻なエッジ(アタック)を100%保持したまま、平坦部のガウスノイズを完全平滑化することを検証"""
    # 階段状信号 (エッジ jump = 1.0)
    y_clean = np.repeat([0.0, 1.0, -0.5, 0.5], 128).astype(np.float32)

    rng = np.random.RandomState(42)
    noise = 0.08 * rng.randn(len(y_clean)).astype(np.float32)
    noisy = y_clean + noise

    tvd = TotalVariationDenoiser(sample_rate=48000.0, lambda_reg=0.08)
    denoised = tvd.process(noisy, s_meter_dbfs=-35.0)

    # 1. 平坦部 (サンプル 20〜100) のノイズ分散低減量
    std_in = float(np.std(noisy[20:100] - y_clean[20:100]))
    std_out = float(np.std(denoised[20:100] - y_clean[20:100]))
    red_db = float(20.0 * np.log10(std_in / max(1e-12, std_out)))

    print(f"[*] TVD Flat Noise Reduction: {red_db:.2f} dB (std: {std_in:.4f} -> {std_out:.4f})")
    assert red_db >= 4.0, f"Expected >=4dB smoothing, got {red_db:.2f} dB"

    # 2. エッジジャンプの保持 (サンプル 127 -> 128)
    jump_clean = float(y_clean[128] - y_clean[127])
    jump_denoised = float(denoised[128] - denoised[127])
    print(f"[*] Edge jump: clean={jump_clean:.3f}, denoised={jump_denoised:.3f}")
    assert abs(jump_denoised - jump_clean) < 0.25, f"Edge was blurred by TVD: {jump_denoised}"
    print("[OK] test_edge_preservation_and_flat_smoothing passed")


def test_stereo_processing():
    """ステレオ 2ch 信号に対する左右独立 TVD 処理検証"""
    fs = 48000.0
    sig_l = np.sin(np.linspace(0, 20, 512)).astype(np.float32)
    sig_r = np.cos(np.linspace(0, 40, 512)).astype(np.float32)
    stereo_in = np.column_stack([sig_l, sig_r])

    tvd = TotalVariationDenoiser(sample_rate=fs)
    out = tvd.process(stereo_in, s_meter_dbfs=-30.0)

    assert out.shape == stereo_in.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    print("[OK] test_stereo_processing passed")


def test_strong_signal_bypass():
    """強電界時の完全バイパス検証"""
    fs = 48000.0
    audio = np.random.randn(512).astype(np.float32)
    tvd = TotalVariationDenoiser(sample_rate=fs)
    out = tvd.process(audio, s_meter_dbfs=-15.0)
    assert np.array_equal(out, audio)
    print("[OK] test_strong_signal_bypass passed")


if __name__ == "__main__":
    print("===== Running Total Variation Denoiser Tests =====")
    test_edge_preservation_and_flat_smoothing()
    test_stereo_processing()
    test_strong_signal_bypass()
    print("ALL TOTAL VARIATION TESTS PASSED!")
