"""
Unit tests for AcousticNonLocalMeans (Acoustic 1D Non-Local Means).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_audio import AcousticNonLocalMeans


def test_periodic_signal_preservation_and_noise_reduction():
    """周期的信号の自己相似パッチ重み付けにより、波形を保ちながらノイズを相殺することを検証"""
    t = np.linspace(0, 1, 512, dtype=np.float32)
    clean = 0.5 * np.sin(2.0 * np.pi * 12.0 * t)

    rng = np.random.RandomState(42)
    noise = 0.05 * rng.randn(len(clean)).astype(np.float32)
    noisy = clean + noise

    nlm = AcousticNonLocalMeans(sample_rate=48000.0, patch_len=5, search_win=24, h_factor=0.08)
    denoised = nlm.process(noisy, s_meter_dbfs=-30.0)

    # ノイズ誤差低減量の測定
    err_in = float(np.std(noisy - clean))
    err_out = float(np.std(denoised - clean))
    red_db = float(20.0 * np.log10(err_in / max(1e-12, err_out)))

    print(f"[*] Acoustic NLM Noise Reduction: {red_db:.2f} dB (std: {err_in:.4f} -> {err_out:.4f})")
    assert red_db >= 1.0, f"Expected >=1dB reduction, got {red_db:.2f} dB"

    # 信号の相関係数 (波形の忠実度)
    corr = float(np.corrcoef(clean, denoised)[0, 1])
    print(f"[*] NLM Waveform Fidelity (correlation): {corr:.4f}")
    assert corr > 0.98, "Signal waveform was distorted by NLM"
    print("[OK] test_periodic_signal_preservation_and_noise_reduction passed")


def test_stereo_processing():
    """ステレオ 2ch 信号に対する左右独立 NLM 処理検証"""
    fs = 48000.0
    sig_l = np.sin(np.linspace(0, 20, 256)).astype(np.float32)
    sig_r = np.cos(np.linspace(0, 40, 256)).astype(np.float32)
    stereo_in = np.column_stack([sig_l, sig_r])

    nlm = AcousticNonLocalMeans(sample_rate=fs)
    out = nlm.process(stereo_in, s_meter_dbfs=-30.0)

    assert out.shape == stereo_in.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    print("[OK] test_stereo_processing passed")


def test_strong_signal_bypass():
    """強電界時の完全バイパス検証"""
    fs = 48000.0
    audio = np.random.randn(256).astype(np.float32)
    nlm = AcousticNonLocalMeans(sample_rate=fs)
    out = nlm.process(audio, s_meter_dbfs=-15.0)
    assert np.array_equal(out, audio)
    print("[OK] test_strong_signal_bypass passed")


if __name__ == "__main__":
    print("===== Running Acoustic Non-Local Means Tests =====")
    test_periodic_signal_preservation_and_noise_reduction()
    test_stereo_processing()
    test_strong_signal_bypass()
    print("ALL ACOUSTIC NLM TESTS PASSED!")
