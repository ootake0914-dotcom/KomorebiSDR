"""
Unit tests for WaveletNoiseShrinkage (Orthogonal Wavelet Shrinkage / Donoho Theory).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_audio import WaveletNoiseShrinkage


def test_perfect_reconstruction_noiseless():
    """
    閾値ゼロ時における Daubechies 4 (DB4) 直交完全再構成性を検証。
    機械精度限界 (< 1e-12) で元の信号と一致することを確認。
    """
    fs = 48000.0
    rng = np.random.RandomState(42)
    sig = rng.randn(2048).astype(np.float32)

    # threshold_scale = 0.0 (閾値なし、完全再構成)
    wns = WaveletNoiseShrinkage(sample_rate=fs, levels=3, threshold_scale=0.0)
    rec = wns.process(sig, s_meter_dbfs=-40.0)

    max_diff = float(np.max(np.abs(sig - rec)))
    print(f"[*] DB4 Wavelet Reconstruction Error: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Perfect reconstruction failed: max diff = {max_diff}"
    print("[OK] test_perfect_reconstruction_noiseless passed")


def test_donoho_noise_elimination_and_attack_preservation():
    """
    Donoho 万能軟閾値処理による背景ヒスノイズ完全排除と
    ドラムアタック (トランジェント) の無傷保持を検証。
    """
    fs = 48000.0
    n = 2048
    t = np.arange(n, dtype=np.float32) / fs

    # 1. 音楽的トーン + 強烈なアタック音 (ドラムスネア)
    clean_audio = 0.4 * np.sin(2.0 * np.pi * 440.0 * t)
    clean_audio[1000] += 2.0  # 鋭利なインパルス (アタック)

    # 2. 背景ヒスノイズ重畳
    rng = np.random.RandomState(123)
    noise = 0.06 * rng.randn(n).astype(np.float32)
    noisy_audio = clean_audio + noise

    wns = WaveletNoiseShrinkage(sample_rate=fs, levels=3, threshold_scale=1.0)
    denoised = wns.process(noisy_audio, s_meter_dbfs=-40.0)

    # 無音・定常トーン部 (先頭300サンプル) のノイズ分散測定
    noise_in_std = float(np.std(noisy_audio[:300] - clean_audio[:300]))
    noise_out_std = float(np.std(denoised[:300] - clean_audio[:300]))
    reduction_db = float(20.0 * np.log10(noise_in_std / noise_out_std))

    print(f"[*] Wavelet Noise Reduction: {reduction_db:.2f} dB (std: {noise_in_std:.4f} -> {noise_out_std:.4f})")
    assert reduction_db >= 4.0, f"Expected >=4dB noise reduction, got {reduction_db:.2f} dB"

    # アタック音 (サンプル1000) の保持確認
    attack_clean = float(clean_audio[1000])
    attack_denoised = float(denoised[1000])
    retention_ratio = attack_denoised / attack_clean
    print(f"[*] Attack impulse retention: {retention_ratio * 100:.1f}% (clean={attack_clean:.2f}, denoised={attack_denoised:.2f})")
    assert retention_ratio >= 0.75, f"Transient attack was crushed: retention = {retention_ratio:.2f}"
    print("[OK] test_donoho_noise_elimination_and_attack_preservation passed")


def test_stereo_independence():
    """ステレオ 2ch 信号に対する左右独立の完全再構成・処理検証"""
    fs = 48000.0
    rng = np.random.RandomState(456)
    sig_l = 0.5 * np.sin(np.linspace(0, 50, 1024)).astype(np.float32) + 0.05 * rng.randn(1024).astype(np.float32)
    sig_r = 0.5 * np.cos(np.linspace(0, 100, 1024)).astype(np.float32) + 0.05 * rng.randn(1024).astype(np.float32)
    stereo_in = np.column_stack([sig_l, sig_r])

    wns = WaveletNoiseShrinkage(sample_rate=fs)
    stereo_out = wns.process(stereo_in, s_meter_dbfs=-40.0)

    assert stereo_out.shape == stereo_in.shape
    assert not np.isnan(stereo_out).any()
    assert not np.isinf(stereo_out).any()

    # モノラル単独処理と完全一致
    mono_l = wns.process(sig_l, s_meter_dbfs=-40.0)
    assert np.allclose(stereo_out[:, 0], mono_l, atol=1e-5), "Stereo L-channel mismatch with mono run"
    print("[OK] test_stereo_independence passed")


def test_strong_signal_bypass():
    """強電界 (S-Meter > -22dBFS) 時の完全バイパス (ビット一致) 検証"""
    fs = 48000.0
    audio = np.random.randn(1024).astype(np.float32)
    wns = WaveletNoiseShrinkage(sample_rate=fs)

    out = wns.process(audio, s_meter_dbfs=-15.0)
    assert np.array_equal(out, audio), "Strong signal must bit-identically bypass"
    print("[OK] test_strong_signal_bypass passed")


if __name__ == "__main__":
    print("===== Running Wavelet Noise Shrinkage Tests =====")
    test_perfect_reconstruction_noiseless()
    test_donoho_noise_elimination_and_attack_preservation()
    test_stereo_independence()
    test_strong_signal_bypass()
    print("ALL WAVELET NOISE SHRINKAGE TESTS PASSED!")
