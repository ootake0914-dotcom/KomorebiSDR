"""
Unit tests for HpssNoiseSeparator (Harmonic-Percussive-Residual Separation).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_audio import HpssNoiseSeparator


def test_harmonic_and_percussive_preservation():
    """調波(メロディ)と打楽器(ドラムアタック)を保持し、等方的背景ノイズを分離消去することを検証"""
    fs = 48000.0
    n = 2048
    t = np.arange(n, dtype=np.float32) / fs

    # 1. 持続トーン (Harmonic: 440Hz)
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    # 2. 打楽器パルス (Percussive: サンプル1000)
    pulse = np.zeros(n, dtype=np.float32)
    pulse[1000] = 1.5
    clean = tone + pulse

    # 3. 等方的背景ヒスノイズ
    rng = np.random.RandomState(42)
    noise = 0.08 * rng.randn(n).astype(np.float32)
    noisy = clean + noise

    hpss = HpssNoiseSeparator(sample_rate=fs, n_fft=256, hop_size=128, kernel_time=9, kernel_freq=9)
    out = hpss.process(noisy, s_meter_dbfs=-40.0)

    # ノイズ低減の確認 (定常トーン部 500〜800サンプルの誤差分散)
    err_in = float(np.std(noisy[500:800] - clean[500:800]))
    err_out = float(np.std(out[500:800] - clean[500:800]))
    red_db = float(20.0 * np.log10(err_in / max(1e-12, err_out)))

    print(f"[*] HPSS Noise Reduction: {red_db:.2f} dB (err: {err_in:.4f} -> {err_out:.4f})")
    assert red_db >= 4.0, f"Expected >=4dB residual noise separation, got {red_db:.2f} dB"

    # 打楽器パルスの保持度
    assert float(out[1000]) > 0.8, f"Percussive pulse crushed: {out[1000]}"
    print("[OK] test_harmonic_and_percussive_preservation passed")


def test_stereo_processing():
    """ステレオ 2ch 信号に対する左右独立の完全分離処理検証"""
    fs = 48000.0
    sig_l = 0.4 * np.sin(np.linspace(0, 30, 1024)).astype(np.float32)
    sig_r = 0.4 * np.cos(np.linspace(0, 60, 1024)).astype(np.float32)
    stereo_in = np.column_stack([sig_l, sig_r])

    hpss = HpssNoiseSeparator(sample_rate=fs)
    out = hpss.process(stereo_in, s_meter_dbfs=-35.0)

    assert out.shape == stereo_in.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    print("[OK] test_stereo_processing passed")


def test_strong_signal_bypass():
    """強電界時の完全バイパス検証"""
    fs = 48000.0
    audio = np.random.randn(1024).astype(np.float32)
    hpss = HpssNoiseSeparator(sample_rate=fs)
    out = hpss.process(audio, s_meter_dbfs=-15.0)
    assert np.array_equal(out, audio)
    print("[OK] test_strong_signal_bypass passed")


if __name__ == "__main__":
    print("===== Running HPSS Noise Separator Tests =====")
    test_harmonic_and_percussive_preservation()
    test_stereo_processing()
    test_strong_signal_bypass()
    print("ALL HPSS TESTS PASSED!")
