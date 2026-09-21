"""
Unit tests for QuaternionMpxDecoupler (Quaternion MPX Stereo Decoupler).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_stereo import QuaternionMpxDecoupler


def test_pure_orthogonal_crosstalk_cancellation():
    """直交軸漏洩 (15° 位相スキュー) を注入し、四元数適応回転によるクロストーク消去を検証"""
    fs = 48000.0
    t = np.arange(2048, dtype=np.float32) / fs

    # 和信号 (L+R: 400Hz)
    sum_m = 0.6 * np.sin(2.0 * np.pi * 400.0 * t)
    # 理想差信号 (L-R: 1kHz)
    diff_true = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)

    # 15° の位相スキューによる直交漏洩をシミュレート
    skew_angle = float(np.radians(15.0))
    diff_i_distorted = diff_true * np.cos(skew_angle)
    diff_q_distorted = diff_true * np.sin(skew_angle)

    decoupler = QuaternionMpxDecoupler(sample_rate=fs, mu_rot=0.08, mu_leak=0.02)

    # 50フレーム連続適応
    for _ in range(50):
        m_out, di_out = decoupler.process(sum_m, diff_i_distorted, diff_q_distorted)

    # 復元された差信号と理想差信号の相関・振幅確認
    amp_true = float(np.max(np.abs(diff_true)))
    amp_restored = float(np.max(np.abs(di_out)))

    print(f"[*] Skew cancellation: angle error phi={np.degrees(decoupler.phi):.2f} deg, "
          f"amp: true={amp_true:.3f}, restored={amp_restored:.3f}")

    # 適応後の回転角が注入スキュー角 (15°) に高精度に収束していること
    assert abs(abs(np.degrees(decoupler.phi)) - 15.0) < 2.5, f"Phi convergence error: {decoupler.phi}"
    # 差信号の振幅が完全復元されていること
    assert abs(amp_restored - amp_true) < 0.05 * amp_true
    print("[OK] test_pure_orthogonal_crosstalk_cancellation passed")


def test_energy_conservation_norm_preservation():
    """四元数直交ローター回転におけるエネルギー (2ノルム) 保存性を検証"""
    fs = 48000.0
    rng = np.random.RandomState(42)

    sum_m = rng.randn(1024).astype(np.float32) * 0.5
    diff_i = rng.randn(1024).astype(np.float32) * 0.4
    diff_q = rng.randn(1024).astype(np.float32) * 0.2

    norm_in = float(np.mean(diff_i ** 2 + diff_q ** 2))

    decoupler = QuaternionMpxDecoupler(sample_rate=fs)
    # 回転角を手動設定
    decoupler.phi = float(np.radians(20.0))
    cos_p = float(np.cos(decoupler.phi))
    sin_p = float(np.sin(decoupler.phi))

    di_rot = diff_i * cos_p - diff_q * sin_p
    dq_rot = diff_i * sin_p + diff_q * cos_p
    norm_out = float(np.mean(di_rot ** 2 + dq_rot ** 2))

    print(f"[*] Quaternion 2-norm: in={norm_in:.6f}, out={norm_out:.6f}")
    assert abs(norm_in - norm_out) < 1e-6, "Quaternion rotation must strictly conserve energy"
    print("[OK] test_energy_conservation_norm_preservation passed")


def test_clean_stereo_bypass_transparency():
    """クリーンなステレオ信号 (直交漏洩なし) において歪み・過剰適応が生じないことの検証"""
    fs = 48000.0
    t = np.arange(2048, dtype=np.float32) / fs
    sum_m = 0.5 * np.sin(2.0 * np.pi * 500.0 * t)
    diff_i = 0.4 * np.sin(2.0 * np.pi * 1200.0 * t)
    diff_q = np.zeros_like(diff_i)  # 漏洩ゼロ

    decoupler = QuaternionMpxDecoupler(sample_rate=fs)
    for _ in range(10):
        m_out, di_out = decoupler.process(sum_m, diff_i, diff_q)

    # 回転角はほぼゼロに留まること
    assert abs(np.degrees(decoupler.phi)) < 1.0
    # 信号は無傷で保たれること
    assert np.allclose(di_out, diff_i, atol=1e-2)
    print("[OK] test_clean_stereo_bypass_transparency passed")


if __name__ == "__main__":
    print("===== Running Quaternion MPX Decoupler Tests =====")
    test_pure_orthogonal_crosstalk_cancellation()
    test_energy_conservation_norm_preservation()
    test_clean_stereo_bypass_transparency()
    print("ALL QUATERNION STEREO TESTS PASSED!")
