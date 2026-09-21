"""
Unit tests for FractionalDeemphasis (Fractional Calculus DSP).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_audio import FractionalDeemphasis


def test_dc_gain_unity():
    """DCおよび低周波 (100Hz) でのゲインが 1.0 (0dB) に規格化されていることを確認"""
    deemph = FractionalDeemphasis(sample_rate=48000.0, tau_us=50.0, alpha=0.8)
    # DC入力
    dc_input = np.ones(2048, dtype=np.float32)
    out = deemph.process(dc_input)
    # 定常状態 (後半500サンプル) の値
    assert np.allclose(out[-500:], 1.0, atol=1e-2), f"DC gain mismatch: mean={np.mean(out[-500:])}"
    print("[OK] test_dc_gain_unity passed")


def test_high_freq_attenuation_and_phase():
    """
    10kHz 高域トーンに対する減衰と位相特性の検証:
    - alpha=1.0 (標準1次) に比べ、alpha=0.7 (非整数階) は高域位相回転が大幅に抑制されること
    """
    fs = 48000.0
    t = np.arange(4800, dtype=np.float64) / fs
    f_test = 10000.0
    sig = np.sin(2.0 * np.pi * f_test * t).astype(np.float32)

    # 1. alpha = 1.0 (標準)
    deemph_1 = FractionalDeemphasis(sample_rate=fs, tau_us=50.0, alpha=1.0)
    out_1 = deemph_1.process(sig)

    # 2. alpha = 0.7 (分数階)
    deemph_frac = FractionalDeemphasis(sample_rate=fs, tau_us=50.0, alpha=0.7)
    out_frac = deemph_frac.process(sig)

    # 定常状態の振幅
    amp_in = 1.0
    amp_1 = float(np.max(out_1[-1000:]))
    amp_frac = float(np.max(out_frac[-1000:]))

    # 10kHz では減衰が生じること (amp < 1.0)
    assert amp_1 < 0.5, f"Expected strong attenuation for alpha=1.0 at 10kHz, got {amp_1}"
    assert amp_frac < 0.7, f"Expected attenuation for alpha=0.7 at 10kHz, got {amp_frac}"
    # 分数階 (alpha=0.7) は alpha=1.0 よりも高域通過量が多く (傾き -14dB/dec vs -20dB/dec)、よりHi-Fi
    assert amp_frac > amp_1, f"Fractional should have flatter roll-off: {amp_frac} vs {amp_1}"

    # 位相遅れの測定 (ゼロクロス検出)
    idx_in = np.where((sig[-200:-1] <= 0) & (sig[-199:] > 0))[0]
    idx_1 = np.where((out_1[-200:-1] <= 0) & (out_1[-199:] > 0))[0]
    idx_frac = np.where((out_frac[-200:-1] <= 0) & (out_frac[-199:] > 0))[0]

    assert len(idx_in) > 0 and len(idx_1) > 0 and len(idx_frac) > 0
    # 位相遅れサンプル数
    delay_1 = (idx_1[0] - idx_in[0]) % (fs / f_test)
    delay_frac = (idx_frac[0] - idx_in[0]) % (fs / f_test)

    phase_deg_1 = delay_1 * (360.0 * f_test / fs)
    phase_deg_frac = delay_frac * (360.0 * f_test / fs)

    # alpha=0.7 の位相回転は alpha=1.0 より小さいこと (位相直線性・群遅延改善)
    print(f"[*] 10kHz Phase Lag: alpha=1.0 -> {phase_deg_1:.1f} deg, alpha=0.7 -> {phase_deg_frac:.1f} deg")
    assert phase_deg_frac < phase_deg_1, "Fractional deemphasis must have less phase lag than standard 1st-order"
    print("[OK] test_high_freq_attenuation_and_phase passed")


def test_stereo_channel_consistency():
    """ステレオ 2ch 信号に対する左右独立フィルタリングの検証"""
    fs = 48000.0
    deemph = FractionalDeemphasis(sample_rate=fs, tau_us=50.0, alpha=0.75)
    t = np.arange(2048, dtype=np.float32) / fs
    sig_l = np.sin(2.0 * np.pi * 1000.0 * t)
    sig_r = np.cos(2.0 * np.pi * 2000.0 * t)
    stereo_in = np.column_stack([sig_l, sig_r])

    out = deemph.process(stereo_in)
    assert out.shape == stereo_in.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

    # モノラルで単独処理した結果とビット級に一致すること
    deemph_single_l = FractionalDeemphasis(sample_rate=fs, tau_us=50.0, alpha=0.75)
    single_l_out = deemph_single_l.process(sig_l)
    assert np.allclose(out[:, 0], single_l_out, atol=1e-5), "Stereo L-channel must match mono run"
    print("[OK] test_stereo_channel_consistency passed")


def test_dynamic_alpha_transition():
    """リアルタイム受信中の alpha / S-meter 変動に対する耐性と無発散・無NaN確認"""
    deemph = FractionalDeemphasis(sample_rate=48000.0, tau_us=50.0, alpha=0.8)
    rng = np.random.RandomState(42)

    # 10フレーム連続処理しながら S-Meter を急変
    s_meters = [-15.0, -20.0, -30.0, -45.0, -50.0, -25.0, -10.0, -40.0, -35.0, -18.0]
    for sm in s_meters:
        chunk = rng.randn(1024).astype(np.float32) * 0.2
        out = deemph.process(chunk, s_meter_dbfs=sm)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        assert np.max(np.abs(out)) < 2.0, "Filter exploded during dynamic transition"

    print("[OK] test_dynamic_alpha_transition passed")


if __name__ == "__main__":
    print("===== Running Fractional Deemphasis Tests =====")
    test_dc_gain_unity()
    test_high_freq_attenuation_and_phase()
    test_stereo_channel_consistency()
    test_dynamic_alpha_transition()
    print("ALL FRACTIONAL DSP TESTS PASSED!")
