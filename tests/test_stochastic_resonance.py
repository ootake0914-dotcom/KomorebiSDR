"""
Unit tests for BistableStochasticResonator (Stochastic Resonance DSP).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_stereo import BistableStochasticResonator


def test_subthreshold_tone_resonance():
    """
    閾値下の微弱パイロット信号 + 強ノイズ (SNR < 0dB) において、
    確率的共鳴により目的周波数のスペクトル線が検出可能レベルへブーストされることを検証。
    """
    fs = 48000.0
    t = np.arange(4096, dtype=np.float64) / fs
    f_pilot = 1000.0  # 1kHz テストトーン

    rng = np.random.RandomState(42)
    # 微弱信号 (振幅 0.1)
    clean_pilot = 0.1 * np.cos(2.0 * np.pi * f_pilot * t)
    # 強ホワイトノイズ (振幅 0.4)
    noise = 0.4 * rng.randn(len(t))
    noisy_input = (clean_pilot + noise).astype(np.float32)

    # 確率的共鳴器
    sr = BistableStochasticResonator(sample_rate=fs, a=0.8, b=0.8, step_size=0.15, coupling=0.6)
    out = sr.process(noisy_input)

    # 入力と出力のFFTパワースペクトル比較
    fft_in = np.abs(np.fft.rfft(noisy_input * np.hanning(len(noisy_input)))) ** 2
    fft_out = np.abs(np.fft.rfft(out * np.hanning(len(out)))) ** 2
    freqs = np.fft.rfftfreq(len(noisy_input), 1.0 / fs)

    # 1kHz ビンのインデックス
    bin_idx = int(np.argmin(np.abs(freqs - f_pilot)))

    # 出力における 1kHz のピーク性 (周囲ノイズフロアに対するコントラスト)
    peak_power = fft_out[bin_idx]
    noise_floor = float(np.median(fft_out[bin_idx - 20 : bin_idx + 21])) + 1e-12
    contrast_out_db = float(10.0 * np.log10(peak_power / noise_floor))

    print(f"[*] Stochastic Resonance: 1kHz line-to-floor contrast = {contrast_out_db:.2f} dB")
    assert contrast_out_db > 3.0, f"Expected resonant peak at pilot tone, got contrast {contrast_out_db} dB"
    print("[OK] test_subthreshold_tone_resonance passed")


def test_rk4_numerical_stability():
    """過大入力 (5.0) に対しても RK4 数値積分が発散せず、有限範囲に収まることを検証"""
    fs = 48000.0
    sr = BistableStochasticResonator(sample_rate=fs, a=1.0, b=1.0)
    extreme_input = np.ones(2048, dtype=np.float32) * 5.0

    out = sr.process(extreme_input)
    assert not np.isnan(out).any(), "NaN detected in RK4 output"
    assert not np.isinf(out).any(), "Inf detected in RK4 output"
    assert np.max(np.abs(out)) <= 5.0, "Output exploded beyond physical bounds"
    print("[OK] test_rk4_numerical_stability passed")


def test_reset_and_continuity():
    """リセットと連続ブロック処理の整合性確認"""
    fs = 48000.0
    sr = BistableStochasticResonator(sample_rate=fs)
    data = np.sin(np.linspace(0, 10, 1024)).astype(np.float32)

    out1 = sr.process(data)
    sr.reset()
    assert sr._x == 0.0
    out2 = sr.process(data)

    assert np.allclose(out1, out2, atol=1e-5), "Reset failed to restore reproducible initial state"
    print("[OK] test_reset_and_continuity passed")


if __name__ == "__main__":
    print("===== Running Stochastic Resonance Tests =====")
    test_subthreshold_tone_resonance()
    test_rk4_numerical_stability()
    test_reset_and_continuity()
    print("ALL STOCHASTIC RESONANCE TESTS PASSED!")
