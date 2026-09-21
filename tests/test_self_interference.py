"""
Unit tests for DigitalSelfInterferenceCanceller (In-Band Full Duplex SIC for SDR).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_rf import DigitalSelfInterferenceCanceller


def test_single_spurious_cancellation():
    """
    PC/USBからアンテナに混入した強力な内部スプリアス (+25kHz) に対し、
    直交基底適応追従により 20dB 以上逆位相消去され、目的信号が無傷であることを検証。
    """
    fs = 288000.0
    f_spur = 25000.0  # +25kHz のPCクロック高調波スプリアス
    sic = DigitalSelfInterferenceCanceller(sample_rate=fs, mu=0.1)
    sic.set_spurious_frequencies([f_spur])

    f_mod = 1000.0
    n = 1024

    # 35フレーム連続ストリーム処理 (NLMS適応収束)
    for b in range(35):
        t = (b * n + np.arange(n, dtype=np.float64)) / fs
        phase_wanted = 1.2 * np.sin(2.0 * np.pi * f_mod * t)
        sig_wanted = 0.4 * np.exp(1j * phase_wanted).astype(np.complex64)
        spurious = 0.6 * np.exp(1j * (2.0 * np.pi * f_spur * t + 0.75)).astype(np.complex64)
        noisy_iq = sig_wanted + spurious
        clean_iq = sic.process(noisy_iq)

    # 処理後のスプリアス残留パワー測定 (25kHz FFTビン)
    fft_in = np.abs(np.fft.fft(noisy_iq))
    fft_out = np.abs(np.fft.fft(clean_iq))
    freqs = np.fft.fftfreq(len(noisy_iq), 1.0 / fs)
    idx_spur = int(np.argmin(np.abs(freqs - f_spur)))

    power_spur_in = fft_in[idx_spur] ** 2
    power_spur_out = fft_out[idx_spur] ** 2
    actual_cancellation_db = float(10.0 * np.log10(power_spur_in / power_spur_out))

    print(f"[*] Spurious cancellation at +25kHz: {actual_cancellation_db:.1f} dB (sic diagnostic: {sic.cancellation_db:.1f} dB)")
    # 20dB 以上のスプリアス消去
    assert actual_cancellation_db >= 20.0, f"Expected >=20dB cancellation, got {actual_cancellation_db:.1f} dB"

    # 目的信号の保持度 (相関係数)
    corr = float(np.abs(np.vdot(clean_iq, sig_wanted) / (np.linalg.norm(clean_iq) * np.linalg.norm(sig_wanted))))
    print(f"[*] Wanted signal correlation after SIC: {corr:.4f}")
    assert corr > 0.98, "Wanted signal was distorted by SIC"
    print("[OK] test_single_spurious_cancellation passed")


def test_auto_detect_spurious():
    """スプリアス周波数の自動検出 (Auto Detect) 機能を検証"""
    fs = 288000.0
    t = np.arange(2048, dtype=np.float64) / fs
    f_target_spur = 32000.0  # +32kHz スプリアス

    rng = np.random.RandomState(42)
    noise_floor = (0.05 * (rng.randn(len(t)) + 1j * rng.randn(len(t)))).astype(np.complex64)
    spur = 0.5 * np.exp(1j * (2.0 * np.pi * f_target_spur * t)).astype(np.complex64)
    iq_input = noise_floor + spur

    sic = DigitalSelfInterferenceCanceller(sample_rate=fs)
    sic.auto_detect_spurious(iq_input, n_fft=1024, prominence_db=15.0)

    print(f"[*] Auto-detected spurious frequencies: {sic.spurious_freqs}")
    assert len(sic.spurious_freqs) >= 1, "Failed to auto-detect prominent spurious tone"
    best_detected = sic.spurious_freqs[0]
    assert abs(best_detected - f_target_spur) <= (fs / 1024), f"Frequency mismatch: {best_detected} vs {f_target_spur}"
    print("[OK] test_auto_detect_spurious passed")


def test_clean_signal_transparency():
    """スプリアスが存在しないクリーン信号に対する完全素通し・無歪み検証"""
    fs = 288000.0
    t = np.arange(1024, dtype=np.float64) / fs
    clean_in = 0.5 * np.exp(1j * (2.0 * np.pi * 5000.0 * t)).astype(np.complex64)

    sic = DigitalSelfInterferenceCanceller(sample_rate=fs)
    # スプリアスリストが空の時は即座にバイパス
    out = sic.process(clean_in)
    assert np.allclose(out, clean_in), "Bypass failed when no spurious list is registered"
    print("[OK] test_clean_signal_transparency passed")


def test_pipeline_sic_integration():
    """
    SdrDspPipeline 全体を通じた SIC の自動検出・適応消去・素通し透明性の統合テスト。
    """
    from dsp import SdrDspPipeline

    pipeline = SdrDspPipeline(sample_rate=1152000, audio_rate=48000)
    assert hasattr(pipeline, "sic_canceller")
    assert pipeline.sic_enabled is True

    # 1. PC内部クロック高調波 (+45kHz) が混入した生IQ信号を作成
    fs_rf = 1152000.0
    n_rf = 115200  # 0.1秒分
    t = np.arange(n_rf, dtype=np.float64) / fs_rf

    # 目的FM信号 (1kHzトーン変調)
    f_mod = 1000.0
    phase_wanted = 1.0 * np.sin(2.0 * np.pi * f_mod * t)
    sig_wanted = 0.4 * np.exp(1j * phase_wanted).astype(np.complex64)

    # PCスプリアス (+45kHz の急峻なビート)
    f_spur = 45000.0
    spurious = 0.6 * np.exp(1j * (2.0 * np.pi * f_spur * t)).astype(np.complex64)
    noisy_iq = sig_wanted + spurious

    # uint8 生バイト列へ変換 (RTL-SDR形式: 0..255, 127.5中心)
    i_u8 = np.clip(np.real(noisy_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    q_u8 = np.clip(np.imag(noisy_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    raw_bytes = np.empty(n_rf * 2, dtype=np.uint8)
    raw_bytes[0::2] = i_u8
    raw_bytes[1::2] = q_u8

    # 複数ブロック処理して SIC を収束・検出させる
    for _ in range(5):
        audio, spec = pipeline.process(raw_bytes, mode="WFM")

    print(f"[*] Pipeline SIC detected spurious freqs: {pipeline.sic_detected_spurious}")
    print(f"[*] Pipeline SIC cancellation: {pipeline.sic_cancellation_db:.1f} dB")

    # +45kHz 近傍が検出されているか
    assert len(pipeline.sic_detected_spurious) >= 1, "Pipeline SIC failed to detect spurious tone"
    detected_f = pipeline.sic_detected_spurious[0]
    assert abs(detected_f - f_spur) < 2000.0, f"Detected spurious freq mismatch: {detected_f} vs {f_spur}"

    # 2. 延長ケーブルで離してスプリアスが消滅したケース（クリーン信号: アンテナ熱雑音フロアのみ存在）
    rng = np.random.RandomState(42)
    thermal_noise = (0.03 * (rng.randn(len(t)) + 1j * rng.randn(len(t)))).astype(np.complex64)
    clean_iq = sig_wanted + thermal_noise
    i_c = np.clip(np.real(clean_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    q_c = np.clip(np.imag(clean_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    raw_clean = np.empty(n_rf * 2, dtype=np.uint8)
    raw_clean[0::2] = i_c
    raw_clean[1::2] = q_c

    # 選局リセット
    pipeline.set_offset_freq(0.0)
    for _ in range(5):
        audio_c, _ = pipeline.process(raw_clean, mode="WFM")

    print(f"[*] Clean signal spurious freqs after reset: {pipeline.sic_detected_spurious}")
    assert len(pipeline.sic_detected_spurious) == 0, "Spurious freqs should be empty for clean signal"
    print("[OK] test_pipeline_sic_integration passed")


if __name__ == "__main__":
    print("===== Running Digital SIC Tests =====")
    test_single_spurious_cancellation()
    test_auto_detect_spurious()
    test_clean_signal_transparency()
    test_pipeline_sic_integration()
    print("ALL DIGITAL SIC TESTS PASSED!")
