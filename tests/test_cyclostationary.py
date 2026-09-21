"""
Unit tests for CyclostationaryFeatureDetector (Cyclostationary Signal Processing).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_rf import CyclostationaryFeatureDetector


def test_sub_noise_cyclic_detection():
    """
    熱雑音フロア下に埋没した微弱パイロット波 (SNR ~ -12dB) を
    2次周期定常性 (スペクトル相関 SCD) によりブラインド検出できることを検証。
    """
    fs = 288000.0
    t = np.arange(8192, dtype=np.float64) / fs
    f_pilot = 19000.0  # 19kHz FMステレオパイロット

    rng = np.random.RandomState(42)
    # 微弱パイロット信号 (振幅 0.08)
    pilot = 0.08 * np.exp(1j * (2.0 * np.pi * f_pilot * t)).astype(np.complex64)
    # 巨大ホワイトノイズ (振幅 0.5、電力比で約 -12dB の劣悪環境)
    noise = (0.5 * (rng.randn(len(t)) + 1j * rng.randn(len(t)))).astype(np.complex64)
    noisy_iq = pilot + noise

    # 1. 通常のFFTパワースペクトルでの確認 (ノイズに埋没)
    fft_raw = np.abs(np.fft.fft(noisy_iq[:2048])) ** 2
    freqs = np.fft.fftfreq(2048, 1.0 / fs)
    idx_p = int(np.argmin(np.abs(freqs - f_pilot)))
    raw_snr_db = 10.0 * np.log10(fft_raw[idx_p] / np.median(fft_raw))
    print(f"[*] Raw FFT SNR at 19kHz: {raw_snr_db:.2f} dB (ノイズフロアに埋没)")

    # 2. 周期定常性検出器
    csd = CyclostationaryFeatureDetector(sample_rate=fs, detection_thresh_db=3.0)
    snr_db, peak_pow, is_detected = csd.compute_cyclic_spectrum(noisy_iq, alpha_target_hz=19000.0, n_fft=1024)

    print(f"[*] Cyclostationary SCD SNR at alpha=19kHz: {snr_db:.2f} dB, detected={is_detected}")
    assert is_detected, f"Failed to detect sub-noise pilot: cyclic SNR = {snr_db:.2f} dB"
    assert snr_db > 3.0, f"Expected strong cyclic correlation peak, got {snr_db:.2f} dB"
    print("[OK] test_sub_noise_cyclic_detection passed")


def test_pure_noise_rejection():
    """純粋なホワイトノイズに対して誤検出 (False Positive) が発生しないことを検証"""
    fs = 288000.0
    rng = np.random.RandomState(123)
    pure_noise = (rng.randn(4096) + 1j * rng.randn(4096)).astype(np.complex64)

    csd = CyclostationaryFeatureDetector(sample_rate=fs, detection_thresh_db=3.0)
    snr_db, _, is_detected = csd.compute_cyclic_spectrum(pure_noise, alpha_target_hz=19000.0)

    print(f"[*] Pure noise cyclic SNR: {snr_db:.2f} dB, detected={is_detected}")
    assert not is_detected, "False positive detection on pure white noise"
    print("[OK] test_pure_noise_rejection passed")


def test_frequency_scanning():
    """巡回周波数帯域スキャンにより目的の変調周期が最高峰として同定されることを検証"""
    fs = 288000.0
    t = np.arange(4096, dtype=np.float64) / fs
    f_target = 19000.0

    rng = np.random.RandomState(99)
    sig = 0.15 * np.exp(1j * (2.0 * np.pi * f_target * t)).astype(np.complex64)
    sig += (0.2 * (rng.randn(len(t)) + 1j * rng.randn(len(t)))).astype(np.complex64)

    csd = CyclostationaryFeatureDetector(sample_rate=fs, detection_thresh_db=3.0)
    peaks = csd.scan_cyclic_frequencies(sig, alpha_range_hz=(18000.0, 20000.0), step_hz=250.0)

    assert len(peaks) > 0, "No peaks discovered in scan"
    best_freq, best_snr = peaks[0]
    print(f"[*] Scan top peak: {best_freq:.1f} Hz (SNR {best_snr:.1f} dB)")
    assert abs(best_freq - f_target) <= 250.0, f"Scan identified wrong peak: {best_freq} vs {f_target}"
    print("[OK] test_frequency_scanning passed")


if __name__ == "__main__":
    print("===== Running Cyclostationary Feature Detector Tests =====")
    test_sub_noise_cyclic_detection()
    test_pure_noise_rejection()
    test_frequency_scanning()
    print("ALL CYCLOSTATIONARY TESTS PASSED!")
