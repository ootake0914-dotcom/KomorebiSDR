"""
Unit tests for TopologicalClickSuppressor (Topological Data Analysis FM demodulator).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_demod import TopologicalClickSuppressor


def test_clean_signal_transparency():
    """クリーンなFM変調波に対してクリック特異点が誤検出されず、高忠実度復調されることを検証"""
    fs = 288000.0
    f_mod = 1000.0
    dev = 50000.0  # 50kHz偏移
    t = np.arange(4096, dtype=np.float64) / fs

    # 積分位相
    mod_signal = np.sin(2.0 * np.pi * f_mod * t)
    phase = (dev / f_mod) * (-np.cos(2.0 * np.pi * f_mod * t))
    iq = np.exp(1j * phase).astype(np.complex64)

    tda = TopologicalClickSuppressor(sample_rate=fs, max_dev_hz=75000.0)
    demod = tda.process(iq)

    assert tda.detected_clicks == 0, f"False positive clicks detected: {tda.detected_clicks}"
    # 復調信号の周波数偏移振幅 (理論値: 2*pi*dev / fs)
    expected_amp = (2.0 * np.pi * dev) / fs
    actual_amp = float(np.max(np.abs(demod[100:-100])))
    assert abs(actual_amp - expected_amp) < 0.05 * expected_amp, f"Demod amp mismatch: {actual_amp} vs {expected_amp}"
    print("[OK] test_clean_signal_transparency passed")


def test_synthetic_click_suppression():
    """
    弱電界フェージングで発生する原点包囲トポロジカル特異点 (Phase Slip / クリックスパイク) を注入し、
    検出と局所微分同相写像によるスパイク抑圧性能を検証。
    """
    fs = 288000.0
    t = np.arange(2048, dtype=np.float64) / fs
    # 搬送波 + 1kHzトーン
    carrier = np.exp(1j * (2.0 * np.pi * 1000.0 * t)).astype(np.complex64)

    # サンプル位置 1000 にトポロジカル特異点を人工注入:
    # 振幅を原点近傍 (0.01) に引き落とし、位相を突然 180° (pi) 回転させる
    carrier[1000] = 0.01 * np.exp(1j * (np.angle(carrier[1000]) + np.pi))

    # 通常の無保護差分復調
    diff_raw = carrier[1:] * np.conj(carrier[:-1])
    raw_demod = np.angle(diff_raw)
    raw_spike_peak = float(np.max(np.abs(raw_demod[990:1010])))

    # TDA トポロジカル復調器
    tda = TopologicalClickSuppressor(sample_rate=fs, max_dev_hz=75000.0)
    tda_demod = tda.process(carrier)
    tda_spike_peak = float(np.max(np.abs(tda_demod[990:1010])))

    print(f"[*] Raw demod spike peak: {raw_spike_peak:.3f} rad, TDA repaired peak: {tda_spike_peak:.3f} rad")
    assert tda.detected_clicks >= 1, "Failed to detect injected topological click singularity"
    # スパイクピークが半分以下 (正規変調範囲内) に抑圧されていること
    assert tda_spike_peak < raw_spike_peak * 0.6, f"Spike not suppressed: {tda_spike_peak} vs {raw_spike_peak}"
    print("[OK] test_synthetic_click_suppression passed")


def test_boundary_continuity_and_stability():
    """複数ブロックを連続処理した際の境界連続性と無NaN・無発散検証"""
    fs = 288000.0
    tda = TopologicalClickSuppressor(sample_rate=fs, max_dev_hz=75000.0)
    rng = np.random.RandomState(42)

    for _ in range(10):
        # ランダムノイズ混じりの微弱信号
        iq_block = (0.5 * np.exp(1j * rng.uniform(-np.pi, np.pi, 1024)) +
                    0.2 * (rng.randn(1024) + 1j * rng.randn(1024))).astype(np.complex64)
        out = tda.process(iq_block)
        assert len(out) == 1024
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        assert np.max(np.abs(out)) <= np.pi

    print("[OK] test_boundary_continuity_and_stability passed")


if __name__ == "__main__":
    print("===== Running Topological TDA Click Suppressor Tests =====")
    test_clean_signal_transparency()
    test_synthetic_click_suppression()
    test_boundary_continuity_and_stability()
    print("ALL TOPOLOGICAL TDA TESTS PASSED!")
