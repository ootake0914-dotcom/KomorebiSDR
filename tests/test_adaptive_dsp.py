"""
Unit tests for adaptive_dsp.py
適応IQインバランス補正およびダイナミックIF帯域トラッカーの単体数理シミュレーション検証。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_dsp import (
    AdaptiveIqCorrector,
    CognitiveSpeechMusicTracker,
    DynamicIfBandwidthTracker,
    UltrasonicSquelchTracker,
)


def test_iq_imbalance_correction():
    print("===== test_iq_imbalance_correction =====")
    fs = 1152000.0
    duration = 0.5  # 0.5秒分
    t = np.arange(int(fs * duration)) / fs

    # 1. 理想的な信号 (100kHzのトーン信号)
    f_sig = 100000.0
    ideal_i = np.cos(2 * np.pi * f_sig * t)
    ideal_q = np.sin(2 * np.pi * f_sig * t)

    # 2. 人工的な不均衡の付与:
    # 振幅比 1.15 (+1.21dB), 直交位相ズレ 4.0度 (0.0698 rad)
    gain_err = 1.15
    phase_err_rad = np.radians(4.0)

    distorted_i = ideal_i
    distorted_q = gain_err * np.sin(2 * np.pi * f_sig * t + phase_err_rad)
    distorted_iq = (distorted_i + 1j * distorted_q).astype(np.complex64)

    # 補正前のイメージ抑圧比 (IRR) を計測
    # 期待される鏡像は -100kHz
    def measure_irr(iq, f_tone):
        spec = np.abs(np.fft.fft(iq))
        freqs = np.fft.fftfreq(len(iq), 1.0 / fs)
        idx_sig = np.argmin(np.abs(freqs - f_tone))
        idx_img = np.argmin(np.abs(freqs - (-f_tone)))
        sig_p = spec[idx_sig] ** 2
        img_p = spec[idx_img] ** 2
        irr_db = 10.0 * np.log10(sig_p / (img_p + 1e-12))
        return irr_db

    initial_irr = measure_irr(distorted_iq[:8192], f_sig)
    print(f"[*] 補正前のイメージ抑圧比 (IRR): {initial_irr:.2f} dB")
    assert initial_irr < 32.0, "補正前は鏡像が強く出ているはず"

    # 3. 適応補正器にブロック単位 (66048サンプル ≈ 57ms) で流し込む
    corrector = AdaptiveIqCorrector(sample_rate=fs, time_constant_sec=0.15)
    block_size = 32768
    corrected_blocks = []

    for start in range(0, len(distorted_iq), block_size):
        chunk = distorted_iq[start:start + block_size]
        corrected_chunk = corrector.process(chunk)
        corrected_blocks.append(corrected_chunk)

    corrected_iq = np.concatenate(corrected_blocks)

    # 収束後 (後半ブロック) での IRR を計測
    final_irr = measure_irr(corrected_iq[-8192:], f_sig)
    print(f"[*] 収束後のイメージ抑圧比 (IRR): {final_irr:.2f} dB (推定位相誤差: {corrector.estimated_phase_error_deg:.2f}°, 推定ゲイン比: {corrector.estimated_gain_imbalance_db:.2f} dB)")

    # 鏡像抑圧比が 30dB 未満から 50dB 以上へ大幅向上していることを検証
    assert final_irr >= 50.0, f"収束後のIRRが不十分です: {final_irr:.2f} dB"
    assert abs(corrector.estimated_phase_error_deg - 4.0) < 1.0, "位相誤差の推定精度が不十分です"
    print("[OK] IQインバランス自動補正テスト成功 (IRR改善: +{:.1f} dB)".format(final_irr - initial_irr))


def test_dynamic_if_bandwidth():
    print("\n===== test_dynamic_if_bandwidth =====")
    tracker = DynamicIfBandwidthTracker(min_bw_hz=85000.0, max_bw_hz=190000.0)

    # 1. 静かなトーク・休符区間 (最大偏移 ±15kHz)
    quiet_demod = np.random.normal(0.0, 5000.0, 4000)  # ピーク ~15kHz
    bw_quiet = 0.0
    for _ in range(10):
        bw_quiet = tracker.update(quiet_demod)

    print(f"[*] 静かなトーク時の追従IF帯域幅: {bw_quiet / 1e3:.1f} kHz (期待値: 85〜100 kHz)")
    assert 80000.0 <= bw_quiet <= 105000.0, "トーク時の帯域が狭まりませんでした"

    # 2. 音楽フォルテシモ区間 (最大偏移 ±75kHz)
    loud_demod = 75000.0 * np.sin(np.linspace(0, 100 * np.pi, 4000))
    bw_loud = 0.0
    for _ in range(15):
        bw_loud = tracker.update(loud_demod)

    print(f"[*] 音楽大音量時の追従IF帯域幅: {bw_loud / 1e3:.1f} kHz (期待値: 175〜190 kHz)")
    assert 170000.0 <= bw_loud <= 190000.0, "大音量時の帯域が広がりませんでした"
    print("[OK] ダイナミックIF帯域幅追従テスト成功")


def test_ultrasonic_squelch():
    print("\n===== test_ultrasonic_squelch =====")
    sr = 288000.0
    squelch = UltrasonicSquelchTracker(sample_rate=sr, noise_threshold_db=-38.0)

    # 1. 局間ノイズ（未変調・ホワイト三角雑音）: 超音波高域パワーが大きい
    noise_demod = np.random.normal(0.0, 0.4, 4000).astype(np.float32)
    gain_mute = 1.0
    is_open_mute = True
    for _ in range(40):
        gain_mute, is_open_mute = squelch.process(noise_demod)

    print(f"[*] 局間ノイズ時のスケルチ状態: is_open={is_open_mute}, gain={gain_mute:.2f} (ノイズ推定: {squelch.noise_db:.1f} dB)")
    assert not is_open_mute, "局間ノイズでスケルチが閉じませんでした"
    assert gain_mute < 0.05, f"ソフトフェードミュートゲインが十分に下がりませんでした: {gain_mute}"

    # 2. 本物のFM変調波キャリア受信時 (クワイエティング効果で超音波ノイズ急減)
    t = np.arange(4000) / sr
    clean_demod = (0.25 * np.sin(2 * np.pi * 1000.0 * t) + np.random.normal(0.0, 0.0005, 4000)).astype(np.float32)
    gain_open = 0.0
    is_open = False
    for _ in range(40):
        gain_open, is_open = squelch.process(clean_demod)

    print(f"[*] キャリア受信時のスケルチ状態: is_open={is_open}, gain={gain_open:.2f} (ノイズ推定: {squelch.noise_db:.1f} dB)")
    assert is_open, "本物のキャリアを受信してもスケルチが開きませんでした"
    assert gain_open > 0.95, f"ソフトフェードゲインが十分に開きませんでした: {gain_open}"
    print("[OK] 超音波ノイズ比追従型コグニティブ・オートスケルチテスト成功")


def test_cognitive_speech_music_tracker():
    print("\n===== test_cognitive_speech_music_tracker =====")
    sr = 48000.0
    eq = CognitiveSpeechMusicTracker(sample_rate=sr)

    # 1. トーク音声区間 (300Hz〜2500Hz中心、低音サブベースなし、ロールオフ低)
    t = np.arange(1024) / sr
    speech_audio = (
        0.4 * np.sin(2 * np.pi * 800.0 * t) +
        0.3 * np.sin(2 * np.pi * 1800.0 * t) +
        0.1 * np.sin(2 * np.pi * 2800.0 * t)
    ).astype(np.float32)

    prob_speech = 0.0
    for _ in range(30):
        prob_speech = eq.analyze(speech_audio)

    print(f"[*] トーク区間の推定音声確率: {prob_speech:.2f} (期待値 > 0.6)")
    assert prob_speech > 0.6, f"トーク区間を音声と認識できませんでした: {prob_speech}"

    # トーク時の了解度EQ処理
    speech_out = eq.process(speech_audio)
    assert speech_out.shape == speech_audio.shape
    assert not np.allclose(speech_out, speech_audio), "トーク用EQが適用されていません"

    # 2. 音楽区間 (重低音50Hz〜60Hzサブベース + 12kHz以上のハイハット)
    music_audio = (
        0.6 * np.sin(2 * np.pi * 55.0 * t) +
        0.3 * np.sin(2 * np.pi * 1000.0 * t) +
        0.3 * np.sin(2 * np.pi * 12000.0 * t)
    ).astype(np.float32)

    prob_music = 1.0
    for _ in range(60):
        prob_music = eq.analyze(music_audio)

    print(f"[*] 音楽区間の推定音声確率: {prob_music:.2f} (期待値 < 0.2)")
    assert prob_music < 0.2, f"音楽区間を音楽と認識できませんでした: {prob_music}"

    # 音楽時は完全フラット（原音維持）
    music_out = eq.process(music_audio)
    assert np.allclose(music_out, music_audio, atol=1e-5), "音楽区間で完全フラットが維持されていません"
    print("[OK] 音声/音楽 認知型オートチルトEQテスト成功")


if __name__ == "__main__":
    test_iq_imbalance_correction()
    test_dynamic_if_bandwidth()
    test_ultrasonic_squelch()
    test_cognitive_speech_music_tracker()
    print("\nALL ADAPTIVE DSP TESTS PASSED!")
