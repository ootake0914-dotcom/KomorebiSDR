"""
Unit tests for audiophile_dsp.py
高級オーディオ数理モジュールの単体検証テスト。
1. TPDFディザー＆音響心理ノイズシェーピングによる微小信号量子化歪みの消滅
2. 最小位相アポダイジング変換によるプリリンギング（予兆波紋）の完全消滅
3. アクティブDCサーボによる可聴低域位相回転ゼロ・直流完全相殺
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from audiophile_dsp import TpdfDitherNoiseShaper, MinimumPhaseApodizer, ActiveDcServo
from dsp import design_fir_kaiser


def test_tpdf_dither():
    print("===== test_tpdf_dither =====")
    sr = 48000.0
    n = 16384
    t = np.arange(n) / sr
    
    # 振幅 -80dBFS の微小信号 (997Hz: 整数倍周期を避けた標準オーディオ測定周波数)
    amp_f = 10.0 ** (-80.0 / 20.0)  # 約 0.0001 (16bitで約 3.2 LSB)
    test_sig = (amp_f * np.sin(2 * np.pi * 997.0 * t)).astype(np.float32)

    # 1. 単純丸め (ディザーなし): 階段状になり高調波歪みが発生
    raw_int = np.round(np.clip(test_sig * 32767.0, -32768.0, 32767.0)).astype(np.int16)

    # 2. TPDFディザー＋ノイズシェーピング
    shaper = TpdfDitherNoiseShaper(sample_rate=sr)
    shaped_int = shaper.process_to_int16(test_sig)

    # FFTスペクトル比較
    win = np.hanning(n)
    spec_raw = np.abs(np.fft.rfft(raw_int * win))
    spec_shaped = np.abs(np.fft.rfft(shaped_int * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # 3次高調波 (2991Hz 付近) のスパイク強度を測定
    mask_3rd = (freqs >= 2970.0) & (freqs <= 3015.0)
    peak_3rd_raw = float(np.max(spec_raw[mask_3rd]))
    peak_3rd_shaped = float(np.max(spec_shaped[mask_3rd]))

    # 可聴帯域 (1kHz〜4kHz) の高調波比率
    harm_suppression_db = 20.0 * np.log10(peak_3rd_shaped / (peak_3rd_raw + 1e-12))
    print(f"[*] 単純丸めの3次高調波スパイク: {peak_3rd_raw:.1f}")
    print(f"[*] TPDFディザー適用後の3次高調波: {peak_3rd_shaped:.1f} (高調波歪み低減: {harm_suppression_db:.1f} dB)")

    # ディザーにより高調波歪みが大幅に抑制されていることを検証
    assert harm_suppression_db < -6.0 or peak_3rd_shaped < peak_3rd_raw * 0.5, "ディザーによる高調波歪みの解消が不十分です"
    print("[OK] TPDFディザー＆ノイズシェーピング単体テスト成功")


def test_minimum_phase_apodizer():
    print("\n===== test_minimum_phase_apodizer =====")
    sr = 48000.0
    # 65タップの直線位相Kaiser窓LPF (8.5kHzカットオフ)
    linear_fir = design_fir_kaiser(num_taps=65, cutoff_norm=8500.0 / sr, beta=6.5)

    # 最小位相フィルタへ変換
    min_fir = MinimumPhaseApodizer.convert_fir_to_minimum_phase(linear_fir, n_fft=4096)

    assert len(min_fir) == len(linear_fir)

    # 1. プリリンギングエネルギーの検証
    # 直線位相FIR: ピークは中央 (idx = 32)
    peak_linear_idx = np.argmax(np.abs(linear_fir))
    pre_energy_linear = np.sum(linear_fir[:peak_linear_idx] ** 2) / np.sum(linear_fir ** 2)

    # 最小位相FIR: ピークは先頭 (idx = 0 または 1)
    peak_min_idx = np.argmax(np.abs(min_fir))
    pre_energy_min = np.sum(min_fir[:peak_min_idx] ** 2) / np.sum(min_fir ** 2)

    print(f"[*] 直線位相FIRのピーク位置: idx={peak_linear_idx}, ピーク前エネルギー(プリリンギング): {pre_energy_linear * 100:.1f}%")
    print(f"[*] 最小位相FIRのピーク位置: idx={peak_min_idx}, ピーク前エネルギー(プリリンギング): {pre_energy_min * 100:.2f}%")

    assert peak_min_idx <= 2, "最小位相FIRのピークが先頭にありません"
    assert pre_energy_min < 0.05, f"プリリンギングが十分に消滅していません: {pre_energy_min * 100:.2f}%"

    # 2. 周波数振幅特性の一致検証 (通過域 0〜6kHz)
    h_lin = np.abs(np.fft.rfft(linear_fir, 1024))
    h_min = np.abs(np.fft.rfft(min_fir, 1024))
    passband_mask = np.fft.rfftfreq(1024, 1.0 / sr) <= 6000.0
    max_diff_db = np.max(np.abs(20.0 * np.log10(h_min[passband_mask] / (h_lin[passband_mask] + 1e-12))))
    print(f"[*] 通過域の振幅特性最大誤差: {max_diff_db:.2f} dB (期待値 < 0.8 dB)")
    assert max_diff_db < 0.8, f"振幅特性が一致していません: {max_diff_db:.2f} dB"
    print("[OK] 最小位相アポダイジング変換単体テスト成功")


def test_active_dc_servo():
    print("\n===== test_active_dc_servo =====")
    sr = 48000.0
    servo = ActiveDcServo(sample_rate=sr, time_constant_sec=1.0)

    # 30Hz低音正弦波 + 大きなDCオフセット (+0.25)
    t = np.arange(int(sr * 6.0)) / sr  # 6.0秒分 (NumPyなら一瞬で演算)
    dc_offset = 0.25
    clean_30hz = 0.5 * np.sin(2 * np.pi * 30.0 * t)
    distorted = (clean_30hz + dc_offset).astype(np.float32)

    # サーボ通過
    block_size = 2048
    out_blocks = []
    for start in range(0, len(distorted), block_size):
        chunk = distorted[start:start + block_size]
        out_blocks.append(servo.process(chunk))
    out = np.concatenate(out_blocks)

    # 収束後 (後半3200サンプル = 30Hzの厳密な2周期) のDCオフセット測定
    tail = out[-3200:]
    remaining_dc = float(np.mean(tail))
    print(f"[*] サーボ通過後の残留DCオフセット: {remaining_dc:.5f} (元: {dc_offset})")
    assert abs(remaining_dc) < 0.005, f"DCオフセットが十分に相殺されていません: {remaining_dc}"

    # 30Hz信号の位相回転 (時間差) を測定: ゼロ度であることを検証
    ref_tail = clean_30hz[-3200:]
    # 相互相関による遅延サンプル数測定
    corr = np.correlate(tail - remaining_dc, ref_tail, mode="full")
    lag = np.argmax(corr) - (len(ref_tail) - 1)
    phase_deg = (lag / sr) * 30.0 * 360.0
    print(f"[*] 30Hzにおける位相回転角: {phase_deg:.2f}° (通常のHPFは約+15°〜+30°回転する)")
    assert abs(phase_deg) < 2.0, f"低域の位相が回転しています: {phase_deg:.2f}°"
    print("[OK] 低域位相回転ゼロ・アクティブDCサーボ単体テスト成功")


if __name__ == "__main__":
    test_tpdf_dither()
    test_minimum_phase_apodizer()
    test_active_dc_servo()
    print("\nALL AUDIOPHILE DSP TESTS PASSED!")
