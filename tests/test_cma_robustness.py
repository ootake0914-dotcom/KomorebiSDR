"""
Stress and robustness test for CMA equalizer and DSP pipeline.
実機接続時の過渡パルス・大振幅入力・ゼロ信号・NaN混入に対する耐性検証テスト。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import SdrDspPipeline, _load_native_core


def test_cma_stability():
    print("===== test_cma_stability =====")
    pipeline = SdrDspPipeline(1152000, 48000)
    pipeline.multipath_cancel_enabled = True
    pipeline.multipath_amount = 0.5  # CMAを強制起動

    # 1. 巨大振幅パルス (10.0 = 定格の10倍以上の過大入力)
    n = 16384
    iq_huge = np.full(n, 10.0 + 10.0j, dtype=np.complex64)
    out = pipeline._apply_cma(iq_huge)
    assert np.all(np.isfinite(out)), "巨大振幅入力でCMAがNaN/Infを発散させました"
    print("[OK] 巨大過大入力 (10.0) に対するCMAの非発散を確認")

    # 2. ゼロ入力 (PLLロック外れ・無信号)
    iq_zero = np.zeros(n, dtype=np.complex64)
    out_zero = pipeline._apply_cma(iq_zero)
    assert np.all(np.isfinite(out_zero)), "ゼロ入力でCMAがNaN/Infを出力しました"
    print("[OK] ゼロ信号 (PLL未ロック) に対するCMAの非発散を確認")

    # 3. 故意のNaN混入入力
    iq_nan = np.zeros(n, dtype=np.complex64)
    iq_nan[100:150] = np.nan + 1j * np.nan
    out_nan = pipeline._apply_cma(iq_nan)
    assert np.all(np.isfinite(out_nan)), "NaN混入時にCMAの安全フォールバックが機能しませんでした"
    print("[OK] NaN混入に対するCMAの自己修復＆原信号フォールバックを確認")


def test_full_pipeline_extreme_inputs():
    print("\n===== test_full_pipeline_extreme_inputs =====")
    pipeline = SdrDspPipeline(1152000, 48000)
    pipeline.set_audiophile_mode(apodizing=True, dc_servo=True, dither=True)

    # 1. オールゼロ (0x00) バイト列 (PLL未ロック時)
    raw_zeros = np.zeros(48 * 200, dtype=np.uint8)
    audio, spec = pipeline.process(raw_zeros, mode="WFM")
    assert np.all(np.isfinite(audio)), "オールゼロIQで復調音声にNaNが混入しました"
    assert np.all(np.isfinite(spec)), "オールゼロIQでスペクトラムにNaNが混入しました"
    print("[OK] オールゼロIQのパイプライン通過確認 (NaNゼロ)")

    # 2. オール255 (0xFF) バイト列 (飽和入力)
    raw_ff = np.full(48 * 200, 255, dtype=np.uint8)
    audio, spec = pipeline.process(raw_ff, mode="WFM")
    assert np.all(np.isfinite(audio)), "飽和IQで復調音声にNaNが混入しました"
    print("[OK] 飽和IQ (0xFF) のパイプライン通過確認 (NaNゼロ)")

    print("[OK] パイプライン極限入力ストレステスト成功")


if __name__ == "__main__":
    test_cma_stability()
    test_full_pipeline_extreme_inputs()
    print("\nALL ROBUSTNESS TESTS PASSED!")
