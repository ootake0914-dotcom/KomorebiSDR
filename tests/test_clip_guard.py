"""過大入力ガード (Phase 1): notch/RMTがADC飽和時に処理しないこと。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from adaptive_notch import AdaptiveNotchCanceller
from rmt_denoiser import SafeRmtDenoiser

FS = 48000.0
N = 2752 * 4


def _hum(clipped=False):
    t = np.arange(N) / FS
    h = 0.05 * np.sin(2 * np.pi * 50.0 * t) + 0.03 * np.sin(2 * np.pi * 100.0 * t)
    x = (h + 0.01 * np.random.default_rng(7).standard_normal(N)).astype(np.float32)
    if clipped:
        x = np.clip(x * 40.0, -1.0, 1.0).astype(np.float32)
    return x


def test_notch_guard():
    nc = AdaptiveNotchCanceller()
    # クリーンハムは検出・除去する (dwell確定のため複数ブロック)
    xh = _hum(False)
    for _ in range(6):
        y, info = nc.process_mono(xh, clip=False)
    assert info["lines"], f"clean hum not detected: {info}"
    # クリップ時は一切触らない
    xc = _hum(True)
    y2, info2 = nc.process_mono(xc, clip=True)
    assert info2["bypass_reason"] == "adc-clip", info2
    assert np.array_equal(np.asarray(y2), xc), "clip input must pass through"
    # ステレオ経路も同様
    (_, _), info3 = nc.process_stereo(xc, xc, clip=True)
    assert info3["bypass_reason"] == "adc-clip", info3
    print("notch guard OK")


def test_rmt_guard():
    dn = SafeRmtDenoiser()
    rng = np.random.default_rng(11)
    t = np.arange(N) / FS
    clean = 0.05 * np.sin(2 * np.pi * 1000.0 * t)
    x = (clean + 0.05 * rng.standard_normal(N)).astype(np.float32)
    # クリップ時はバイパス (遅延整合出力のみ)
    xc = np.clip(x * 40.0, -1.0, 1.0).astype(np.float32)
    y, info = dn.process_mono(xc, s_meter_dbfs=-40.0, snr_db=12.0, ch="g", clip=True)
    assert info["bypass_reason"] == "adc-clip", info
    assert len(y) == len(xc)
    # クリーン時は通常処理 (adc-clipではない)
    dn2 = SafeRmtDenoiser()
    _, info2 = dn2.process_mono(x, s_meter_dbfs=-40.0, snr_db=12.0, ch="g", clip=False)
    assert info2["bypass_reason"] != "adc-clip", info2
    # ステレオ経路
    (_, _), info3 = dn.process_stereo(xc, xc, s_meter_dbfs=-40.0, clip=True)
    assert info3["bypass_reason"] == "adc-clip", info3
    print("rmt guard OK")


if __name__ == "__main__":
    test_notch_guard()
    test_rmt_guard()
    print("ALL CLIP-GUARD TESTS PASSED!")
