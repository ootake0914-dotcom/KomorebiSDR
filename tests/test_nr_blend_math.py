"""ステレオNRブレンド則の単体テスト (合成信号、HW不要)。

仕様 (機械判定で一段外しを採用):
  呼び出し側 (dsp_wfm.py) は blend 引数に nr_gain を掛けない。
  _freq_dependent_blend は低域=blend、高域=blend^2*nr_gain。
  旧二重適用 (高域nr^3) からの変更根拠: 同一合成弱ステレオの対比較で
  SI-SDR side +1.22dB (番組+1.7dB、ヒス+1.0dB)、Scoreq MOS向上。
  docstringの「高域=blend^2*nr_gain」の blend はNR込み前の値。

検証:
  1) 低域トーン (200Hz) の利得 ≒ blend
  2) 高域トーン (12kHz) の利得 ≒ lo_frac*blend + (1-lo_frac)*blend^2*nr
     (1次相補クロスオーバーの理論値。xoは実属性から取得)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

FS = 48000


def _tone_gain(dsp, freq_hz, blend, nr_gain):
    n = FS
    t = np.arange(n) / float(FS)
    diff = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    dsp.stereo_nr_gain = float(nr_gain)
    y = dsp._freq_dependent_blend(diff, float(blend)).astype(np.float64)
    k0 = 4800  # クロスオーバー状態の過渡を捨てる
    a = 2.0 * abs(np.mean(
        y[k0:] * np.exp(-2j * np.pi * freq_hz * np.arange(k0, n) / float(FS))))
    return float(a)  # 入力振幅1.0に対する利得


def _fresh(blend_raw=0.5):
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.freq_blend_enabled = True
    dsp.multipath_gain = 1.0
    dsp.aci_gain = 1.0
    dsp._stereo_blend = float(blend_raw)
    return dsp


def test_blend_arg_convention():
    """呼び出し側の約束: 引数 = _stereo_blend * mp * aci (nr_gainなし)。
    低域利得が blend に一致することで確認する (nr込みなら blend*nr になる)。"""
    print("===== test_blend_arg_convention =====")
    b, g = 0.5, 0.5
    dsp = _fresh(b)
    # 修正後の呼び出し側を再現 (nr_gainは掛けない)
    arg = dsp._stereo_blend * dsp.multipath_gain * dsp.aci_gain
    lo = _tone_gain(dsp, 200.0, arg, g)
    print(f"[*] lo gain={lo:.3f} (expect ~{b:.3f})")
    assert abs(lo - b) < 0.03, f"低域利得が blend と不一致: {lo:.3f} vs {b}"
    print("[OK] blend引数はNR込み前 (低域=blend)")


def test_highband_matches_docstring():
    """高域利得が blend^2 * nr_gain (理論値±0.035) であること。
    二重適用 (nr^3) なら約0.09になり、この帯に入らない。"""
    print("===== test_highband_matches_docstring =====")
    b, g = 0.5, 0.5
    dsp = _fresh(b)
    xo = float(dsp.freq_blend_xo_hz)
    lo_frac = 1.0 / np.sqrt(1.0 + (12000.0 / xo) ** 2)
    expect = lo_frac * b + (1.0 - lo_frac) * b * b * g
    arg = dsp._stereo_blend * dsp.multipath_gain * dsp.aci_gain
    hi = _tone_gain(dsp, 12000.0, arg, g)
    print(f"[*] hi gain={hi:.3f} (expect {expect:.3f}, xo={xo:.0f}Hz)")
    assert abs(hi - expect) < 0.035, \
        f"高域利得が設計値と不一致: {hi:.3f} vs {expect:.3f}"
    print("[OK] 高域=blend^2*nr (二重適用なし)")


def test_clean_passthrough_unchanged():
    """blend=1, nr=1 では旧スカラー動作と一致 (高速経路)。"""
    print("===== test_clean_passthrough_unchanged =====")
    dsp = _fresh(1.0)
    for f in (200.0, 5000.0, 12000.0):
        gv = _tone_gain(dsp, f, 1.0, 1.0)
        print(f"[*] {f:.0f}Hz gain={gv:.4f}")
        assert abs(gv - 1.0) < 0.02, f"クリーン時素通しでない: {f:.0f}Hz {gv:.4f}"
    print("[OK] clean時は素通し")


def main() -> int:
    try:
        test_blend_arg_convention()
        test_highband_matches_docstring()
        test_clean_passthrough_unchanged()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL NR BLEND MATH TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
