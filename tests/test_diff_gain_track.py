"""diff_gain deviation-tracking test (improvement doc diff_gain 4-2).

stereo_diff_gain=1.03 is calibrated at 100% deviation only; at normal
program deviation (30-60%) it costs 10-14dB separation. The pipeline now
tracks peak deviation and interpolates the measured table, keeping
exactly 1.03 at 100% (golden-stable) while using 1.00 at 30-60%.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "audio_ab"))

import numpy as np

from dsp import SdrDspPipeline
from regress import _wfm_tone_raw, _decode_stereo, _tone_pow, BLOCK_WFM, FS


def _run(dev_hz, gain=None):
    dsp = SdrDspPipeline(1152000, FS)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    dsp.filter_mode = "wide"
    dsp.set_stereo_nr(False)
    if gain is not None:
        dsp.stereo_diff_gain = float(gain)
    y = _decode_stereo(dsp, _wfm_tone_raw(1000.0, dev_hz=dev_hz, dur=4.0),
                       BLOCK_WFM)[-FS:]
    sep = float(10.0 * np.log10((_tone_pow(y[:, 0], 1000.0) + 1e-24)
                                / (_tone_pow(y[:, 1], 1000.0) + 1e-24)))
    return dsp, sep


def test_tracks_low_deviation():
    _, sep = _run(22500.0)
    print(f"[*] 30% dev: sep={sep:.1f}dB (was 35.1 fixed-gain)")
    assert sep > 40.0, f"tracking did not restore 30% separation ({sep:.1f})"
    print("[OK] 30% deviation tracked")


def test_holds_full_deviation():
    dsp, sep = _run(75000.0)
    print(f"[*] 100% dev: eff={dsp._diff_gain_eff():.4f} sep={sep:.1f}dB")
    assert abs(dsp._diff_gain_eff() - 1.03) < 0.005, "100% gain moved"
    assert sep > 50.0, f"100% separation regressed ({sep:.1f})"
    print("[OK] 100% deviation holds calibration")


def test_manual_gain_respected():
    dsp, _ = _run(22500.0, gain=1.06)
    eff = dsp._diff_gain_eff()
    print(f"[*] manual 1.06 @30%: eff={eff:.4f}")
    assert abs(eff - 1.06 * (1.000 / 1.03)) < 0.005, \
        "manual calibration overridden"
    print("[OK] manual gain scales relatively")


def main() -> int:
    try:
        test_tracks_low_deviation()
        test_holds_full_deviation()
        test_manual_gain_respected()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL DIFF-GAIN TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
