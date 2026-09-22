"""Dynamic WFM IF morph regression: SNR-linked slew limiting must preserve targets."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline


def test_wfm_if_snr_estimator():
    n = 1024
    strong = np.full(n, -90.0)
    strong[512 - 75:512 + 76] = 5.0
    q = SdrDspPipeline._wfm_if_snr_db(strong, 1152000.0)
    assert q is not None and q > 40.0, f"strong in-band SNR unreadable: {q}"
    flat = np.full(n, -80.0)
    q0 = SdrDspPipeline._wfm_if_snr_db(flat, 1152000.0)
    assert q0 is not None and abs(q0) < 3.0, f"flat spectrum misread: {q0}"
    assert SdrDspPipeline._wfm_if_snr_db(np.zeros(10), 1152000.0) is None
    print("[OK] IF SNR estimator")


def test_if_morph_snr_slew():
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.cognitive_enabled = True
    dsp.target_if_bw_hz = 130000.0
    dsp.applied_if_bw_hz = 190000.0
    dsp._update_cognitive_morph(2.0, "WFM")
    assert dsp.applied_if_bw_hz == 187500.0, dsp.applied_if_bw_hz
    assert dsp._if_snr_db == 2.0
    dsp.applied_if_bw_hz = 130000.0
    dsp.target_if_bw_hz = 190000.0
    dsp._update_cognitive_morph(20.0, "WFM")
    assert dsp.applied_if_bw_hz == 138000.0, dsp.applied_if_bw_hz
    dsp.applied_if_bw_hz = 190000.0
    dsp.target_if_bw_hz = 130000.0
    dsp._update_cognitive_morph(2.0, "AM")
    assert abs(dsp.applied_if_bw_hz - 179200.0) < 1e-6, dsp.applied_if_bw_hz
    print("[OK] IF morph slew")


if __name__ == "__main__":
    test_wfm_if_snr_estimator()
    test_if_morph_snr_slew()
    print("\nALL DYNAMIC IF TESTS PASSED!")
