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


def test_wfm_aci_estimator():
    n = 1024
    spec = np.full(n, -90.0)
    c = n // 2
    spec[c - 75:c + 76] = 5.0        # 自局 (±85kHz)
    spec[c + 100:c + 160] = 15.0     # 右隣接が自局より10dB強い
    r = SdrDspPipeline._wfm_aci_db(spec, 1152000.0)
    assert r is not None
    du_l, du_r, ab_l, ab_r = r
    assert du_r < -5.0, f"right ACI not detected: {du_r:.1f} dB"
    assert du_l > 20.0, f"left side should be clean: {du_l:.1f} dB"
    assert ab_r > ab_l, "adjacent-above-floor should flag the right side"
    flat = np.full(n, -80.0)
    r0 = SdrDspPipeline._wfm_aci_db(flat, 1152000.0)
    assert r0 is not None and abs(r0[0]) < 3.0 and abs(r0[1]) < 3.0
    assert SdrDspPipeline._wfm_aci_db(np.zeros(10), 1152000.0) is None
    print("[OK] ACI estimator")


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
    test_wfm_aci_estimator()
    test_if_morph_snr_slew()
    print("\nALL DYNAMIC IF TESTS PASSED!")
