"""PPM auto-calibration tests (no hardware required).

- Calibrator math: median over stations, outlier rejection, gating
- Sign convention end-to-end: injected carrier offset -> AFC -> PPM sign
- Driver SW fallback frequency compensation math
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ppm_cal import PpmCalibrator


def test_median_with_outlier():
    print("===== test_median_with_outlier =====")
    cal = PpmCalibrator()
    # 真値 +40ppm のドングル。3局は正常、1局は局側が+500Hzズレた外れ値
    truth = 40.0
    for f in (83200000, 88100000, 94600000):
        cal.collect(f, -(truth * f / 1e6))
    cal.collect(80000000, -((truth * 80000000 / 1e6) + 500.0))
    ppm, n, confident = cal.estimate()
    print(f"[*] estimated {ppm:+.2f} ppm (n={n}, confident={confident})")
    assert confident and n == 4, "should be confident with 4 stations"
    assert abs(ppm - truth) < 1.0, f"outlier rejection failed: {ppm}"
    print("[OK] median with outlier")


def test_gating():
    print("===== test_gating =====")
    cal = PpmCalibrator()
    ppm, n, confident = cal.estimate()
    assert (ppm, confident) == (None, False), "empty must not be confident"
    cal.collect(83200000, -100.0)
    cal.collect(83200000, -110.0)  # 同一局は上書きで1票のまま
    ppm, n, confident = cal.estimate()
    assert n == 1 and not confident, f"need 3 stations, got n={n}"
    cal.collect(83300000, -100.0)  # 近接周波数は分散不足
    cal.collect(83400000, -100.0)
    ppm, n, confident = cal.estimate()
    assert not confident, "narrow spread must not be confident"
    print("[OK] gating (min samples, spread)")


def test_absurd_rejected():
    print("===== test_absurd_rejected =====")
    cal = PpmCalibrator()
    for f in (76000000, 85000000, 95000000):
        cal.collect(f, -(300.0 * f / 1e6))  # 300ppm相当の異常値
    ppm, n, confident = cal.estimate()
    assert not confident, "absurd PPM must not be confident"
    print("[OK] absurd value rejected")


def test_sign_convention_end_to_end():
    print("===== test_sign_convention_end_to_end =====")
    from dsp import SdrDspPipeline
    RF = 1152000
    N = 66048
    t = np.arange(N) / RF
    # +2000Hz の搬送波誤差を100MHz受信として模擬 -> +20ppm のはず
    err_hz = 2000.0
    freq_hz = 100000000
    p = SdrDspPipeline(RF, 48000)
    p.set_offset_freq(0.0)
    p.afc_enabled = True
    p.cognitive_enabled = False
    iq = np.exp(1j * 2 * np.pi * err_hz * t).astype(np.complex64)
    raw = np.empty(2 * N, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 40 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 40 + 127.5), 0, 255).astype(np.uint8)
    for _ in range(120):  # AFC収束 (α=0.05 -> 120ブロックで99.8%)
        p.process(raw, mode="WFM")
    cal = PpmCalibrator()
    cal.collect(freq_hz, p.afc_offset_hz)
    # 単局では分散不足のため直接換算で符号を確認
    got = PpmCalibrator.err_to_ppm(-p.afc_offset_hz, freq_hz)
    want = err_hz * 1e6 / freq_hz
    print(f"[*] afc={p.afc_offset_hz:+.0f}Hz -> {got:+.1f}ppm (want {want:+.1f}ppm)")
    assert got > 0 and abs(got - want) / want < 0.15, "sign/scale convention broken"
    print("[OK] sign convention end-to-end")


def test_driver_sw_fallback_math():
    print("===== test_driver_sw_fallback_math =====")
    from rtlsdr_driver import RtlSdrDriver
    d = RtlSdrDriver()
    assert d.get_ppm_correction() == 0
    assert d.compensated_freq(100000000) == 100000000
    d.set_ppm_correction(50)  # 未open: 保持のみ、SWフォールバック
    assert d.get_ppm_correction() == 50
    assert d.compensated_freq(100000000) == 99995000
    d.set_ppm_correction(-30)
    assert d.compensated_freq(100000000) == 100003000
    d.set_ppm_correction(0)
    assert d.compensated_freq(100000000) == 100000000
    print("[OK] driver SW fallback math")


def test_pick_ppm_stations():
    print("===== test_pick_ppm_stations =====")
    from ppm_cal import pick_ppm_stations
    stations = [
        {"freq_hz": 1197000, "freq_mhz": 1.197, "name": "x", "snr_db": 30.0},  # 短波は除外
        {"freq_hz": 80000000, "freq_mhz": 80.0, "name": "Unknown FM Station", "snr_db": 9.0},
        {"freq_hz": 81300000, "freq_mhz": 81.3, "name": "J-WAVE", "snr_db": 25.0},
        {"freq_hz": 82500000, "freq_mhz": 82.5, "name": "NHK", "snr_db": 30.0},
        {"freq_hz": 83200000, "freq_mhz": 83.2, "name": "Unknown FM Station", "snr_db": 28.0},
        {"freq_hz": 90000000, "freq_mhz": 90.0, "name": "weak", "snr_db": 5.0},  # SNR不足で除外
        {"freq_hz": "bad", "name": "bad", "snr_db": 30.0},  # 破損エントリ除外
    ]
    picked = pick_ppm_stations(stations, n=5)
    freqs = [s["freq_hz"] for s in picked]
    print(f"[*] picked {freqs}")
    assert 1197000 not in freqs, "HF must be excluded"
    assert 90000000 not in freqs, "weak station must be excluded"
    assert all(isinstance(f, int) for f in freqs)
    # 既知局優先: 先頭2件は既知局のはず
    assert picked[0]["freq_hz"] == 82500000 and picked[1]["freq_hz"] == 81300000
    assert pick_ppm_stations([], n=5) == []
    assert pick_ppm_stations(None, n=5) == []
    print("[OK] pick_ppm_stations")


def main() -> int:
    try:
        test_median_with_outlier()
        test_gating()
        test_absurd_rejected()
        test_sign_convention_end_to_end()
        test_driver_sw_fallback_math()
        test_pick_ppm_stations()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL PPM CAL TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
