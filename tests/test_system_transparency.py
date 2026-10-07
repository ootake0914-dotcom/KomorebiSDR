"""Whole-system transparency test (no hardware required).

Contract: in clean conditions (high C/N, no multipath, no interference),
the shipped cognitive path must be indistinguishable from the same path
with all condition-dependent subsystems disabled (NR, multipath, ACI,
mono NR, BSS). Audited subsystems:
  stereo NR (incl. STFT Wiener + diff LPF), mono NR, BSS, multipath
  narrowing, ACI guard, SIC.
Calibration stages (inv-sinc, diff-gain deviation tracking, deemphasis,
trim servo, DC servo) are always-on by design and excluded by keeping
them on in both runs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

RF = 1152000.0
FS = 48000
BLK = 132096


def _to_raw(iq):
    n = len(iq)
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _fm_raw(l, r):
    n = len(l)
    t = np.arange(n) / RF
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    return _to_raw(0.6 * np.exp(1j * ph).astype(np.complex64))


def _tonal_music(dur=3.0, seed=3):
    n = int(dur * RF)
    t = np.arange(n) / RF
    rng = np.random.default_rng(seed)
    lf = [220, 440, 660, 1320, 2640, 5280, 7920, 10560, 13200]
    rfq = [330, 550, 880, 1760, 3520, 6160, 8800, 11880, 14080]
    l = sum((0.12 + 0.05 * rng.random()) *
            np.sin(2 * np.pi * f * t + rng.random() * 6.28) for f in lf)
    r = sum((0.12 + 0.05 * rng.random()) *
            np.sin(2 * np.pi * f * t + rng.random() * 6.28) for f in rfq)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
    return _fm_raw(l * am * 0.66, r * am * 0.66)


def _independent_noise(dur=3.0, seed=21):
    n = int(dur * RF)
    fr = np.fft.rfftfreq(n, 1.0 / RF)
    outs = []
    for sd in (seed, seed + 1):
        r2 = np.random.default_rng(sd)
        X = np.fft.rfft(r2.standard_normal(n))
        X[(np.abs(fr) < 300.0) | (np.abs(fr) > 15000.0)] = 0
        x = np.fft.irfft(X, n=n)
        outs.append(x / (np.max(np.abs(x)) + 1e-9) * 0.7)
    return _fm_raw(outs[0], outs[1])


def _make(cognitive=True, adaptive_off=False):
    d = SdrDspPipeline(1152000, FS)
    d.set_offset_freq(0.0)
    d.afc_enabled = False
    d.slow_agc_enabled = False
    d.filter_mode = "wide"
    if cognitive:
        d.cognitive_enabled = True
        d.target_cutoff_hz = 15000.0
        d.applied_cutoff_hz = 15000.0
        d.target_if_bw_hz = 145000.0
        d.applied_if_bw_hz = 145000.0
        d.target_hf_gain = 1.0
        d.hf_gain_applied = 1.0
    if adaptive_off:
        d.multipath_enabled = False
        d.aci_depth = 0.0
        d.set_stereo_nr(False)
        d.mono_nr.enabled = False
        d.bss_separator.enabled = False
    return d


def _decode(d, raw):
    outs = []
    for k in range(len(raw) // BLK):
        a, _ = d.process(raw[k * BLK:(k + 1) * BLK], "WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
    return np.concatenate(outs, axis=0)


def _band(x, lo, hi):
    n = len(x)
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    W = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    return 10 * np.log10(float(W[(fr >= lo) & (fr <= hi)].sum()) + 1e-24)


def _check_program(raw, label, cog=True):
    d_def = _make(cognitive=cog)
    d_ref = _make(cognitive=cog, adaptive_off=True)
    y_def = _decode(d_def, raw)
    y_ref = _decode(d_ref, raw)
    n = min(len(y_def), len(y_ref))
    start = FS  # skip settling
    worst = 0.0
    for name, fn in (("Mid", lambda y: (y[:, 0].astype(np.float64)
                                        + y[:, 1]) * 0.5),
                     ("Side", lambda y: (y[:, 0].astype(np.float64)
                                         - y[:, 1]) * 0.5)):
        a = fn(y_def[start:n])
        b = fn(y_ref[start:n])
        for lo, hi in ((300, 3000), (3000, 8000), (8000, 15000)):
            dv = _band(a, lo, hi) - _band(b, lo, hi)
            worst = max(worst, abs(dv))
            print(f"[*] {label} {name} {lo}-{hi}: d={dv:+.2f} dB")
    print(f"[*] {label}: worst |d|={worst:.2f} dB")
    assert worst < 0.25, f"{label}: adaptive systems altered output ({worst:.2f}dB)"
    return d_def


def test_clean_tonal_music_transparent():
    d = _check_program(_tonal_music(), "tonal music")
    assert d.multipath_gain > 0.999
    assert d.aci_gain > 0.999
    assert d.stereo_nr_gain > 0.999
    assert d._nr_cut_eff >= 14900.0
    assert not list(d.sic_detected_spurious)
    assert d.is_stereo
    print("[OK] tonal music system-transparent")


def test_clean_independent_noise_transparent():
    _check_program(_independent_noise(), "independent L/R noise")
    print("[OK] independent noise system-transparent")


def test_non_cognitive_path_transparent():
    _check_program(_tonal_music(seed=7), "tonal music (non-cog)",
                   cog=False)
    print("[OK] non-cognitive path system-transparent")


def main() -> int:
    try:
        test_clean_tonal_music_transparent()
        test_clean_independent_noise_transparent()
        test_non_cognitive_path_transparent()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL SYSTEM TRANSPARENCY TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
