"""Multipath detector + protection tests (FM doc section 2d).

Before the fix, the envelope-variance detector never fired on real-world
echoes (needs 3km+ path difference) and there was no test watching
multipath_amount itself. Covers:
- static echo (10us): trim-excursion path fires
- drifting echo (6us/3us): envelope peak-hold path fires
- clean + weak-field program: stays silent
- strong echo (30us): canceller bypass restores separation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

RF = 1152000.0
BLK = 132096
DUR_S = 4.0


def _synth_echo(freq_hz, tau_us, alpha, snr_db=40.0, seed=99,
                drift_hz=0.0):
    """1kHz tone L-only @30% deviation with optional delayed echo at RF."""
    n = int(DUR_S * RF)
    t = np.arange(n) / RF
    l = 0.95 * np.sin(2 * np.pi * freq_hz * t)
    r = np.zeros(n)
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 22500.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    if tau_us > 0:
        d = max(1, int(tau_us * 1e-6 * RF))
        echo = np.zeros_like(iq)
        echo[d:] = (iq[:-d] * alpha
                    * np.exp(1j * 2 * np.pi * drift_hz * t[d:]))
        iq = iq + echo
    rng = np.random.default_rng(seed)
    p = 0.36 / 10 ** (float(snr_db) / 10.0)
    iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _synth_hf_prog_echo(tau_us, alpha, seed=3):
    """高域のみ番組 (3.5kHz以上) ＋静止エコー。検証指摘§2-2の回帰用。
    NRが潰れても (nr_gain→0) トリムは動かねばならない。"""
    n = int(DUR_S * RF)
    t = np.arange(n) / RF
    prog_hf = (0.25 * np.sin(2 * np.pi * 3500.0 * t)
               + 0.25 * np.sin(2 * np.pi * 4000.0 * t)
               + 0.20 * np.sin(2 * np.pi * 6000.0 * t)
               + 0.15 * np.sin(2 * np.pi * 8000.0 * t))
    body = 0.2 * (0.003 * np.sin(2 * np.pi * 500.0 * t)
                  + 0.003 * np.sin(2 * np.pi * 1500.0 * t))
    m = prog_hf * 0.5 + body
    s = prog_hf
    l, r = (m + s) * 0.5, (m - s) * 0.5
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    if tau_us > 0:
        d = max(1, int(tau_us * 1e-6 * RF))
        echo = np.zeros_like(iq)
        echo[d:] = iq[:-d] * alpha
        iq = iq + echo
    rng = np.random.default_rng(seed)
    p = 0.36 / 10 ** 40.0
    iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _decode(raw):
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    outs = []
    for k in range(len(raw) // BLK):
        a, _ = dsp.process(raw[k * BLK:(k + 1) * BLK], "WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
    y = np.concatenate(outs, axis=0)
    return y[len(y) // 2:], dsp


def _separation(y, freq_hz=1000.0, fs=48000.0):
    L = y[:, 0].astype(np.float64)
    R = y[:, 1].astype(np.float64)
    n = len(L)
    WL = np.abs(np.fft.rfft(L * np.hanning(n))) ** 2
    WR = np.abs(np.fft.rfft(R * np.hanning(n))) ** 2
    i = int(round(freq_hz * n / fs))
    fund = np.sum(WL[max(0, i - 2):i + 3])
    leak = np.sum(WR[max(0, i - 2):i + 3])
    return 10 * np.log10((fund + 1e-24) / (leak + 1e-24))


def test_detector_fires_static_echo():
    _, dsp = _decode(_synth_echo(1000.0, 10.0, 0.6))
    print(f"[*] static 10us: amount={dsp.multipath_amount:.3f}")
    assert dsp.multipath_amount > 0.3, "static echo missed by detector"
    print("[OK] static echo fires detector")


def test_detector_fires_drift_echo():
    _, dsp = _decode(_synth_echo(1000.0, 6.0, 0.6, drift_hz=0.5))
    print(f"[*] drift 6us: amount={dsp.multipath_amount:.3f}")
    assert dsp.multipath_amount > 0.3, "drifting echo missed by detector"
    _, dsp3 = _decode(_synth_echo(1000.0, 3.0, 0.6, drift_hz=0.5))
    print(f"[*] drift 3us: amount={dsp3.multipath_amount:.3f}")
    assert dsp3.multipath_amount > 0.05, "short drifting echo missed"
    print("[OK] drifting echo fires detector")


def test_detector_silent_clean():
    _, dsp = _decode(_synth_echo(1000.0, 0.0, 0.0))
    print(f"[*] clean: amount={dsp.multipath_amount:.3f}")
    assert dsp.multipath_amount < 0.05, "false fire on clean signal"
    _, dspw = _decode(_synth_echo(1000.0, 0.0, 0.0, seed=7, snr_db=12.0))
    print(f"[*] weak-field: amount={dspw.multipath_amount:.3f}")
    assert dspw.multipath_amount < 0.05, "false fire on weak-field program"
    print("[OK] detector silent on clean/weak-field")


def test_canceller_bypass_restores():
    y, dsp = _decode(_synth_echo(1000.0, 30.0, 0.6))
    sep = _separation(y)
    print(f"[*] 30us echo: amount={dsp.multipath_amount:.3f} sep={sep:.1f}dB")
    assert dsp.multipath_amount > 0.3
    assert sep > 3.0, f"canceller bypass did not restore stereo ({sep:.1f}dB)"
    print("[OK] strong-echo stereo partly restored")


def test_trim_runs_on_hf_program_echo():
    """高域のみ番組＋静止エコーでも検出器が発動すること (検証指摘§2-2)。

    旧トリムゲート (nr_gain>0.7) は高域番組で0に落ち、静止10usが
    あっても amount=0 のままだった。ゲート撤廃後は発動する。"""
    _, dsp = _decode(_synth_hf_prog_echo(10.0, 0.6))
    print(f"[*] HFprog+echo10: nr_gain={dsp.stereo_nr_gain:.3f} "
          f"trim={dsp.stereo_phase_offset:+.4f} "
          f"amount={dsp.multipath_amount:.3f}")
    assert dsp.multipath_amount > 0.3, "HF-program echo missed (trim frozen?)"
    print("[OK] HF-program echo fires detector")


def main() -> int:
    try:
        test_detector_fires_static_echo()
        test_detector_fires_drift_echo()
        test_detector_silent_clean()
        test_canceller_bypass_restores()
        test_trim_runs_on_hf_program_echo()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL MULTIPATH TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
