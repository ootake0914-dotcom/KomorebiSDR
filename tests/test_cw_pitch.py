"""CW自動ピッチ (第2章#1): 650Hz収束・振動なし・同調時透明・選局リセット。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

FS = 48000
RF = 1152000
BLK = 132096


def _cw_iq(freq_hz, snr_db=20.0, secs=3.0, seed=7):
    n = int(RF * secs)
    t = np.arange(n) / RF
    iq = 0.5 * np.exp(2j * np.pi * freq_hz * t)
    rng = np.random.default_rng(seed)
    p = 0.25 / (10 ** (snr_db / 10.0))
    iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _peak(audio):
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    m = (f >= 400.0) & (f <= 1000.0)
    band, fr = spec[m], f[m]
    k = int(np.argmax(band))
    if 0 < k < len(band) - 1:
        a, b, c = float(band[k - 1]), float(band[k]), float(band[k + 1])
        d = a - 2.0 * b + c
        s = 0.5 * (a - c) / d if abs(d) > 1e-18 else 0.0
        s = min(max(s, -1.0), 1.0)
    else:
        s = 0.0
    return float(fr[k] + s * (fr[1] - fr[0]))


def _run(raw, flag=True):
    dsp = SdrDspPipeline(RF, FS)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    dsp.cw_auto_pitch = flag
    peaks, bfos, outs = [], [], []
    for k in range(len(raw) // BLK):
        audio, _ = dsp.process(raw[k * BLK:(k + 1) * BLK], mode="CW")
        a = np.asarray(audio, dtype=np.float64).reshape(-1)
        outs.append(a)
        peaks.append(_peak(a))
        bfos.append(float(dsp.bfo_offset_hz))
    return np.concatenate(outs), peaks, bfos


def test_converges():
    for delta in (80.0, -80.0):
        raw = _cw_iq(650.0 + delta)
        _, peaks, bfos = _run(raw, True)
        tail = peaks[-5:]
        print(f"d={delta:+.0f}: peaks_end={np.mean(tail):.1f} "
              f"std={np.std(tail):.1f} bfo_end={bfos[-1]:.0f}")
        assert abs(np.mean(tail) - 650.0) < 20.0, (delta, tail)
        assert np.std(tail) < 15.0, (delta, tail)
        assert abs(bfos[-1]) <= 800.0, bfos[-1]
    print("converge OK")


def test_tuned_transparent():
    raw = _cw_iq(650.0)
    y_off, _, _ = _run(raw, False)
    y_on, _, _ = _run(raw, True)
    assert np.array_equal(np.asarray(y_off), np.asarray(y_on))
    print("tuned transparent OK")


def test_tune_reset():
    raw = _cw_iq(730.0)
    dsp = SdrDspPipeline(RF, FS)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    for k in range(10):
        dsp.process(raw[k * BLK:(k + 1) * BLK], mode="CW")
    assert abs(float(dsp.bfo_offset_hz)) > 1.0, dsp.bfo_offset_hz
    dsp.set_offset_freq(1000.0)
    assert float(dsp.bfo_offset_hz) == 0.0, dsp.bfo_offset_hz
    # 手動BFOは維持
    dsp.cw_auto_pitch = False
    dsp.bfo_offset_hz = 123.0
    dsp.set_offset_freq(2000.0)
    assert float(dsp.bfo_offset_hz) == 123.0, dsp.bfo_offset_hz
    print("tune reset OK")


if __name__ == "__main__":
    test_converges()
    test_tuned_transparent()
    test_tune_reset()
    print("ALL CW-PITCH TESTS PASSED!")
