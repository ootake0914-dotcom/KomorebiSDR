"""Inverse-sinc aperture compensation test (no hardware required).

Verifies dsp_filters.design_inverse_sinc and its WFM wiring:
- filter matches x/sin(x) within 0.02dB over 0-53kHz, DC gain exactly 1,
  symmetric (linear phase: stereo timing untouched)
- stereo separation does not regress with compensation ON (clean synth)
- mono transparency: ON/OFF difference bounded (small correction only)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp_filters import design_inverse_sinc
from dsp import SdrDspPipeline

RF = 1152000
BLOCK = 132096
NBLK = 20
N = (BLOCK // 2) * NBLK


def _resp(h, f, fs):
    n = np.arange(len(h))
    H = np.sum(h * np.exp(-2j * np.pi * f / fs * n))
    return 20.0 * np.log10(abs(H) + 1e-12)


def test_filter_shape():
    h = design_inverse_sinc(5, 288000.0)
    assert len(h) == 5
    # symmetric (linear phase)
    assert float(np.max(np.abs(h - h[::-1]))) == 0.0
    # DC gain exactly 1
    assert abs(float(np.sum(h)) - 1.0) < 1e-6
    worst = 0.0
    for f in np.linspace(0.0, 53000.0, 200):
        x = np.pi * f / 288000.0
        target = 20.0 * np.log10(x / np.sin(x)) if x > 1e-9 else 0.0
        worst = max(worst, abs(_resp(h, f, 288000.0) - target))
    print(f"[*] inv-sinc max err {worst:.4f} dB over 0-53kHz")
    assert worst < 0.02, worst
    print("[OK] inverse-sinc filter shape")


def make_raw(stereo=True, seed=7):
    n = (BLOCK // 2) * NBLK
    t = np.arange(n) / RF
    left = np.sin(2 * np.pi * 1000.0 * t)
    right = np.sin(2 * np.pi * 5000.0 * t)
    if stereo:
        mpx = (0.45 * (left + right)
               + 0.45 * (left - right) * np.sin(2 * np.pi * 38000.0 * t)
               + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    else:
        mpx = left + right
    mpx = mpx / (np.max(np.abs(mpx)) + 1e-9)
    iq = 0.6 * np.exp(1j * 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF)
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    return raw


def run_blocks(raw, inv_sinc, diff_gain=None):
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    dsp.inv_sinc_enabled = bool(inv_sinc)
    if diff_gain is not None:
        dsp.stereo_diff_gain = float(diff_gain)
    chunks = []
    for k in range(NBLK):
        audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        chunks.append(audio)
    return dsp, np.concatenate(chunks, axis=0)


def amp(x, freq, sr=48000):
    seg = x[len(x) // 3: len(x) // 3 + sr]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs > freq - 60) & (freqs < freq + 60)
    return float(np.max(spectrum[mask]) + 1e-12)


def test_separation_no_regress():
    # 各経路の校正済みペアで比較 (OFF+1.06 vs ON+1.03)。合成30kHz偏移。
    raw = make_raw(stereo=True)
    _, pcm_on = run_blocks(raw, True, 1.03)
    _, pcm_off = run_blocks(raw, False, 1.06)

    def worst(pcm):
        l, r = pcm[:, 0], pcm[:, 1]
        return max(20 * np.log10(amp(r, 1000.0) / amp(l, 1000.0)),
                   20 * np.log10(amp(l, 5000.0) / amp(r, 5000.0)))
    w_on, w_off = worst(pcm_on), worst(pcm_off)
    print(f"[*] separation OFF+1.06 {w_off:+.1f} dB -> ON+1.03 {w_on:+.1f} dB")
    assert w_on <= -18.0, "separation broken"
    assert w_on <= w_off + 1.0, "compensation regressed separation"
    print("[OK] separation preserved")


def test_mono_transparency():
    # 群遅延2spl (6.9us) の時間ズレは避けられないため、サンプル一致では
    # なく音色 (トーン別レベル・全帯域RMS) で透明性を検証する
    raw = make_raw(stereo=False)
    _, pcm_on = run_blocks(raw, True, 1.02)
    _, pcm_off = run_blocks(raw, False, 1.06)
    m_on = pcm_on[:, 0]
    m_off = pcm_off[:, 0]
    for f in (1000.0, 5000.0):
        d = 20.0 * np.log10(amp(m_on, f) / amp(m_off, f))
        print(f"[*] mono tone {f:.0f}Hz ON/OFF {d:+.3f} dB")
        assert abs(d) < 0.3, f"tonal shift at {f}Hz"
    rms_d = 20.0 * np.log10(
        (float(np.sqrt(np.mean(m_on ** 2))) + 1e-12)
        / (float(np.sqrt(np.mean(m_off ** 2))) + 1e-12))
    print(f"[*] mono RMS ON/OFF {rms_d:+.3f} dB")
    assert abs(rms_d) < 0.5
    print("[OK] mono transparency")


def main() -> int:
    try:
        test_filter_shape()
        test_separation_no_regress()
        test_mono_transparency()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL INV-SINC TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
