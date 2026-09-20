"""Synthetic FM stereo separation test (no hardware required).

Generates a standard stereo MPX (L=1kHz, R=5kHz, 9% pilot), FM-modulates it,
runs it through the DSP pipeline and checks that the decoded channels are
correctly separated (crosstalk <= -18 dB) and that a mono signal stays mono.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline

RF = 1152000
BLOCK = 132096
NBLK = 20
N = (BLOCK // 2) * NBLK


def make_raw(stereo: bool) -> np.ndarray:
    t = np.arange(N) / RF
    left = np.sin(2 * np.pi * 1000.0 * t)
    right = np.sin(2 * np.pi * 5000.0 * t)
    if stereo:
        mpx = (0.45 * (left + right)
               + 0.45 * (left - right) * np.cos(2 * np.pi * 38000.0 * t)
               + 0.09 * np.cos(2 * np.pi * 19000.0 * t))
    else:
        mpx = left + right
    mpx = mpx / (np.max(np.abs(mpx)) + 1e-9)
    phase = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * phase)
    raw = np.empty(2 * N, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    return raw


def decode(raw: np.ndarray):
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    chunks = []
    for k in range(NBLK):
        audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        chunks.append(audio)
    return dsp, np.concatenate(chunks, axis=0)


def amp(x: np.ndarray, freq: float, sr: int = 48000) -> float:
    seg = x[len(x) // 3: len(x) // 3 + sr]
    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs > freq - 60) & (freqs < freq + 60)
    return float(np.max(spectrum[mask]) + 1e-12)


def main() -> int:
    dsp, pcm = decode(make_raw(stereo=True))
    left, right = pcm[:, 0], pcm[:, 1]
    crosstalk_lr = 20 * np.log10(amp(right, 1000.0) / amp(left, 1000.0))
    crosstalk_rl = 20 * np.log10(amp(left, 5000.0) / amp(right, 5000.0))
    worst = max(crosstalk_lr, crosstalk_rl)
    print(f"is_stereo={dsp.is_stereo} blend={dsp.stereo_blend:.2f} "
          f"crosstalk L->R {crosstalk_lr:+.1f} dB, R->L {crosstalk_rl:+.1f} dB")

    ok = dsp.is_stereo and worst <= -18.0
    print(f"[{'OK' if ok else 'FAIL'}] stereo separation (need <= -18 dB)")

    dsp_mono, _ = decode(make_raw(stereo=False))
    mono_ok = (not dsp_mono.is_stereo) and dsp_mono.stereo_blend < 0.1
    print(f"[{'OK' if mono_ok else 'FAIL'}] mono station stays mono (blend={dsp_mono.stereo_blend:.2f})")

    ok &= mono_ok
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
