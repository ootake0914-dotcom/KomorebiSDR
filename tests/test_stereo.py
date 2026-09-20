"""Synthetic FM stereo test (no hardware required).

Generates a BS.450-compliant stereo MPX (L=1kHz, R=5kHz, 9% pilot;
pilot and 38kHz subcarrier both sine-phase with aligned zero crossings),
FM-modulates it, runs it through the DSP pipeline and checks that the
decoded channels are correctly separated (crosstalk <= -18 dB), that a
mono signal stays mono and that the stereo noise reduction cuts the
(L-R) hiss on noisy signals while staying inactive on clean ones.
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


def make_raw(stereo: bool, snr_db: float = None, seed: int = 7) -> np.ndarray:
    t = np.arange(N) / RF
    left = np.sin(2 * np.pi * 1000.0 * t)
    right = np.sin(2 * np.pi * 5000.0 * t)
    if stereo:
        # BS.450: 副搬送波・パイロットともsin系 (cos系は非標準で分離度0dBになる)
        mpx = (0.45 * (left + right)
               + 0.45 * (left - right) * np.sin(2 * np.pi * 38000.0 * t)
               + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    else:
        mpx = left + right
    mpx = mpx / (np.max(np.abs(mpx)) + 1e-9)
    phase = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * phase)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p_noise = 0.36 / (10 ** (snr_db / 10.0))
        iq = iq + np.sqrt(p_noise / 2.0) * (
            rng.standard_normal(N) + 1j * rng.standard_normal(N))
    raw = np.empty(2 * N, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    return raw


def decode(raw: np.ndarray, nr: bool = True):
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.set_stereo_nr(nr)
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


def band_rms(x: np.ndarray, lo: float, hi: float, sr: int = 48000) -> float:
    # 収束後の定常状態を評価 (NRは約2秒で定常に達する)
    seg = x[-2 * sr:]
    seg = seg - seg.mean()
    window = np.hanning(len(seg))
    power = np.abs(np.fft.rfft(seg * window)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs > lo) & (freqs < hi)
    return float(np.sqrt(np.mean(power[mask]) + 1e-18))


def main() -> int:
    ok = True
    dsp, pcm = decode(make_raw(stereo=True))
    left, right = pcm[:, 0], pcm[:, 1]
    crosstalk_lr = 20 * np.log10(amp(right, 1000.0) / amp(left, 1000.0))
    crosstalk_rl = 20 * np.log10(amp(left, 5000.0) / amp(right, 5000.0))
    worst = max(crosstalk_lr, crosstalk_rl)
    print(f"is_stereo={dsp.is_stereo} blend={dsp.stereo_blend:.2f} "
          f"crosstalk L->R {crosstalk_lr:+.1f} dB, R->L {crosstalk_rl:+.1f} dB")

    sep_ok = dsp.is_stereo and worst <= -18.0
    print(f"[{'OK' if sep_ok else 'FAIL'}] stereo separation (need <= -18 dB)")
    nr_idle = dsp.stereo_nr_gain > 0.95
    print(f"[{'OK' if nr_idle else 'FAIL'}] clean signal keeps full stereo "
          f"(NR gain={dsp.stereo_nr_gain:.2f}, cut={dsp.stereo_cut_hz:.0f}Hz)")
    ok &= sep_ok and nr_idle

    noisy = make_raw(stereo=True, snr_db=12.0)
    dsp_nr, pcm_nr = decode(noisy, nr=True)
    _, pcm_off = decode(noisy, nr=False)
    hiss_nr = band_rms(pcm_nr[:, 0] - pcm_nr[:, 1], 10000.0, 14000.0)
    hiss_off = band_rms(pcm_off[:, 0] - pcm_off[:, 1], 10000.0, 14000.0)
    reduction = 20 * np.log10((hiss_nr + 1e-18) / (hiss_off + 1e-18))
    nr_ok = dsp_nr.stereo_wiener_gain < 0.8 and reduction <= -8.0
    print(f"[{'OK' if nr_ok else 'FAIL'}] noisy stereo hiss cut "
          f"(wiener_gain={dsp_nr.stereo_wiener_gain:.2f}, cut={dsp_nr.stereo_cut_hz:.0f}Hz, "
          f"(L-R) 10-14kHz {reduction:+.1f} dB)")
    ok &= nr_ok

    dsp_mono, _ = decode(make_raw(stereo=False))
    mono_ok = (not dsp_mono.is_stereo) and dsp_mono.stereo_blend < 0.1
    print(f"[{'OK' if mono_ok else 'FAIL'}] mono station stays mono (blend={dsp_mono.stereo_blend:.2f})")

    ok &= mono_ok
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
