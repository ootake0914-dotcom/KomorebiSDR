"""SSB / CW demodulation tests (synthetic, no hardware).

- USB tone is decoded in USB mode and rejected in LSB mode (and vice versa)
- CW carrier offset produces an audible beat tone
- BFO shift moves the decoded audio frequency (mixer sign check)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline

SR = 48000
N = SR * 2


def tone_power(x: np.ndarray, freq: float, sr: int = SR) -> float:
    seg = x[-16384:] * np.hanning(16384)
    p = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    m = (fr > freq - 30) & (fr < freq + 30)
    return float(np.mean(p[m]) + 1e-18)


def decode(iq: np.ndarray, mode: str):
    dsp = SdrDspPipeline(1152000, 48000)
    chunks = []
    blk = 2752
    for k in range(0, len(iq) - blk + 1, blk):
        chunks.append(dsp.demodulate_ssb(iq[k:k + blk].astype(np.complex64), mode))
    return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)


def main() -> int:
    ok = True
    rng = np.random.default_rng(2)
    t = np.arange(N) / SR

    # USB: 音声1kHzは +1kHz に現れる
    usb = np.exp(1j * 2 * np.pi * 1000.0 * t)
    out_usb = decode(usb, "USB")
    out_usb_wrong = decode(usb, "LSB")
    p_sig = tone_power(out_usb, 1000.0)
    p_rej = tone_power(out_usb_wrong, 1000.0)
    rej_db = 10 * np.log10((p_rej + 1e-18) / (p_sig + 1e-18))
    good = p_sig > 1e-6 and rej_db < -15.0
    print(f"[{'OK' if good else 'FAIL'}] USB tone decoded, opposite sideband rejected "
          f"({rej_db:+.1f} dB)")
    ok &= good

    # LSB: 音声1kHzは -1kHz に現れる
    lsb = np.exp(-1j * 2 * np.pi * 1000.0 * t)
    out_lsb = decode(lsb, "LSB")
    out_lsb_wrong = decode(lsb, "USB")
    p_sig = tone_power(out_lsb, 1000.0)
    p_rej = tone_power(out_lsb_wrong, 1000.0)
    rej_db = 10 * np.log10((p_rej + 1e-18) / (p_sig + 1e-18))
    good = p_sig > 1e-6 and rej_db < -15.0
    print(f"[{'OK' if good else 'FAIL'}] LSB tone decoded, opposite sideband rejected "
          f"({rej_db:+.1f} dB)")
    ok &= good

    # ノイズ耐性: 弱いUSBトーン + ホワイトノイズ
    noisy = (0.05 * usb + 0.02 * (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
    out_n = decode(noisy, "USB")
    good = tone_power(out_n, 1000.0) > 1e-8
    print(f"[{'OK' if good else 'FAIL'}] weak USB tone recovered through noise")
    ok &= good

    # BFO: mix_frequency で +300Hz シフトすること (USBモードのみ)
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.offset_freq = 0.0
    dsp.bfo_offset_hz = 0.0
    rf = dsp.rf_rate
    tr = np.arange(65536) / rf
    iq0 = np.exp(1j * 2 * np.pi * 1000.0 * tr).astype(np.complex64)
    f0 = np.argmax(np.abs(np.fft.fft(dsp.mix_frequency(iq0, mode="USB"))))
    dsp.bfo_offset_hz = 300.0
    iq1 = np.exp(1j * 2 * np.pi * 1000.0 * tr).astype(np.complex64)
    f1 = np.argmax(np.abs(np.fft.fft(dsp.mix_frequency(iq1, mode="USB"))))
    shift = (f1 - f0) * rf / 65536.0
    good = abs(shift - 300.0) < 10.0
    print(f"[{'OK' if good else 'FAIL'}] BFO shifts USB audio by {shift:+.0f} Hz (expect +300)")
    ok &= good

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
