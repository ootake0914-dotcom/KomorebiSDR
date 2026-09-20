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

    # BFO: 音声ドメインでUSB/LSBとも正方向にピッチ移動し、帯域外無音化しない
    # (旧RF方式はLSB逆転＋CW無音化のため廃止)
    for m, tone in (("USB", np.exp(1j * 2 * np.pi * 1000.0 * t)),
                    ("LSB", np.exp(-1j * 2 * np.pi * 1000.0 * t))):
        dsp = SdrDspPipeline(1152000, 48000)
        dsp.bfo_offset_hz = 0.0
        ref = decode(tone, m)
        dsp2 = SdrDspPipeline(1152000, 48000)
        dsp2.bfo_offset_hz = 300.0
        chunks = []
        blk = 2752
        for k in range(0, len(tone) - blk + 1, blk):
            chunks.append(dsp2.demodulate_ssb(tone[k:k + blk].astype(np.complex64), m))
        shifted = np.concatenate(chunks)
        p_ref = tone_power(ref, 1000.0)
        p_up = tone_power(shifted, 1300.0)
        p_dn = tone_power(shifted, 700.0)
        good = p_up > p_ref * 0.25 and p_up > p_dn * 4.0
        print(f"[{'OK' if good else 'FAIL'}] BFO +300 moves {m} audio 1000->1300 Hz "
              f"(up={p_up:.2e}, ref={p_ref:.2e}, down={p_dn:.2e})")
        ok &= good

    # CW: BFO+1500でも帯域外無音化しない (旧RF方式は650±350Hz外で消音)
    dsp3 = SdrDspPipeline(1152000, 48000)
    dsp3.bfo_offset_hz = 1500.0
    cw = np.exp(1j * 2 * np.pi * 850.0 * t)
    chunks = []
    blk = 2752
    for k in range(0, len(cw) - blk + 1, blk):
        chunks.append(dsp3.demodulate_ssb(cw[k:k + blk].astype(np.complex64), "CW"))
    out_cw = np.concatenate(chunks)
    p_cw = tone_power(out_cw, 2350.0)
    good = p_cw > 1e-6
    print(f"[{'OK' if good else 'FAIL'}] CW survives BFO+1500 (2350Hz power={p_cw:.2e})")
    ok &= good

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
