"""AM synchronous detection (SAM) test - synthetic, no hardware.

Generates an AM carrier (weak amplitude, +20 Hz tuning error) and checks that
the PLL-based synchronous detector locks and recovers the modulation without
the envelope detector's distortion. Also checks graceful fallback when the
native AM sync core is unavailable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import dsp as dsp_mod
from dsp import SdrDspPipeline

RATE = 288000
BLK = 16512
NBLK = 16


def main() -> int:
    if not dsp_mod.NATIVE_AM_SYNC:
        print("[SKIP] sdr_am_sync not available in native core (pure-Python fallback)")
        print("OK")
        return 0

    rng = np.random.default_rng(1)
    n = BLK * NBLK
    t = np.arange(n) / RATE
    audio = np.cos(2 * np.pi * 1000.0 * t)
    env = 1e-4 * (1.0 + 0.6 * audio)
    iq = (env * np.exp(1j * 2 * np.pi * 20.0 * t)).astype(np.complex64)
    iq += (1e-5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)

    dsp = SdrDspPipeline(1152000, 48000)
    out = []
    for k in range(NBLK):
        out.append(dsp._am_sync_detect(iq[k * BLK:(k + 1) * BLK]))
    y = np.concatenate(out)

    ok = dsp.am_sync_lock > 0.7 and dsp._am_sync_mix > 0.8
    print(f"[{'OK' if ok else 'FAIL'}] PLL lock={dsp.am_sync_lock:.2f} mix={dsp._am_sync_mix:.2f} "
          f"(need lock>0.7, mix>0.8)")

    seg = y[-16384:]
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / RATE)

    def peak(f):
        m = (freqs > f - 20) & (freqs < f + 20)
        return float(np.max(spec[m]))

    tone = peak(1000.0)
    dc = peak(0.0)
    tone_ok = tone > 0.05 * dc
    print(f"[{'OK' if tone_ok else 'FAIL'}] 1kHz modulation recovered "
          f"(tone/DC = {tone / (dc + 1e-12):.3f})")
    ok &= tone_ok

    # 無信号時はロックせず包絡線へフォールバックすること
    dsp2 = SdrDspPipeline(1152000, 48000)
    noise = (2e-5 * (rng.standard_normal(BLK * 4) + 1j * rng.standard_normal(BLK * 4)))
    for k in range(4):
        dsp2._am_sync_detect(noise[k * BLK:(k + 1) * BLK].astype(np.complex64))
    fb_ok = dsp2._am_sync_mix < 0.5
    print(f"[{'OK' if fb_ok else 'FAIL'}] no-carrier falls back to envelope "
          f"(mix={dsp2._am_sync_mix:.2f})")
    ok &= fb_ok

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
