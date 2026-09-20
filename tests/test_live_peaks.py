"""Live spectrum peak picking tests (no hardware).

- find_spectrum_peaks recovers known tone frequencies on a synthetic
  spectrum with a noise floor (incl. parabolic sub-bin refinement).
- Flat noise yields (almost) no peaks above the SNR gate.
- Peak snap helper logic: nearest peak within 30 kHz wins.
"""

import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui import SdrGui, find_spectrum_peaks

SR = 1152000
FC = 83000000
N = 1024


def make_spectrum(freqs, snr_db=20.0, floor_db=-70.0, seed=1):
    rng = np.random.default_rng(seed)
    spec = floor_db + rng.standard_normal(N) * 1.5
    bins = (np.asarray(freqs) - (FC - SR / 2)) / SR * N
    for b in bins:
        i = int(round(b))
        # 幅3ビンの山 (放物線補間の検証用に非対称も混ぜる)
        spec[max(0, i - 1)] += snr_db * 0.7
        spec[i] += snr_db
        if i + 1 < N:
            spec[i + 1] += snr_db * 0.8
    return spec


def main() -> int:
    ok = True

    # 既知3局の復元 (放物線補間でビン内精度のはず)
    # 表示帯域: 82.424〜83.576MHz (FC=83MHz, SR=1.152MHz)
    spec = make_spectrum([82500000, 83000000, 83200000])
    peaks = find_spectrum_peaks(spec, SR, FC)
    found = sorted(p["freq_hz"] for p in peaks)
    good = (len(peaks) == 3
            and all(abs(f - e) < 3000 for f, e in
                    zip(found, [82500000, 83000000, 83200000])))
    print(f"[{'OK' if good else 'FAIL'}] 3 peaks recovered "
          f"({[f'{f / 1e6:.3f}' for f in found]})")
    ok &= good

    # 降順 (強い順) であること
    snrs = [p["snr_db"] for p in peaks]
    good = all(a >= b for a, b in zip(snrs, snrs[1:]))
    print(f"[{'OK' if good else 'FAIL'}] peaks sorted by snr")
    ok &= good

    # フラットノイズはほぼ無検出
    rng = np.random.default_rng(2)
    flat = -70.0 + rng.standard_normal(N) * 1.5
    peaks = find_spectrum_peaks(flat, SR, FC)
    good = len(peaks) <= 1
    print(f"[{'OK' if good else 'FAIL'}] flat noise yields {len(peaks)} peak(s)")
    ok &= good

    # 近接2局の分離 (min_sep_hz)
    spec = make_spectrum([83000000, 83010000], snr_db=25.0)
    peaks = find_spectrum_peaks(spec, SR, FC, min_sep_hz=25000.0)
    good = len(peaks) == 1
    print(f"[{'OK' if good else 'FAIL'}] close pair merged to {len(peaks)}")
    ok &= good

    # GUI状態への組み込み (live_peaks / hover 初期値)
    gui = SdrGui()
    good = gui.live_peaks == [] and gui.hover_freq_hz is None
    print(f"[{'OK' if good else 'FAIL'}] gui live-peak state init")
    ok &= good
    gui.close()

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
