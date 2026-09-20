"""Shortwave band definitions / scan planning tests (no hardware required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (SHORTWAVE_BANDS, SW_MAX_HZ, shortwave_band_name,
                    shortwave_scan_centers)

RATE = 1152000


def main() -> int:
    ok = True

    # 1) バンド定義の妥当性
    prev_hi = 0
    for name, lo_khz, hi_khz in SHORTWAVE_BANDS:
        good = (
            isinstance(name, str) and name.endswith("m")
            and 500 <= lo_khz < hi_khz <= SW_MAX_HZ / 1000
            and lo_khz >= prev_hi  # 帯域は昇順・非重複
        )
        print(f"[{'OK' if good else 'FAIL'}] {name}: {lo_khz}-{hi_khz} kHz")
        ok &= good
        prev_hi = hi_khz

    # 2) スキャンセンタが全バンドを窓幅内に収めること
    centers = shortwave_scan_centers(RATE)
    half = RATE / 2.0 - 100000
    covered = True
    for name, lo_khz, hi_khz in SHORTWAVE_BANDS:
        lo, hi = lo_khz * 1000, min(hi_khz * 1000, SW_MAX_HZ)
        best = any(fc - half <= lo and fc + half >= hi for fc in centers)
        print(f"[{'OK' if best else 'FAIL'}] {name} covered by scan window")
        covered &= best
    ok &= covered
    ok &= bool(centers) and centers == sorted(centers)
    print(f"[{'OK' if ok else 'FAIL'}] {len(centers)} scan centers: "
          f"{centers[0]/1e6:.2f}-{centers[-1]/1e6:.2f} MHz")
    print(f"     centers = {[round(c/1e6, 2) for c in centers]}")

    # 3) バンド名マッピング
    checks = [
        (6055000, "49m"),
        (9750000, "31m"),
        (8330000, "SW"),   # 放送バンド外
        (400000, "SW"),
    ]
    for freq, expect in checks:
        got = shortwave_band_name(freq)
        good = got == expect
        print(f"[{'OK' if good else 'FAIL'}] {freq/1e6:.3f}MHz -> {got} (expect {expect})")
        ok &= good

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
