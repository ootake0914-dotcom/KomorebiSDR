"""Region profile / config tests (no hardware required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import region_profile, REGIONS


def main() -> int:
    checks = [
        ("JP", "JP", 76.0, 95.0, 50.0),
        ("US", "US", 87.5, 108.0, 75.0),
        ("DE", "CCIR", 87.5, 108.0, 50.0),
        ("RU", "OIRT", 65.8, 74.0, 50.0),
        ("BR", "CCIR", 87.5, 108.0, 50.0),
    ]
    ok = True
    for country, region, start, end, deemph in checks:
        p = region_profile(country)
        good = (p["region"] == region and p["fm_start"] == start
                and p["fm_end"] == end and p["deemphasis_us"] == deemph)
        print(f"[{'OK' if good else 'FAIL'}] {country} -> {p['region']} "
              f"{p['fm_start']}-{p['fm_end']}MHz de-emph {p['deemphasis_us']}us")
        ok &= good

    # 全プロファイルに必須キーが存在すること
    for name, prof in REGIONS.items():
        for key in ("fm_start", "fm_end", "fm_step_mhz", "deemphasis_us", "default_freq_hz"):
            if key not in prof:
                print(f"[FAIL] {name} missing {key}")
                ok = False

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
