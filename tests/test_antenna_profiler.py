"""AntennaProfiler unit tests (pure logic, no hardware)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antenna_profiler import AntennaProfiler, band_key


def main() -> int:
    ok = True

    # 帯域キー分類
    cases = [(1000000, "MW"), (6000000, "HF"), (83200000, "FM"),
             (145800000, "VHF"), (439000000, "UHF"), ("bad", "UNK")]
    for f, want in cases:
        got = band_key(f)
        good = got == want
        print(f"[{'OK' if good else 'FAIL'}] band_key({f!r})={got} (want {want})")
        ok &= good

    # 未学習は空助言
    p = AntennaProfiler()
    good = p.advice("FM") == ""
    print(f"[{'OK' if good else 'FAIL'}] no advice before learning")
    ok &= good

    # 非有限は無視
    p.update("FM", float("nan"), -20.0)
    good = p.advice("FM") == ""
    print(f"[{'OK' if good else 'FAIL'}] NaN ignored")
    ok &= good

    # 好調帯
    for _ in range(20):
        p.update("FM", 20.0, -30.0)
    a = p.advice("FM")
    good = "好調" in a
    print(f"[{'OK' if good else 'FAIL'}] strong band advice: {a}")
    ok &= good

    # 弱電界帯
    for _ in range(20):
        p.update("HF", 1.0, -5.0)
    a = p.advice("HF")
    good = "ATU" in a or "弱電界" in a
    print(f"[{'OK' if good else 'FAIL'}] weak band advice: {a}")
    ok &= good

    # summary形状
    s = p.summary()
    good = set(s) == {"FM", "HF"} and all("snr_db" in v for v in s.values())
    print(f"[{'OK' if good else 'FAIL'}] summary shape")
    ok &= good

    # バイアス: 未学習0、強0、弱は負 (第2段・自動選択用)
    q = AntennaProfiler()
    good = q.bias_hz("FM") == 0.0
    print(f"[{'OK' if good else 'FAIL'}] bias unleared = 0")
    ok &= good
    for _ in range(25):
        q.update("HF", 2.0, -5.0)
    b = q.bias_hz("HF")
    good = -1500.0 <= b < 0.0
    print(f"[{'OK' if good else 'FAIL'}] bias weak = {b:.0f} Hz (want -1500..0)")
    ok &= good
    for _ in range(25):
        q.update("FM", 20.0, -30.0)
    good = q.bias_hz("FM") == 0.0
    print(f"[{'OK' if good else 'FAIL'}] bias strong = 0")
    ok &= good

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
