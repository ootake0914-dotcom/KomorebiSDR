"""RtProfile (RT deadline profiling) tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rt_profile import RtProfile


def main() -> int:
    ok = True
    p = RtProfile(budget_ms=57.3, size=64)
    for v in [10, 12, 11, 13, 12, 50, 11, 12, 80, 10, 11, 12]:
        p.add(v)
    p50, p95, p99 = p.percentiles()
    expect_p50 = 12.0  # 12個の中央値
    good = abs(p50 - expect_p50) < 1.0 and p99 >= p95 >= p50
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] percentiles p50={p50:.1f} p95={p95:.1f} p99={p99:.1f}")
    good = p.misses == 1 and p.max_consecutive_misses == 1  # 80msのみ超過 (50msは予算内)
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] misses={p.misses} max_consecutive={p.max_consecutive_misses}")
    good = p.headroom > 1.0 and "DSP" in p.summary() and "MISS" in p.summary()
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] summary '{p.summary()}'")

    # 連続超過の検出
    p2 = RtProfile(budget_ms=20.0, size=32)
    for v in [10, 30, 30, 30, 10]:
        p2.add(v)
    good = p2.max_consecutive_misses == 3
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] consecutive miss tracking = "
          f"{p2.max_consecutive_misses} (expect 3)")

    # リングが固定長であること (メモリ増加なし)
    p3 = RtProfile(budget_ms=10.0, size=16)
    for i in range(1000):
        p3.add(float(i % 7))
    good = p3._buf.shape == (16,) and p3._n == 16 and p3.blocks == 1000
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] bounded ring (size={p3._buf.shape[0]}, blocks={p3.blocks})")

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
