"""EiBi shortwave schedule parser / lookup tests (offline)."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sw_schedule import parse_schedule, _day_active, LANG_NAMES

SAMPLE = """kHz:75;Time(UTC):93;Days:59;ITU:49;Station:201;Lng:49;Target:62;Remarks:135;P:35;Start:60;Stop:60;
6070;0300-0400;12345;D;Deutsche Welle;D;Eu;1;100;;
9750;0900-1000;;CHN;China Radio Int;M;EAs;1;100;;
9750;1000-1100;;CHN;China Radio Int;J;EAs;1;100;;
11810;0000-2400;;ROU;Radio Romania Int;E;Eu;1;100;;
"""


def main() -> int:
    entries = parse_schedule(SAMPLE)
    ok = len(entries) == 4
    print(f"[{'OK' if ok else 'FAIL'}] parsed {len(entries)} entries")
    for e in entries:
        if not (e["freq_hz"] and e["start_min"] is not None):
            ok = False

    by_freq = [e for e in entries if e["freq_hz"] == 9750000]
    good = len(by_freq) == 2 and by_freq[0]["language"] == "M"
    print(f"[{'OK' if good else 'FAIL'}] 9750kHz parsed with language codes")
    ok &= good

    good = _day_active("", 3) and _day_active("12345", 3) and not _day_active("671", 3)
    print(f"[{'OK' if good else 'FAIL'}] day-of-week filtering")
    ok &= good

    good = LANG_NAMES.get("M") == "中国語" and LANG_NAMES.get("J") == "日本語"
    print(f"[{'OK' if good else 'FAIL'}] language display map")
    ok &= good

    # lookup の時間帯判定 (time 09:30 UTC -> 9750kHz の zh 放送)
    from sw_schedule import lookup
    import sw_schedule as sw
    sw._schedule = entries
    when = datetime(2026, 6, 10, 9, 30, tzinfo=timezone.utc)  # 水曜
    hit = lookup(9750000, when, tolerance_hz=1000)
    good = hit is not None and "China" in hit["station"] and hit["language"] == "中国語"
    print(f"[{'OK' if good else 'FAIL'}] 09:30 UTC lookup -> {hit['name'] if hit else None}")
    ok &= good

    when2 = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
    hit2 = lookup(9750000, when2, tolerance_hz=1000)
    good = hit2 is None
    print(f"[{'OK' if good else 'FAIL'}] out-of-schedule time returns None")
    ok &= good

    # 月曜のみの放送が水曜にヒットしないこと
    hit3 = lookup(6070000, when, tolerance_hz=1000)
    good = hit3 is None
    print(f"[{'OK' if good else 'FAIL'}] weekday-only entry ignored on other days")
    ok &= good

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
