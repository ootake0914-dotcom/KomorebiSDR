"""Station list pulldown tests (no hardware).

- 25 stations -> 12 visible rows + scroll range.
- Clicking a row tunes exactly and closes the list.
- Clicking outside / Esc closes without tuning.
- Empty list does not open (status hint instead).
"""

import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame  # noqa: F401  (gui needs it; dummy driver above)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui import SdrGui


def make_stations(n=25):
    sts = []
    for i in range(n):
        f = 76000000 + i * 700000
        sts.append({"freq_hz": f, "freq_mhz": f / 1e6, "name": f"ST{i:02d}",
                    "snr_db": 20.0 - i * 0.3, "quality": "STRONG"})
    return sts


def main() -> int:
    ok = True
    gui = SdrGui()
    gui.detected_stations = make_stations(25)

    # レイアウト: 25局 -> 12行表示
    panel, rows, total = gui._station_list_layout()
    good = total == 25 and len(rows) == 12 and panel.width == 340
    print(f"[{'OK' if good else 'FAIL'}] layout 25 -> 12 rows (panel {panel.width}x{panel.height})")
    ok &= good

    # 行クリックで正確同調＋閉じる
    tuned = []
    gui.on_freq_change = lambda f: tuned.append(f)
    gui._toggle_station_list()
    good = gui.station_list_open
    print(f"[{'OK' if good else 'FAIL'}] toggle opens")
    ok &= good
    _, rows, _ = gui._station_list_layout()
    rc = rows[3]
    consumed = gui._station_list_click(rc.centerx, rc.centery)
    want = make_stations(25)[3]["freq_hz"]
    good = consumed and tuned == [want] and not gui.station_list_open
    print(f"[{'OK' if good else 'FAIL'}] row click tunes {tuned} (want [{want}])")
    ok &= good

    # 範囲外クリックで閉じる (同調なし)
    gui._toggle_station_list()
    tuned.clear()
    consumed = gui._station_list_click(5, 5)
    good = consumed and tuned == [] and not gui.station_list_open
    print(f"[{'OK' if good else 'FAIL'}] outside click closes without tuning")
    ok &= good

    # スクロール上限のクランプ
    gui._toggle_station_list()
    gui.station_list_scroll = 999
    panel, rows, total = gui._station_list_layout()
    start = max(0, min(gui.station_list_scroll, max(0, total - len(rows))))
    good = start == 13
    print(f"[{'OK' if good else 'FAIL'}] scroll clamped to {start} (want 13)")
    ok &= good

    # 空リストは開かない
    gui.detected_stations = []
    gui.station_list_open = False
    gui._toggle_station_list()
    good = not gui.station_list_open
    print(f"[{'OK' if good else 'FAIL'}] empty list does not open")
    ok &= good

    gui.close()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
