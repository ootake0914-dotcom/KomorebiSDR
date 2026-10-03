"""AB結果の集計表示 (audio_abプロジェクト)。

scores.jsonを読み、off/on・CER・P835の対比を一覧にする。
key.jsonが無くても動く (winner表示だけスキップ)。

Usage:
  python audio_ab/summary.py [--out audio_ab/out] [--csv]
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    out = os.path.join(ROOT, "audio_ab", "out")
    to_csv = "--csv" in argv
    i = 0
    while i < len(argv):
        if argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        else:
            i += 1
    with open(os.path.join(out, "scores.json"), encoding="utf-8") as f:
        scores = json.load(f)
    try:
        with open(os.path.join(out, "key.json"), encoding="utf-8") as f:
            key = json.load(f)
    except Exception:
        key = {}
    rows = []
    for item in sorted(scores):
        v = scores[item]
        if "error" in v:
            rows.append((item, "ERROR", "", "", "", ""))
            continue
        k = key.get(item, {}) or {}
        a_is = k.get("A", "?")
        b_is = k.get("B", "?")
        mos_a = v.get("mos_A", "-")
        mos_b = v.get("mos_B", "-")
        winner = "-"
        if isinstance(mos_a, (int, float)) and isinstance(mos_b, (int, float)):
            if abs(mos_a - mos_b) < 1e-9:
                winner = "tie"
            else:
                winner = a_is if mos_a > mos_b else b_is
        cer = ""
        if "cer_off" in v:
            cer = f"{v['cer_off']}->{v['cer_on']}"
        p = ""
        pa, pb = v.get("p835_A"), v.get("p835_B")
        if isinstance(pa, dict) and isinstance(pb, dict):
            oa, ob = pa.get("OVRL"), pb.get("OVRL")
            p = f"A:{oa} B:{ob}"
        rows.append((item, str(mos_a), str(mos_b), winner, cer, p))
    w = max((len(r[0]) for r in rows), default=10) + 2
    print(f"{'item':<{w}} {'MOS_A':>7} {'MOS_B':>7} {'winner':>8} "
          f"{'CER(off->on)':>14} {'P835_A/B':>16}")
    for r in rows:
        print(f"{r[0]:<{w}} {r[1]:>7} {r[2]:>7} {r[3]:>8} {r[4]:>14} {r[5]:>16}")
    if to_csv:
        import csv
        p = os.path.join(out, "summary.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["item", "mos_A", "mos_B", "winner", "cer_off_on", "p835"])
            wr.writerows(rows)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
