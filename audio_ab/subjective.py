"""主観ログと機械指標の突き合わせ (audio_ab)。

listen_log.csv (prefer=A/B/same) と scores.json / key.json を突き合わせ、
各指標の「良し悪しの向き」が主観と一致するか (符号一致率) と
順位相関 (Spearman) を出す。どの指標が耳を予測するかをデータで決める。

主観が未記入でも --xcorr で指標同士の相関行列は出せる (指標の冗長性検査)。

Usage:
  python audio_ab/subjective.py [--out audio_ab/out] [--xcorr]
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "audio_ab", "listen_log.csv")


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _side_value(v, k, field):
    """A/B割付からoff/on値を取り出す。"""
    fa, fb = v.get(f"{field}_A"), v.get(f"{field}_B")
    if fa is None and fb is None:
        return None, None
    a_is = (k or {}).get("A", "?")
    if a_is == "off":
        return fb, fa
    if a_is == "on":
        return fa, fb
    return None, None


def metric_deltas(v, k):
    """{metric: (delta_better_positive, n_valid)}。改善方向へ正規化する。"""
    out = {}
    if "mos_on" in v and "mos_off" in v:
        out["scoreq"] = float(v["mos_on"]) - float(v["mos_off"])
    else:
        on, off = _side_value(v, k, "mos")
        if on is not None:
            out["scoreq"] = float(on) - float(off)
    for f in ("OVRL", "SIG", "BAK"):
        pa, pb = v.get("p835_A"), v.get("p835_B")
        if isinstance(pa, dict) and isinstance(pb, dict):
            a_is = (k or {}).get("A", "?")
            onv = pa.get(f) if a_is == "on" else pb.get(f)
            offv = pb.get(f) if a_is == "on" else pa.get(f)
            if onv is not None and offv is not None:
                out[f"p835_{f}"] = float(onv) - float(offv)
    if "stoi_clean_on" in v and "stoi_clean_off" in v:
        out["stoi"] = float(v["stoi_clean_on"]) - float(v["stoi_clean_off"])
    if "hiss_on" in v and "hiss_off" in v:
        ho = float(v["hiss_off"]) + 1e-18
        hn = float(v["hiss_on"]) + 1e-18
        out["hiss_red_db"] = 10.0 * float(np.log10(ho / hn))
    if "cer_on" in v and "cer_off" in v:
        out["cer_improve"] = float(v["cer_off"]) - float(v["cer_on"])
    return out


def _load(out):
    with open(os.path.join(out, "scores.json"), encoding="utf-8") as f:
        scores = json.load(f)
    try:
        with open(os.path.join(out, "key.json"), encoding="utf-8") as f:
            key = json.load(f)
    except Exception:
        key = {}
    return scores, key


def _ratings():
    rows = {}
    with open(LOG, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = (r.get("prefer") or "").strip().lower()
            if p in ("a", "b", "same"):
                rows[r["item"].strip()] = p
    return rows


def main(argv):
    out = os.path.join(ROOT, "audio_ab", "out")
    xcorr = "--xcorr" in argv
    i = 0
    while i < len(argv):
        if argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        else:
            i += 1
    scores, key = _load(out)
    ratings = _ratings()
    print(f"scores items={len(scores)}  filled ratings={len(ratings)}")
    if ratings:
        per_metric = {}
        for item, p in ratings.items():
            v = scores.get(item)
            if not isinstance(v, dict) or "error" in v:
                continue
            k = key.get(item, {})
            a_is = (k or {}).get("A", "?")
            if p == "same":
                subj = 0.0
            elif p == "a":
                subj = 1.0 if a_is == "on" else (-1.0 if a_is == "off" else 0.0)
            else:  # b
                subj = 1.0 if a_is == "off" else (-1.0 if a_is == "on" else 0.0)
            if subj == 0.0:
                continue
            for m, d in metric_deltas(v, k).items():
                per_metric.setdefault(m, []).append((subj, d))
        print(f"\n{'metric':14s} {'n':>3s} {'sign_agree':>11s} {'spearman':>9s}")
        for m, pairs in sorted(per_metric.items()):
            s = np.array([p[0] for p in pairs])
            d = np.array([p[1] for p in pairs])
            agree = float(np.mean(np.sign(s) == np.sign(d))) if len(s) else float("nan")
            rho = _spearman(s, d)
            print(f"{m:14s} {len(s):3d} {agree:11.2f} {rho:9.3f}")
    else:
        print("listen_log.csvは未記入。run_ab.pyでAB生成→試聴→preferにA/B/sameを"
              "記入すると、指標と主観の一致率・Spearmanを出せます。")
    if xcorr:
        dl = {}
        for item, v in scores.items():
            if not isinstance(v, dict) or "error" in v:
                continue
            for m, d in metric_deltas(v, key.get(item, {})).items():
                dl.setdefault(m, {})[item] = d
        names = [m for m, dd in dl.items() if len(dd) >= 4]
        print(f"\n== 指標間Spearman (改善方向へ正規化, n>=4) ==")
        print(f"{'':14s}" + "".join(f"{n[:10]:>11s}" for n in names))
        for a in names:
            cells = []
            for b in names:
                common = sorted(set(dl[a]) & set(dl[b]))
                if a == b:
                    cells.append(f"{'—':>11s}")
                else:
                    rho = _spearman([dl[a][i] for i in common],
                                    [dl[b][i] for i in common])
                    cells.append(f"{rho:11.2f}" if np.isfinite(rho) else f"{'-':>11s}")
            print(f"{a:14s}" + "".join(cells))
        counts = {m: len(dl[m]) for m in names}
        print("n:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
