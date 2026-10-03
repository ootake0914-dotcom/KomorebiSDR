"""既存ABペアのScoreq-MOS一括採点 (audio_abプロジェクト)。

耳の代わりの総合点 (NR-MOS) をout/*_{A,B}.wavに付ける。
順位付け用 (絶対値の解釈はしない)。

Usage:
  python audio_ab/score_mos.py [--out audio_ab/out]
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "audio_ab"))

from score_noref import MosScorer


def _to16k_tmp(path):
    """DNSMOS用に16k mono wavへ変換 (temp)。戻り値はパス。"""
    import wave as _w
    import tempfile
    import numpy as np
    from score_noref import load_wav_mono, fft_resample
    x, sr = load_wav_mono(path)
    y = fft_resample(x, sr, 16000)
    fd, tp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with _w.open(tp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return tp


def main(argv):
    out = os.path.join(ROOT, "audio_ab", "out")
    p835 = False
    i = 0
    while i < len(argv):
        if argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        elif argv[i] == "--p835":
            p835 = True
            i += 1
        else:
            i += 1
    mos = MosScorer()
    dns = None
    if p835:
        from score_noref import DnsmosScorer
        dns = DnsmosScorer()
    try:
        with open(os.path.join(out, "scores.json"), encoding="utf-8") as f:
            scores = json.load(f)
    except Exception:
        scores = {}
    try:
        with open(os.path.join(out, "key.json"), encoding="utf-8") as f:
            key = json.load(f)
    except Exception:
        key = {}
    pairs = sorted(glob.glob(os.path.join(out, "*_A.wav")))
    for pa in pairs:
        item = os.path.basename(pa)[:-len("_A.wav")]
        pb = os.path.join(out, item + "_B.wav")
        if not os.path.exists(pb):
            continue
        try:
            ma = round(float(mos.score_wav(pa)), 4)
        except Exception as e:
            ma = f"ERR {e}"
        try:
            mb = round(float(mos.score_wav(pb)), 4)
        except Exception as e:
            mb = f"ERR {e}"
        row = scores.get(item, {})
        row["mos_A"] = ma
        row["mos_B"] = mb
        # key開封は採点後に行う (ここでは開封せず、順位だけ付ける)
        ka = (key.get(item, {}) or {}).get("A", "?")
        row["mos_better"] = ("A" if isinstance(ma, float) and isinstance(mb, float)
                             and ma > mb else
                             ("B" if isinstance(ma, float) and isinstance(mb, float)
                              and mb > ma else "tie"))
        row["mos_winner_is"] = ka if row["mos_better"] == "A" else (
            (key.get(item, {}) or {}).get("B", "?") if row["mos_better"] == "B"
            else "tie")
        scores[item] = row
        print(f"{item}: MOS A={ma} B={mb} -> {row['mos_better']} "
              f"(={row['mos_winner_is']})", flush=True)
        if dns is not None:
            tmps = []
            for tag, p in (("A", pa), ("B", pb)):
                try:
                    tp = _to16k_tmp(p)
                    tmps.append(tp)
                    d = dns.score_wav16k(tp)
                    row[f"p835_{tag}"] = {k: round(float(v), 3)
                                          for k, v in d.items()
                                          if k in ("SIG", "BAK", "OVRL", "P808_MOS")}
                except Exception as e:
                    row[f"p835_{tag}"] = f"ERR {str(e)[:80]}"
            for tp in tmps:
                try:
                    os.unlink(tp)
                except Exception:
                    pass
            print(f"  P835 A={row.get('p835_A')} B={row.get('p835_B')}", flush=True)
    with open(os.path.join(out, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print("scores.json updated (mos_*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
