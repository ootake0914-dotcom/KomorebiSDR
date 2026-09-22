"""試聴用WAV生成 (主観評価用)。

録音IQを指定bm条件で復調し、48kHzステレオWAVを書き出す。
耳審査 (RMT単独AB等) の素材作り用。

Usage:
  python tools/make_listen_wav.py testdata/weak_775_10s.npy --bm rmt --out out.wav
  bmはカンマ区切り (例: --bm cyclo,notch)。空文字でOFF。
"""

import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ab_benchmark import load_iq, decode, BM_FLAGS


def main(argv):
    src = None
    bm = ()
    out = None
    i = 0
    while i < len(argv):
        if argv[i] == "--bm" and i + 1 < len(argv):
            v = argv[i + 1]
            bm = tuple(x for x in v.split(",") if x in BM_FLAGS)
            if "all" in bm:
                bm = ("all",)
            i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            src = argv[i]
            i += 1
    if src is None or out is None:
        print(__doc__)
        return 1
    raw = load_iq(src)
    res = decode(raw, bm)
    a = np.clip(np.asarray(res["audio"], dtype=np.float64), -1.0, 1.0)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes((a * 32767.0).astype(np.int16).tobytes())
    print(f"saved {out} (bm={list(bm) if bm else 'OFF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
