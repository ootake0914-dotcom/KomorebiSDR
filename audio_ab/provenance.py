"""結果の来歴 (audio_ab)。

どのコード・モデル版で測ったかを結果JSONに刻む。数値の比較は
同一来歴の間でのみ意味を持つため、後から追跡できるようにする。
"""

import datetime
import importlib.metadata as md
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(args):
    return subprocess.run(["git"] + args, cwd=ROOT, capture_output=True,
                          text=True, timeout=5)


def stamp() -> dict:
    out = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
           "python": sys.version.split()[0],
           "numpy": np.__version__}
    try:
        head = _git(["rev-parse", "--short", "HEAD"])
        out["git_head"] = head.stdout.strip()
        dirty = _git(["status", "--porcelain"])
        out["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        pass
    for pkg, key in (("faster-whisper", "faster_whisper"),
                     ("onnxruntime", "onnxruntime"),
                     ("torch", "torch"),
                     ("voicevox", None)):
        if key is None:
            continue
        try:
            out[key] = md.version(pkg)
        except Exception:
            pass
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(stamp(), ensure_ascii=False, indent=2))
