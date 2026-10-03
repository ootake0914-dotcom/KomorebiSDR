"""実素材レジストリ (audio_ab)。

合成だけでなく実放送を第1級の測定素材にする。ファイルは巨大で
gitignore対象 (testdata/day_*、night_*、*_rec_* 等) のため、
無い環境では available() から落ちて呼び出し側がskipできる。

Usage:
  from realdata import available, path, decode_wfm
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ITEMS = {
    "fm_lucky60": {
        "file": "day_fm_lucky_94.600_60s.npy", "mode": "WFM",
        "offset": 150000.0, "labels": {"band": "FM", "content": "music",
                                       "station": "LuckyFM 94.6"}},
    "fm_nhk60": {
        "file": "day_fm_nhk_83.200_60s.npy", "mode": "WFM",
        "offset": 150000.0, "labels": {"band": "FM", "content": "mixed",
                                       "station": "NHK-FM 83.2"}},
    "fm_lucky120": {
        "file": "fm_rec_lucky_94.600_120s.npy", "mode": "WFM",
        "offset": 150000.0, "labels": {"band": "FM", "content": "music",
                                       "station": "LuckyFM 94.6"}},
}


def path(name: str) -> str:
    return os.path.join(ROOT, "testdata", ITEMS[name]["file"])


def available() -> list:
    return [k for k, v in ITEMS.items()
            if os.path.exists(os.path.join(ROOT, "testdata", v["file"]))]


def load(name: str) -> np.ndarray:
    return np.load(path(name))


def decode_wfm(name: str, nr: bool, block: int = 132096,
               telemetry: bool = False):
    """WFM実録をNR on/offでデコード。returns (stereo_audio, info)。"""
    from dsp import SdrDspPipeline
    raw = load(name)
    d = SdrDspPipeline(1152000, 48000)
    d.set_offset_freq(float(ITEMS[name].get("offset", 0.0)))
    d.set_stereo_nr(bool(nr))
    d.cognitive_enabled = False
    d.slow_agc_enabled = False
    outs = []
    g = []
    for k in range(len(raw) // block):
        a, _ = d.process(raw[k * block:(k + 1) * block], mode="WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
        if telemetry:
            g.append(float(d.stereo_nr_gain))
    info = {}
    if telemetry:
        g = np.asarray(g)
        warm = g[20:] if len(g) > 40 else g
        info["nr_gain_min"] = float(warm.min()) if len(warm) else 1.0
        info["nr_gain_mean"] = float(warm.mean()) if len(warm) else 1.0
    return np.concatenate(outs, axis=0), info
