"""Ear-AB batch runner (audio_ab project).

生放送は二度と同じ条件で聴けないため、耳ABはすべて「同一録音・
同一合成素材の往復比較」で行う。このスクリプトは保留中の試聴項目を
一括で回し、ブラインドABペア (A/Bランダム割付)＋スコアを出す。
聴くのは人間 (listen_log.csv に記録)。測るのは機械 (scores.json)。

Usage:
  python audio_ab/run_ab.py [--items E1,E2] [--out audio_ab/out]
  python audio_ab/run_ab.py --list
"""

import json
import os
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dsp import SdrDspPipeline
from score_wav import stoi, band_pow

BLOCK = 132096
FS = 48000
IF = 288000

ITEMS = [
    "E1_nr_ssb_white", "E2_nr_ssb_pink", "E3_nr_nfm",
    "E4_sband_synth", "E5_sband_sw7300", "E6_notch_prog",
    "E7_rmt_music", "E8_rmt_talk", "E9_apod_strong",
    "E10_wfm_rmt_weak", "R1_day_nbm_transp", "R4_day94_rmt", "R5_day94_apod",
]


# ---------------- proxies ----------------
def _vowel_paused(n, seed=0, f0=120.0):
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    f0t = f0 + 30.0 * np.sin(2 * np.pi * 0.7 * t + seed)
    ph = 2 * np.pi * np.cumsum(f0t) / FS
    v = np.zeros(n)
    for h, a in [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]:
        v += a * np.sin(h * ph + rng.uniform(0, np.pi))
    v = v * (0.8 + 0.2 * np.sin(2 * np.pi * 3.0 * t))
    gate = np.where((t % 1.1) < 0.8, 1.0, 0.0)
    edge = int(FS * 0.01)
    k = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    for start in np.arange(0.8, t[-1], 1.1):
        i = int(start * FS)
        v[i:i + edge] *= k
        j = max(0, i - edge)
        v[j:j + edge] *= k[::-1]
    v = v * gate
    return (v / max(float(np.max(np.abs(v))), 1e-9)).astype(np.float32)


def _music_pad(n, seed=0):
    """音楽もどき: 2コード往復のパッド＋ゆるいトレモロ (RMT/apodizing用)。"""
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    roots = [110.0, 130.81]  # A2/C3
    y = np.zeros(n)
    half = n // 2
    for (s, e), root in zip([(0, half), (half, n)], roots):
        seg = np.arange(s, e)
        chord = [1.0, 1.25, 1.5, 2.0]
        for m in chord:
            y[s:e] += (0.25 / m) * np.sin(2 * np.pi * root * m * t[s:e]
                                          + rng.uniform(0, np.pi))
    y = y * (0.85 + 0.15 * np.sin(2 * np.pi * 4.5 * t))
    return (y / max(float(np.max(np.abs(y))), 1e-9)).astype(np.float32)


def _white(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _pink_kellet(n, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    b = np.zeros(7)
    y = np.zeros(n)
    for i, x in enumerate(w):
        b[0] = 0.99886 * b[0] + x * 0.0555179
        b[1] = 0.99332 * b[1] + x * 0.0750759
        b[2] = 0.96900 * b[2] + x * 0.1538520
        b[3] = 0.86650 * b[3] + x * 0.3104856
        b[4] = 0.55000 * b[4] + x * 0.5329522
        b[5] = -0.7616 * b[5] - x * 0.0168980
        y[i] = (b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + x * 0.5362)
        b[6] = x * 0.115926
    y = (y * 0.11).astype(np.float32)
    return (y / max(float(np.max(np.abs(y))), 1e-9)).astype(np.float32)


def _to_mono(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        a = np.mean(a, axis=1)
    return a.reshape(-1)


def _stereo_wav(path, audio):
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    peak = float(np.max(np.abs(a)))
    clipped = peak >= 1.0
    a = np.clip(a, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((a * 32767.0).astype(np.int16).tobytes())
    return peak, clipped


def _scores(off, on, clean=None):
    offm, onm = _to_mono(off), _to_mono(on)
    n = min(len(offm), len(onm))
    offm, onm = offm[:n], onm[:n]
    s = {
        "stoi_offref": round(float(stoi(offm, onm)), 4),
        "rms_off": round(float(np.sqrt(np.mean(offm ** 2))), 5),
        "rms_on": round(float(np.sqrt(np.mean(onm ** 2))), 5),
        "peak_off": round(float(np.max(np.abs(offm))), 4),
        "peak_on": round(float(np.max(np.abs(onm))), 4),
        "hiss_off": round(float(band_pow(offm, FS, 6000, 12000)), 6),
        "hiss_on": round(float(band_pow(onm, FS, 6000, 12000)), 6),
    }
    if clean is not None:
        cm = _to_mono(clean)[:n]
        s["stoi_clean_off"] = round(float(stoi(cm, offm)), 4)
        s["stoi_clean_on"] = round(float(stoi(cm, onm)), 4)
    try:
        a = np.asarray(on)
        if a.ndim == 2 and a.shape[1] == 2:
            mid = (a[:, 0] + a[:, 1]) / 2.0
            sid = (a[:, 0] - a[:, 1]) / 2.0
            with np.errstate(divide="ignore", invalid="ignore"):
                r = float(np.mean(sid ** 2) / (np.mean(mid ** 2) + 1e-18))
            if r > 0 and np.isfinite(r):
                s["side_mid_ratio_db"] = round(float(10 * np.log10(r)), 2)
    except Exception:
        pass
    return s


# ---------------- treatments ----------------
def _demod_blocks(dsp, data, fn, nblk, blen):
    outs = []
    for k in range(nblk):
        outs.append(fn(dsp, data[k * blen:(k + 1) * blen]))
    y = np.concatenate([np.asarray(o).reshape(-1) for o in outs])
    return y.astype(np.float32)


def t_nr_ssb(noise="white", secs=10, snr_db=5.0):
    n = FS * secs
    v = _vowel_paused(n, seed=2) * 0.5
    nz = _white(n, seed=3) if noise == "white" else _pink_kellet(n, seed=3)
    nz = nz / (float(np.sqrt(np.mean(nz.astype(np.float64) ** 2))) + 1e-12)
    nz = (nz * np.sqrt(float(np.mean(v.astype(np.float64) ** 2))
                       / (10.0 ** (snr_db / 10.0)))).astype(np.float32)
    t = np.arange(n) / FS
    iq = ((v + nz) * np.exp(1j * 2 * np.pi * 1500.0 * t)).astype(np.complex64)
    bl = 2208
    nb = n // bl
    d0 = SdrDspPipeline(1152000, FS)
    off = _demod_blocks(d0, iq, lambda d, s: d.demodulate_ssb(s, "USB"), nb, bl)
    d1 = SdrDspPipeline(1152000, FS)
    d1.nbm_nr_enabled = True
    on = _demod_blocks(d1, iq, lambda d, s: d.demodulate_ssb(s, "USB"), nb, bl)
    return off, on, {"mode": "SSB", "noise": noise, "snr": snr_db}, v


def t_nr_nfm(secs=10, snr_db=5.0):
    n = IF * secs
    t = np.arange(n) / IF
    v48 = _vowel_paused(FS * secs, seed=4)
    v_if = np.interp(t, np.arange(FS * secs) / FS, v48).astype(np.float32)
    ph = 2 * np.pi * 3000.0 * np.cumsum(v_if) / IF
    car = (0.5 * np.exp(1j * ph)).astype(np.complex64)
    rng = np.random.default_rng(5)
    nz = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    nz = nz / (float(np.sqrt(np.mean(np.abs(nz) ** 2))) + 1e-12)
    sig_pow = float(np.mean(np.abs(car) ** 2))
    iq = (car + nz * np.sqrt(sig_pow / (10.0 ** (snr_db / 10.0)))).astype(np.complex64)
    bl = 16512
    nb = n // bl
    d0 = SdrDspPipeline(1152000, FS)
    d0.nbm_nr_enabled = False
    off = _demod_blocks(d0, iq, lambda d, s: d.demodulate_nfm(s), nb, bl)
    d1 = SdrDspPipeline(1152000, FS)
    d1.nbm_nr_enabled = True
    on = _demod_blocks(d1, iq, lambda d, s: d.demodulate_nfm(s), nb, bl)
    return off, on, {"mode": "NFM", "snr": snr_db}, None


def t_sband_synth(secs=8):
    n = IF * secs
    t = np.arange(n) / IF
    v48 = _vowel_paused(FS * secs, seed=6)
    v_if = np.interp(t, np.arange(FS * secs) / FS, v48)
    env = 1e-4 * (1.0 + 0.5 * v_if)
    iq = (env * np.exp(1j * 2 * np.pi * 15.0 * t)).astype(np.complex64)
    gate = np.ones(n)
    gate[:len(gate) // 2] = 0.0
    iq = (iq + (gate * 3e-5 * np.exp(1j * 2 * np.pi * 2500.0 * t)).astype(np.complex64))
    rng = np.random.default_rng(7)
    iq = (iq + (5e-6 * (rng.standard_normal(n)
                        + 1j * rng.standard_normal(n))).astype(np.complex64))
    bl = 16512
    nb = n // bl
    d0 = SdrDspPipeline(1152000, FS)
    off = _demod_blocks(d0, iq, lambda d, s: d.demodulate_am(s), nb, bl)
    d1 = SdrDspPipeline(1152000, FS)
    d1.am_sideband_enabled = True
    on = _demod_blocks(d1, iq, lambda d, s: d.demodulate_am(s), nb, bl)
    return off, on, {"mode": "AM-synth-onesided"}, None


def _iqfile(path):
    a = np.load(path)
    return np.asarray(a, dtype=np.uint8).reshape(-1)


def _proc_file(raw, mode, setup=None, offset=0.0):
    dsp = SdrDspPipeline(1152000, FS)
    dsp.set_offset_freq(offset)
    if setup:
        setup(dsp)
    outs = []
    nblk = len(raw) // BLOCK
    for k in range(nblk):
        audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode=mode)
        a = np.asarray(audio)
        outs.append(a if a.ndim == 2 else np.stack([a, a], axis=1))
    return np.concatenate(outs, axis=0)


def _pick_offset(raw, mode, setup=None):
    """録音のオフセット同調 (0 / +150k) をRMSで自動選択する。"""
    best, best_rms = 0.0, -1.0
    for off in (0.0, 150000.0):
        try:
            y = _proc_file(raw[:BLOCK * 6], mode, setup, offset=off)
            r = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
        except Exception:
            r = -1.0
        if r > best_rms:
            best, best_rms = off, r
    return best


def t_sband_sw7300():
    raw = _iqfile(os.path.join(ROOT, "testdata", "sw_7300_10s.npy"))

    def setup(d, on=False):
        if on:
            d.am_sideband_enabled = True

    off_cfg = _pick_offset(raw, "AM")
    off = _proc_file(raw, "AM", offset=off_cfg)
    on = _proc_file(raw, "AM", setup=lambda d: setup(d, True), offset=off_cfg)
    return off, on, {"mode": "AM", "src": "sw_7300_10s", "offset": off_cfg}, None


def t_notch_prog(secs=8):
    n = IF * secs
    t = np.arange(n) / IF
    v48 = _vowel_paused(FS * secs, seed=8)
    v_if = np.interp(t, np.arange(FS * secs) / FS, v48)
    env = 1e-4 * (1.0 + 0.5 * v_if)
    hum = 2e-5 * (np.sin(2 * np.pi * 50.0 * t) + 0.5 * np.sin(2 * np.pi * 100.0 * t))
    iq = ((env + hum) * np.exp(1j * 2 * np.pi * 10.0 * t)).astype(np.complex64)
    bl = 16512
    nb = n // bl
    d0 = SdrDspPipeline(1152000, FS)
    off = _demod_blocks(d0, iq, lambda d, s: d.demodulate_am(s), nb, bl)
    d1 = SdrDspPipeline(1152000, FS)
    d1.black_magic_enabled = True
    d1.bm_notch_enabled = True
    on = _demod_blocks(d1, iq, lambda d, s: d.demodulate_am(s), nb, bl)
    return off, on, {"mode": "AM", "hum": "50+100Hz"}, None


def t_rmt_prog(kind="music", secs=12):
    from rmt_denoiser import SafeRmtDenoiser
    n = FS * secs
    prog = _music_pad(n, seed=9) if kind == "music" else _vowel_paused(n, seed=10)
    rng = np.random.default_rng(11)
    nz = rng.standard_normal(n).astype(np.float32)
    nz = nz / (float(np.sqrt(np.mean(nz.astype(np.float64) ** 2))) + 1e-12)
    nz = (nz * np.sqrt(float(np.mean(prog.astype(np.float64) ** 2)) / 10.0)).astype(np.float32)
    x = (prog * 0.3 + nz).astype(np.float32)
    den = SafeRmtDenoiser(max_strength=0.65)
    on, info = den.process_mono(x, s_meter_dbfs=-40.0, ch="ab")
    on = np.asarray(on, dtype=np.float32)
    return x, on, {"denoiser": "rmt-direct", "prog": kind,
                   "bypass": str(info.get("bypass_reason", ""))}, prog * 0.3


def t_apod_strong():
    raw = _iqfile(os.path.join(ROOT, "testdata", "strong_946_3s.npy"))

    def setup(d, on=False):
        d.set_audiophile_mode(apodizing=bool(on))

    off_cfg = _pick_offset(raw, "WFM")
    off = _proc_file(raw, "WFM", offset=off_cfg)
    on = _proc_file(raw, "WFM", setup=lambda d: setup(d, True), offset=off_cfg)
    return off, on, {"mode": "WFM", "src": "strong_946", "apodizing": "off/on",
                     "offset": off_cfg}, None


def t_wfm_rmt_weak():
    raw = _iqfile(os.path.join(ROOT, "testdata", "weak_775_10s.npy"))

    def setup(d, on=False):
        if on:
            d.black_magic_enabled = True
            d.bm_rmt_enabled = True

    off_cfg = _pick_offset(raw, "WFM")
    off = _proc_file(raw, "WFM", offset=off_cfg)
    on = _proc_file(raw, "WFM", setup=lambda d: setup(d, True), offset=off_cfg)
    return off, on, {"mode": "WFM", "src": "weak_775", "rmt": "off/on",
                     "offset": off_cfg}, None


def t_day_nbm_transp():
    """昼デッドエアでのNR透過性 (実RFノイズでバイパスすることを確認)。
    7.1MHz LSB録音をSSB復調±NRで比較する。"""
    raw = _iqfile(os.path.join(ROOT, "testdata", "day_7.100_LSB_12s.npy"))
    # 正式受信経路 (dsp.process LSB) で比較する
    off = _proc_file(raw, "LSB", offset=150000.0)
    on = _proc_file(raw, "LSB", setup=lambda d: setattr(d, "nbm_nr_enabled", True),
                    offset=150000.0)
    return off, on, {"mode": "LSB", "src": "day_7.100(dead-air)",
                     "expect": "identical-or-bypass"}, None


def t_day94_rmt():
    """昼FM音楽でのRMT定位 (R4)。実音楽のステレオ像・人工物の有無を聴く。"""
    raw = _iqfile(os.path.join(ROOT, "testdata", "day_94.600_WFM_15s.npy"))

    def setup(d, on=False):
        if on:
            d.black_magic_enabled = True
            d.bm_rmt_enabled = True

    off_cfg = _pick_offset(raw, "WFM")
    off = _proc_file(raw, "WFM", offset=off_cfg)
    on = _proc_file(raw, "WFM", setup=lambda d: setup(d, True), offset=off_cfg)
    return off, on, {"mode": "WFM", "src": "day_94.600(music?)",
                     "rmt": "off/on", "offset": off_cfg}, None


def t_day94_apod():
    """昼FM音楽でのapodizing (R4続き)。"""
    raw = _iqfile(os.path.join(ROOT, "testdata", "day_94.600_WFM_15s.npy"))

    def setup(d, on=False):
        d.set_audiophile_mode(apodizing=bool(on))

    off_cfg = _pick_offset(raw, "WFM")
    off = _proc_file(raw, "WFM", offset=off_cfg)
    on = _proc_file(raw, "WFM", setup=lambda d: setup(d, True), offset=off_cfg)
    return off, on, {"mode": "WFM", "src": "day_94.600(music?)",
                     "apodizing": "off/on", "offset": off_cfg}, None


TREATMENTS = {
    "E1_nr_ssb_white": lambda: t_nr_ssb("white"),
    "E2_nr_ssb_pink": lambda: t_nr_ssb("pink"),
    "E3_nr_nfm": t_nr_nfm,
    "E4_sband_synth": t_sband_synth,
    "E5_sband_sw7300": t_sband_sw7300,
    "E6_notch_prog": t_notch_prog,
    "E7_rmt_music": lambda: t_rmt_prog("music"),
    "E8_rmt_talk": lambda: t_rmt_prog("talk"),
    "E9_apod_strong": t_apod_strong,
    "E10_wfm_rmt_weak": t_wfm_rmt_weak,
    "R1_day_nbm_transp": t_day_nbm_transp,
    "R4_day94_rmt": t_day94_rmt,
    "R5_day94_apod": t_day94_apod,
}


def main(argv):
    sel = list(ITEMS)
    out = os.path.join(ROOT, "audio_ab", "out")
    i = 0
    while i < len(argv):
        if argv[i] == "--items" and i + 1 < len(argv):
            sel = [x.strip() for x in argv[i + 1].split(",") if x.strip() in TREATMENTS]
            i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        elif argv[i] == "--list":
            for k in TREATMENTS:
                print(k)
            return 0
        else:
            i += 1
    os.makedirs(out, exist_ok=True)
    import random
    # 既存結果に追記する (項目指定の部分回しで他項目を消さない)
    scores_all, key = {}, {}
    for fn, dst in (("scores.json", scores_all), ("key.json", key)):
        try:
            with open(os.path.join(out, fn), encoding="utf-8") as f:
                dst.update(json.load(f))
        except Exception:
            pass
    for item in sel:
        t0 = time.perf_counter()
        print(f"--- {item} ---", flush=True)
        try:
            res = TREATMENTS[item]()
            off, on, meta = res[0], res[1], res[2]
            clean = res[3] if len(res) > 3 else None
        except Exception as e:
            print(f"[ERROR] {item}: {e}")
            scores_all[item] = {"error": str(e)}
            continue
        sc = _scores(off, on, clean)
        sc["meta"] = meta
        identical = bool(np.array_equal(np.asarray(off), np.asarray(on)))
        sc["identical"] = identical
        flip = random.Random(item).random() < 0.5
        A, B = (off, on) if flip else (on, off)
        key[item] = {"A": ("on" if flip else "off"),
                     "B": ("off" if flip else "on")}
        pa, ca = _stereo_wav(os.path.join(out, f"{item}_A.wav"), A)
        pb, cb = _stereo_wav(os.path.join(out, f"{item}_B.wav"), B)
        sc["peak_A"], sc["peak_B"] = round(pa, 4), round(pb, 4)
        sc["clipped"] = bool(ca or cb)
        sc["seconds"] = round((time.perf_counter() - t0), 1)
        scores_all[item] = sc
        print(f"  identical={identical} stoi_offref={sc['stoi_offref']} "
              f"rms {sc['rms_off']}->{sc['rms_on']} ({sc['seconds']}s)", flush=True)
    with open(os.path.join(out, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores_all, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    print(f"wrote {out}/scores.json + key.json (keyは聴取後に開封)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
