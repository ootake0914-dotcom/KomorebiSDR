"""録音IQのABハーネス (Phase 0-1)。

同一録音データをblack_magic OFF/ONで2回通し、以下を比較する:
- パイロットロック率 (ON: cyclo present率・平均confidence、OFF: native lock平均)
- blend平均・チャタリング回数
- 音声差分 (RMS比・高域10-15kエネルギー差)
- ブロック処理時間 p50/p95/p99・57.3ms超過率 (初回2ブロックは除外)
- RMT処理時間・バイパス率 (rmt有効時のみ)

対応形式: .npy (uint8 raw)、.cs16/.complex16 (int16 IQ)、.wav (int16 mono/stereo)。
RDS成功率・BER/MERはWFM音声に既知系列がないため未対応 (将来: RDS CRC率)。

Usage:
  python tools/ab_benchmark.py testdata/strong_946_3s.npy [--bm cyclo|rmt|sr|all] [--out result.json]
"""

import json
import os
import struct
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsp import SdrDspPipeline

BLOCK = 132096
BLOCK_MS = 57.3
WARMUP = 2
BM_FLAGS = ("cyclo", "rmt", "sr", "notch", "all")


def load_iq(path):
    """IQ録音→uint8 rawバイト列。形式は拡張子で判定する。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
        if a.dtype == np.uint8:
            return np.asarray(a, dtype=np.uint8).reshape(-1)
        a = np.asarray(a)
        if np.iscomplexobj(a):
            iq = a.reshape(-1)
            raw = np.empty(2 * len(iq), dtype=np.uint8)
            raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
            raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
            return raw
        raise ValueError(f"unsupported npy content: {a.dtype}")
    if ext in (".cs16", ".complex16", ".s16"):
        iq = np.fromfile(path, dtype=np.int16).astype(np.float64)
        if len(iq) % 2:
            iq = iq[:-1]
        c = (iq[0::2] + 1j * iq[1::2]) / 32768.0
        raw = np.empty(2 * len(c), dtype=np.uint8)
        raw[0::2] = np.clip(np.round(c.real * 127.5 + 127.5), 0, 255)
        raw[1::2] = np.clip(np.round(c.imag * 127.5 + 127.5), 0, 255)
        return raw
    if ext == ".wav":
        with wave.open(path, "rb") as w:
            n = w.getnframes()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            data = w.readframes(n)
        if sw == 1:
            iq = np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 127.5
            iq = iq / 127.5
        elif sw == 2:
            iq = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
        else:
            raise ValueError(f"unsupported wav width: {sw}")
        if ch == 2:
            c = iq[0::2] + 1j * iq[1::2]
        else:
            c = iq + 0j
        raw = np.empty(2 * len(c), dtype=np.uint8)
        raw[0::2] = np.clip(np.round(c.real * 127.5 + 127.5), 0, 255)
        raw[1::2] = np.clip(np.round(c.imag * 127.5 + 127.5), 0, 255)
        return raw
    raise ValueError(f"unsupported extension: {ext}")


def decode(raw, bm=()):
    """1パス復調→ブロック毎メトリクス。bmは有効化する黒魔法の集合。"""
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(0.0)
    dsp.cognitive_enabled = True
    if bm:
        dsp.black_magic_enabled = True
        dsp.bm_cyclo_enabled = "cyclo" in bm or "all" in bm
        dsp.bm_rmt_enabled = "rmt" in bm or "all" in bm
        dsp.bm_sr_enabled = "sr" in bm or "all" in bm
        dsp.bm_notch_enabled = "notch" in bm or "all" in bm
    nblk = len(raw) // BLOCK
    blends, locks, lats, presents, confs = [], [], [], [], []
    bypass = {}
    chunks = []
    for k in range(nblk):
        t0 = time.perf_counter()
        audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
        lats.append((time.perf_counter() - t0) * 1000.0)
        blends.append(float(dsp.stereo_blend))
        locks.append(float(dsp.stereo_pilot_lock))
        cyc = getattr(dsp, "cyclo_detector", None)
        presents.append(bool(getattr(cyc, "pilot_present", False)) if cyc else False)
        confs.append(float(getattr(dsp, "bm_cyclo_confidence", 0.0)))
        info = getattr(dsp, "_bm_last_rmt_info", None) or {}
        if dsp.bm_rmt is None:
            r = "rmt-disabled"
        else:
            r = str(info.get("bypass_reason", "active"))
        bypass[r] = bypass.get(r, 0) + 1
        a = np.asarray(audio)
        chunks.append(a if a.ndim == 2 else np.stack([a, a], axis=1))
    lat = np.array(lats[WARMUP:]) if len(lats) > WARMUP else np.array(lats)
    return {"blends": np.array(blends), "locks": np.array(locks),
            "lat": lat, "presents": np.array(presents),
            "confs": np.array(confs), "bypass": bypass,
            "audio": np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2))}


def chatter(blends, thr=0.5):
    s = (np.asarray(blends) > thr).astype(int)
    return int(np.sum(np.abs(np.diff(s)))) if len(s) > 1 else 0


def band_energy(x, lo, hi, sr=48000):
    n = len(x)
    if n < 256:
        return 1e-18
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f >= lo) & (f < hi)
    return float(np.sum(spec[m] ** 2) + 1e-18)


def pct(lat, q):
    a = np.asarray(lat, dtype=np.float64)
    return float(np.percentile(a, q)) if len(a) else 0.0


def align_delay(a, b, maxlag=32):
    """bをaに最大±maxlagで整合 (RMTの15サンプル固定遅延を吸収)。
    遅延差を「差分」と誤認しないための前処理。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best_lag, best_e = 0, float(np.mean((a - b) ** 2))
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            e = float(np.mean((a[lag:] - b[:n - lag]) ** 2))
        else:
            e = float(np.mean((a[:n + lag] - b[-lag:]) ** 2))
        if e < best_e:
            best_e, best_lag = e, lag
    if best_lag >= 0:
        return a[best_lag:], b[:n - best_lag], best_lag
    return a[:n + best_lag], b[-best_lag:], best_lag


def compare(off, on):
    """OFF/ON結果→指標辞書 (音声系は後半のみで定常比較・遅延整合つき)。"""
    ao = np.asarray(off["audio"][:, 0], dtype=np.float64)
    an = np.asarray(on["audio"][:, 0], dtype=np.float64)
    n = min(len(ao), len(an))
    tail = slice(n // 2, n)
    ao_t, an_t, lag = align_delay(ao[tail], an[tail])
    rms = 20.0 * float(np.log10((np.sqrt(np.mean(an_t ** 2)) + 1e-18)
                                / (np.sqrt(np.mean(ao_t ** 2)) + 1e-18)))
    hf = 10.0 * float(np.log10(band_energy(an_t, 10000, 15000)
                               / band_energy(ao_t, 10000, 15000)))
    return {
        "present_rate_on": float(np.mean(on["presents"])) if len(on["presents"]) else 0.0,
        "conf_mean_on": float(np.mean(on["confs"])) if len(on["confs"]) else 0.0,
        "lock_mean_off": float(np.mean(off["locks"])),
        "lock_mean_on": float(np.mean(on["locks"])),
        "blend_mean_off": float(np.mean(off["blends"])),
        "blend_mean_on": float(np.mean(on["blends"])),
        "chatter_off": chatter(off["blends"]),
        "chatter_on": chatter(on["blends"]),
        "rms_diff_db": rms,
        "hf_diff_db": hf,
        "align_lag": int(lag),
        "lat_off": {"p50": pct(off["lat"], 50), "p95": pct(off["lat"], 95),
                    "p99": pct(off["lat"], 99),
                    "overrun": float(np.mean(np.asarray(off["lat"]) > BLOCK_MS))},
        "lat_on": {"p50": pct(on["lat"], 50), "p95": pct(on["lat"], 95),
                   "p99": pct(on["lat"], 99),
                   "overrun": float(np.mean(np.asarray(on["lat"]) > BLOCK_MS))},
        "rmt_bypass_on": on["bypass"],
    }


def run_file(path, bm=("cyclo",)):
    raw = load_iq(path)
    nblk = len(raw) // BLOCK
    if nblk < 4:
        raise ValueError(f"too short: {path} ({nblk} blocks)")
    off = decode(raw, ())
    on = decode(raw, bm)
    m = compare(off, on)
    m.update({"file": os.path.basename(path), "blocks": nblk, "bm": list(bm)})
    return m


def fmt_lat(d):
    return (f"p50 {d['p50']:.1f}/p95 {d['p95']:.1f}/p99 {d['p99']:.1f}ms "
            f"超過率 {d['overrun']:.2f}")


def main(argv):
    bm = ("cyclo",)
    out = None
    files = []
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
            files.append(argv[i])
            i += 1
    if not files:
        print(__doc__)
        return 1
    results = []
    for f in files:
        m = run_file(f, bm)
        results.append(m)
        print(f"=== {m['file']} ({m['blocks']}blk, bm={m['bm']}) ===")
        print(f"  present率(ON) {m['present_rate_on']:.2f} conf(ON) {m['conf_mean_on']:.2f} "
              f"lock OFF {m['lock_mean_off']:.2f}/ON {m['lock_mean_on']:.2f}")
        print(f"  blend OFF {m['blend_mean_off']:.2f}/ON {m['blend_mean_on']:.2f} "
              f"chatter OFF {m['chatter_off']}/ON {m['chatter_on']}")
        print(f"  RMS差 {m['rms_diff_db']:+.2f}dB 高域差 {m['hf_diff_db']:+.2f}dB "
              f"(lag {m.get('align_lag', 0)})")
        print(f"  遅延 OFF {fmt_lat(m['lat_off'])}")
        print(f"  遅延 ON  {fmt_lat(m['lat_on'])}")
        if m["rmt_bypass_on"]:
            print(f"  RMT(ON) {m['rmt_bypass_on']}")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
