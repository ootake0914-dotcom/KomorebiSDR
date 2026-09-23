"""プロファイリングスナップショット (第2章一手目)。
パイプライン無改変: メソッドを外部ラッパで包み、排他時間を集計する。
出力: ブロック内訳 / BM ON-OFF差分 / p50-p99 / モード別 / tracemalloc上位。
Usage: python tools/profile_snapshot.py [--blocks 30]
"""
import os
import sys
import time
import tracemalloc

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsp import SdrDspPipeline

STACK = []
STATS = {}


def wrap(cls_or_obj, meth_name, stage):
    orig = getattr(cls_or_obj, meth_name, None)
    if orig is None or getattr(orig, "_prof_wrapped", False):
        return False

    def fn(*a, **k):
        frame = {"stage": stage, "child": 0.0}
        STACK.append(frame)
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            STACK.pop()
            s = STATS.setdefault(stage, {"incl": 0.0, "excl": 0.0, "n": 0})
            s["incl"] += dt
            s["excl"] += dt - frame["child"]
            s["n"] += 1
            if STACK:
                STACK[-1]["child"] += dt

    fn._prof_wrapped = True
    try:
        setattr(cls_or_obj, meth_name, fn)
    except (TypeError, AttributeError):
        return False
    return True


def install():
    from rmt_denoiser import SafeRmtDenoiser
    from adaptive_notch import AdaptiveNotchCanceller
    from stochastic_resonance import StochasticResonanceDetector
    from cyclostationary_detector import CyclostationaryPilotDetector
    from adaptive_audio import (CognitiveSpeechMusicTracker,
                                HolographicAudioEnhancer)
    from adaptive_stereo import SuperSpatialBssStereoSeparator
    P = SdrDspPipeline
    wrap(P, "raw_to_iq", "front/raw_to_iq")
    wrap(P, "mix_frequency", "front/mix")
    wrap(P, "compute_spectrum", "front/spectrum")
    wrap(P, "decimate", "chan/decimate")
    wrap(P, "decimate_with_history", "chan/decim_hist")
    wrap(P, "demodulate_wfm", "demod/wfm")
    wrap(P, "demodulate_nfm", "demod/nfm")
    wrap(P, "demodulate_am", "demod/am")
    wrap(P, "demodulate_ssb", "demod/ssb")
    wrap(P, "_slow_agc_level", "post/agc")
    wrap(P, "_apply_cma", "demod/cma")
    wrap(P, "_apply_hard_limiter", "wfm/limiter")
    wrap(P, "_update_cma_auto_gate", "wfm/cma_gate")
    wrap(P, "_decode_stereo_pair", "wfm/stereo_pair")
    wrap(P, "_freq_dependent_blend", "wfm/freq_blend")
    wrap(P, "_update_stereo_trim", "wfm/trim")
    wrap(P, "_update_cognitive_morph", "wfm/cog_morph")
    wrap(P, "_bm_attack_limit", "wfm/bm_attack")
    wrap(P, "_delay_mono", "pair/delay_mono")
    wrap(P, "_update_stereo_nr", "pair/nr_est")
    wrap(P, "_diff_lowpass", "pair/diff_lp")
    wrap(P, "_wiener_diff", "pair/wiener")
    wrap(P, "_post_process_wfm", "pair/post")
    wrap(P, "_update_stereo_trim", "pair/trim")
    wrap(SafeRmtDenoiser, "process_mono", "bm/rmt_mono")
    wrap(SafeRmtDenoiser, "process_stereo", "bm/rmt_stereo")
    wrap(AdaptiveNotchCanceller, "process_mono", "bm/notch_mono")
    wrap(AdaptiveNotchCanceller, "process_stereo", "bm/notch_stereo")
    wrap(StochasticResonanceDetector, "assess", "bm/sr")
    wrap(CyclostationaryPilotDetector, "update", "bm/cyclo")
    wrap(CognitiveSpeechMusicTracker, "process", "post/cog_eq")
    wrap(CognitiveSpeechMusicTracker, "analyze", "post/cog_an")
    wrap(HolographicAudioEnhancer, "process", "post/holo")
    wrap(SuperSpatialBssStereoSeparator, "process", "post/bss")


def synth_iq(mode, n, seed=5):
    RF = 1152000
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RF
    if mode == "NFM":
        m = np.sin(2 * np.pi * 1000.0 * t)
        ph = 2 * np.pi * 5000.0 * np.cumsum(m) / RF
        iq = 0.6 * np.exp(1j * ph)
    elif mode == "AM":
        m = 0.5 + 0.4 * np.sin(2 * np.pi * 1000.0 * t)
        iq = m * np.exp(2j * np.pi * 0.0 * t)
    elif mode == "USB":
        iq = 0.5 * np.exp(2j * np.pi * 3000.0 * t) * (
            0.7 + 0.3 * np.sin(2 * np.pi * 5.0 * t))
    else:
        m = np.sin(2 * np.pi * 1000.0 * t)
        ph = 2 * np.pi * 30000.0 * np.cumsum(m) / RF
        iq = 0.6 * np.exp(1j * ph)
    iq = iq + 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def load_or_synth(mode, nblocks, block):
    P = os.path.join(ROOT, "testdata",
                     {"WFM": "weak_775_10s.npy",
                      "AM": "sw_7300_10s.npy"}.get(mode, ""))
    if P and os.path.isfile(P):
        a = np.load(P)
        if a.dtype != np.uint8:
            a = np.asarray(a).reshape(-1)
            raw = np.empty(2 * len(a), dtype=np.uint8)
            raw[0::2] = np.clip(np.round(a.real * 127.5 + 127.5), 0, 255)
            raw[1::2] = np.clip(np.round(a.imag * 127.5 + 127.5), 0, 255)
            a = raw
        need = nblocks * block
        if len(a) < need:
            rep = int(np.ceil(need / len(a)))
            a = np.tile(a, rep)
        return np.ascontiguousarray(a[:need])
    return synth_iq(mode, (nblocks * block) // 2)


def run_case(mode, bm_all, nblocks, block):
    STATS.clear()
    del STACK[:]
    raw = load_or_synth(mode, nblocks, block)
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = True
    dsp.slow_agc_enabled = False
    if bm_all:
        dsp.black_magic_enabled = True
        dsp.bm_cyclo_enabled = True
        dsp.bm_rmt_enabled = True
        dsp.bm_sr_enabled = True
        dsp.bm_notch_enabled = True
    totals = []
    for k in range(nblocks):
        t0 = time.perf_counter()
        dsp.process(raw[k * block:(k + 1) * block], mode=mode)
        totals.append((time.perf_counter() - t0) * 1000.0)
    totals = np.array(totals[2:])
    rows = {s: dict(v) for s, v in STATS.items()}
    return totals, rows


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def main(argv):
    nblocks = 30
    if "--blocks" in argv:
        nblocks = int(argv[argv.index("--blocks") + 1])
    block = 132096
    install()
    modes = ("WFM", "NFM", "AM", "USB")
    if "--modes" in argv:
        modes = tuple(argv[argv.index("--modes") + 1].split(","))
    print(f"=== snapshot: {nblocks}blk/mode (warmup 2 excluded) ===")
    summary = {}
    for mode in modes:
        for bm in (False, True):
            tot, rows = run_case(mode, bm, nblocks, block)
            tag = f"{mode} bm={'ALL' if bm else 'off'}"
            p50, p95, p99 = pct(tot, 50), pct(tot, 95), pct(tot, 99)
            print(f"--- {tag}: total p50 {p50:.1f}/p95 {p95:.1f}/p99 {p99:.1f}ms")
            # 内訳 (排他ms/block、上位8)
            items = sorted(((s, v["excl"] / nblocks) for s, v in rows.items()),
                           key=lambda kv: -kv[1])[:8]
            for s, ms in items:
                n = rows[s]["n"] / nblocks
                print(f"    {s:18s} {ms:6.2f}ms/blk x{n:.1f}/blk")
            summary[tag] = {"p50": round(p50, 2), "p95": round(p95, 2),
                            "p99": round(p99, 2),
                            "top": [(s, round(ms, 3)) for s, ms in items]}
    if "--mem" in argv:
        # アロケーション計数は別走 (tracemalloc自体が遅いため計時と分離)
        tracemalloc.start()
        run_case("WFM", True, 6, block)
        snap = tracemalloc.take_snapshot()
        tracemalloc.stop()
        print("--- alloc top (WFM bm=ALL, 6blk) ---")
        tops = [st for st in snap.statistics("lineno")
                if "site-packages" not in st.traceback[0].filename][:8]
        for st in tops:
            fn = st.traceback[0].filename
            short = fn[fn.rfind("\\") + 1:] if "\\" in fn else fn.split("/")[-1]
            print(f"    {short}:{st.traceback[0].lineno} count={st.count} "
                  f"{st.size / 1e6:.1f}MB")
        summary["alloc_top"] = [
            {"loc": f"{st.traceback[0].filename}:"
                    f"{st.traceback[0].lineno}",
             "count": st.count, "size_mb": round(st.size / 1e6, 2)}
            for st in snap.statistics("filename")[:8]]
    import json
    out = os.path.join(ROOT, "docs", "profile_snapshot.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
