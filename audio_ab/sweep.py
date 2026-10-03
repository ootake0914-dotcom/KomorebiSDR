"""パラメータスイープランナー (audio_ab)。

これまでのスイープはtempスクリプトの書き捨てだった。本ツールは
「シナリオ×パラメータ軸」を宣言するだけで、
同一雑音での対比較・文単位CER・CI・実用ゲートを一括で回す。

Usage:
  python audio_ab/sweep.py --list
  python audio_ab/sweep.py --scenario ssb --axis nr.over_sub=1.0,1.5,2.0 \
      --utts 4 --seeds 1 --snrs 0,5
  python audio_ab/sweep.py --scenario ssb-clicks --base nr.on=0 \
      --axis ssb.blank_k=4,6,8,10 --snrs 15
"""

import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audio_ab"))

from cer_ab import cer, decode_trial  # noqa: E402
from corpus import FS, load as load_corpus  # noqa: E402
from stats import bootstrap_ci, paired, verdict  # noqa: E402

_ORIG_BLANK = [None]


def _nr_obj(dsp, ctx):
    from narrowband_nr import NarrowbandNr
    if getattr(dsp, "nbm_nr", None) is None:
        preset = {"NFM": "NFM", "CW": "CW"}.get(ctx.get("mode", ""), "SSB")
        dsp.nbm_nr = NarrowbandNr(preset)
    return dsp.nbm_nr


def _patch_blank_k(v):
    import dsp_nfm
    if _ORIG_BLANK[0] is None:
        _ORIG_BLANK[0] = dsp_nfm.blank_impulses_iq
    orig = _ORIG_BLANK[0]

    def wrapper(iq, thr_k=None, **kw):
        return orig(iq, thr_k=float(v), **kw)
    dsp_nfm.blank_impulses_iq = wrapper


HOOKS = {
    "nr.on": lambda d, v, c: setattr(d, "nbm_nr_enabled", bool(v)),
    "nr.over_sub": lambda d, v, c: _nr_obj(d, c).p.__setitem__("over_sub", float(v)),
    "nr.floor_db": lambda d, v, c: _nr_obj(d, c).p.__setitem__("floor_db", float(v)),
    "nr.gain_smooth": lambda d, v, c: _nr_obj(d, c).p.__setitem__("gain_smooth", float(v)),
    "nr.noise_beta": lambda d, v, c: _nr_obj(d, c).p.__setitem__("noise_beta", float(v)),
    "nr.dd_alpha": lambda d, v, c: _nr_obj(d, c).p.__setitem__("dd_alpha", float(v)),
    "ssb.blank_k": lambda d, v, c: _patch_blank_k(v),
    "am.on": lambda d, v, c: setattr(d, "am_sideband_enabled", bool(v)),
    "wfm.on": lambda d, v, c: d.set_stereo_nr(bool(v)),
    "wf.mask_scale": lambda d, v, c: setattr(
        d, "_wf_mask_offset_vec",
        (d._wf_mask_offset_vec * np.float32(v)).astype(np.float32)),
    "wf.gmin": lambda d, v, c: setattr(d, "_nr_gmin", float(v)),
    "agc.hyst": lambda d, v, c: setattr(d, "_agc_hyst_db", float(v)),
}
BASE_HOOK = {"SSB": "nr.on", "NFM": "nr.on", "CW": "nr.on",
             "AM": "am.on", "WFM": "wfm.on"}


def parse_pair(s):
    if "=" not in s:
        raise ValueError(f"expected name=value: {s}")
    k, v = s.split("=", 1)
    try:
        return k, float(v)
    except ValueError:
        return k, v


def main(argv):
    from dsp import SdrDspPipeline
    from score_noref import AsrSpotter
    from simulate import SCENARIOS, build as build_scenario

    if "--list" in argv:
        for k in HOOKS:
            print(k)
        return 0
    scenario = "ssb"
    snrs, utts_n, seeds_n = [0.0], 4, 1
    speakers = (3, 2, 8, 11)
    axes = {}
    base_over = {}
    out = os.path.join(ROOT, "audio_ab", "out")
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scenario" and i + 1 < len(argv):
            scenario = argv[i + 1]
            i += 2
        elif a == "--snrs" and i + 1 < len(argv):
            snrs = [float(x) for x in argv[i + 1].split(",")]
            i += 2
        elif a == "--utts" and i + 1 < len(argv):
            utts_n = int(argv[i + 1])
            i += 2
        elif a == "--seeds" and i + 1 < len(argv):
            seeds_n = int(argv[i + 1])
            i += 2
        elif a == "--speakers" and i + 1 < len(argv):
            speakers = tuple(int(x) for x in argv[i + 1].split(","))
            i += 2
        elif a == "--axis" and i + 1 < len(argv):
            k, vals = argv[i + 1].split("=", 1)
            axes[k] = [parse_pair(f"{k}={v}")[1] for v in vals.split(",")]
            i += 2
        elif a == "--base" and i + 1 < len(argv):
            k, v = parse_pair(argv[i + 1])
            base_over[k] = v
            i += 2
        elif a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        else:
            i += 1
    if scenario not in SCENARIOS:
        print(f"unknown scenario {scenario}; use --list")
        return 2
    for k in list(axes) + list(base_over):
        if k not in HOOKS:
            print(f"unknown hook {k}; use --list")
            return 2
    sc = SCENARIOS[scenario]
    ctx = {"mode": sc["mode"], "scenario": scenario}
    base_hook = BASE_HOOK.get(sc["mode"])

    # バリアント: base + 軸の直積
    variants = [("base", dict(base_over))]
    keys = list(axes)
    for combo in itertools.product(*[axes[k] for k in keys]):
        label = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in zip(keys, combo))
        over = dict(base_over)
        over.update(zip(keys, combo))
        variants.append((label, over))

    corpus = load_corpus(utts_n, speakers=speakers)
    n_utts = len(corpus)
    spotter = AsrSpotter("small")
    os.makedirs(out, exist_ok=True)

    trials = []
    for snr in snrs:
        for ui, utt in enumerate(corpus):
            pair = [corpus[ui], corpus[(ui + 1) % n_utts]] if sc["utts"] == 2 else [utt]
            for s in range(seeds_n):
                seed = 5000 + ui * 37 + s
                trials.append((snr, ui, seed, utt, build_scenario(scenario, pair, snr, seed)))
    print(f"scenario={scenario} variants={len(variants)} trials={len(trials)} "
          f"snrs={snrs}", flush=True)

    t00 = time.perf_counter()
    res = {}
    for label, over in variants:
        if _ORIG_BLANK[0] is not None:
            import dsp_nfm
            dsp_nfm.blank_impulses_iq = _ORIG_BLANK[0]
        cers = []
        for snr, ui, seed, utt, data in trials:
            d = SdrDspPipeline(1152000, FS)
            if base_hook:
                HOOKS[base_hook](d, True, ctx)
            for k, v in over.items():
                HOOKS[k](d, v, ctx)
            y = decode_trial(d, sc, data)
            c = cer(utt["text"], spotter.transcribe(y, FS))
            cers.append((snr, ui, seed, c))
        res[label] = cers
        m = bootstrap_ci([c for *_, c in cers], seed=11)
        print(f"  {label:28s} CER {m['mean']:.3f} "
              f"[{m['lo']:.3f},{m['hi']:.3f}] n={m['n']} "
              f"({time.perf_counter() - t00:.0f}s)", flush=True)

    base_c = np.array([c for *_, c in res["base"]])
    summary = {}
    print(f"\n{'variant':28s} {'CER':>6s} {'delta':>8s} {'CI':>18s} "
          f"{'p':>6s}  verdict")
    for label, _ in variants:
        arr = np.array([c for *_, c in res[label]])
        d = paired(base_c, arr, seed=13)
        v = verdict(d, min_effect=0.01)
        summary[label] = {"cer_mean": round(float(arr.mean()), 4),
                          "delta": {k: round(float(d[k]), 4)
                                    for k in ("mean", "lo", "hi", "p_sign")},
                          "significant": d["significant"], "verdict": v}
        print(f"{label:28s} {arr.mean():6.3f} {d['mean']:+8.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}] {d['p_sign']:6.3f}  {v}")

    path = os.path.join(out, f"sweep_{scenario}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scenario": scenario, "snrs": snrs, "axes": axes,
                   "base": base_over, "summary": summary,
                   "trials": {k: [{"snr": s, "utt": u, "seed": sd, "cer": c}
                                  for s, u, sd, c in v] for k, v in res.items()}},
                  f, ensure_ascii=False, indent=2)
    if _ORIG_BLANK[0] is not None:
        import dsp_nfm
        dsp_nfm.blank_impulses_iq = _ORIG_BLANK[0]
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
