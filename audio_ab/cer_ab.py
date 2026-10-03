"""CER了解度AB (audio_ab) v2: 文単位試行 × 統計ゲート。

旧版は16.8秒連結原稿で1条件1サンプル (量子化±0.014) だった。
本版は corpus の文一つを1試行とし、チャネルは simulate の標準
シナリオ、集計は stats のブートストラップCI＋符号検定で行う。
「1文字差」を有意と誤認しないための実用ゲート付き。

Usage:
  python audio_ab/cer_ab.py [--scenario ssb] [--snrs 0,5] [--utts 6]
                            [--seeds 2] [--speakers 3,2,8,11] [--mos]
                            [--out audio_ab/out] [--list]
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audio_ab"))

from corpus import FS, load as load_corpus, norm_ja  # noqa: E402
from metrics import evaluate  # noqa: E402
from stats import bootstrap_ci, paired, verdict  # noqa: E402


def cer(truth: str, hyp: str) -> float:
    a, b = norm_ja(truth), norm_ja(hyp)
    n, m = len(a), len(b)
    if n == 0:
        return 1.0 if m else 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        ndp = [i] + [0] * m
        for j in range(1, m + 1):
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1,
                         dp[j - 1] + (a[i - 1] != b[j - 1]))
        dp = ndp
    return dp[m] / n


def decode_trial(dsp, sc, data):
    bl = int(sc["block"])
    nb = len(data) // bl
    demod, args = sc["demod"]
    outs = []
    for k in range(nb):
        blk = data[k * bl:(k + 1) * bl]
        if demod == "process":
            a, _ = dsp.process(blk, *args)
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 2:
                # WFM: 左ch = mid+side。side処理 (NR) の影響を直接受ける。
                a = a[:, 0]
        else:
            a = getattr(dsp, demod)(blk, *args)
            a = np.asarray(a, dtype=np.float32).reshape(-1)
        outs.append(a)
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def apply_cond(dsp, sc, on: bool):
    kind, name = sc["enable"]
    if kind == "method":
        getattr(dsp, name)(bool(on))
    else:
        setattr(dsp, name, bool(on))


def main(argv):
    from simulate import SCENARIOS, build as build_scenario

    scenario = "ssb"
    snrs = [0.0, 5.0]
    utts_n = 6
    seeds_n = 2
    speakers = (3, 2, 8, 11)
    mos = False
    beam = 5
    use_cache = True
    ref_metrics = True
    out = os.path.join(ROOT, "audio_ab", "out")
    i = 0
    while i < len(argv):
        if argv[i] == "--list":
            for name in SCENARIOS:
                print(name)
            return 0
        if argv[i] == "--scenario" and i + 1 < len(argv):
            scenario = argv[i + 1]
            i += 2
        elif argv[i] == "--snrs" and i + 1 < len(argv):
            snrs = [float(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--utts" and i + 1 < len(argv):
            utts_n = int(argv[i + 1])
            i += 2
        elif argv[i] == "--seeds" and i + 1 < len(argv):
            seeds_n = int(argv[i + 1])
            i += 2
        elif argv[i] == "--speakers" and i + 1 < len(argv):
            speakers = tuple(int(x) for x in argv[i + 1].split(","))
            i += 2
        elif argv[i] == "--mos":
            mos = True
            i += 1
        elif argv[i] == "--beam" and i + 1 < len(argv):
            beam = int(argv[i + 1])
            i += 2
        elif argv[i] == "--no-cache":
            use_cache = False
            i += 1
        elif argv[i] == "--no-ref":
            ref_metrics = False
            i += 1
        elif argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        else:
            i += 1
    if scenario not in SCENARIOS:
        print(f"unknown scenario {scenario}; use --list")
        return 2
    sc = SCENARIOS[scenario]
    os.makedirs(out, exist_ok=True)

    from dsp import SdrDspPipeline
    from fast import CachedAsr

    corpus = load_corpus(utts_n, speakers=speakers)
    n_utts = len(corpus)
    spotter = CachedAsr("small", beam=beam, cache=use_cache)
    scoreq = dnsmos = None
    if mos:
        from score_noref import DnsmosScorer, MosScorer
        scoreq, dnsmos = MosScorer(), DnsmosScorer()

    print(f"scenario={scenario} mode={sc['mode']} utts={n_utts} "
          f"seeds={seeds_n} snrs={snrs}", flush=True)
    result = {"scenario": scenario, "mode": sc["mode"], "utts": n_utts,
              "seeds": seeds_n, "speakers": list(speakers), "snrs": {}}
    t00 = time.perf_counter()
    for snr in snrs:
        trials = []  # (utt_idx, seed, cer_off, cer_on, y_off, y_on)
        for ui, utt in enumerate(corpus):
            pair = [corpus[ui], corpus[(ui + 1) % n_utts]] if sc["utts"] == 2 else [utt]
            for s in range(seeds_n):
                seed = 1000 + ui * 37 + s
                data = build_scenario(scenario, pair, snr, seed)
                row = {"utt": ui, "seed": seed}
                outs = {}
                for tag, on in (("off", False), ("on", True)):
                    d = SdrDspPipeline(1152000, FS)
                    apply_cond(d, sc, on)
                    y = decode_trial(d, sc, data)
                    outs[tag] = y
                    row[f"cer_{tag}"] = round(cer(utt["text"],
                                                  spotter.transcribe(y, FS)), 4)
                if ref_metrics:
                    for tag in ("off", "on"):
                        try:
                            row[f"ref_{tag}"] = evaluate(utt["x"], outs[tag], FS)
                        except Exception as e:
                            row[f"ref_{tag}"] = {"err": str(e)}
                trials.append((row, outs))
            print(f"  snr={snr:g} utt={ui} done "
                  f"(off={trials[-1][0]['cer_off']} on={trials[-1][0]['cer_on']})",
                  flush=True)
        off = [t[0]["cer_off"] for t in trials]
        on = [t[0]["cer_on"] for t in trials]
        ci_off = bootstrap_ci(off, seed=1)
        ci_on = bootstrap_ci(on, seed=2)
        delta = paired(off, on, seed=3, tol=0.0)
        ref_summary = {}
        if ref_metrics:
            for m in ("si_sdr", "seg_snr", "stoi"):
                ov = [t[0].get("ref_off", {}).get(m) for t in trials]
                nv = [t[0].get("ref_on", {}).get(m) for t in trials]
                ov = [v for v in ov if isinstance(v, (int, float))]
                nv = [v for v in nv if isinstance(v, (int, float))]
                if len(ov) >= 2 and len(nv) == len(ov):
                    ref_summary[m] = {
                        "off": bootstrap_ci(ov, seed=4),
                        "on": bootstrap_ci(nv, seed=5),
                        "delta": paired(ov, nv, seed=6)}
        row = {
            "n_trials": len(trials),
            "off": ci_off, "on": ci_on, "delta_on_minus_off": delta,
            "verdict": verdict(delta, min_effect=0.01),
            "ref_metrics": ref_summary,
            "trials": [t[0] for t in trials],
        }
        if mos:
            for tag in ("off", "on"):
                sq, dm = [], []
                for _, outs in trials:
                    y = outs[tag]
                    sq.append(float(scoreq.score_array(y, FS)))
                    try:
                        dval = dnsmos.score_array(y, FS)
                        dm.append((float(dval["SIG"]), float(dval["BAK"]),
                                   float(dval["OVRL"])))
                    except Exception:
                        pass
                if sq:
                    row[f"scoreq_{tag}"] = round(float(np.mean(sq)), 4)
                if dm:
                    arr = np.asarray(dm)
                    row[f"p835_{tag}"] = {
                        "SIG": round(float(arr[:, 0].mean()), 3),
                        "BAK": round(float(arr[:, 1].mean()), 3),
                        "OVRL": round(float(arr[:, 2].mean()), 3)}
        result["snrs"][str(snr)] = row
        print(f"snr={snr:g}  off {ci_off['mean']:.3f} "
              f"[{ci_off['lo']:.3f},{ci_off['hi']:.3f}]  "
              f"on {ci_on['mean']:.3f} [{ci_on['lo']:.3f},{ci_on['hi']:.3f}]  "
              f"delta {delta['mean']:+.4f} "
              f"[{delta['lo']:+.4f},{delta['hi']:+.4f}] "
              f"p={delta['p_sign']:.3f} -> {row['verdict']}", flush=True)
        for m, s in ref_summary.items():
            print(f"       ref {m:8s} off {s['off']['mean']:+8.2f} "
                  f"on {s['on']['mean']:+8.2f} "
                  f"delta {s['delta']['mean']:+.3f} "
                  f"[{s['delta']['lo']:+.3f},{s['delta']['hi']:+.3f}]",
                  flush=True)
        if mos:
            print(f"       scoreq {row.get('scoreq_off')}->{row.get('scoreq_on')}"
                  f"  p835 {row.get('p835_off')}->{row.get('p835_on')}",
                  flush=True)

    path = os.path.join(out, f"cer_stats_{scenario}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {path} ({time.perf_counter() - t00:.0f}s)  {spotter.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
