"""測定基盤v2 (audio_ab) の回帰テスト。

VOICEVOX/Whisperのモデルロードなしで回る範囲を守る:
- stats: CI・符号検定・実用ゲートの数学
- simulate: 標準シナリオの型/長さ/レート (過去のNFMレート・AM複素キャスト
  バグの再発防止) と、既知トーンが復調に残ること
- fast: ディスクキャッシュのヒット/ミス (スタブASR)
- sweep: フックがパイプライン属性/プリセットへ正しく効くこと
- regress: goldenが現状と一致すること (決定的DSP指標, 約6s)
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_ab"))

import numpy as np  # noqa: E402

from dsp import SdrDspPipeline  # noqa: E402
from simulate import (SCENARIOS, build, add_clicks)  # noqa: E402
from stats import bootstrap_ci, paired, sign_test, verdict  # noqa: E402


def _tone(secs=2.0, f=1000.0, fs=48000):
    t = np.arange(int(secs * fs)) / fs
    return 0.5 * np.sin(2 * np.pi * f * t).astype(np.float32)


def _utt(x):
    return {"text": "", "x": np.asarray(x, dtype=np.float32),
            "speaker": 0, "index": 0}


def test_stats() -> bool:
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.2, 200)
    ci = bootstrap_ci(x, n_boot=2000, seed=1)
    ok1 = (ci["n"] == 200 and ci["lo"] < ci["mean"] < ci["hi"]
           and ci["hi"] - ci["lo"] < 0.2)
    p_all = sign_test([0.1] * 6)
    p_mix = sign_test([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    ok2 = abs(p_all - 0.03125) < 1e-9 and p_mix == 1.0
    a = rng.normal(1.0, 0.1, 50)
    d = paired(a, a + 0.5, n_boot=2000, seed=2)
    ok3 = d["significant"] and d["mean"] > 0.4 and verdict(d) == "regression"
    d2 = paired(a, a + 0.005, n_boot=2000, seed=3)
    ok4 = verdict(d2) == "no meaningful difference"
    d3 = {"mean": 0.05, "lo": 0.01, "hi": 0.09, "p_sign": 1.0,
          "significant": True}
    ok5 = verdict(d3) == "trend only (sign test n.s.)"
    from stats import holm, n_for_effect
    adj = holm([0.01, 0.02, 0.03, 0.04])
    ok6 = bool(np.allclose(adj, [0.04, 0.06, 0.06, 0.06]))
    ok7 = n_for_effect(0.1, 0.1) == 8 and n_for_effect(0.1, 0.2) < 8
    ok = bool(all((ok1, ok2, ok3, ok4, ok5, ok6, ok7)))
    print(f"[{'OK' if ok else 'FAIL'}] "
          f"stats (ci={ok1} sign={ok2} paired={ok3} gate={ok4} {ok5} "
          f"holm={ok6} power={ok7})")
    return ok


def test_scenarios_shape() -> bool:
    u = _utt(_tone())
    x = u["x"]
    iq = build("ssb", [u], 10.0, 1)
    ok1 = (iq.dtype == np.complex64 and abs(len(iq) - len(x)) < 2)
    cl = build("ssb-clicks", [u], 10.0, 1)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    ok2 = float(np.max(np.abs(cl))) > 4.0 * rms  # 12x rmsのクリック
    nf = build("nfm", [u], 10.0, 1)
    ok3 = abs(len(nf) - len(x) * 288000 / 48000) <= 2  # 288kレート (旧バグは4倍)
    am = build("am", [u], 10.0, 1)
    ok4 = (am.dtype == np.complex64
           and abs(len(am) - len(x) * 288000 / 48000) <= 2)
    # AM雑音が複素であること (旧バグ: .astype(float64)で虚部消失)
    z = _utt(np.zeros(24000, dtype=np.float32))
    am0 = build("am", [z], 10.0, 2, carrier_hz=0.0)
    nz = am0 - 1e-4
    ratio = float(np.std(nz.imag) / (np.std(nz.real) + 1e-30))
    ok5 = 0.3 < ratio < 3.0
    raw = build("wfm", [u, u], 20.0, 3)
    ok6 = (raw.dtype == np.uint8 and len(raw) % 2 == 0
           and abs(len(raw) // 2 - len(x) * 1152000 / 48000) <= 2)
    ok7 = True
    for name in ("ssb-fade", "ssb-step", "wfm-fade", "wfm-multipath",
                 "wfm-adjacent"):
        sc = SCENARIOS[name]
        d = build(name, [u, u], 10.0, 4)
        if sc["io"] == "raw":
            good = d.dtype == np.uint8 and len(d) % 2 == 0 and len(d) >= sc["block"]
        else:
            good = d.dtype == np.complex64 and len(d) >= sc["block"]
        ok7 = ok7 and bool(good)
    ok = all((ok1, ok2, ok3, ok4, ok5, ok6, ok7))
    print(f"[{'OK' if ok else 'FAIL'}] scenario shapes "
          f"(ssb={ok1} clicks={ok2} nfm288k={ok3} am={ok4} am_iq={ok5} "
          f"wfm={ok6} new={ok7}, am imag/real={ratio:.2f})")
    return bool(ok)


def test_demod_tone() -> bool:
    """NFM/SSBのレートが正しいと既知トーンが正しい周波数に残る。"""
    u = _utt(_tone(secs=2.0, f=1000.0))
    out = {}
    for name, demod in (("ssb", "demodulate_ssb"), ("nfm", "demodulate_nfm")):
        iq = build(name, [u], 20.0, 5)
        d = SdrDspPipeline(1152000, 48000)
        bl = SCENARIOS[name]["block"]
        ys = []
        for k in range(len(iq) // bl):
            a = getattr(d, demod)(iq[k * bl:(k + 1) * bl],
                                  *SCENARIOS[name]["demod"][1])
            ys.append(np.asarray(a, dtype=np.float32).reshape(-1))
        y = np.concatenate(ys)
        seg = y[48000:48000 + 48000] if len(y) > 96000 else y
        P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1.0 / 48000)
        band1k = float(np.sum(P[(f > 900) & (f < 1100)]))
        band0 = float(np.sum(P[(f > 200) & (f < 400)])
                      + np.sum(P[(f > 1600) & (f < 1800)]))
        out[name] = band1k / (band0 + 1e-18)
    ok = out["ssb"] > 10.0 and out["nfm"] > 5.0
    print(f"[{'OK' if ok else 'FAIL'}] tone lands at 1kHz "
          f"(ssb 1k/other={out['ssb']:.1f}, nfm={out['nfm']:.1f})")
    return bool(ok)


def test_am_auto_engage() -> bool:
    """片側妨害で側波帯合成が自動発動し、両側では発動しない。"""
    u = _utt(_tone(secs=2.0, f=1000.0))
    iq = build("am-onesided", [u], 10.0, 7)
    d = SdrDspPipeline(1152000, 48000)
    bl = 16512
    for k in range(len(iq) // bl):
        d.demodulate_am(iq[k * bl:(k + 1) * bl])
    w_one = float(getattr(d, "_am_sb_w", 0.0))
    iq2 = build("am", [u], 10.0, 7, intf_db=-10.0, sides="both")
    d2 = SdrDspPipeline(1152000, 48000)
    for k in range(len(iq2) // bl):
        d2.demodulate_am(iq2[k * bl:(k + 1) * bl])
    w_both = float(getattr(d2, "_am_sb_w", 0.0))
    ok = w_one > 0.9 and w_both == 0.0
    print(f"[{'OK' if ok else 'FAIL'}] AM auto engage "
          f"(one-sided w={w_one:.2f}, both w={w_both:.2f})")
    return bool(ok)


def test_fast_cache() -> bool:
    from fast import CachedAsr

    calls = []

    class Stub:
        def transcribe(self, x, sr, beam_size=5):
            calls.append(len(x))
            return "stub-text"

    tmp = tempfile.mkdtemp(prefix="asr_cache_")
    tmp2 = tempfile.mkdtemp(prefix="asr_evict_")
    try:
        asr = CachedAsr(spotter=Stub(), beam=5, cache=True, cache_dir=tmp)
        x = _tone(secs=0.5)
        t1 = asr.transcribe(x, 48000)
        t2 = asr.transcribe(x, 48000)
        ok = (t1 == t2 == "stub-text" and len(calls) == 1
              and asr.hits == 1 and asr.misses == 1)
        asr.cache = False
        asr.transcribe(x, 48000)
        ok = ok and len(calls) == 2
        for j in range(6):
            with open(os.path.join(tmp2, f"k{j}.txt"), "w") as f:
                f.write("x")
        CachedAsr(spotter=Stub(), cache=True, cache_dir=tmp2, max_files=3)
        ok_ev = len(os.listdir(tmp2)) <= 3
        ok = ok and ok_ev
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)
    print(f"[{'OK' if ok else 'FAIL'}] asr disk cache "
          f"(calls={len(calls)}, evict={ok_ev})")
    return bool(ok)


def test_sweep_hooks() -> bool:
    from sweep import HOOKS, parse_pair
    d = SdrDspPipeline(1152000, 48000)
    HOOKS["agc.hyst"](d, 0.7, {})
    HOOKS["am.on"](d, False, {})
    HOOKS["wf.gmin"](d, 0.03, {})
    HOOKS["wfm.on"](d, True, {"mode": "WFM"})
    ok1 = (d._agc_hyst_db == 0.7 and d.am_sideband_enabled is False
           and d._nr_gmin == 0.03 and d.stereo_nr_enabled)
    HOOKS["nr.over_sub"](d, 1.5, {"mode": "SSB"})
    HOOKS["nr.floor_db"](d, -18.0, {"mode": "SSB"})
    ok2 = (d.nbm_nr is not None and d.nbm_nr.p["over_sub"] == 1.5
           and d.nbm_nr.p["floor_db"] == -18.0)
    k, v = parse_pair("nr.floor_db=-18")
    ok3 = k == "nr.floor_db" and v == -18.0
    ok = bool(ok1 and ok2 and ok3)
    print(f"[{'OK' if ok else 'FAIL'}] sweep hooks "
          f"(attrs={ok1} nr={ok2} parse={ok3})")
    return ok


def test_reference_metrics() -> bool:
    from metrics import evaluate, seg_snr, shift, si_sdr, stoi_score
    from metrics import estoi_score, si_sar_sir
    rng = np.random.default_rng(0)
    t = np.arange(48000) / 48000.0
    ref = (0.4 * np.sin(2 * np.pi * 440 * t)
           + 0.2 * np.sin(2 * np.pi * 1200 * t)
           + 0.1 * rng.standard_normal(len(t))).astype(np.float32)
    ok1 = si_sdr(ref, ref) > 60.0 and stoi_score(ref, ref) > 0.99
    pad = np.zeros(600, dtype=np.float32)
    ref2 = np.concatenate([ref, pad])          # 末尾も伸ばして重なりを保つ
    delayed = np.concatenate([pad, ref2])      # 600サンプル遅れたコピー
    ok2 = si_sdr(ref2, delayed) > 60.0
    noisy = (ref + 0.25 * rng.standard_normal(len(ref))).astype(np.float32)
    ok3 = (si_sdr(ref, noisy) < 15.0
           and seg_snr(ref, noisy) < seg_snr(ref2, delayed))
    ev = evaluate(ref2, delayed)
    ok4 = (set(ev) == {"si_sdr", "seg_snr", "stoi"} and ev["stoi"] > 0.98)
    nz = 0.25 * rng.standard_normal(len(ref))
    noisy2 = ref + nz
    d3 = si_sar_sir(ref, noisy2, nz)
    ok5 = (abs(d3["sdr"] - si_sdr(ref, noisy2)) < 0.5
           and d3["sir"] < d3["sar"] - 5.0
           and si_sar_sir(ref, noisy2)["sir"] is None)
    ok6 = (estoi_score(ref, ref) > 0.99
           and estoi_score(ref, noisy) < estoi_score(ref2, delayed))
    ok = bool(ok1 and ok2 and ok3 and ok4 and ok5 and ok6)
    print(f"[{'OK' if ok else 'FAIL'}] reference metrics "
          f"(clean={ok1} delay={ok2} noisy={ok3} keys={ok4} "
          f"sir/sar={ok5} estoi={ok6})")
    return ok


def test_realdata_registry() -> bool:
    import realdata
    ok1 = ("fm_lucky60" in realdata.ITEMS
           and all("file" in v and "mode" in v
                   for v in realdata.ITEMS.values()))
    avail = realdata.available()
    ok2 = isinstance(avail, list) and set(avail) <= set(realdata.ITEMS)
    ok3 = True
    if "fm_lucky60" not in avail:
        try:
            realdata.load("fm_lucky60")
            ok3 = False  # 無いのに読めたらおかしい
        except FileNotFoundError:
            ok3 = True
    else:
        ok3 = bool(os.path.exists(realdata.path("fm_lucky60")))
    ok = bool(ok1 and ok2 and ok3)
    print(f"[{'OK' if ok else 'FAIL'}] realdata registry "
          f"(items={len(realdata.ITEMS)}, available={avail})")
    return ok


def test_provenance_and_loudness() -> bool:
    from provenance import stamp
    s = stamp()
    ok1 = all(k in s for k in ("date", "python", "numpy")) \
        and ("git_head" in s)
    from run_ab import _loudness_match
    from audiophile_dsp import LoudnessNormalizer

    def lufs(x):
        ln = LoudnessNormalizer(48000)
        v = None
        for k in range(0, len(x), 2752):
            r = ln.push(np.asarray(x[k:k + 2752], dtype=np.float64))
            if r is not None:
                v = r
        return v if v is not None else -99.0
    t = np.arange(int(48000 * 4)) / 48000.0
    prog = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 1200 * t)
    a = (prog * 0.5).astype(np.float32)
    b = (prog * 0.05).astype(np.float32)
    A, B = _loudness_match(a, b)
    ok2 = abs(lufs(A) - lufs(B)) < 0.5 and float(np.max(np.abs(A))) <= 1.0
    ok = bool(ok1 and ok2)
    print(f"[{'OK' if ok else 'FAIL'}] provenance+loudness "
          f"(stamp={ok1}, LUFS diff={abs(lufs(A) - lufs(B)):.2f})")
    return ok


def test_aes_scorer() -> bool:
    # 合成トーンではPQが逆転することがある (実測: noiseでPQ上昇)。
    # 実音声 (TTSキャッシュ) で感度を検証する。無い場合はSKIP。
    from score_noref import AesScorer
    try:
        from corpus import load as load_corpus
        u = load_corpus(1)[0]
        x = np.asarray(u["x"], dtype=np.float32)
    except Exception as e:
        print(f"[SKIP] aes scorer (corpus unavailable: {e})")
        return True
    try:
        sc = AesScorer()
        r1 = sc.score_array(x, 48000)
    except Exception as e:
        print(f"[SKIP] aes scorer (model unavailable: {e})")
        return True
    ok1 = (set(r1) == {"PQ", "CE", "CU", "PC"}
           and all(1.0 <= float(r1[k]) <= 10.0 for k in r1))
    r2 = sc.score_array(x, 48000)  # キャッシュヒットで同一
    ok2 = all(abs(float(r2[k]) - float(r1[k])) < 1e-9 for k in r1)
    nz = (x + np.random.default_rng(1).standard_normal(len(x)).astype(np.float32)
          * float(np.sqrt(np.mean(x ** 2)))).astype(np.float32)
    r3 = sc.score_array(nz, 48000)
    ok3 = float(r3["PQ"]) < float(r1["PQ"]) - 1.0
    ok = bool(ok1 and ok2 and ok3)
    print(f"[{'OK' if ok else 'FAIL'}] aes scorer "
          f"(keys={ok1} cache={ok2} noise<{ok3}, PQ={r1['PQ']:.2f}->{r3['PQ']:.2f})")
    return ok


def test_embed_dist() -> bool:
    # 小集合の分布距離: 同一集合≈0、0dB雑音集合は大きく離れる
    try:
        import embed_dist
        from corpus import load as load_corpus
        utts = load_corpus(1)
    except Exception as e:
        print(f"[SKIP] embed dist (unavailable: {e})")
        return True
    try:
        ref = [np.asarray(u["x"], dtype=np.float32) for u in utts]
        rng = np.random.default_rng(2)
        nz = [x + rng.standard_normal(len(x)).astype(np.float32)
              * float(np.sqrt(np.mean(x ** 2))) for x in ref]
        s_same = embed_dist.fad_sets(ref, ref, 48000)
        s_diff = embed_dist.fad_sets(ref, nz, 48000)
    except Exception as e:
        print(f"[SKIP] embed dist (model unavailable: {e})")
        return True
    ok = bool(s_same < 1.0 and s_diff > s_same + 10.0)
    print(f"[{'OK' if ok else 'FAIL'}] embed dist "
          f"(same={s_same:.2f} noisy={s_diff:.1f})")
    return ok


def test_regress_golden() -> bool:
    import regress
    ttsdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "audio_ab", "tts")
    need = [os.path.join(ttsdir, f) for f in ("u03_00.wav", "u02_01.wav")]
    if not all(os.path.exists(p) for p in need):
        print("[SKIP] golden regression (corpus cache missing; "
              "run cer_ab/regress once with VOICEVOX up)")
        return True
    if not os.path.exists(regress.GOLDEN):
        print("[FAIL] golden.json missing (run regress.py --update)")
        return False
    rc = regress.main([])
    ok = rc == 0
    print(f"[{'OK' if ok else 'FAIL'}] golden regression passes current tree")
    return bool(ok)


def main() -> int:
    ok = test_stats()
    ok &= test_scenarios_shape()
    ok &= test_demod_tone()
    ok &= test_am_auto_engage()
    ok &= test_fast_cache()
    ok &= test_sweep_hooks()
    ok &= test_reference_metrics()
    ok &= test_realdata_registry()
    ok &= test_provenance_and_loudness()
    ok &= test_aes_scorer()
    ok &= test_embed_dist()
    ok &= test_regress_golden()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
