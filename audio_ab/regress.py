"""Golden regression (audio_ab)。

標準シナリオの指標を audio_ab/golden.json に保持し、現在値との
ドリフトを検出する。DSP指標は決定的で高速 (既定)、--full でCERも測る。
run_all.py の二値テストでは拾えない「静かな悪化」(例: 抑圧量が1dB変わる)
を検出するのが目的。

Usage:
  python audio_ab/regress.py --update [--full]   # 基準を更新
  python audio_ab/regress.py [--full]            # 検査 (ドリフトでexit 1)
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audio_ab"))

from corpus import FS, load as load_corpus  # noqa: E402

GOLDEN = os.path.join(ROOT, "audio_ab", "golden.json")
BLOCK_WFM, BLOCK_IF, BLOCK_SSB = 132096, 16512, 2208

TOLS = {
    "ssb_clean_blank_changed": 120.0,
    "ssb_click_resid_db": 0.8,
    "wfm_side_hiss_db": 0.8,
    "wfm_mid_err_db": 1.5,
    "am_sinr_gain_db": 1.0,
    "nr_ssb_delta_rms_db": 0.5,
    "ssb_cer_off_0db": 0.08,
    "ssb_cer_on_0db": 0.08,
}


def _decode(dsp, data, bl, demod, args=()):
    outs = []
    for k in range(len(data) // bl):
        if demod == "process":
            a, _ = dsp.process(data[k * bl:(k + 1) * bl], *args)
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 2:
                a = a[:, 0]
        else:
            a = getattr(dsp, demod)(data[k * bl:(k + 1) * bl], *args)
        outs.append(np.asarray(a, dtype=np.float32).reshape(-1))
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def _decode_stereo(dsp, data, bl):
    outs = []
    for k in range(len(data) // bl):
        a, _ = dsp.process(data[k * bl:(k + 1) * bl], "WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
    return np.concatenate(outs, axis=0)


def _band(x, lo, hi, n=2752):
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    sel = (fr >= lo) & (fr <= hi)
    vals = [float(np.mean(np.abs(np.fft.rfft(
        x[k * n:(k + 1) * n] * np.hanning(n)))[sel] ** 2))
        for k in range(len(x) // n)]
    return 10 * np.log10(np.mean(vals) + 1e-24)


def measure(full=False):
    from dsp import SdrDspPipeline
    from simulate import SCENARIOS, build, add_clicks

    m = {}
    utt = load_corpus(1)[0]
    utts2 = load_corpus(2)

    # 1) SSBクリーンでブランカ偽検出量 (thr_k=8)
    from dsp_filters import blank_impulses_iq
    iq = build("ssb", [utt], 40.0, seed=7)
    ch = 0
    for k in range(len(iq) // BLOCK_SSB):
        b = iq[k * BLOCK_SSB:(k + 1) * BLOCK_SSB]
        ch += int(np.count_nonzero(blank_impulses_iq(b, thr_k=8.0, max_width=8) != b))
    m["ssb_clean_blank_changed"] = float(ch)

    # 2) 既知位置クリックの残差 (クリーンIQに注入→ブランク→窓内誤差)
    from ssb_synth import ssb_usb_iq
    iq0 = ssb_usb_iq(np.asarray(utt["x"], dtype=np.float32))
    iq1 = add_clicks(iq0, FS, seed=71, rate_per_s=15.0, amp=12.0)
    iq2 = iq1.copy()
    for k in range(len(iq2) // BLOCK_SSB):
        b = iq2[k * BLOCK_SSB:(k + 1) * BLOCK_SSB]
        iq2[k * BLOCK_SSB:(k + 1) * BLOCK_SSB] = blank_impulses_iq(
            b, thr_k=8.0, max_width=8)
    diff = iq1 - iq2
    rms0 = float(np.sqrt(np.mean(np.abs(iq0) ** 2))) + 1e-18
    m["ssb_click_resid_db"] = float(
        20.0 * np.log10(float(np.sqrt(np.mean(np.abs(diff) ** 2))) / rms0 + 1e-12))

    # 3-4) WFMステレオ: sideヒス抑圧とmid透明性 (テストと同じ隔離設定)
    raw = build("wfm", utts2, 20.0, seed=99)

    def _wf_pipe(nr):
        d = SdrDspPipeline(1152000, FS)
        d.set_offset_freq(0.0)
        d.afc_enabled = False
        d.cognitive_enabled = False
        d.slow_agc_enabled = False
        d.set_stereo_nr(bool(nr))
        return d

    d_off = _wf_pipe(False)
    y0 = _decode_stereo(d_off, raw, BLOCK_WFM)
    d_on = _wf_pipe(True)
    y1 = _decode_stereo(d_on, raw, BLOCK_WFM)
    n = min(len(y0), len(y1))
    sid0 = (y0[:n, 0] - y0[:n, 1]) * 0.5
    sid1 = (y1[:n, 0] - y1[:n, 1]) * 0.5
    mid0 = (y0[:n, 0] + y0[:n, 1]) * 0.5
    mid1 = (y1[:n, 0] + y1[:n, 1]) * 0.5
    m["wfm_side_hiss_db"] = _band(sid1, 10000, 14000) - _band(sid0, 10000, 14000)
    # 遅延は仮定せず相互相関で合わせる (符号の取り違え防止)
    a = mid0 - mid0.mean()
    b = mid1 - mid1.mean()
    xc = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
    lag = int(np.argmax(xc))
    if lag > n:
        lag -= 2 * n
    if lag > 0:
        mid1a = np.concatenate((np.zeros(lag, dtype=np.float32), mid1))[:n]
    elif lag < 0:
        mid1a = np.concatenate((mid1[-lag:], np.zeros(-lag, dtype=np.float32)))[:n]
    else:
        mid1a = mid1
    rng = slice(FS, n - FS)
    m["wfm_mid_err_db"] = float(10.0 * np.log10(
        np.mean((mid1a[rng] - mid0[rng]) ** 2) / (np.mean(mid0[rng] ** 2) + 1e-24) + 1e-24))

    # 5) AM側波帯: 片側妨害のSINR利得 (自動発動)
    iq = build("am", [utt], 10.0, seed=77, intf_db=-10.0, gate_frac=0.35)
    d0 = SdrDspPipeline(1152000, FS)
    d0.am_sideband_enabled = False
    a0 = _decode(d0, iq, BLOCK_IF, "demodulate_am")
    d1 = SdrDspPipeline(1152000, FS)
    a1 = _decode(d1, iq, BLOCK_IF, "demodulate_am")

    def _am_sinr(y):
        tail = np.asarray(y[-2 * FS:], dtype=np.float64)
        win = tail * np.hanning(len(tail))
        P = np.abs(np.fft.rfft(win)) ** 2
        f = np.fft.rfftfreq(len(tail), 1.0 / FS)
        prog = float(np.sum(P[(f > 300) & (f < 3000)]))
        intf = float(np.sum(P[(f > 2455) & (f < 2515)]))
        return 10.0 * np.log10((prog + 1e-24) / (intf + 1e-24))
    m["am_sinr_gain_db"] = _am_sinr(a1) - _am_sinr(a0)

    # 6) 狭帯域NRの指纹: on/off出力RMS差 (0dB)
    iq = build("ssb", [utt], 0.0, seed=55)
    d0 = SdrDspPipeline(1152000, FS)
    b0 = _decode(d0, iq, BLOCK_SSB, "demodulate_ssb", ("USB",))
    d1 = SdrDspPipeline(1152000, FS)
    d1.nbm_nr_enabled = True
    b1 = _decode(d1, iq, BLOCK_SSB, "demodulate_ssb", ("USB",))
    nn = min(len(b0), len(b1))
    m["nr_ssb_delta_rms_db"] = 20.0 * np.log10(
        float(np.sqrt(np.mean(b1[:nn] ** 2))) /
        (float(np.sqrt(np.mean(b0[:nn] ** 2))) + 1e-18) + 1e-18)

    if full:
        from score_noref import AsrSpotter
        from cer_ab import cer
        spotter = AsrSpotter("small")
        utts3 = load_corpus(3)
        for tag, on in (("off", False), ("on", True)):
            cs = []
            for ui, u in enumerate(utts3):
                data = build("ssb", [u], 0.0, 7000 + ui)
                dd = SdrDspPipeline(1152000, FS)
                dd.nbm_nr_enabled = on
                y = _decode(dd, data, BLOCK_SSB, "demodulate_ssb", ("USB",))
                cs.append(cer(u["text"], spotter.transcribe(y, FS)))
            m[f"ssb_cer_{tag}_0db"] = float(np.mean(cs))
    return m


def main(argv):
    update = "--update" in argv
    full = "--full" in argv
    t0 = time.perf_counter()
    cur = measure(full=full)
    if update:
        gold = {"version": 1, "full": full,
                "metrics": {k: {"value": float(v), "tol": float(TOLS.get(k, 0.1))}
                            for k, v in cur.items()}}
        with open(GOLDEN, "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=2)
        print(f"golden updated ({GOLDEN}, {time.perf_counter() - t0:.0f}s)")
        for k, v in cur.items():
            print(f"  {k:26s} {v:10.4f}")
        return 0
    try:
        with open(GOLDEN, encoding="utf-8") as f:
            gold = json.load(f)
    except Exception:
        print("no golden.json; run --update first")
        return 2
    bad = []
    print(f"{'metric':26s} {'golden':>10s} {'current':>10s} {'delta':>9s} "
          f"{'tol':>6s}  status")
    for k, gv in gold["metrics"].items():
        if k not in cur:
            print(f"{k:26s} {'-':>10s} {'-':>10s} {'-':>9s} {'-':>6s}  skipped")
            continue
        v = float(cur[k])
        dlt = v - float(gv["value"])
        ok = abs(dlt) <= float(gv["tol"])
        if not ok:
            bad.append(k)
        print(f"{k:26s} {gv['value']:10.4f} {v:10.4f} {dlt:+9.4f} "
              f"{gv['tol']:6.2f}  {'OK' if ok else 'DRIFT'}")
    print(f"{'PASS' if not bad else 'DRIFT: ' + ','.join(bad)} "
          f"({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
