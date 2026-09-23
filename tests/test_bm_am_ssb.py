"""AM/NFM/SSB黒魔法横展開 (第2章#1): RMT/SR配線のAB。
弱局STOI非退行・強局透明・p99<40ms・既定OFF。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline
from tools.score_wav import stoi

FS = 48000
RF = 1152000
BLK = 132096


def _talk(seed=31, secs=2.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * secs)) / FS
    vow = sum(a * np.sin(2 * np.pi * f * t) for f, a in
              [(350.0, 0.5), (800.0, 0.4), (1500.0, 0.3), (2400.0, 0.2)])
    syl = (t % 0.25 < 0.18)
    cons = rng.standard_normal(len(t)) * ((t % 0.2) < 0.02) * 0.6
    x = (vow * (0.4 + 0.6 * syl) + cons).astype(np.float64)
    return ((x / (np.sqrt(np.mean(x ** 2)) + 1e-18)) * 0.2).astype(np.float32)


def _analytic(x):
    X = np.fft.rfft(x)
    H = np.zeros(len(x) // 2 + 1)
    H[0] = H[-1] = 1.0
    H[1:-1] = 2.0
    return np.fft.irfft(X * H, len(x))


def _to_u8(iq):
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _gt(mode, weak=True, seed=7):
    prog = _talk()
    n = int(len(prog) * RF / FS)
    t = np.arange(n) / RF
    m = np.interp(t, np.arange(len(prog)) / FS, prog.astype(np.float64))
    rng = np.random.default_rng(seed)
    if mode == "AM":
        iq = (0.5 + 0.4 * m) * np.exp(2j * np.pi * 0.0 * t)
    elif mode == "NFM":
        iq = 0.6 * np.exp(1j * 2 * np.pi * 3000.0 * np.cumsum(m) / RF)
    else:  # USB
        an = _analytic(m)
        iq = (an * np.exp(2j * np.pi * 1500.0 * t)).astype(np.complex128)
    if weak:
        iq = iq * 0.05  # s-meter約-26〜-30dBFS (RMT作動域)
        p = 0.0025 / (10 ** (8.0 / 10.0))
        iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                      + 1j * rng.standard_normal(n))
    else:
        iq = iq * 0.6
    return _to_u8(iq), prog


def _decode(raw, mode, bm=()):
    dsp = SdrDspPipeline(RF, FS)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    if bm:
        dsp.black_magic_enabled = True
        dsp.bm_rmt_enabled = "rmt" in bm or "all" in bm
        dsp.bm_sr_enabled = "sr" in bm or "all" in bm
        dsp.bm_notch_enabled = "notch" in bm or "all" in bm
    ch = []
    for k in range(len(raw) // BLK):
        audio, _ = dsp.process(raw[k * BLK:(k + 1) * BLK], mode=mode)
        a = np.asarray(audio, dtype=np.float64).reshape(-1)
        ch.append(a)
    return np.concatenate(ch)


def _align(ref, out):
    n = min(len(ref), len(out))
    r = ref[:n] - np.mean(ref[:n])
    o = out[:n] - np.mean(out[:n])
    if np.sum(o ** 2) < 1e-18 or np.sum(r ** 2) < 1e-18:
        return r, o
    corr = np.correlate(o, r, mode="full")
    lag = max(-480, min(480, int(np.argmax(corr)) - (n - 1)))
    if lag >= 0:
        return r[:n - lag], o[lag:]
    return r[-lag:], o[:n + lag]


def test_weak_no_regression():
    # RMT横展開は不採用のため、("rmt",) でもビット一致する (配線なし)。
    # SRは音声に触れない。将来の狭帯域tune時に再評価する。
    for mode in ("AM", "NFM", "USB"):
        raw, prog = _gt(mode, weak=True)
        off = _decode(raw, mode, ())
        on = _decode(raw, mode, ("rmt", "sr"))
        assert np.array_equal(off, on), (mode, "must be bit-identical")
        r0, o0 = _align(prog.astype(np.float64), off)
        print(f"{mode} weak: STOI={stoi(r0, o0):.3f} (identical)")
    print("weak no-regression OK")


def test_strong_transparent():
    for mode in ("AM", "NFM", "USB"):
        raw, prog = _gt(mode, weak=False)
        off = _decode(raw, mode, ())
        on = _decode(raw, mode, ("rmt", "sr"))
        r, o0 = _align(prog.astype(np.float64), off)
        _, o1 = _align(prog.astype(np.float64), on)
        p = stoi(o0, o1)
        print(f"{mode} strong: pseudo={p:.3f}")
        assert p >= 0.95, (mode, p)
    print("strong transparent OK")


def test_sr_probe_sane():
    raw, _ = _gt("AM", weak=True)
    dsp = SdrDspPipeline(RF, FS)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    dsp.black_magic_enabled = True
    dsp.bm_sr_enabled = True
    for k in range(len(raw) // BLK):
        audio, _ = dsp.process(raw[k * BLK:(k + 1) * BLK], mode="AM")
        assert np.all(np.isfinite(np.asarray(audio)))
    c = float(dsp.bm_sr_confidence)
    assert 0.0 <= c <= 1.0, c
    print(f"sr probe OK (conf={c:.2f})")


def test_p99_budget():
    import json as _j  # noqa
    for mode in ("AM", "NFM", "USB"):
        raw, _ = _gt(mode, weak=True)
        dsp = SdrDspPipeline(RF, FS)
        dsp.set_offset_freq(0.0)
        dsp.afc_enabled = False
        dsp.cognitive_enabled = True
        dsp.slow_agc_enabled = False
        dsp.black_magic_enabled = True
        dsp.bm_rmt_enabled = True
        dsp.bm_sr_enabled = True
        dsp.bm_notch_enabled = True
        ts = []
        for k in range(len(raw) // BLK):
            t0 = time.perf_counter()
            dsp.process(raw[k * BLK:(k + 1) * BLK], mode=mode)
            ts.append((time.perf_counter() - t0) * 1000.0)
        p99 = float(np.percentile(np.array(ts[2:]), 99))
        print(f"{mode} ALL p99={p99:.1f}ms")
        assert p99 < 40.0, (mode, p99)
    print("p99 OK")


def test_real_am_golden():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "testdata", "sw_7300_10s.npy")
    if not os.path.exists(p):
        print("real AM golden missing: skip")
        return
    raw = np.load(p)
    if raw.dtype != np.uint8:
        a = np.asarray(raw).reshape(-1)
        t = np.empty(2 * len(a), dtype=np.uint8)
        t[0::2] = np.clip(np.round(a.real * 127.5 + 127.5), 0, 255)
        t[1::2] = np.clip(np.round(a.imag * 127.5 + 127.5), 0, 255)
        raw = t
    off = _decode(raw, "AM", ())
    on = _decode(raw, "AM", ("rmt", "sr"))
    assert np.array_equal(off, on)
    print("real AM OK (bit-identical)")


if __name__ == "__main__":
    test_weak_no_regression()
    test_strong_transparent()
    test_sr_probe_sane()
    test_p99_budget()
    test_real_am_golden()
    print("ALL BM-AM-SSB TESTS PASSED!")
