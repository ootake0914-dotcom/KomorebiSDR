"""R128ラウドネスAGC (Phase 4): 局間均一±1LU・ポンピング低減・既定OFF。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from audiophile_dsp import LoudnessNormalizer
from dsp import SdrDspPipeline

FS = 48000
BLK = 2752


def _program(rms, seed=3, secs=8.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * secs)) / FS
    # 番組らしく振幅変調した多トーン＋無音区間
    m = (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)) * (t % 2.0 < 1.7)
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in
            [(300.0, 0.5), (800.0, 0.3), (2000.0, 0.2)])
    x = (x * m).astype(np.float64)
    x *= rms / (np.sqrt(np.mean(x ** 2)) + 1e-18)
    return x.astype(np.float32)


def _run(dsp, x):
    gains = []
    outs = []
    for k in range(0, len(x), BLK):
        y = dsp._slow_agc_level(x[k:k + BLK])
        outs.append(np.asarray(y, dtype=np.float64).reshape(-1))
        gains.append(float(dsp.slow_agc_gain))
    return np.concatenate(outs), np.array(gains)


def _lufs(x):
    ln = LoudnessNormalizer(FS)
    v = None
    for k in range(0, len(x), BLK):
        r = ln.push(x[k:k + BLK])
        if r is not None:
            v = r
    return v


def test_lufs_disabled_by_default():
    dsp = SdrDspPipeline(1152000, FS)
    assert dsp.lufs_agc_enabled is False
    print("default OFF OK")


def test_station_leveling():
    # 大音量局と小音量局 → 収束後の出力LUFS差が±1LU以内 (両方式)
    res = {}
    for mode in ("rms", "lufs"):
        outs = []
        for rms, seed in ((0.20, 3), (0.05, 4)):
            dsp = SdrDspPipeline(1152000, FS)
            dsp.slow_agc_enabled = True
            dsp.slow_agc_release = 1.0  # 試験短縮 (復元不要・局所品)
            dsp.slow_agc_attack = 0.5
            dsp.lufs_agc_enabled = (mode == "lufs")
            y, _ = _run(dsp, _program(rms, seed))
            outs.append(_lufs(y[len(y) // 2:]))
        res[mode] = abs(outs[0] - outs[1])
        print(f"{mode}: loud={outs[0]:.1f} quiet={outs[1]:.1f} diff={res[mode]:.2f}LU")
    assert res["lufs"] <= 1.0, res
    print(f"(参考: 旧RMS方式の局間差 {res['rms']:.2f}LU — LUFS化の意義)")
    print("leveling OK")


def test_pumping():
    # 断続番組: LUFS版のゲイン分散がRMS版以下 (ゲートで無音区間を無視)
    x = _program(0.15, seed=9, secs=10.0)
    var = {}
    for mode in ("rms", "lufs"):
        dsp = SdrDspPipeline(1152000, FS)
        dsp.slow_agc_enabled = True
        dsp.slow_agc_release = 1.0
        dsp.slow_agc_attack = 0.5
        dsp.lufs_agc_enabled = (mode == "lufs")
        _, gains = _run(dsp, x)
        var[mode] = float(np.std(gains[len(gains) // 3:]))
        print(f"{mode}: gain std={var[mode]:.4f}")
    assert var["lufs"] <= var["rms"] * 1.1, var
    print("pumping OK")


def test_silence_freeze():
    for mode in (False, True):
        dsp = SdrDspPipeline(1152000, FS)
        dsp.lufs_agc_enabled = mode
        g0 = float(dsp.slow_agc_gain)
        dsp._slow_agc_level(np.zeros(BLK, dtype=np.float32))
        assert float(dsp.slow_agc_gain) == g0, mode
    print("silence freeze OK")


if __name__ == "__main__":
    test_lufs_disabled_by_default()
    test_station_leveling()
    test_pumping()
    test_silence_freeze()
    print("ALL LUFS-AGC TESTS PASSED!")
