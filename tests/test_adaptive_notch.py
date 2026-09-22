"""2-1: 適応ノッチキャンセラのテスト (合成信号のみ・実機不要)。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_notch import AdaptiveNotchCanceller

FS = 48000.0
N = 2752


def _hum(freqs=(50.0, 100.0, 150.0), amps=(0.02, 0.012, 0.006), seed=0, n=N * 4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2.0 * np.pi * f * t)
    return x + 0.002 * rng.standard_normal(n)


def _amp(x, freq):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f > freq - 8) & (f < freq + 8)
    return float(np.max(spec[m]) + 1e-12)


def test_hum_suppressed_tone_preserved():
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    nblk = 8
    t = np.arange(N * nblk) / FS
    prog = 0.05 * np.sin(2.0 * np.pi * 1000.0 * t)
    # 実機的なハム量 (番組-24dB)。dwell到達後の定常4ブロックで評価する
    x = (prog + _hum(amps=(0.003, 0.0018, 0.0009), seed=5, n=N * nblk)).astype(np.float32)
    outs = []
    for k in range(nblk):
        y, info = nc.process_mono(x[k * N:(k + 1) * N])
        outs.append(y)
    y = np.concatenate(outs)[4 * N:]
    xs = x[4 * N:]
    sup50 = 20 * np.log10(_amp(y, 50.0) / _amp(xs, 50.0))
    sup100 = 20 * np.log10(_amp(y, 100.0) / _amp(xs, 100.0))
    keep = 20 * np.log10(_amp(y, 1000.0) / _amp(xs, 1000.0))
    print(f"[*] hum抑制 50Hz {sup50:.1f}dB 100Hz {sup100:.1f}dB, 番組 {keep:+.2f}dB")
    assert sup50 <= -15.0, f"50Hz抑制不足: {sup50:.1f}"
    assert sup100 <= -10.0, f"100Hz抑制不足: {sup100:.1f}"
    assert abs(keep) <= 0.5, f"番組変動: {keep:+.2f}"
    assert set(info["lines"]) >= {50.0, 100.0}


def test_loud_hum_partial():
    # 過大ハムは部分除去に留めて番組を守る
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    nblk = 8
    t = np.arange(N * nblk) / FS
    prog = 0.05 * np.sin(2.0 * np.pi * 1000.0 * t)
    x = (prog + _hum(seed=6, n=N * nblk)).astype(np.float32)
    for k in range(nblk):
        y, info = nc.process_mono(x[k * N:(k + 1) * N])
    assert info["bypass_reason"] == "partial", f"部分除去にならない: {info}"
    y = np.asarray(y)
    keep = 20 * np.log10(_amp(y, 1000.0) / _amp(x[(nblk - 1) * N:], 1000.0))
    assert abs(keep) <= 1.0, f"番組変動: {keep:+.2f}"


def test_no_hum_bypass():
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    rng = np.random.default_rng(0)
    x = (0.05 * np.sin(2 * np.pi * 1000.0 * np.arange(N * 4) / FS)
         + 0.01 * rng.standard_normal(N * 4)).astype(np.float32)
    for k in range(4):
        y, info = nc.process_mono(x[k * N:(k + 1) * N])
    assert info["bypass_reason"] == "no-hum"
    assert np.array_equal(np.asarray(y), x[3 * N:4 * N])


def test_single_tone_not_eaten():
    # 音楽の単一100Hzトーンはハムと誤認しない (2本以上ルール)
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    t = np.arange(N * 6) / FS
    x = (0.05 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
    for k in range(6):
        y, info = nc.process_mono(x[k * N:(k + 1) * N])
    assert info["bypass_reason"] == "no-hum", f"単一トーン誤検出: {info}"
    assert np.array_equal(np.asarray(y), x[5 * N:6 * N])


def test_60hz_auto_select():
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    # EMA分離＋4tick投票のため10ブロック回す
    x = _hum(freqs=(60.0, 120.0, 180.0), amps=(0.02, 0.012, 0.006), n=N * 10)
    for k in range(10):
        y, info = nc.process_mono(x[k * N:(k + 1) * N])
    assert nc._base == 60.0, f"60Hz未選択: {nc._base}"
    assert 60.0 in info["lines"]


def test_nan_safe_and_clicks():
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    y, info = nc.process_mono(np.full(N, np.nan, dtype=np.float32))
    assert info["bypass_reason"] == "non-finite"
    x = _hum(n=N * 4)
    outs = [nc.process_mono(x[k * N:(k + 1) * N])[0] for k in range(4)]
    cat = np.concatenate([np.asarray(o) for o in outs])
    step = float(np.max(np.abs(np.diff(cat.astype(np.float64)))))
    print(f"[*] notch最大段差: {step:.5f}")
    assert step < 0.05, f"境界クリック疑い: {step}"


def test_stereo_common_mode():
    nc = AdaptiveNotchCanceller(sample_rate=FS)
    rng = np.random.default_rng(1)
    t = np.arange(N * 4) / FS
    hum = 0.02 * np.sin(2 * np.pi * 50.0 * t)
    l = (0.05 * np.sin(2 * np.pi * 1000.0 * t) + hum
         + 0.002 * rng.standard_normal(len(t))).astype(np.float32)
    r = (0.05 * np.sin(2 * np.pi * 1000.0 * t + 0.3) + hum
         + 0.002 * rng.standard_normal(len(t))).astype(np.float32)
    yl = yr = None
    for k in range(4):
        (yl, yr), info = nc.process_stereo(l[k * N:(k + 1) * N],
                                           r[k * N:(k + 1) * N])
    assert yl.shape == l[:N].shape and yr.shape == r[:N].shape
    sup = 20 * np.log10(_amp(np.asarray(yl), 50.0) / _amp(l, 50.0))
    assert sup <= -10.0, f"ステレオ抑制不足: {sup:.1f}"
    # ステレオ差が残る (片ch処理落ちなし)
    assert float(np.mean((np.asarray(yl) - np.asarray(yr)) ** 2)) > 0.0


def main() -> int:
    try:
        test_hum_suppressed_tone_preserved()
        print("[*] ハム抑制・番組保存 OK")
        test_loud_hum_partial()
        print("[*] 過大ハム部分除去 OK")
        test_no_hum_bypass()
        print("[*] 無ハム素通し OK")
        test_single_tone_not_eaten()
        print("[*] 単一トーン誤認防止 OK")
        test_60hz_auto_select()
        print("[*] 60Hz自動選択 OK")
        test_nan_safe_and_clicks()
        print("[*] NaN安全・無クリック OK")
        test_stereo_common_mode()
        print("[*] ステレオ同相 OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL ADAPTIVE NOTCH TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
