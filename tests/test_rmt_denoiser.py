"""B: RMT安全層のテスト (合成信号のみ・実機不要)。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmt_denoiser import SafeRmtDenoiser, strength_for_snr

FS = 48000.0
N = 2752  # 実オーディオブロック相当


def _noisy_tone(freq=1000.0, amp=0.05, noise=0.05, seed=0, n=N):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    clean = amp * np.sin(2.0 * np.pi * freq * t)
    return (clean + noise * rng.standard_normal(n)).astype(np.float32), clean


def _snr_vs_clean(out, clean):
    err = out.astype(np.float64) - clean.astype(np.float64)
    return float(10.0 * np.log10(np.mean(clean ** 2) / (np.mean(err ** 2) + 1e-18)))


def test_snr_not_degraded():
    dn = SafeRmtDenoiser(sample_rate=FS)
    x, clean = _noisy_tone()
    # 弱信号条件 (Sメータ-40dBFS、SNR15dB帯) で複数ブロック処理
    y = None
    for i in range(8):
        xi, _ = _noisy_tone(seed=i)
        y, info = dn.process_mono(xi, s_meter_dbfs=-40.0, snr_db=15.0)
    snr_in = _snr_vs_clean(x, clean)
    snr_out = _snr_vs_clean(y, _noisy_tone(seed=7)[1])
    print(f"[*] RMT SNR: in {snr_in:.2f} -> out {snr_out:.2f} dB")
    assert snr_out >= snr_in - 1.0, f"SNR悪化: {snr_in:.2f}->{snr_out:.2f}"
    assert info["retained_rank"] >= 0


def test_nan_safe():
    dn = SafeRmtDenoiser(sample_rate=FS)
    x = np.full(N, np.nan, dtype=np.float32)
    y, info = dn.process_mono(x, s_meter_dbfs=-40.0, snr_db=15.0)
    assert info["bypass_reason"] == "non-finite"
    assert y.shape == x.shape  # クラッシュせず形状維持
    xi = np.zeros(N, dtype=np.float32)
    xi[N // 2] = np.inf
    y2, info2 = dn.process_mono(xi, s_meter_dbfs=-40.0, snr_db=15.0)
    assert info2["bypass_reason"] == "non-finite"


def test_strong_signal_bypass():
    dn = SafeRmtDenoiser(sample_rate=FS)
    x, _ = _noisy_tone()
    y, info = dn.process_mono(x, s_meter_dbfs=-20.0, snr_db=35.0)
    assert info["bypass_reason"] == "strong-signal"
    assert np.array_equal(np.asarray(y), np.asarray(x))


def test_silence_not_amplified():
    dn = SafeRmtDenoiser(sample_rate=FS)
    x = np.zeros(N, dtype=np.float32)
    for _ in range(4):
        y, info = dn.process_mono(x, s_meter_dbfs=-40.0, snr_db=5.0)
    assert float(np.max(np.abs(y))) == 0.0
    assert info["bypass_reason"] == "silent"


def test_no_clicks_at_boundaries():
    dn = SafeRmtDenoiser(sample_rate=FS)
    t = np.arange(N * 4) / FS
    full = (0.3 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    outs = []
    for i in range(4):
        y, _ = dn.process_mono(full[i * N:(i + 1) * N], s_meter_dbfs=-40.0,
                               snr_db=15.0)
        outs.append(y)
    cat = np.concatenate(outs)
    step = np.max(np.abs(np.diff(cat.astype(np.float64))))
    print(f"[*] RMT 最大段差: {step:.5f}")
    assert step < 0.1, f"境界クリック疑い: {step}"


def test_cpu_budget_bypass():
    dn = SafeRmtDenoiser(sample_rate=FS, cpu_budget_percent=0.0)
    x, _ = _noisy_tone()
    y, info = dn.process_mono(x, s_meter_dbfs=-40.0, snr_db=15.0)
    assert info["bypass_reason"] == "cpu-budget"
    assert np.array_equal(np.asarray(y), np.asarray(x))


def test_stereo_mid_side():
    dn = SafeRmtDenoiser(sample_rate=FS)
    rng = np.random.default_rng(0)
    t = np.arange(N) / FS
    l = (0.05 * np.sin(2.0 * np.pi * 1000.0 * t) + 0.03 * rng.standard_normal(N)).astype(np.float32)
    r = (0.05 * np.sin(2.0 * np.pi * 1000.0 * t + 0.5) + 0.03 * rng.standard_normal(N)).astype(np.float32)
    (yl, yr), info = dn.process_stereo(l, r, s_meter_dbfs=-40.0, snr_db=12.0)
    assert yl.shape == l.shape and yr.shape == r.shape
    # ステレオが潰れていない (差信号エネルギーが残る)
    assert float(np.mean((yl - yr) ** 2)) > 0.0
    assert np.all(np.isfinite(yl)) and np.all(np.isfinite(yr))


def test_strength_map():
    assert strength_for_snr(35.0) <= 0.1
    assert 0.1 <= strength_for_snr(25.0) <= 0.35
    assert 0.35 <= strength_for_snr(15.0) <= 0.65
    assert 0.65 <= strength_for_snr(5.0) <= 0.85
    assert strength_for_snr(float("nan")) == 0.0


def main() -> int:
    try:
        test_snr_not_degraded()
        print("[*] SNR非悪化 OK")
        test_nan_safe()
        print("[*] NaN安全 OK")
        test_strong_signal_bypass()
        print("[*] 強信号バイパス OK")
        test_silence_not_amplified()
        print("[*] 無音非増幅 OK")
        test_no_clicks_at_boundaries()
        print("[*] 境界無クリック OK")
        test_cpu_budget_bypass()
        print("[*] CPU予算バイパス OK")
        test_stereo_mid_side()
        print("[*] Mid/Side OK")
        test_strength_map()
        print("[*] 強度マップ OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL RMT DENOISER TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
