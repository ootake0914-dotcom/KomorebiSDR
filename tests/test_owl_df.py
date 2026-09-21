"""Brutal tests for owl direction finding (2-antenna AoA + reflection separation).

前提: 単一ドングルでは空間サンプルが1点で到来方向は原理的に不可。
本テストは2ch合成IQでアルゴリズムを縛り、将来の2台目ドングルに備える。
先に全部Red (import不能) → owl_df.py実装でGreenにするTDD。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import owl_df

FS = 200000.0
FC = 80000000.0
LAM = 3e8 / FC
D = LAM / 2.0  # 半波長間隔


def fm_source(n, seed=0, dev_hz=30000.0, mod_f=1000.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    m = 0.5 * np.sin(2 * np.pi * mod_f * t)
    ph = 2 * np.pi * dev_hz * np.cumsum(m) / FS
    return np.exp(1j * ph).astype(np.complex64)


def array_snapshot(s, theta_deg, gain=1.0, cal_gain=1.0, cal_phase=0.0, delay_s=0.0, snr_db=None, seed=1):
    """2素子ULAのスナップショット。ch2にAoA位相＋較正誤差＋遅延を付与。"""
    phi = 2.0 * np.pi * D * np.sin(np.deg2rad(theta_deg)) / LAM
    n = len(s)
    shift = delay_s * FS
    k = np.arange(n)
    # サブサンプル遅延は周波数領域で付与 (e^{-j2πfτ}、τは秒。Hz×sampleの
    # 無次元ミスに注意: shiftはサンプル数なのでFSで割って秒に戻す)
    S = np.fft.fft(s)
    freqs = np.fft.fftfreq(n, 1.0 / FS)
    S = S * np.exp(-2j * np.pi * freqs * shift / FS)
    s_d = np.fft.ifft(S)
    ch1 = gain * np.asarray(s_d, dtype=np.complex128)
    ch2 = gain * cal_gain * np.exp(1j * (phi + cal_phase)) * s_d
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p = 10.0 ** (-snr_db / 10.0)
        ch1 = ch1 + np.sqrt(p / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        ch2 = ch2 + np.sqrt(p / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return ch1, ch2


def err_deg(est, true):
    return abs(float(est) - float(true))


def test_clean_aoa_sweep():
    print("\n===== test_clean_aoa_sweep =====")
    n = 40000
    s = fm_source(n)
    for th in (-60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0):
        ch1, ch2 = array_snapshot(s, th)
        est = owl_df.estimate_aoa(ch1, ch2, D, LAM)
        assert err_deg(est, th) < 1.0, f"th={th}: est={est:.2f}"
    print("[OK] clean sweep <1deg")


def test_direct_vs_reflection():
    print("\n===== test_direct_vs_reflection =====")
    n = 60000
    s = fm_source(n, mod_f=700.0)
    ch1d, ch2d = array_snapshot(s, 20.0, gain=1.0)
    ch1r, ch2r = array_snapshot(s, -35.0, gain=0.6, delay_s=3.0e-6)
    ch1, ch2 = ch1d + ch1r, ch2d + ch2r
    paths = owl_df.resolve_multipath(ch1, ch2, D, LAM, fs=FS)
    assert len(paths) >= 2, f"only {len(paths)} paths"
    angs = sorted([p["aoa_deg"] for p in paths[:2]])
    assert min(abs(a - 20.0) for a in angs) < 3.0, f"direct lost: {angs}"
    assert min(abs(a + 35.0) for a in angs) < 5.0, f"reflection lost: {angs}"
    direct = max(paths[:2], key=lambda p: p["power"])
    assert abs(direct["aoa_deg"] - 20.0) < 3.0, f"direct not strongest: {paths}"
    print(f"[OK] direct/reflection separated: {angs}")


def test_low_snr():
    print("\n===== test_low_snr =====")
    n = 60000
    s = fm_source(n)
    ch1, ch2 = array_snapshot(s, -15.0, snr_db=0.0)
    est = owl_df.estimate_aoa(ch1, ch2, D, LAM)
    assert err_deg(est, -15.0) < 5.0, f"0dB SNR: est={est:.2f}"
    print(f"[OK] 0dB SNR err={err_deg(est, -15.0):.2f}deg")


def test_miscalibration():
    print("\n===== test_miscalibration =====")
    # 正規手順: 既知ビーコン (0°) で静的誤差を較正 → 未知源 (25°) を測定。
    # 静的オフセットは単一源では到来方向と原理的に分離不能なため、
    # 較正なしでの分離を要求した旧テストは物理的に不当だった (修正)。
    n = 60000
    s = fm_source(n)
    cg, cp = 10.0 ** (2.0 / 20.0), np.deg2rad(20.0)
    c1, c2 = array_snapshot(s, 0.0, cal_gain=cg, cal_phase=cp)
    t = np.arange(n) / FS
    c2 = c2 * np.exp(1j * 2 * np.pi * 0.5 * t)  # 0.5Hzドリフト
    cal = owl_df.estimate_calibration(c1, c2, 0.0, D, LAM, fs=FS)
    m1, m2 = array_snapshot(s, 25.0, cal_gain=cg, cal_phase=cp)
    m2 = m2 * np.exp(1j * 2 * np.pi * 0.5 * (t + n / FS))  # ドリフト継続
    est = owl_df.estimate_aoa_calibrated(m1, m2, D, LAM, fs=FS, cal=cal,
                                         dt_since_cal=n / FS)
    assert err_deg(est, 25.0) < 5.0, f"miscalibrated: est={est:.2f}"
    print(f"[OK] miscalibrated err={err_deg(est, 25.0):.2f}deg")


def test_coherent_subsample_multipath():
    print("\n===== test_coherent_subsample_multipath =====")
    n = 60000
    s = fm_source(n, mod_f=1500.0)
    ch1d, ch2d = array_snapshot(s, 10.0, gain=1.0)
    # 0.3サンプル遅延のコヒーレント反射 (MUSIC殺し)
    ch1r, ch2r = array_snapshot(s, -40.0, gain=0.5, delay_s=0.3 / FS)
    ch1, ch2 = ch1d + ch1r, ch2d + ch2r
    est = owl_df.estimate_aoa_robust(ch1, ch2, D, LAM, fs=FS)
    assert err_deg(est, 10.0) < 5.0, f"coherent MP: est={est:.2f}"
    print(f"[OK] coherent multipath direct err={err_deg(est, 10.0):.2f}deg")


if __name__ == "__main__":
    test_clean_aoa_sweep()
    test_direct_vs_reflection()
    test_low_snr()
    test_miscalibration()
    test_coherent_subsample_multipath()
    print("\nALL OWL DF TESTS PASSED!")
