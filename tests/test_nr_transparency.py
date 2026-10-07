"""Strict NR transparency tests (no hardware required).

Transparency contract (audited 2026-10-07):
1. Clean strong field: NR on/off must be indistinguishable across 5-15kHz.
   (Regression: unconditional -1.5dB Side micro-mask >10kHz and -6dB@15kHz
   top diff-LPF used to collapse L-only HF separation 50->25dB at 10kHz,
   43->22dB at 12kHz, and Side 15k by -6dB even with NR "clean".)
2. Cognitive path (mono NR active) same.
3. Mild-hiss field: program content above the hiss floor must be retained
   (NR has to work on the noise, not on the program).
4. Noise-like program with no measured hiss must pass through clean field
   unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

RF = 1152000.0
FS = 48000
BLK = 132096


def _tone_raw(fl, fr, dev=22500.0, snr_db=None, dur=3.0, seed=99):
    n = int(dur * RF)
    t = np.arange(n) / RF
    l = 0.95 * np.sin(2 * np.pi * fl * t)
    r = 0.95 * np.sin(2 * np.pi * fr * t) if fr else np.zeros(n)
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * float(dev) * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p = 0.36 / 10 ** (float(snr_db) / 10.0)
        iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                      + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _run(raw, nr_on, cognitive=False, freeze_gain=True):
    d = SdrDspPipeline(1152000, FS)
    d.set_offset_freq(0.0)
    d.afc_enabled = False
    d.slow_agc_enabled = False
    d.filter_mode = "wide"
    d.sic_enabled = False  # NR audit isolated from SIC (covered elsewhere)
    d.set_stereo_nr(nr_on)
    if cognitive:
        d.cognitive_enabled = True
        d.target_cutoff_hz = 15000.0
        d.applied_cutoff_hz = 15000.0
        d.target_if_bw_hz = 145000.0
        d.applied_if_bw_hz = 145000.0
        d.target_hf_gain = 1.0
        d.hf_gain_applied = 1.0
    else:
        d.mono_nr.enabled = False
    if freeze_gain:
        d.stereo_diff_gain = 1.0
        d._diff_gain_eff = lambda: 1.0
    outs = []
    for k in range(len(raw) // BLK):
        a, _ = d.process(raw[k * BLK:(k + 1) * BLK], "WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
    y = np.concatenate(outs, axis=0)
    return d, y[len(y) // 3:]


def _tone_pow(x, f, fs=FS):
    n = len(x)
    W = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    i = int(round(f * n / fs))
    return float(W[max(0, i - 2):i + 3].sum())


def _metrics(y, f):
    L = y[:, 0].astype(np.float64)
    R = y[:, 1].astype(np.float64)
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    pl, pr = _tone_pow(L, f), _tone_pow(R, f)
    return {"sep": 10 * np.log10((pl + 1e-24) / (pr + 1e-24)),
            "side": 10 * np.log10(_tone_pow(S, f) + 1e-24),
            "mid": 10 * np.log10(_tone_pow(M, f) + 1e-24)}


def test_clean_strong_field_transparent():
    """クリーン強電界: NR on/offが5〜15kHzで一致すること。"""
    for f in (5000.0, 8000.0, 10000.0, 12000.0, 14000.0, 15000.0):
        raw = _tone_raw(f, 0.0)
        _, y_off = _run(raw, nr_on=False)
        _, y_on = _run(raw, nr_on=True)
        a, b = _metrics(y_off, f), _metrics(y_on, f)
        d_sep = abs(a["sep"] - b["sep"])
        d_side = abs(a["side"] - b["side"])
        d_mid = abs(a["mid"] - b["mid"])
        print(f"[*] {f / 1000:.0f}k clean: dsep={d_sep:.2f} "
              f"dSide={d_side:.3f} dMid={d_mid:.3f}")
        assert d_sep < 0.7, f"NR altered separation at {f:.0f}Hz"
        assert d_side < 0.2, f"NR altered Side at {f:.0f}Hz"
        assert d_mid < 0.1, f"NR altered Mid at {f:.0f}Hz"
    print("[OK] clean strong field transparent")


def test_cognitive_path_transparent():
    """認知経路 (mono NR込み出荷構成) も同様に透明であること。"""
    for f in (10000.0, 12000.0, 14000.0):
        raw = _tone_raw(f, 0.0)
        d_off, y_off = _run(raw, nr_on=False, cognitive=True)
        d_on, y_on = _run(raw, nr_on=True, cognitive=True)
        a, b = _metrics(y_off, f), _metrics(y_on, f)
        d_sep = abs(a["sep"] - b["sep"])
        d_side = abs(a["side"] - b["side"])
        print(f"[*] {f / 1000:.0f}k cog: dsep={d_sep:.2f} dSide={d_side:.3f}")
        assert d_sep < 0.5, f"cognitive NR altered separation at {f:.0f}Hz"
        assert d_side < 0.15, f"cognitive NR altered Side at {f:.0f}Hz"
    print("[OK] cognitive path transparent")


def test_weak_field_program_retained():
    """実ヒス下 (C/N15) でもヒス床より上の番組は保持されること。"""
    raw = _tone_raw(1000.0, 5000.0, snr_db=15.0)
    d_off, y_off = _run(raw, nr_on=False)
    d_on, y_on = _run(raw, nr_on=True)
    a, b = _metrics(y_off, 5000.0), _metrics(y_on, 5000.0)
    d_side = b["side"] - a["side"]
    m_off, m_on = _metrics(y_off, 1000.0), _metrics(y_on, 1000.0)
    d_mid = m_on["mid"] - m_off["mid"]
    print(f"[*] C/N15 program: dSide@5k={d_side:+.2f} dMid@1k={d_mid:+.2f} "
          f"cut={d_on._nr_cut_eff:.0f} nr_w={d_on._nr_s_w:.2f}")
    assert d_side > -0.7, "NR removed program Side tone under mild hiss"
    assert d_mid > -0.2, "NR removed program Mid tone under mild hiss"
    assert d_on._nr_cut_eff > 10000.0, "cut collapsed in mild hiss"
    print("[OK] weak-field program retained")


def test_modulated_program_clean_field_untouched():
    """変調された高域ステレオ番組はクリーン場でNRが触らないこと。"""
    n = int(3.0 * RF)
    t = np.arange(n) / RF
    hf = (0.3 * np.sin(2 * np.pi * 6000.0 * t)
          + 0.3 * np.sin(2 * np.pi * 10000.0 * t)
          + 0.25 * np.sin(2 * np.pi * 14000.0 * t))
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    l = hf * am * 0.7
    r = -l  # pure Side program (anti-phase)
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 22500.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    _, y_off = _run(raw, nr_on=False)
    d_on, y_on = _run(raw, nr_on=True)
    S_off = (y_off[:, 0].astype(np.float64) - y_off[:, 1]) * 0.5
    S_on = (y_on[:, 0].astype(np.float64) - y_on[:, 1]) * 0.5

    def band(x, lo, hi):
        nn = len(x)
        frq = np.fft.rfftfreq(nn, 1.0 / FS)
        W = np.abs(np.fft.rfft(x * np.hanning(nn))) ** 2
        return 10 * np.log10(float(W[(frq >= lo) & (frq <= hi)].sum()) + 1e-24)

    d_hf = band(S_on, 10000.0, 15000.0) - band(S_off, 10000.0, 15000.0)
    print(f"[*] modulated program clean: dSide10-15k={d_hf:+.2f} "
          f"cut={d_on._nr_cut_eff:.0f} nr_w={d_on._nr_s_w:.2f}")
    assert abs(d_hf) < 0.4, "NR touched modulated program without hiss"
    print("[OK] modulated program untouched in clean field")


def test_stationary_noise_program_bounded():
    """定常ノイズ様番組は物理的にヒスと不可分 (既知限界)。

    プローブ: ヒス計測ゼロでもNRはSide高域を抑圧し得る (実測: -2.5dB
    @cut 13.7k)。ここでは「壊さない」ことを境界で保証する:
    Side高域の損失は4dB未満、Mid低域 (番組の芯) は不変、全帯域モノラル化
    (gain→0) には至らないこと。"""
    n = int(3.0 * RF)
    t = np.arange(n) / RF
    rng = np.random.default_rng(5)
    fr = np.fft.rfftfreq(n, 1.0 / RF)
    lf = np.fft.rfft(rng.standard_normal(n))
    lf[np.abs(fr) > 15000.0] = 0
    lf[np.abs(fr) < 500.0] = 0
    prog = np.fft.irfft(lf, n=n)
    prog = prog / (np.max(np.abs(prog)) + 1e-9) * 0.7
    l = prog
    r = np.roll(prog, 129) * 0.8
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 22500.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    _, y_off = _run(raw, nr_on=False)
    d_on, y_on = _run(raw, nr_on=True)
    S_off = (y_off[:, 0].astype(np.float64) - y_off[:, 1]) * 0.5
    S_on = (y_on[:, 0].astype(np.float64) - y_on[:, 1]) * 0.5
    M_off = (y_off[:, 0].astype(np.float64) + y_off[:, 1]) * 0.5
    M_on = (y_on[:, 0].astype(np.float64) + y_on[:, 1]) * 0.5

    def band(x, lo, hi):
        nn = len(x)
        frq = np.fft.rfftfreq(nn, 1.0 / FS)
        W = np.abs(np.fft.rfft(x * np.hanning(nn))) ** 2
        return 10 * np.log10(float(W[(frq >= lo) & (frq <= hi)].sum()) + 1e-24)

    d_hf = band(S_on, 10000.0, 15000.0) - band(S_off, 10000.0, 15000.0)
    d_mid = band(M_on, 500.0, 3000.0) - band(M_off, 500.0, 3000.0)
    print(f"[*] stationary noise program: dSide10-15k={d_hf:+.2f} "
          f"dMid0.5-3k={d_mid:+.2f} cut={d_on._nr_cut_eff:.0f} "
          f"gain={d_on.stereo_nr_gain:.2f}")
    assert d_hf > -4.0, f"noise program Side over-suppressed ({d_hf:+.1f}dB)"
    assert abs(d_mid) < 0.3, "Mid core of noise program altered"
    assert d_on.stereo_nr_gain > 0.2, "noise program driven to mono"
    print("[OK] stationary noise program bounded (documented ambiguity)")


def test_hiss_recovery_releases_fast():
    """弱電界ヒス→強電界クリーン番組の切替でNR残留が速やかに消えること。

    透明性ゲート (帯域外15.5-16.4kにヒス証拠なし) が閉じたら、in-band HFを
    ヒスと誤認した残留をτ1.5/2.5sのゆっくり復帰で残してはならない。
    実測 (復帰加速の修正前): 15k帯域制限した独立L/R番組で Side 14-15.5k
    -3.3dB が残留 (cut 14.4k / _nr_s_w 0.49)。放送帯域 (15k) に無い帯域外
    証拠が無い以上、NRは無動作が透明性の要件。
    """
    n = int(5.0 * RF)
    t = np.arange(n) / RF

    def band_noise(seed):
        r = np.random.default_rng(seed)
        fr = np.fft.rfftfreq(n, 1.0 / RF)
        f = np.fft.rfft(r.standard_normal(n))
        f[np.abs(fr) > 15000.0] = 0
        f[np.abs(fr) < 300.0] = 0
        x = np.fft.irfft(f, n=n)
        return x / (np.max(np.abs(x)) + 1e-9) * 0.6

    l = band_noise(7)
    r = band_noise(8)
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 22500.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    # 前半2秒だけ弱電界 (C/N12) → 後半は強電界クリーン
    rng = np.random.default_rng(11)
    p = 0.36 / 10 ** (12.0 / 10.0)
    nz = np.sqrt(p / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    k = int(2.0 * RF)
    iq[:k] += nz[:k]
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)

    d_off, y_off = _run(raw, nr_on=False)
    d_on, y_on = _run(raw, nr_on=True)
    tail = 3 * FS // 2  # 後半クリーン区間の終端1.5秒
    S_off = (y_off[-tail:, 0].astype(np.float64) - y_off[-tail:, 1]) * 0.5
    S_on = (y_on[-tail:, 0].astype(np.float64) - y_on[-tail:, 1]) * 0.5

    def band(x, lo, hi):
        nn = len(x)
        frq = np.fft.rfftfreq(nn, 1.0 / FS)
        W = np.abs(np.fft.rfft(x * np.hanning(nn))) ** 2
        return 10 * np.log10(float(W[(frq >= lo) & (frq <= hi)].sum()) + 1e-24)

    d_hf = band(S_on, 10000.0, 15000.0) - band(S_off, 10000.0, 15000.0)
    print(f"[*] hiss->clean recovery: dSide10-15k={d_hf:+.2f} "
          f"cut={d_on._nr_cut_eff:.0f} gain={d_on.stereo_nr_gain:.3f} "
          f"nr_w={d_on._nr_s_w:.3f} gate={d_on._nr_hiss_gate:.3f}")
    assert d_on._nr_cut_eff > 14900.0, "NR cut did not reopen after hiss"
    assert d_on.stereo_nr_gain > 0.98, "NR gain did not release after hiss"
    assert abs(d_hf) < 0.5, "NR left residual suppression on clean program"
    print("[OK] hiss recovery releases fast")


def main() -> int:
    try:
        test_clean_strong_field_transparent()
        test_cognitive_path_transparent()
        test_weak_field_program_retained()
        test_modulated_program_clean_field_untouched()
        test_stationary_noise_program_bounded()
        test_hiss_recovery_releases_fast()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL NR TRANSPARENCY TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
