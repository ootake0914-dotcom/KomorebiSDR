"""Whole-system transparency test (no hardware required).

Contract: in clean conditions (high C/N, no multipath, no interference),
the shipped cognitive path must be indistinguishable from the same path
with all condition-dependent subsystems disabled (NR, multipath, ACI,
mono NR, BSS). Audited subsystems:
  stereo NR (incl. STFT Wiener + diff LPF), mono NR, BSS, multipath
  narrowing, ACI guard, SIC.
Calibration stages (inv-sinc, diff-gain deviation tracking, deemphasis,
trim servo, DC servo) are always-on by design and excluded by keeping
them on in both runs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

RF = 1152000.0
FS = 48000
BLK = 132096


def _to_raw(iq):
    n = len(iq)
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _fm_raw(l, r):
    n = len(l)
    t = np.arange(n) / RF
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    return _to_raw(0.6 * np.exp(1j * ph).astype(np.complex64))


def _tonal_music(dur=3.0, seed=3):
    n = int(dur * RF)
    t = np.arange(n) / RF
    rng = np.random.default_rng(seed)
    lf = [220, 440, 660, 1320, 2640, 5280, 7920, 10560, 13200]
    rfq = [330, 550, 880, 1760, 3520, 6160, 8800, 11880, 14080]
    l = sum((0.12 + 0.05 * rng.random()) *
            np.sin(2 * np.pi * f * t + rng.random() * 6.28) for f in lf)
    r = sum((0.12 + 0.05 * rng.random()) *
            np.sin(2 * np.pi * f * t + rng.random() * 6.28) for f in rfq)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
    return _fm_raw(l * am * 0.66, r * am * 0.66)


def _independent_noise(dur=3.0, seed=21):
    n = int(dur * RF)
    fr = np.fft.rfftfreq(n, 1.0 / RF)
    outs = []
    for sd in (seed, seed + 1):
        r2 = np.random.default_rng(sd)
        X = np.fft.rfft(r2.standard_normal(n))
        X[(np.abs(fr) < 300.0) | (np.abs(fr) > 15000.0)] = 0
        x = np.fft.irfft(X, n=n)
        outs.append(x / (np.max(np.abs(x)) + 1e-9) * 0.7)
    return _fm_raw(outs[0], outs[1])


def _make(cognitive=True, adaptive_off=False):
    d = SdrDspPipeline(1152000, FS)
    d.set_offset_freq(0.0)
    d.afc_enabled = False
    d.slow_agc_enabled = False
    d.filter_mode = "wide"
    if cognitive:
        d.cognitive_enabled = True
        d.target_cutoff_hz = 15000.0
        d.applied_cutoff_hz = 15000.0
        d.target_if_bw_hz = 145000.0
        d.applied_if_bw_hz = 145000.0
        d.target_hf_gain = 1.0
        d.hf_gain_applied = 1.0
    if adaptive_off:
        d.multipath_enabled = False
        d.aci_depth = 0.0
        d.set_stereo_nr(False)
        d.mono_nr.enabled = False
        d.bss_separator.enabled = False
    return d


def _decode(d, raw):
    outs = []
    for k in range(len(raw) // BLK):
        a, _ = d.process(raw[k * BLK:(k + 1) * BLK], "WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
    return np.concatenate(outs, axis=0)


def _band(x, lo, hi):
    n = len(x)
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    W = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    return 10 * np.log10(float(W[(fr >= lo) & (fr <= hi)].sum()) + 1e-24)


def _check_program(raw, label, cog=True):
    d_def = _make(cognitive=cog)
    d_ref = _make(cognitive=cog, adaptive_off=True)
    y_def = _decode(d_def, raw)
    y_ref = _decode(d_ref, raw)
    n = min(len(y_def), len(y_ref))
    start = FS  # skip settling
    worst = 0.0
    for name, fn in (("Mid", lambda y: (y[:, 0].astype(np.float64)
                                        + y[:, 1]) * 0.5),
                     ("Side", lambda y: (y[:, 0].astype(np.float64)
                                         - y[:, 1]) * 0.5)):
        a = fn(y_def[start:n])
        b = fn(y_ref[start:n])
        for lo, hi in ((300, 3000), (3000, 8000), (8000, 15000)):
            dv = _band(a, lo, hi) - _band(b, lo, hi)
            worst = max(worst, abs(dv))
            print(f"[*] {label} {name} {lo}-{hi}: d={dv:+.2f} dB")
    print(f"[*] {label}: worst |d|={worst:.2f} dB")
    assert worst < 0.25, f"{label}: adaptive systems altered output ({worst:.2f}dB)"
    return d_def


def test_clean_tonal_music_transparent():
    d = _check_program(_tonal_music(), "tonal music")
    assert d.multipath_gain > 0.999
    assert d.aci_gain > 0.999
    assert d.stereo_nr_gain > 0.999
    assert d._nr_cut_eff >= 14900.0
    assert not list(d.sic_detected_spurious)
    assert d.is_stereo
    assert d._nr_hiss_gate < 0.05, "clean field must show no hiss evidence"
    print("[OK] tonal music system-transparent")


def test_resampler_bit_transparent_at_zero_drift():
    """ドリフト≈0では分数補間せず、入力をそのまま返す (ビット透過)。"""
    from dsp_resampler import AdaptiveDriftResampler
    r = AdaptiveDriftResampler()
    r.update_feedback(8.0)  # 不感帯中央 → ratio 1.0
    assert r.current_ratio == 1.0
    rng = np.random.default_rng(1)
    for stereo in (False, True):
        x = (rng.standard_normal((2752, 2) if stereo else 2752)
             * 0.5).astype(np.float32)
        y = r.process(x)
        assert np.array_equal(y, x), \
            f"resampler altered clean audio (stereo={stereo})"
    print("[OK] resampler bit-transparent at zero drift")


def test_bss_gate_transparent():
    """BSSはhiss_gate=0で遅延素通し (透明) になること。

    原理ゲート: 帯域外プローブがヒス無しを示す場では、BSSの抑圧を
    無効化して入力の24サンプル遅延コピーを返す (独立L/R番組の
    デコリレート成分をヒスと誤認しない)。"""
    from adaptive_stereo import SuperSpatialBssStereoSeparator
    rng = np.random.default_rng(3)
    n = 2752
    l = rng.standard_normal(n).astype(np.float64) * 0.3
    r = rng.standard_normal(n).astype(np.float64) * 0.3
    b = SuperSpatialBssStereoSeparator(sample_rate=48000.0)
    out_l, out_r = b.process(l.copy(), r.copy(), stereo_blend=1.0,
                             hiss_gate=0.0)
    d = int(b.delay)
    # 素通し (遅延d) との一致を確認 (履歴ゼロ初期なので末尾のみ有効)
    err = float(np.max(np.abs(out_l[d:] - l[:-d])))
    print(f"[*] BSS gate=0 passthrough max err={err:.2e} (delay={d})")
    assert err < 1e-6, f"BSS gate=0 not transparent ({err:.2e})"
    print("[OK] BSS gate transparency")


def test_clean_independent_noise_transparent():
    _check_program(_independent_noise(), "independent L/R noise")
    print("[OK] independent noise system-transparent")


def test_non_cognitive_path_transparent():
    _check_program(_tonal_music(seed=7), "tonal music (non-cog)",
                   cog=False)
    print("[OK] non-cognitive path system-transparent")


def test_demod_helpers_strong_field_noop():
    """強電界ではEKF/Riemann/TDAが一切呼ばれず、出力がビット一致すること。

    境界 (EKF C/N26-20 / Riemann 28-22 / TDA gate30) より十分上では
    重み0で完全無動作。サブシステムon/offでビット同一を要求する。"""
    raw = _tonal_music()
    calls = {"ekf": 0, "riemann": 0, "tda": 0}

    def run(disable):
        d = _make(cognitive=True)
        if disable:
            d.ekf_enabled = False
            d.riemann_demodulator.enabled = False
            d.tda_click.enabled = False
        else:
            _e = d.ekf_demod.demodulate

            def ekf_spy(x, _e=_e):
                calls["ekf"] += 1
                return _e(x)
            d.ekf_demod.demodulate = ekf_spy
            _r = d.riemann_demodulator.demodulate

            def r_spy(x, _r=_r):
                calls["riemann"] += 1
                return _r(x)
            d.riemann_demodulator.demodulate = r_spy
            _t = d.tda_click.process_with_mask

            def t_spy(x, _t=_t):
                calls["tda"] += 1
                return _t(x)
            d.tda_click.process_with_mask = t_spy
        return _decode(d, raw), d

    y_on, d_on = run(False)
    y_off, _ = run(True)
    print(f"[*] strong field: calls={calls} if_snr={d_on._if_snr_db:.1f} "
          f"bit-identical={np.array_equal(y_on, y_off)}")
    assert calls == {"ekf": 0, "riemann": 0, "tda": 0}, \
        f"demod helpers called in strong field: {calls}"
    assert np.array_equal(y_on, y_off), "strong-field output not bit-identical"
    print("[OK] EKF/Riemann/TDA strong-field no-op")


def test_ekf_mono_strong_noop():
    """EKFゲート条件 (blend<=0.05) が成立する強電界モノラルでも無動作。"""
    n = int(2.0 * RF)
    t = np.arange(n) / RF
    mono = 0.9 * np.sin(2 * np.pi * 1000.0 * t)
    ph = 2 * np.pi * 22500.0 * np.cumsum(mono) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    raw = _to_raw(iq)
    calls = [0]

    def run(ekf_on):
        d = _make(cognitive=False)
        d.ekf_enabled = ekf_on
        if ekf_on:
            _e = d.ekf_demod.demodulate

            def spy(x, _e=_e):
                calls[0] += 1
                return _e(x)
            d.ekf_demod.demodulate = spy
        outs = []
        for k in range(len(raw) // BLK):
            d._stereo_blend = 0.0  # モノラル時と同条件を強制
            a, _ = d.process(raw[k * BLK:(k + 1) * BLK], "WFM")
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 1:
                a = np.stack([a, a], axis=1)
            outs.append(a)
        return np.concatenate(outs), d

    y_on, d_on = run(True)
    y_off, _ = run(False)
    print(f"[*] mono strong: ekf_calls={calls[0]} "
          f"if_snr={d_on._if_snr_db:.1f} "
          f"bit-identical={np.array_equal(y_on, y_off)}")
    assert calls[0] == 0, "EKF ran in strong field"
    assert np.array_equal(y_on, y_off), "mono strong-field output altered"
    print("[OK] EKF mono strong-field no-op")


def test_ultra_squelch_strong_transparent():
    """ultra squelch有効でも強電界ではゲイン1.0のビット透過。"""
    raw = _tonal_music(seed=5)
    d_off = _make(cognitive=False)
    d_on = _make(cognitive=False)
    d_on.ultra_squelch.enabled = True
    y_off = _decode(d_off, raw)
    y_on = _decode(d_on, raw)
    print(f"[*] ultra on: gain={d_on.ultra_squelch.current_gain:.4f} "
          f"bit-identical={np.array_equal(y_on, y_off)}")
    assert d_on.ultra_squelch.current_gain > 0.999
    assert np.array_equal(y_on, y_off), "ultra squelch altered strong field"
    print("[OK] ultra squelch strong-field transparent")


def test_fir_audio_flat_dly_pure_delay():
    """最大帯域開放時の素通しFIRが完全な単一遅延 (リップル0) であること。"""
    d = _make(cognitive=True)
    h = np.asarray(d.fir_audio_flat_dly)
    idx = int(np.argmax(np.abs(h)))
    print(f"[*] flat_dly: taps={len(h)} peak_idx={idx} sum={h.sum():.4f}")
    assert abs(float(h.sum()) - 1.0) < 1e-6
    assert abs(float(np.abs(h).sum()) - 1.0) < 1e-6, "not a single tap"
    assert idx == (len(h) - 1) // 2, "delay tap not centered"
    print("[OK] flat delay is ripple-free")


def main() -> int:
    try:
        test_clean_tonal_music_transparent()
        test_clean_independent_noise_transparent()
        test_non_cognitive_path_transparent()
        test_resampler_bit_transparent_at_zero_drift()
        test_bss_gate_transparent()
        test_demod_helpers_strong_field_noop()
        test_ekf_mono_strong_noop()
        test_ultra_squelch_strong_transparent()
        test_fir_audio_flat_dly_pure_delay()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL SYSTEM TRANSPARENCY TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
