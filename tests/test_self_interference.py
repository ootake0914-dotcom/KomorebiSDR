"""
Unit tests for DigitalSelfInterferenceCanceller (In-Band Full Duplex SIC for SDR).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from adaptive_rf import DigitalSelfInterferenceCanceller


def test_single_spurious_cancellation():
    """
    PC/USBからアンテナに混入した強力な内部スプリアス (+25kHz) に対し、
    直交基底適応追従により 20dB 以上逆位相消去され、目的信号が無傷であることを検証。
    """
    fs = 288000.0
    f_spur = 25000.0  # +25kHz のPCクロック高調波スプリアス
    sic = DigitalSelfInterferenceCanceller(sample_rate=fs, mu=0.1)
    sic.set_spurious_frequencies([f_spur])

    f_mod = 1000.0
    n = 1024

    # 35フレーム連続ストリーム処理 (NLMS適応収束)
    for b in range(35):
        t = (b * n + np.arange(n, dtype=np.float64)) / fs
        phase_wanted = 1.2 * np.sin(2.0 * np.pi * f_mod * t)
        sig_wanted = 0.4 * np.exp(1j * phase_wanted).astype(np.complex64)
        spurious = 0.6 * np.exp(1j * (2.0 * np.pi * f_spur * t + 0.75)).astype(np.complex64)
        noisy_iq = sig_wanted + spurious
        clean_iq = sic.process(noisy_iq)

    # 処理後のスプリアス残留パワー測定 (25kHz FFTビン)
    fft_in = np.abs(np.fft.fft(noisy_iq))
    fft_out = np.abs(np.fft.fft(clean_iq))
    freqs = np.fft.fftfreq(len(noisy_iq), 1.0 / fs)
    idx_spur = int(np.argmin(np.abs(freqs - f_spur)))

    power_spur_in = fft_in[idx_spur] ** 2
    power_spur_out = fft_out[idx_spur] ** 2
    actual_cancellation_db = float(10.0 * np.log10(power_spur_in / power_spur_out))

    print(f"[*] Spurious cancellation at +25kHz: {actual_cancellation_db:.1f} dB (sic diagnostic: {sic.cancellation_db:.1f} dB)")
    # 20dB 以上のスプリアス消去
    assert actual_cancellation_db >= 20.0, f"Expected >=20dB cancellation, got {actual_cancellation_db:.1f} dB"

    # 目的信号の保持度 (相関係数)
    corr = float(np.abs(np.vdot(clean_iq, sig_wanted) / (np.linalg.norm(clean_iq) * np.linalg.norm(sig_wanted))))
    print(f"[*] Wanted signal correlation after SIC: {corr:.4f}")
    assert corr > 0.98, "Wanted signal was distorted by SIC"
    print("[OK] test_single_spurious_cancellation passed")


def test_auto_detect_spurious():
    """スプリアス周波数の自動検出 (Auto Detect) 機能を検証"""
    fs = 288000.0
    t = np.arange(2048, dtype=np.float64) / fs
    f_target_spur = 32000.0  # +32kHz スプリアス

    rng = np.random.RandomState(42)
    noise_floor = (0.05 * (rng.randn(len(t)) + 1j * rng.randn(len(t)))).astype(np.complex64)
    spur = 0.5 * np.exp(1j * (2.0 * np.pi * f_target_spur * t)).astype(np.complex64)
    iq_input = noise_floor + spur

    sic = DigitalSelfInterferenceCanceller(sample_rate=fs)
    sic.auto_detect_spurious(iq_input, n_fft=1024, prominence_db=15.0)

    print(f"[*] Auto-detected spurious frequencies: {sic.spurious_freqs}")
    assert len(sic.spurious_freqs) >= 1, "Failed to auto-detect prominent spurious tone"
    best_detected = sic.spurious_freqs[0]
    assert abs(best_detected - f_target_spur) <= (fs / 1024), f"Frequency mismatch: {best_detected} vs {f_target_spur}"
    print("[OK] test_auto_detect_spurious passed")


def test_clean_signal_transparency():
    """スプリアスが存在しないクリーン信号に対する完全素通し・無歪み検証"""
    fs = 288000.0
    t = np.arange(1024, dtype=np.float64) / fs
    clean_in = 0.5 * np.exp(1j * (2.0 * np.pi * 5000.0 * t)).astype(np.complex64)

    sic = DigitalSelfInterferenceCanceller(sample_rate=fs)
    # スプリアスリストが空の時は即座にバイパス
    out = sic.process(clean_in)
    assert np.allclose(out, clean_in), "Bypass failed when no spurious list is registered"
    print("[OK] test_clean_signal_transparency passed")


def test_pipeline_sic_integration():
    """
    SdrDspPipeline 全体を通じた SIC の自動検出・適応消去・素通し透明性の統合テスト。
    """
    from dsp import SdrDspPipeline

    pipeline = SdrDspPipeline(sample_rate=1152000, audio_rate=48000)
    assert hasattr(pipeline, "sic_canceller")
    assert pipeline.sic_enabled is True

    # 1. PC内部クロック高調波 (+45kHz) が混入した生IQ信号を作成
    fs_rf = 1152000.0
    n_rf = 115200  # 0.1秒分
    t = np.arange(n_rf, dtype=np.float64) / fs_rf

    # 目的FM信号 (1kHzトーン変調)
    f_mod = 1000.0
    phase_wanted = 1.0 * np.sin(2.0 * np.pi * f_mod * t)
    sig_wanted = 0.4 * np.exp(1j * phase_wanted).astype(np.complex64)

    # PCスプリアス (+45kHz の急峻なビート)
    f_spur = 45000.0
    spurious = 0.6 * np.exp(1j * (2.0 * np.pi * f_spur * t)).astype(np.complex64)
    noisy_iq = sig_wanted + spurious

    # uint8 生バイト列へ変換 (RTL-SDR形式: 0..255, 127.5中心)
    i_u8 = np.clip(np.real(noisy_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    q_u8 = np.clip(np.imag(noisy_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    raw_bytes = np.empty(n_rf * 2, dtype=np.uint8)
    raw_bytes[0::2] = i_u8
    raw_bytes[1::2] = q_u8

    # 複数ブロック処理して SIC を収束・検出させる
    for _ in range(5):
        audio, spec = pipeline.process(raw_bytes, mode="WFM")

    print(f"[*] Pipeline SIC detected spurious freqs: {pipeline.sic_detected_spurious}")
    print(f"[*] Pipeline SIC cancellation: {pipeline.sic_cancellation_db:.1f} dB")

    # +45kHz 近傍が検出されているか
    assert len(pipeline.sic_detected_spurious) >= 1, "Pipeline SIC failed to detect spurious tone"
    detected_f = pipeline.sic_detected_spurious[0]
    assert abs(detected_f - f_spur) < 2000.0, f"Detected spurious freq mismatch: {detected_f} vs {f_spur}"

    # 2. 延長ケーブルで離してスプリアスが消滅したケース（クリーン信号: アンテナ熱雑音フロアのみ存在）
    rng = np.random.RandomState(42)
    thermal_noise = (0.03 * (rng.randn(len(t)) + 1j * rng.randn(len(t)))).astype(np.complex64)
    clean_iq = sig_wanted + thermal_noise
    i_c = np.clip(np.real(clean_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    q_c = np.clip(np.imag(clean_iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    raw_clean = np.empty(n_rf * 2, dtype=np.uint8)
    raw_clean[0::2] = i_c
    raw_clean[1::2] = q_c

    # 選局リセット
    pipeline.set_offset_freq(0.0)
    for _ in range(5):
        audio_c, _ = pipeline.process(raw_clean, mode="WFM")

    print(f"[*] Clean signal spurious freqs after reset: {pipeline.sic_detected_spurious}")
    assert len(pipeline.sic_detected_spurious) == 0, "Spurious freqs should be empty for clean signal"
    print("[OK] test_pipeline_sic_integration passed")


def _offgrid_trial(f_spur, nblk=40, n=4096, redetect_at=()):
    """格子外スプリアスの検出＋消去量を測る共通部。"""
    fs = 288000.0
    sic = DigitalSelfInterferenceCanceller(sample_rate=fs, mu=0.08)
    outs = []
    for b in range(nblk):
        t = (b * n + np.arange(n, dtype=np.float64)) / fs
        iq = (0.4 * np.exp(1j * 1.2 * np.sin(2 * np.pi * 1000.0 * t))
              + 0.6 * np.exp(1j * (2 * np.pi * f_spur * t + 0.75))).astype(np.complex64)
        if b == 0 or b in redetect_at:
            sic.auto_detect_spurious(iq)
        outs.append(sic.process(iq))
    y = outs[-1]
    F = np.abs(np.fft.fft(y * np.hanning(n))) ** 2
    fr = np.fft.fftfreq(n, 1.0 / fs)
    i = int(np.argmin(np.abs(fr - f_spur)))
    tL = ((nblk - 1) * n + np.arange(n, dtype=np.float64)) / fs
    ref = np.abs(np.fft.fft(
        0.6 * np.exp(1j * (2 * np.pi * f_spur * tL + 0.75))
        * np.hanning(n))) ** 2
    lo, hi = max(0, i - 2), i + 3
    cancel = float(10.0 * np.log10(np.sum(ref[lo:hi]) / np.sum(F[lo:hi])))
    return sic, cancel


def test_offgrid_spurious_cancellation():
    """FFT格子から外れたスプリアス (+10Hz・半ビン) も消去できること。
    旧実装は格子量子化で消去量≈0dBだった (放物線補間＋追従で修復)。"""
    for f_spur in (28125.0, 28135.0, 28265.0):
        sic, cancel = _offgrid_trial(f_spur)
        print(f"[*] spur={f_spur:.0f}Hz detected={sic.spurious_freqs} "
              f"cancel={cancel:.1f} dB")
        assert abs(sic.spurious_freqs[0] - f_spur) < 140.0, \
            f"freq estimate off: {sic.spurious_freqs[0]} vs {f_spur}"
        assert cancel >= 15.0, \
            f"off-grid cancellation too weak: {cancel:.1f} dB @ {f_spur}Hz"
    print("[OK] test_offgrid_spurious_cancellation passed")


def test_redetect_weight_carryover():
    """0.8秒ごとの再検出で重みが維持され、消去が崩壊しないこと。
    旧実装は毎回0リセットで再収束過渡 (6.5dBまで低下) だった。"""
    sic0, c0 = _offgrid_trial(28125.0)
    sic1, c1 = _offgrid_trial(28125.0, redetect_at=(10, 20, 30))
    print(f"[*] no-redetect={c0:.1f} dB, with-redetect={c1:.1f} dB, "
          f"weights={sic1.weights}")
    assert abs(sic1.weights[0]) > 0.1, "weights were reset on re-detection"
    assert c1 >= 15.0, f"cancellation collapsed after re-detection: {c1:.1f} dB"
    print("[OK] test_redetect_weight_carryover passed")


def test_pilot_exclusion():
    """WFM無音＋パイロットのみのFMで、パイロット由来線 (±19/38/57/76kHzの
    ベッセル側波帯) をPCノイズと誤検出しないこと。対称ペア拒否 (±f が
    両方突出する候補は搬送波変調積) により除外リスト無しでも弾かれ、
    exclude_bands はマルチパス等で対称性が崩れた場合の保険として機能する。
    真のスプリアス (±片側単独) は検出される。"""
    fs = 288000.0
    n = int(0.5 * fs)
    t = np.arange(n, dtype=np.float64) / fs
    mpx = 0.09 * np.sin(2 * np.pi * 19000.0 * t)
    ph = 2 * np.pi * 75000.0 * np.cumsum(mpx) / fs
    iq = (0.6 * np.exp(1j * ph)).astype(np.complex64)
    excl = [(19000.0, 1500.0), (38000.0, 1500.0),
            (57000.0, 1500.0), (76000.0, 1500.0)]
    s0 = DigitalSelfInterferenceCanceller(sample_rate=fs)
    s0.auto_detect_spurious(iq)
    assert not s0.spurious_freqs, \
        f"symmetric modulation comb adopted without exclusion {s0.spurious_freqs}"
    s1 = DigitalSelfInterferenceCanceller(sample_rate=fs)
    s1.auto_detect_spurious(iq, exclude_bands=excl)
    bad = [f for f in s1.spurious_freqs
           if any(abs(abs(f) - c) < hw for (c, hw) in excl)]
    assert not bad, f"pilot falsely detected with exclusion: {s1.spurious_freqs}"
    spur = 0.4 * np.exp(1j * 2 * np.pi * 25000.0 * t)
    s2 = DigitalSelfInterferenceCanceller(sample_rate=fs)
    s2.auto_detect_spurious((0.6 * np.exp(1j * ph) + spur).astype(np.complex64),
                            exclude_bands=excl)
    assert any(abs(f - 25000.0) < 1500.0 for f in s2.spurious_freqs), \
        f"real spur missed with exclusion: {s2.spurious_freqs}"
    print(f"[*] no-excl={s0.spurious_freqs} excl={s1.spurious_freqs} "
          f"spur+excl={s2.spurious_freqs}")
    print("[OK] test_pilot_exclusion passed")


def _sparse_tone_raw(f, dev, snr_db=40.0, dur=4.0):
    """高域トーン (疎なベッセル線) のFM生IQ。実ノイズ付き。"""
    rf = 1152000.0
    n = int(dur * rf)
    t = np.arange(n) / rf
    l = 0.95 * np.sin(2 * np.pi * f * t)
    mpx = (0.45 * l + 0.45 * l * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * float(dev) * np.cumsum(mpx) / rf
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    rng = np.random.default_rng(99)
    p = 0.36 / 10 ** (float(snr_db) / 10.0)
    iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def test_sparse_program_not_adopted():
    """出荷既定 (sic_enabled=True) で疎な高域トーン番組の側波帯を
    スプリアス誤採用しないこと (回帰: ±8k/±16kを消去しSideが消失)。"""
    from dsp import SdrDspPipeline
    raw = _sparse_tone_raw(8000.0, 22500.0, snr_db=40.0, dur=1.0)
    d = SdrDspPipeline(1152000, 48000)
    d.set_offset_freq(0.0)
    d.afc_enabled = False
    d.cognitive_enabled = False
    d.slow_agc_enabled = False
    assert d.sic_enabled is True
    blk = 132096
    for k in range(min(3, len(raw) // blk)):
        d.process(raw[k * blk:(k + 1) * blk], "WFM")
    got = list(d.sic_detected_spurious)
    print(f"[*] adopted spurious (must be empty): {got}")
    assert not any(6000.0 < abs(f) < 26000.0 for f in got), \
        f"program sidebands adopted as spurious: {got}"


def _lonly_tone_sep(f, dev, sic_on, dur=2.0):
    """L-onlyトーンFM (パイロット+DSB含む) のSIC経路込み分離度。"""
    from dsp import SdrDspPipeline
    raw = _sparse_tone_raw(f, dev, snr_db=50.0, dur=dur)
    d = SdrDspPipeline(1152000, 48000)
    d.set_offset_freq(0.0)
    d.afc_enabled = False
    d.cognitive_enabled = False
    d.slow_agc_enabled = False
    d.filter_mode = "wide"
    d.set_stereo_nr(False)
    d.sic_enabled = bool(sic_on)
    blk = 132096
    outs = []
    for k in range(len(raw) // blk):
        a, _ = d.process(raw[k * blk:(k + 1) * blk], "WFM")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        outs.append(a)
    y = np.concatenate(outs, axis=0)
    y = y[len(y) // 4:]
    L = y[:, 0].astype(np.float64)
    R = y[:, 1].astype(np.float64)
    n = len(L)
    WL = np.abs(np.fft.rfft(L * np.hanning(n))) ** 2
    WR = np.abs(np.fft.rfft(R * np.hanning(n))) ** 2
    i = int(round(f * n / 48000.0))
    return float(-20.0 * np.log10(
        np.sqrt(np.sum(WR[max(0, i - 2):i + 3])
                / (np.sum(WL[max(0, i - 2):i + 3]) + 1e-24)) + 1e-12))


def test_dsb_sideband_not_adopted():
    """38kHz周りのDSB側波帯 (38k±f) を誤採用しないこと。

    回帰: 10kHz/dev22.5k→−48k採用で分離度 -22.7dB、12kHz→−26k採用で
    -25.1dB。0Hz対称でも整数倍でもないため第3規則 (DSB対称) が必要。
    採用ゼロ、かつSIC on/offの分離度が一致することを確認。
    """
    for f, dev in ((10000.0, 22500.0), (12000.0, 22500.0)):
        sep_on = _lonly_tone_sep(f, dev, True)
        sep_off = _lonly_tone_sep(f, dev, False)
        print(f"[*] {f / 1000:.0f}k dev={dev / 1000:.1f}k: "
              f"SICon={sep_on:.2f} SICoff={sep_off:.2f}")
        assert sep_off - sep_on < 1.5, \
            f"DSB sideband adopted ({f:.0f}Hz): {sep_on:.2f} vs {sep_off:.2f}"


if __name__ == "__main__":
    print("===== Running Digital SIC Tests =====")
    test_single_spurious_cancellation()
    test_auto_detect_spurious()
    test_clean_signal_transparency()
    test_pipeline_sic_integration()
    test_offgrid_spurious_cancellation()
    test_redetect_weight_carryover()
    test_pilot_exclusion()
    test_sparse_program_not_adopted()
    test_dsb_sideband_not_adopted()
    print("ALL DIGITAL SIC TESTS PASSED!")
