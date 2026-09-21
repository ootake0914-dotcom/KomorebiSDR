"""Tests for audio-quality enhancements (no hardware required).

- Frequency-dependent stereo blend (highs go mono first on weak signals)
- 38kHz carrier phase auto-trim (recovers separation under phase error)
- Slow AGC leveling (compresses station-to-station loudness gaps)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline

RF = 1152000
BLOCK = 132096
NBLK = 20
N = (BLOCK // 2) * NBLK


def make_raw(stereo=True, snr_db=None, seed=7, sub_phase_deg=0.0, depth=1.0, nblk=NBLK):
    n = (BLOCK // 2) * nblk
    t = np.arange(n) / RF
    left = np.sin(2 * np.pi * 1000.0 * t)
    right = np.sin(2 * np.pi * 5000.0 * t)
    if stereo:
        phi = float(np.deg2rad(sub_phase_deg))
        mpx = (0.45 * (left + right)
               + 0.45 * (left - right) * np.sin(2 * np.pi * 38000.0 * t + phi)
               + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    else:
        mpx = left + right
    mpx = depth * mpx / (np.max(np.abs(mpx)) + 1e-9)
    phase = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * phase)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p_noise = 0.36 / (10 ** (snr_db / 10.0))
        iq = iq + np.sqrt(p_noise / 2.0) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    return raw


def run_blocks(raw, nblk=NBLK, **kw):
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    for k, v in kw.items():
        setattr(dsp, k, v)
    chunks = []
    for k in range(nblk):
        audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        chunks.append(audio)
    return dsp, np.concatenate(chunks, axis=0)


def band_rms(x, lo, hi, sr=48000):
    seg = x[-2 * sr:]
    seg = seg - seg.mean()
    window = np.hanning(len(seg))
    power = np.abs(np.fft.rfft(seg * window)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs > lo) & (freqs < hi)
    return float(np.sqrt(np.mean(power[mask]) + 1e-18))


def amp(x, freq, sr=48000):
    seg = x[len(x) // 3: len(x) // 3 + sr]
    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs > freq - 60) & (freqs < freq + 60)
    return float(np.max(spectrum[mask]) + 1e-12)


def test_freq_dependent_blend():
    print("===== test_freq_dependent_blend =====")
    noisy = make_raw(stereo=True, snr_db=9.0)
    _, pcm_on = run_blocks(noisy, freq_blend_enabled=True)
    _, pcm_off = run_blocks(noisy, freq_blend_enabled=False)
    lr_on = pcm_on[:, 0] - pcm_on[:, 1]
    lr_off = pcm_off[:, 0] - pcm_off[:, 1]
    hi_on = band_rms(lr_on, 10000.0, 14000.0)
    hi_off = band_rms(lr_off, 10000.0, 14000.0)
    lo_on = band_rms(lr_on, 800.0, 1200.0)
    lo_off = band_rms(lr_off, 800.0, 1200.0)
    hi_cut = 20 * np.log10((hi_on + 1e-18) / (hi_off + 1e-18))
    lo_cut = 20 * np.log10((lo_on + 1e-18) / (lo_off + 1e-18))
    print(f"[*] high-band extra cut {hi_cut:+.1f} dB, low-band change {lo_cut:+.1f} dB")
    assert hi_cut < -1.5, f"highs should go mono first (got {hi_cut:+.1f} dB)"
    assert lo_cut > -1.0, f"lows should keep stereo (got {lo_cut:+.1f} dB)"
    # clean signal must stay within inaudible tolerance (nr_gain≈0.96 on
    # synthetic clean, so gentle action is correct behavior, not bit-identity)
    clean = make_raw(stereo=True)
    _, pcm_c_on = run_blocks(clean, freq_blend_enabled=True)
    _, pcm_c_off = run_blocks(clean, freq_blend_enabled=False)
    diff = float(np.max(np.abs(pcm_c_on - pcm_c_off)))
    print(f"[*] clean on/off max diff {diff:.2e}")
    assert diff < 0.05, "clean stereo audibly altered by freq blend"
    print("[OK] frequency-dependent blend")


def test_phase_trim():
    print("===== test_phase_trim =====")
    # 38kHz副搬送波に+12°の位相誤差を持たせたMPX (80ブロック分生成)
    raw = make_raw(stereo=True, sub_phase_deg=12.0, nblk=80)
    dsp, pcm = run_blocks(raw, nblk=80, stereo_trim_enabled=True)
    trim_deg = float(np.rad2deg(dsp.stereo_phase_offset))
    print(f"[*] trim converged to {trim_deg:+.1f} deg (expect ≈ +12 deg)")
    assert abs(trim_deg - 12.0) < 3.0, f"trim did not track phase error ({trim_deg:+.1f})"
    left, right = pcm[:, 0], pcm[:, 1]
    xt_lr = 20 * np.log10(amp(right, 1000.0) / amp(left, 1000.0))
    xt_rl = 20 * np.log10(amp(left, 5000.0) / amp(right, 5000.0))
    worst = max(xt_lr, xt_rl)
    print(f"[*] crosstalk after trim L->R {xt_lr:+.1f} dB, R->L {xt_rl:+.1f} dB")
    assert worst <= -18.0, f"separation not recovered ({worst:+.1f} dB)"
    # trim無効時は劣化したまま
    dsp2, pcm2 = run_blocks(raw, nblk=20, stereo_trim_enabled=False)
    l2, r2 = pcm2[:, 0], pcm2[:, 1]
    worst2 = max(20 * np.log10(amp(r2, 1000.0) / amp(l2, 1000.0)),
                 20 * np.log10(amp(l2, 5000.0) / amp(r2, 5000.0)))
    print(f"[*] crosstalk without trim {worst2:+.1f} dB (expect worse than -18)")
    assert worst2 > -18.0, "test setup broken: no degradation without trim"
    print("[OK] 38kHz phase auto-trim")


def test_slow_agc():
    print("===== test_slow_agc =====")
    quiet = make_raw(stereo=False, depth=0.25)
    loud = make_raw(stereo=False, depth=1.0)
    # 同一インスタンスで quiet x20ブロック → loud x20ブロック
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = True
    nblk = 120  # スローAGC (attack 2s/release 10s) の収束を見るため長めに滞留
    bpb = N // NBLK  # bytes per original block section; reuse BLOCK slicing
    assert len(quiet) == len(loud) == 2 * N
    q_chunks, l_chunks = [], []
    for rep in range(nblk // NBLK + 1):
        for k in range(NBLK):
            a, _ = dsp.process(quiet[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
            q_chunks.append(a)
            if len(q_chunks) >= nblk:
                break
        if len(q_chunks) >= nblk:
            break
    g_quiet = float(dsp.slow_agc_gain)
    for rep in range(nblk // NBLK + 1):
        for k in range(NBLK):
            a, _ = dsp.process(loud[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
            l_chunks.append(a)
            if len(l_chunks) >= nblk:
                break
        if len(l_chunks) >= nblk:
            break
    g_loud = float(dsp.slow_agc_gain)
    q_out = np.concatenate([np.atleast_2d(c) if c.ndim == 1 else c for c in q_chunks])
    l_out = np.concatenate([np.atleast_2d(c) if c.ndim == 1 else c for c in l_chunks])
    in_ratio = 1.0 / 0.25
    out_ratio = float(np.sqrt(np.mean(l_out[-48000:] ** 2)) / np.sqrt(np.mean(q_out[-48000:] ** 2)))
    print(f"[*] input ratio {in_ratio:.1f}x -> output ratio {out_ratio:.2f}x "
          f"(gain quiet {g_quiet:.2f} -> loud {g_loud:.2f})")
    assert out_ratio < in_ratio * 0.8, "AGC did not compress station gap"
    assert 0.5 <= g_quiet <= 2.0 and 0.5 <= g_loud <= 2.0, "gain out of bounds"
    assert g_loud < g_quiet, "gain should fall for loud station"
    print("[OK] slow AGC leveling")


def test_nr_program_decoupling():
    print("===== test_nr_program_decoupling =====")
    from dsp import SdrDspPipeline
    dsp = SdrDspPipeline(1152000, 48000)
    rng = np.random.default_rng(0)
    n = 2048
    t = np.arange(n) / 48000
    hiss = (rng.standard_normal(n).astype(np.float32) * 0.15)
    loud = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    quiet = (0.05 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(10):
        dsp._update_stereo_nr(hiss, loud)
    g_loud = float(dsp.stereo_nr_gain)
    dsp._update_stereo_nr(hiss, quiet)
    g_step = float(dsp.stereo_nr_gain)
    print(f"[*] loud {g_loud:.3f} -> 1 block quiet {g_step:.3f} "
          f"(delta {abs(g_step - g_loud):.3f})")
    assert abs(g_step - g_loud) < 0.10, "NR must not jump on program dip"
    for _ in range(60):
        dsp._update_stereo_nr(hiss, quiet)
    g_settled = float(dsp.stereo_nr_gain)
    print(f"[*] after 60 quiet blocks {g_settled:.3f}")
    assert abs(g_settled - g_loud) > 0.15, "NR must still adapt to sustained change"
    print("[OK] NR program decoupling")


def test_pilot_flywheel():
    print("===== test_pilot_flywheel =====")
    from dsp import SdrDspPipeline
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    stereo = make_raw(stereo=True)
    mono = make_raw(stereo=False, nblk=80)
    for k in range(NBLK):
        dsp.process(stereo[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
    assert dsp.stereo_blend > 0.9, f"stereo did not lock ({dsp.stereo_blend})"
    # パイロット喪失直後25ブロックはフライホイールで凍結
    for k in range(25):
        dsp.process(mono[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
    held = float(dsp.stereo_blend)
    print(f"[*] blend after 25 pilot-less blocks {held:.2f} (expect held ≈ 1.0)")
    assert held > 0.9, "flywheel did not hold blend"
    # 持続喪失では解放ランプで落ちる
    for k in range(25, 80):
        dsp.process(mono[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
    fallen = float(dsp.stereo_blend)
    print(f"[*] blend after 80 pilot-less blocks {fallen:.2f} (expect < 0.3)")
    assert fallen < 0.3, "blend did not release on sustained loss"
    # 選局で即時リセット (前局ブレンドの持ち越し防止)
    dsp.set_offset_freq(0.0)
    assert dsp.stereo_blend == 0.0, "tune must reset blend"
    print("[OK] pilot flywheel")


def main() -> int:
    try:
        test_freq_dependent_blend()
        test_phase_trim()
        test_slow_agc()
        test_nr_program_decoupling()
        test_pilot_flywheel()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL AUDIO ENHANCE TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
