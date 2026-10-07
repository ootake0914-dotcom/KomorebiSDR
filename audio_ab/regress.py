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
from provenance import stamp  # noqa: E402

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
    "real_fm_lucky60_hiss_db": 1.0,
    "real_fm_lucky60_nr_gain_min": 0.10,
    "real_fm_nhk60_hiss_db": 1.0,
    "real_fm_nhk60_nr_gain_min": 0.15,
    "wfm_fr_excess_15k_db": 0.8,
    "wfm_sep_1k_db": 1.5,
    "wfm_sep_1k_mod30_db": 1.5,
    "wfm_sep_1k_mod60_db": 1.5,
    "wfm_thd_overmod_db": 1.5,
    "wfm_clip_pin_frac": 0.002,
    "real_fm_lucky60_clip_flat_frac": 0.002,
    "real_fm_nhk60_clip_flat_frac": 0.002,
    "wfm_tda_click_err_db": 1.5,
    "wfm_if_flat_75k_db": 0.5,
    "wfm_if_flat_94k_db": 0.5,
    "wfm_if_stop_106k_db": 3.0,
    "wfm_if_alias_200k_db": 3.0,
    "wfm_adj_sisdr_db": 1.0,
    "wfm_aci_du_r_db": 2.0,
    "wfm_aci_above_r_db": 3.0,
    "wfm_aci_gain_on": 0.10,
    "wfm_aci_side_gain_db": 2.0,
    "sic_offgrid_cancel_db": 3.0,
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


def _band_floor(x, lo, hi, q=10, n=2752):
    """q10パーセンタイル床 (過去のFM解析と同じ定義)。番組HFに鈍い。"""
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    sel = (fr >= lo) & (fr <= hi)
    vals = [float(np.mean(np.abs(np.fft.rfft(
        x[k * n:(k + 1) * n] * np.hanning(n)))[sel] ** 2))
        for k in range(len(x) // n)]
    return 10 * np.log10(np.percentile(vals, q) + 1e-24)


DEEMPH_US = 50.0


def _wfm_tone_raw(freq_hz, dev_hz=75000.0, l_amp=0.95, r_amp=0.0, dur=4.0,
                  pilot=0.09):
    """ラボ用: プリエンファシス無しステレオMPXトーン→FM→ADC生uint8。"""
    rf = 1152000.0
    n = int(dur * rf)
    t = np.arange(n) / rf
    l = l_amp * np.sin(2 * np.pi * freq_hz * t)
    r = r_amp * np.sin(2 * np.pi * freq_hz * t)
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + pilot * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * float(dev_hz) * np.cumsum(mpx) / rf
    iq = 0.6 * np.exp(1j * ph)
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _tone_pow(x, f, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    i = int(round(f * n / fs))
    return float(np.sum(X[max(0, i - 2):i + 3]))


def _thd_plus_n(x, f, fs=FS, lo=20.0, hi=20000.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    i0, i1 = int(lo * n / fs), int(hi * n / fs)
    fund = res = 0.0
    for i in range(i0, i1 + 1):
        if abs(fr[i] - f) <= 3.0:
            fund += X[i]
        else:
            res += X[i]
    return float(10.0 * np.log10(res / (fund + 1e-24) + 1e-24))


def _fir_resp_db(h, f, fs):
    H = np.fft.rfft(np.asarray(h, dtype=np.float64), 1 << 16)
    fr = np.fft.rfftfreq(1 << 16, 1.0 / fs)
    i = int(round(f / fs * (1 << 16)))
    i = min(max(i, 0), len(H) - 1)
    return float(20.0 * np.log10(abs(H[i]) + 1e-18))


def _deemph_db(f, tau_us=DEEMPH_US):
    return -10.0 * np.log10(1.0 + (2 * np.pi * f * tau_us * 1e-6) ** 2)


def _clip_flat_frac(x, tol=1e-4):
    x = np.asarray(x)
    return float(np.mean(np.abs(np.abs(x) - 1.0) < tol))


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

    # 7) WFM前面特性 (wide設定・NR/AGC/認知OFF): 15k超過減衰/分離度/過変調歪
    def _wf_front():
        d = SdrDspPipeline(1152000, FS)
        d.set_offset_freq(0.0)
        d.afc_enabled = False
        d.cognitive_enabled = False
        d.slow_agc_enabled = False
        d.filter_mode = "wide"
        d.set_stereo_nr(False)
        # SICは純音FMのベッセル側波帯を誤消去するため隔離する
        # (SIC自体は sic_offgrid_cancel_db と単体テストで検証)。
        d.sic_enabled = False
        if hasattr(d, "set_stereo_enabled"):
            d.set_stereo_enabled(True)
        return d

    y_low = _decode_stereo(_wf_front(), _wfm_tone_raw(1000.0), BLOCK_WFM)[:, 0]
    y_hi = _decode_stereo(_wf_front(), _wfm_tone_raw(15000.0), BLOCK_WFM)[:, 0]
    meas = 10.0 * np.log10(_tone_pow(y_hi[-FS:], 15000.0)
                           / (_tone_pow(y_low[-FS:], 1000.0) + 1e-24) + 1e-24)
    m["wfm_fr_excess_15k_db"] = float(
        meas - (_deemph_db(15000.0) - _deemph_db(1000.0)))
    y_sep = _decode_stereo(_wf_front(), _wfm_tone_raw(1000.0, dur=5.0),
                           BLOCK_WFM)
    seg = y_sep[-FS:]
    m["wfm_sep_1k_db"] = float(10.0 * np.log10(
        (_tone_pow(seg[:, 0], 1000.0) + 1e-24)
        / (_tone_pow(seg[:, 1], 1000.0) + 1e-24)))
    # 変調深度軸 (改善案diff_gain 4-1): 1k分離度は100%変調の一点校正で、
    # 通常番組変調 (30〜60%) の分離劣化を検出できない。追従gain導入後の
    # 値をベースライン化する (修正前: 30%→34.7 / 60%→35.4dB)。
    for _dev, _tag in ((22500.0, "30"), (45000.0, "60")):
        y_m = _decode_stereo(_wf_front(), _wfm_tone_raw(1000.0, dev_hz=_dev,
                                                        dur=5.0),
                            BLOCK_WFM)[-FS:]
        m[f"wfm_sep_1k_mod{_tag}_db"] = float(10.0 * np.log10(
            (_tone_pow(y_m[:, 0], 1000.0) + 1e-24)
            / (_tone_pow(y_m[:, 1], 1000.0) + 1e-24)))
    y_om = _decode_stereo(_wf_front(), _wfm_tone_raw(1000.0, dev_hz=100000.0),
                          BLOCK_WFM)[:, 0]
    m["wfm_thd_overmod_db"] = _thd_plus_n(y_om[-FS:], 1000.0)
    # ハードクリップ固定率は process のリサンプラで平滑化されるため、
    # 復調直後 (demodulate_wfm) を直接見る
    d = _wf_front()
    raw = _wfm_tone_raw(1000.0, dev_hz=100000.0)
    iq_aa = d.decimate(d.mix_frequency(d.raw_to_iq(raw), mode="WFM"),
                       d.fir_if_aa, d.if_decim)
    iq_if = d.decimate_with_history(iq_aa, d.fir_if, 1, "history_if_ch")
    y_pin = np.asarray(d.demodulate_wfm(iq_if))
    if y_pin.ndim == 2:
        y_pin = y_pin[:, 0]
    m["wfm_clip_pin_frac"] = _clip_flat_frac(y_pin[-FS:])

    # 7b) IFチャンネル選択度 (設計応答。1段目×2段目、折り返しを含む実効値)
    d_if = SdrDspPipeline(1152000, FS)

    def _if_resp_db(f):
        f2 = f - round(f / d_if.if_rate) * d_if.if_rate
        return (_fir_resp_db(d_if.fir_if_aa, f, d_if.rf_rate)
                + _fir_resp_db(d_if.fir_if, abs(f2), d_if.if_rate))

    ref0 = _if_resp_db(0.0)
    m["wfm_if_flat_75k_db"] = _if_resp_db(75000.0) - ref0
    m["wfm_if_flat_94k_db"] = _if_resp_db(94000.0) - ref0
    m["wfm_if_stop_106k_db"] = _if_resp_db(106000.0) - ref0
    m["wfm_if_alias_200k_db"] = _if_resp_db(200000.0) - ref0

    # 7c) ACI: ±200kHz妨害時のモノラル番組の劣化 (mid SI-SDR。ガードの
    # L-R絞りに影響されない。高いほど妨害に強い)
    from metrics import si_sdr
    wf_adj = build("wfm-adjacent", utts2, 20.0, seed=99)
    ya = _decode_stereo(_wf_pipe(False), wf_adj, BLOCK_WFM)
    nn_adj = min(len(y0), len(ya))
    mid_c = (y0[:nn_adj, 0] + y0[:nn_adj, 1]) * 0.5
    mid_a = (ya[:nn_adj, 0] + ya[:nn_adj, 1]) * 0.5
    m["wfm_adj_sisdr_db"] = si_sdr(mid_c, mid_a, align=True)

    # 7d) ACI検出: +200kHz・+10dBの強妨害で右側D/Uが下がる (床ガード付き)
    wf_adj10 = build("wfm-adjacent", utts2, 20.0, seed=99,
                     intf_db=10.0, offset_hz=200000.0)
    d_a10 = _wf_pipe(False)
    _decode_stereo(d_a10, wf_adj10, BLOCK_WFM)
    m["wfm_aci_du_r_db"] = float(d_a10._aci_r_db)
    m["wfm_aci_above_r_db"] = float(d_a10._aci_r_above_db)

    # 7e) ACIガード効果: 定常区間の側波帯妨害/モノラル番組比 (depth 0.6 vs 0)
    def _side_ratio(yc, ya, tail=2 * FS):
        n = min(len(yc), len(ya))
        t = slice(tail, n)
        sc = (yc[:n, 0] - yc[:n, 1]) * 0.5
        mc = (yc[:n, 0] + yc[:n, 1]) * 0.5
        sa = (ya[:n, 0] - ya[:n, 1]) * 0.5
        a = float(np.dot(sa[t], sc[t]) / (np.dot(sc[t], sc[t]) + 1e-18))
        resid = sa[t] - a * sc[t]
        return float(10.0 * np.log10(
            (np.mean(resid ** 2) + 1e-24) / (np.mean(mc[t] ** 2) + 1e-24)))

    y_clean2 = _decode_stereo(_wf_pipe(True),
                              build("wfm", utts2, 20.0, seed=99), BLOCK_WFM)
    d_g0 = _wf_pipe(True)
    d_g0.aci_depth = 0.0
    y_g0 = _decode_stereo(d_g0, wf_adj10, BLOCK_WFM)
    d_g6 = _wf_pipe(True)
    d_g6.aci_depth = 0.6
    y_g6 = _decode_stereo(d_g6, wf_adj10, BLOCK_WFM)
    m["wfm_aci_gain_on"] = float(d_g6.aci_gain)
    m["wfm_aci_side_gain_db"] = (_side_ratio(y_clean2, y_g6)
                                 - _side_ratio(y_clean2, y_g0))

    # 9) SIC: 格子外スプリアス (+10Hz) の検出＋消去 (放物線補間＋追従)。
    # auto_detect→processの直結で、FFT格子量子化の回帰を検出する。
    from adaptive_rf import DigitalSelfInterferenceCanceller
    sic = DigitalSelfInterferenceCanceller(sample_rate=288000.0, mu=0.08)
    f_spur = 28135.0
    n_sic, nb_sic = 4096, 40
    outs_sic = []
    for b in range(nb_sic):
        t = (b * n_sic + np.arange(n_sic, dtype=np.float64)) / 288000.0
        iq = (0.4 * np.exp(1j * 1.2 * np.sin(2 * np.pi * 1000.0 * t))
              + 0.6 * np.exp(
                  1j * (2 * np.pi * f_spur * t + 0.75))).astype(np.complex64)
        if b == 0:
            sic.auto_detect_spurious(iq)
        outs_sic.append(sic.process(iq))
    y_sic = outs_sic[-1]
    F = np.abs(np.fft.fft(y_sic * np.hanning(n_sic))) ** 2
    fr = np.fft.fftfreq(n_sic, 1.0 / 288000.0)
    i_sic = int(np.argmin(np.abs(fr - f_spur)))
    tL = ((nb_sic - 1) * n_sic + np.arange(n_sic, dtype=np.float64)) / 288000.0
    ref_sic = np.abs(np.fft.fft(
        0.6 * np.exp(1j * (2 * np.pi * f_spur * tL + 0.75))
        * np.hanning(n_sic))) ** 2
    lo_s, hi_s = max(0, i_sic - 2), i_sic + 3
    m["sic_offgrid_cancel_db"] = float(
        10.0 * np.log10(np.sum(ref_sic[lo_s:hi_s]) / np.sum(F[lo_s:hi_s])))

    # 8) TDA位相スリップ補修: 既知スリップ注入時のクリーン基準との誤差低減
    rf = 1152000.0
    dur = 3.0
    n = int(dur * rf)
    t = np.arange(n) / rf
    l = 0.95 * np.sin(2 * np.pi * 1000.0 * t)
    mpx = 0.45 * (l + l) + 0.09 * np.sin(2 * np.pi * 19000.0 * t)
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 75000.0 * np.cumsum(mpx) / rf
    # スリップはRFで16サンプル (IF 288kで4サンプル) に分散させる。RF4=IF1
    # では2πが折り返して不可視になり、単発スリップの検証にならない。
    spread = 16
    centers = np.arange(int(0.2 * rf), n - int(0.1 * rf), int(0.12 * rf))[:24]
    ph_s = ph.copy()
    for c in centers:
        ph_s[c:c + spread] += 2 * np.pi * np.arange(1, spread + 1) / spread
        ph_s[c + spread:] += 2 * np.pi
    iq_ref = 0.6 * np.exp(1j * ph)
    iq_slip = 0.6 * np.exp(1j * ph_s)
    for c in centers:
        iq_slip[c - 2:c + spread + 2] *= 0.06
    rng = np.random.default_rng(11)
    nz = (np.sqrt(0.36 / 10 ** 0.6 / 2.0)
          * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))

    def _to_raw(iq):
        raw = np.empty(2 * len(iq), dtype=np.uint8)
        raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
        raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
        return raw

    def _wf_tda(raw, on):
        d = _wf_front()
        d.tda_click.enabled = bool(on)
        return _decode_stereo(d, raw, BLOCK_WFM)[:, 0]

    y_ref = _wf_tda(_to_raw(iq_ref + nz), False)
    y_off = _wf_tda(_to_raw(iq_slip + nz), False)
    y_on = _wf_tda(_to_raw(iq_slip + nz), True)
    nn = min(len(y_ref), len(y_off), len(y_on))
    e_off = float(np.mean((y_off[:nn] - y_ref[:nn]) ** 2))
    e_on = float(np.mean((y_on[:nn] - y_ref[:nn]) ** 2))
    m["wfm_tda_click_err_db"] = float(
        10.0 * np.log10(e_on / (e_off + 1e-24) + 1e-24))

    if full:
        from fast import CachedAsr
        from cer_ab import cer
        spotter = CachedAsr("small")
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
    if full:
        # 実録 (gitignore) は存在時のみ。NR抑圧量と幅推定器の下限を固定化。
        try:
            from realdata import available, decode_wfm
            avail = available()
            for name in ("fm_lucky60", "fm_nhk60"):
                if name not in avail:
                    continue
                y0, _ = decode_wfm(name, False)
                y1, info = decode_wfm(name, True, telemetry=True)
                nn = min(len(y0), len(y1))
                s0 = (y0[:nn, 0] - y0[:nn, 1]) * 0.5
                s1 = (y1[:nn, 0] - y1[:nn, 1]) * 0.5
                m[f"real_{name}_hiss_db"] = (_band_floor(s1, 8000, 12000)
                                             - _band_floor(s0, 8000, 12000))
                m[f"real_{name}_nr_gain_min"] = float(info["nr_gain_min"])
                m[f"real_{name}_clip_flat_frac"] = _clip_flat_frac(y0[FS * 2:])
        except Exception as e:
            print(f"real metrics skipped: {e}")
    return m


def main(argv):
    update = "--update" in argv
    full = "--full" in argv
    t0 = time.perf_counter()
    cur = measure(full=full)
    if update:
        old = {}
        try:
            with open(GOLDEN, encoding="utf-8") as f:
                old = json.load(f).get("metrics", {})
        except Exception:
            pass
        metrics = dict(old)
        for k, v in cur.items():
            metrics[k] = {"value": float(v), "tol": float(TOLS.get(k, 0.1))}
        gold = {"version": 1,
                "full": bool(full or any(k.startswith(("ssb_cer", "real_"))
                                         for k in metrics)),
                "provenance": stamp(), "metrics": metrics}
        with open(GOLDEN, "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=2)
        print(f"golden updated ({GOLDEN}, {time.perf_counter() - t0:.0f}s)")
        for k, v in sorted(cur.items()):
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
            print(f"{k:26s} {'-':>10s} {'-':>10s} {'-':>9s} {'-':>6s}  "
                  f"skipped (--full only)")
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
