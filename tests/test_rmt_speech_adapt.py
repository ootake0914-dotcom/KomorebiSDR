"""番組適応RMT (Phase ③): トークで半減・音楽で全開・音楽経路は旧一致。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from adaptive_audio import CognitiveSpeechMusicTracker
from dsp import SdrDspPipeline
from rmt_denoiser import SafeRmtDenoiser
from tools.score_wav import stoi

FS = 48000
BLK = 2752


def _speech_like(seed=11, secs=6.0):
    # 母音 (300-2500Hz)＋子音バースト＝トーク近似 (分類器でprob>0.5を確認済み)
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * secs)) / FS
    vow = sum(a * np.sin(2 * np.pi * f * t) for f, a in
              [(350.0, 0.5), (800.0, 0.4), (1500.0, 0.3), (2400.0, 0.2)])
    syl = (t % 0.25 < 0.18)
    cons = rng.standard_normal(len(t)) * ((t % 0.2) < 0.02) * 0.6
    x = (vow * (0.4 + 0.6 * syl) + cons).astype(np.float64)
    return ((x / (np.sqrt(np.mean(x ** 2)) + 1e-18)) * 0.2).astype(np.float32)


def _music_like(seed=12, secs=6.0):
    # 明るめ広帯域＝音楽近似 (分類器でprob=0.0を確認済み)
    t = np.arange(int(FS * secs)) / FS
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in
            [(150.0, 0.25), (440.0, 0.3), (1500.0, 0.3), (5000.0, 0.3),
             (9000.0, 0.25), (13000.0, 0.15)])
    return ((x * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t))).astype(np.float64)
            * 0.2).astype(np.float32)


def test_classifier_separates():
    ps, pm = None, None
    for x, slot in ((_speech_like(), "s"), (_music_like(), "m")):
        tr = CognitiveSpeechMusicTracker(FS)
        v = 0.0
        for k in range(0, len(x), BLK):
            seg = np.stack([x[k:k + BLK], x[k:k + BLK]], axis=1)
            v = tr.analyze(seg)
        if slot == "s":
            ps = v
        else:
            pm = v
    print(f"speech_prob: talk={ps:.2f} music={pm:.2f}")
    assert ps > 0.5 and pm < 0.3, (ps, pm)
    print("classifier OK")


def test_cap_mapping():
    dsp = SdrDspPipeline(1152000, FS)
    dsp.cognitive_eq.speech_prob = 0.0
    assert abs(dsp._bm_rmt_strength_cap() - 0.65) < 1e-12
    dsp.cognitive_eq.speech_prob = 1.0
    assert abs(dsp._bm_rmt_strength_cap() - 0.325) < 1e-12
    dsp.bm_rmt_speech_adapt = False
    assert abs(dsp._bm_rmt_strength_cap() - 0.65) < 1e-12
    print("cap mapping OK")


def _denoise(x, cap):
    dn = SafeRmtDenoiser(max_strength=cap)
    outs = [dn.process_mono(x[k:k + BLK], s_meter_dbfs=-40.0, snr_db=12.0)[0]
            for k in range(0, len(x), BLK)]
    return np.concatenate(outs)


def test_mechanism_stoi():
    # 変調5トーン＋ヒス (STOIに余裕のある信号): 強度半減でSTOI損失が半減する
    rng = np.random.default_rng(5)
    t = np.arange(FS * 4) / FS
    clean = sum(0.02 * np.sin(2 * np.pi * f * t)
                * (0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t))
                for f in [300.0, 600.0, 900.0, 1400.0, 2100.0])
    clean = np.asarray(clean, dtype=np.float32)
    noisy = (clean + 0.03 * rng.standard_normal(len(clean))).astype(np.float32)
    D = 15
    s_noisy = stoi(clean, noisy)
    s_full = stoi(clean[:len(clean) - D], _denoise(noisy, 0.65)[D:])
    s_half = stoi(clean[:len(clean) - D], _denoise(noisy, 0.325)[D:])
    print(f"STOI noisy={s_noisy:.3f} full={s_full:.3f} half={s_half:.3f}")
    assert s_half > s_full + 0.01, (s_half, s_full)
    print("mechanism OK")


def _fm_iq(program, snr_db=12.0, seed=7):
    # 簡易WFM変調 (benchmark make_iq相当・L/R同一番組)。
    # 実FM準拠の50μsプリエンファシス付き (復調ディエンファでフラットに戻る)。
    RF = 1152000
    t = np.arange(len(program)) / FS
    ti = np.arange(int(len(program) * RF / FS)) / RF
    m = np.interp(ti, t, program.astype(np.float64))
    a = 50e-6 * RF
    mp = np.concatenate(([m[0]], m[1:] * (1.0 + a) - m[:-1] * a))
    mp = mp / (np.max(np.abs(mp)) + 1e-9)
    phase = 2 * np.pi * 30000.0 * np.cumsum(mp) / RF
    iq = 0.6 * np.exp(1j * phase)
    rng = np.random.default_rng(seed)
    p = 0.36 / (10 ** (snr_db / 10.0))
    iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(len(iq))
                                  + 1j * rng.standard_normal(len(iq)))
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _decode(raw, adapt):
    dsp = SdrDspPipeline(1152000, FS)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = True
    dsp.slow_agc_enabled = False
    dsp.black_magic_enabled = True
    dsp.bm_rmt_enabled = True
    dsp.bm_rmt_speech_adapt = adapt
    ch = []
    nblk = len(raw) // 132096
    for k in range(nblk):
        audio, _ = dsp.process(raw[k * 132096:(k + 1) * 132096], mode="WFM")
        a = np.asarray(audio)
        ch.append(a if a.ndim == 2 else np.stack([a, a], axis=1))
    return np.concatenate(ch, axis=0), float(dsp.cognitive_eq.speech_prob)


def test_pipeline_music_identical():
    raw = _fm_iq(_music_like(secs=4.0))
    y_off, p_off = _decode(raw, False)
    y_on, p_on = _decode(raw, True)
    print(f"music probs: adapt-off={p_off:.2f} adapt-on={p_on:.2f}")
    assert np.array_equal(np.asarray(y_off), np.asarray(y_on)), \
        "music path must be bit-identical"
    print("music identical OK")


def test_pipeline_speech_adapts():
    raw = _fm_iq(_speech_like(secs=4.0))
    y_off, p_off = _decode(raw, False)
    y_on, p_on = _decode(raw, True)
    print(f"speech probs: adapt-off={p_off:.2f} adapt-on={p_on:.2f}")
    assert p_on > 0.5, p_on
    assert not np.array_equal(np.asarray(y_off), np.asarray(y_on)), \
        "speech path must adapt"
    print("speech adapt OK")


if __name__ == "__main__":
    test_classifier_separates()
    test_cap_mapping()
    test_mechanism_stoi()
    test_pipeline_music_identical()
    test_pipeline_speech_adapts()
    print("ALL RMT-SPEECH-ADAPT TESTS PASSED!")
