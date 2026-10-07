"""NR hiss-estimator denominator guard test (no hardware required).

_update_stereo_nr's ratio floor/mf explodes when the 300-3kHz program band
is empty (HF-only content: high woodwinds, cymbals, applause), driving
nr_gain to 0 and wiping Side highs on clean signals. The guard freezes
estimation (holding last state) instead of computing garbage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

SR = 48000
BLK = 2752


def test_hf_only_clean():
    """高域のみのクリーン信号でNRが全閉しないこと (回帰: hiss +48dB)"""
    dsp = SdrDspPipeline(1152000, SR)
    n = BLK
    t = np.arange(n) / SR
    mono = (0.001 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    diff = (0.5 * np.sin(2 * np.pi * 6000.0 * t)).astype(np.float32)
    for _ in range(30):
        dsp._update_stereo_nr(diff, mono)
    print(f"[*] HF-only: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"gain={dsp.stereo_nr_gain:.3f} cut={dsp.stereo_cut_hz:.0f}Hz")
    assert dsp.stereo_hiss_db < 8.0, \
        f"non-physical hiss estimate ({dsp.stereo_hiss_db:+.1f}dB)"
    assert dsp.stereo_nr_gain > 0.9, "clean HF wiped by NR"
    assert dsp.stereo_cut_hz > 12000.0, "cut collapsed on clean HF"
    print("[OK] HF-only clean preserved")


def test_legit_hiss_still_caught():
    """通常のヒス (強い番組＋高域ノイズ) は従来通りNRが効くこと"""
    rng = np.random.default_rng(5)
    dsp = SdrDspPipeline(1152000, SR)
    n = BLK
    t = np.arange(n) / SR
    mono = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(60):
        hiss = (rng.standard_normal(n) * 0.15).astype(np.float32)
        dsp._update_stereo_nr((mono * 0.1 + hiss).astype(np.float32), mono)
    print(f"[*] hissy: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"wiener_w={dsp._nr_s_w:.3f} (blend gain={dsp.stereo_nr_gain:.3f})")
    # ヒス推定はWiener系 (-46〜-26dB) で駆動する。ブレンド系 (-18〜-4dB) は
    # 極端な弱電界専用のため、このレベルでは動かなくて正しい。
    assert dsp._nr_s_w > 0.5, "Wiener path no longer sees real hiss"
    print("[OK] legitimate hiss still suppressed")


def test_silence_bounded():
    """無音で発散・クラッシュしないこと"""
    dsp = SdrDspPipeline(1152000, SR)
    z = np.zeros(BLK, dtype=np.float32)
    for _ in range(10):
        dsp._update_stereo_nr(z, z)
    assert np.isfinite(dsp.stereo_hiss_db)
    assert 0.0 <= dsp.stereo_nr_gain <= 1.0
    print("[OK] silence bounded")


def test_weak_field_hiss_not_frozen():
    """弱電界の真性ヒスは緩和ガードで凍結されずNRが効くこと

    緩和ガード (mf<floor*10) は強電界 (gate_hi以上) 限定。
    弱電界では真性ヒスがあり得るため素通しし、Wienerが立たねばならない。
    """
    rng = np.random.default_rng(7)
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 15.0  # 弱電界: 割引なし
    n = BLK
    t = np.arange(n) / SR
    mono = (0.05 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(60):
        hiss = (rng.standard_normal(n) * 0.15).astype(np.float32)
        dsp._update_stereo_nr((mono * 0.1 + hiss).astype(np.float32), mono)
    print(f"[*] weak-field hissy: wiener_w={dsp._nr_s_w:.3f}")
    assert dsp._nr_s_w > 0.5, "weak-field hiss missed (over-frozen)"
    print("[OK] weak-field hiss still suppressed")


def test_hf_program_strong_field():
    """実素材の高域寄り番組 (3.5kHz以上) でSideが削られないこと。

    追加改善案§3の被害ケース (mf/floor=2.40、旧ガード素通しで
    nr_gain=0.089/cut=5514Hzまで崩壊)。K=15ガード＋強電界条件で
    凍結され、gain/cut共に保全される。Wiener側 (slow) も初回
    プライムで-60に倒すため、Side抑圧は残らない。
    """
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 52.0  # C/N=40dB強電界の実測値
    n = BLK
    t = np.arange(n) / SR
    prog_hf = (0.25 * np.sin(2 * np.pi * 3500.0 * t)
               + 0.25 * np.sin(2 * np.pi * 4000.0 * t)
               + 0.20 * np.sin(2 * np.pi * 6000.0 * t)
               + 0.15 * np.sin(2 * np.pi * 8000.0 * t)).astype(np.float32)
    body_mf = (0.003 * np.sin(2 * np.pi * 500.0 * t)
               + 0.003 * np.sin(2 * np.pi * 1500.0 * t)
               + 0.002 * np.sin(2 * np.pi * 2500.0 * t)).astype(np.float32)
    mono = (prog_hf * 0.5 + body_mf).astype(np.float32)
    diff = prog_hf.astype(np.float32)
    for _ in range(100):
        dsp._update_stereo_nr(diff, mono)
    print(f"[*] HF-program: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"gain={dsp.stereo_nr_gain:.3f} cut_eff={dsp._nr_cut_eff:.0f}Hz "
          f"wiener_w={dsp._nr_s_w:.3f}")
    assert dsp.stereo_nr_gain > 0.8, "strong-field HF program wiped Side"
    assert dsp._nr_cut_eff > 10000.0, "cut collapsed on HF program"
    assert dsp._nr_s_w < 0.1, "Wiener still suppresses HF program"
    print("[OK] HF program in strong field preserved")


def test_normal_cn10_unchanged():
    """通常番組C/N10ではガードが発動せず従来通りNRが効くこと。

    追加改善案§3の正常側ケース (mf/floor=33)。K=15でも余裕で
    素通しし、変更前と同一の振る舞い (gain=0.980) になることの確認。
    """
    rng = np.random.default_rng(11)
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 23.0  # C/N=10dBの推定値
    n = BLK
    t = np.arange(n) / SR
    mono = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(100):
        hiss = (rng.standard_normal(n) * 0.5 / (10 ** (10.0 / 20.0))).astype(np.float32)
        dsp._update_stereo_nr((mono * 0.1 + hiss).astype(np.float32), mono)
    print(f"[*] CN10 normal: gain={dsp.stereo_nr_gain:.3f} "
          f"hiss={dsp.stereo_hiss_db:+.1f}dB")
    assert 0.95 <= dsp.stereo_nr_gain <= 1.0, \
        f"CN10 behavior changed ({dsp.stereo_nr_gain:.3f} vs 0.980)"
    print("[OK] CN10 normal program unchanged")


def test_side_decode_preserved():
    """高域番組の実デコードでSideがNR off参照と一致すること。

    追加改善案の検証項目そのまま (Side 9-12kHzが参照比-20dB以内)。
    L=4k+9k / R=3.5k+11kのHFステレオ番組をRF合成→process()全段で
    デコードし、NR on/offのSide帯域パワーを比べる。
    """
    RF = 1152000.0
    CHUNK = 132096
    dur_s = 3.0
    n = int(dur_s * RF)
    t = np.arange(n) / RF
    l = (0.30 * np.sin(2 * np.pi * 4000.0 * t)
         + 0.20 * np.sin(2 * np.pi * 9000.0 * t)
         + 0.004 * np.sin(2 * np.pi * 1000.0 * t))
    r = (0.30 * np.sin(2 * np.pi * 3500.0 * t)
         + 0.20 * np.sin(2 * np.pi * 11000.0 * t)
         + 0.004 * np.sin(2 * np.pi * 1000.0 * t))
    mpx = (0.45 * (l + r) + 0.45 * (l - r) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph).astype(np.complex64)
    rng = np.random.default_rng(3)
    p = 0.36 / 10 ** (40.0 / 10.0)
    iq = iq + np.sqrt(p / 2.0) * (rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n))
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)

    def decode(nr_on):
        dsp = SdrDspPipeline(1152000, SR)
        dsp.set_offset_freq(0.0)
        dsp.afc_enabled = False
        dsp.cognitive_enabled = False
        dsp.slow_agc_enabled = False
        dsp.set_stereo_nr(nr_on)
        outs = []
        for k in range(len(raw) // CHUNK):
            a, _ = dsp.process(raw[k * CHUNK:(k + 1) * CHUNK], "WFM")
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 1:
                a = np.stack([a, a], axis=1)
            outs.append(a)
        y = np.concatenate(outs, axis=0)
        return y[len(y) // 2:]

    def band(x, lo, hi, nn=2752):
        fr = np.fft.rfftfreq(nn, 1.0 / SR)
        sel = (fr >= lo) & (fr <= hi)
        vals = [float(np.mean(np.abs(np.fft.rfft(
            x[k * nn:(k + 1) * nn] * np.hanning(nn)))[sel] ** 2))
            for k in range(len(x) // nn)]
        return 10 * np.log10(float(np.mean(vals)) + 1e-24)

    y_on = decode(True)
    y_off = decode(False)
    for lo, hi in ((6000, 9000), (9000, 12000), (12000, 15000)):
        s_on = band((y_on[:, 0] - y_on[:, 1]) * 0.5, lo, hi)
        s_off = band((y_off[:, 0] - y_off[:, 1]) * 0.5, lo, hi)
        print(f"[*] Side {lo}-{hi}: on={s_on:+.1f}dB off={s_off:+.1f}dB "
              f"ratio={s_on - s_off:+.1f}dB")
        assert s_on - s_off > -20.0, \
            f"Side wiped in {lo}-{hi}Hz ({s_on - s_off:+.1f}dB)"
    print("[OK] decoded Side preserved vs NR-off reference")


def test_silence_freeze_holds_and_recovers():
    """長時間無音でNRが全閉せず、番組復帰が即時であること。

    改善案§6-1: 無音 (mono_rms<0.01) では推定・学習とも凍結する。
    凍結なしでは20秒超の無音で全閉し、番組復帰後も約3.8秒潰れる。
    """
    rng = np.random.default_rng(21)
    dsp = SdrDspPipeline(1152000, SR)
    dsp._if_snr_db = 37.0
    n = BLK
    w = (rng.standard_normal(n) * 0.004).astype(np.float32)
    for _ in range(400):  # 約23秒の無音
        dsp._update_stereo_nr(w, w)
    print(f"[*] silence 23s: hiss={dsp.stereo_hiss_db:+.1f}dB "
          f"gain={dsp.stereo_nr_gain:.3f}")
    assert dsp.stereo_nr_gain > 0.9, "silence collapsed NR"
    t = np.arange(n) / SR
    mono = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    for _ in range(5):
        dsp._update_stereo_nr((mono * 0.1).astype(np.float32), mono)
    print(f"[*] program back 5blk: gain={dsp.stereo_nr_gain:.3f}")
    assert dsp.stereo_nr_gain > 0.9, "slow recovery after silence"
    print("[OK] silence frozen, instant recovery")


def main() -> int:
    try:
        test_hf_only_clean()
        test_legit_hiss_still_caught()
        test_silence_bounded()
        test_weak_field_hiss_not_frozen()
        test_hf_program_strong_field()
        test_normal_cn10_unchanged()
        test_side_decode_preserved()
        test_silence_freeze_holds_and_recovers()
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("ALL NR-FLOOR TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
