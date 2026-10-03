"""Narrowband speech NR tests (synthetic, no hardware).

案1 (SSB/CW/NFM音声帯NR) の回帰テスト:
1) LSA利得の健全性 (E1 LUT精度・利得範囲・クリーン高SNRでほぼ1)
2) クリーンバイパスのビット等価性 (ONでも素通し)
3) 効果: 母音状信号＋白/ピンク雑音 SNR 0/5/10dBでSTOI非悪化＋改善
4) musical noise: 指標の感度検証 (人工トーン性残差は検出)＋NR出力は合格
5) p99: 2752サンプル/ブロック処理時間の予算確認
6) 配線: 既定OFF・遅延生成・SSB/NFM経路で例外なく動作
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import numpy as np
from narrowband_nr import NarrowbandNr, expint_e1, _lsa_factor
from score_wav import stoi
from dsp import SdrDspPipeline

FS = 48000


def _vowel(n, seed=0, f0=120.0):
    """母音状プロキシ: f0＋倍音＋緩やかな振幅動揺＋語間ポーズ (了解度評価用)。
    注意1: STOI実装は無音フレームを除外するため、バーストは0.5秒以上にする
    (0.8秒発声/0.3秒休止の繰り返し)。連続発声のみだとNRの語間学習も
    働かないため、ポーズ入りが実態に近い。
    注意2: ポーズは真の無音でなく低レベル雑音のみ (語間残留ノイズの実態)。"""
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    f0t = f0 + 30.0 * np.sin(2 * np.pi * 0.7 * t + seed)
    ph = 2 * np.pi * np.cumsum(f0t) / FS
    v = np.zeros(n)
    for h, a in [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]:
        v += a * np.sin(h * ph + rng.uniform(0, np.pi))
    v = v * (0.8 + 0.2 * np.sin(2 * np.pi * 3.0 * t))
    cyc = t % 1.1
    gate = np.where(cyc < 0.8, 1.0, 0.0)
    # ゲート縁のクリック防止に10ms立ち上げ/立ち下げ
    edge = int(FS * 0.01)
    k = np.ones(edge) * 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    for start in np.arange(0.8, t[-1], 1.1):
        i = int(start * FS)
        v[i:i + edge] *= k
        j = max(0, i - edge)
        v[j:j + edge] *= k[::-1]
    v = v * gate
    return (v / max(float(np.max(np.abs(v))), 1e-9)).astype(np.float32)


def _pink(n, seed=1):
    """真性ピンク雑音 (-3dB/oct。Paul Kelletフィルタ)。
    注意: cumsumはブラウン (-6dB/oct) になるため使わない。
    ブラウンは低域が強すぎてLSAの想定外であり、実環境のピンクとは別物。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    b = np.zeros(7)
    y = np.zeros(n)
    for i, x in enumerate(w):
        b[0] = 0.99886 * b[0] + x * 0.0555179
        b[1] = 0.99332 * b[1] + x * 0.0750759
        b[2] = 0.96900 * b[2] + x * 0.1538520
        b[3] = 0.86650 * b[3] + x * 0.3104856
        b[4] = 0.55000 * b[4] + x * 0.5329522
        b[5] = -0.7616 * b[5] - x * 0.0168980
        y[i] = (b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + x * 0.5362)
        b[6] = x * 0.115926
    y = y * 0.11
    y = y - np.mean(y)
    return (y / max(float(np.max(np.abs(y))), 1e-9)).astype(np.float32)


def _peakiness(x, sr=FS):
    """残差のトーン性 (musical noise指標): 最大ビン/中央値 (dB)。
    広帯域残差は低く、孤立トーンは高く出る。"""
    n = 4096
    if len(x) < n:
        x = np.concatenate((x, np.zeros(n - len(x))))
    seg = x[:n] - float(np.mean(x[:n]))
    spec = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2 + 1e-24
    # DC・ナイキスト除外
    spec = spec[1:-1]
    return 10.0 * float(np.log10(float(np.max(spec)) / float(np.median(spec))))


def test_lsa_sane() -> bool:
    v = np.array([0.01, 0.1, 1.0, 5.0, 20.0, 49.0])
    e = expint_e1(v)
    # 真値 (高精度表): E1(0.01)=4.037929, E1(0.1)=1.822924, E1(1)=0.219384,
    # E1(5)=0.001148, E1(20)=2.061e-9, E1(49)≈1e-24
    want = np.array([4.037929, 1.822924, 0.219384, 0.001148, 2.061e-9, 0.0])
    # E1(20)級の微小値は相対誤差が大きく出るが、利得exp(0.5*E1)への寄与は
    # 無視できるため絶対1e-6で判定する (音響的に意味のある精度)。
    ok = bool(np.allclose(e, want, rtol=2e-6, atol=1e-6))
    print(f"[{'OK' if ok else 'FAIL'}] E1 LUT source accurate (max err "
          f"{float(np.max(np.abs(e - want))):.2e})")
    # LSA補正係数の性質: v小で大きくv大で1へ単調減少 (factor単体は
    # v->0で発散するが、利得 g=xi/(1+xi)*factor はclipで1以下に抑える)。
    # ここでは単調性と v=1 での値 exp(0.5*0.219)=1.116 を見る。
    f = _lsa_factor(np.array([0.05, 0.5, 1.0, 5.0, 50.0]))
    mono = bool(np.all(np.diff(f) < 0.0))
    one = abs(float(f[2]) - 1.1157) < 1e-3
    print(f"[{'OK' if mono and one else 'FAIL'}] LSA factor monotone, f(1)="
          f"{float(f[2]):.4f} (want 1.1157)")
    # 実利得は1以下 (無音ビン×中程度事前SNRの爆発ケースを含む)
    from narrowband_nr import NarrowbandNr as _N
    nr = _N("SSB")
    rng = np.random.default_rng(0)
    X = (rng.standard_normal(129) + 1j * rng.standard_normal(129)) * 0.01
    g = nr._frame_gains(X)
    bounded = bool(np.all(g <= 1.0) and np.all(g >= 0.0) and np.all(np.isfinite(g)))
    print(f"[{'OK' if bounded else 'FAIL'}] frame gains bounded in [0,1]")
    return bool(ok and mono and one and bounded)


def test_clean_bypass() -> bool:
    n = FS * 2
    t = np.arange(n) / FS
    clean = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    ok_all = True
    for pre in ("NFM", "SSB", "CW"):
        nr = NarrowbandNr(pre)
        y = nr.process(clean)
        same = bool(np.array_equal(y, clean))
        print(f"[{'OK' if same else 'FAIL'}] {pre} clean bypass bit-equal "
              f"(bypass={nr.last_bypass})")
        ok_all &= same
    return bool(ok_all)


def _segsnr(ref, x, tail, sr=FS):
    """区間SNR (0.25秒窓平均。了解度ではなく音質・残留ヒス量の指標)。
    単一チャネル強調はSTOIを上げないのが文献的定石のため、効果判定は
    segSNR改善＋STOI非悪化の二本立てにする。"""
    r = ref[tail].astype(np.float64)
    e = x[tail].astype(np.float64) - r
    L = sr // 4
    vals = []
    for s in range(0, len(r) - L, L):
        seg = r[s:s + L]
        er = e[s:s + L]
        if np.mean(seg ** 2) < 1e-8:
            continue
        vals.append(10.0 * np.log10(np.mean(seg ** 2) / (np.mean(er ** 2) + 1e-12)))
    return float(np.mean(vals)) if vals else -99.0


def test_effect() -> bool:
    ok_all = True
    for noise_name in ("white", "pink"):
        for snr_db in (0.0, 5.0, 10.0):
            n = FS * 4
            v = _vowel(n) * 0.5
            rng = np.random.default_rng(7)
            if noise_name == "white":
                nz = rng.standard_normal(n)
            else:
                nz = _pink(n, seed=7)
            nz = nz / (float(np.sqrt(np.mean(nz ** 2))) + 1e-12)
            sig_pow = float(np.mean(v ** 2))
            nz = (nz * np.sqrt(sig_pow / (10.0 ** (snr_db / 10.0)))).astype(np.float32)
            mix = (v + nz).astype(np.float32)
            # ストリーミング実態に合わせ0.5秒ブロックで逐次処理する。
            # (単発processだと初回はノイズ未学習でバイパス素通しになるのが仕様。
            # 評価は後半2秒のみ)
            nr = NarrowbandNr("SSB")
            bl = FS // 2
            ys = [nr.process(mix[k * bl:(k + 1) * bl]) for k in range(8)]
            y = np.concatenate(ys)
            # NR出力と参照は同位相 (WOLA零位相。128は立上り過渡のみ)。
            tail = slice(FS * 2, FS * 4)
            s_in = stoi(v[tail], mix[tail])
            s_out = stoi(v[tail], y[tail])
            g_in = _segsnr(v, mix, tail)
            g_out = _segsnr(v, y, tail)
            # 非悪化 (STOI) が必須、効果はsegSNRで測る。
            # 合格線+2.5dBは可聴な改善の下限 (JND約1dBの倍以上)。
            # ピンクは番組と帯域が重なるため白より伸びない (+2.7〜3.2dB級)。
            # SNR10では床だけ残るため+1dB以上を要求する。
            no_harm = s_out >= s_in - 0.02
            need = 2.5 if snr_db <= 5.0 else 1.0
            improved = (g_out - g_in) >= need
            ok = bool(no_harm and improved)
            print(f"[{'OK' if ok else 'FAIL'}] {noise_name} SNR{snr_db:.0f}: "
                  f"STOI {s_in:.3f}->{s_out:.3f} "
                  f"segSNR {g_in:.1f}->{g_out:.1f}dB")
            ok_all &= ok
    return bool(ok_all)


def test_musical() -> bool:
    # 指標の感度: 人工トーン性残差は検出できること
    n = FS * 2
    rng = np.random.default_rng(3)
    tone_resid = (0.01 * rng.standard_normal(n)
                  + 0.05 * np.sin(2 * np.pi * 3333.0 * np.arange(n) / FS)).astype(np.float32)
    pk_bad = _peakiness(tone_resid)
    # 広帯域残差は低いこと
    pk_good = _peakiness((0.01 * rng.standard_normal(n)).astype(np.float32))
    sensitive = pk_bad > pk_good + 10.0
    print(f"[{'OK' if sensitive else 'FAIL'}] metric sensitive "
          f"(tonal={pk_bad:.1f}dB broadband={pk_good:.1f}dB)")
    # NR出力の残差はトーン性なし
    v = _vowel(n) * 0.5
    nz = rng.standard_normal(n)
    nz = nz / (float(np.sqrt(np.mean(nz ** 2))) + 1e-12)
    nz = (nz * np.sqrt(float(np.mean(v ** 2)) / (10.0 ** (5.0 / 10.0)))).astype(np.float32)
    mix = (v + nz).astype(np.float32)
    nr = NarrowbandNr("SSB")
    y = nr.process(mix)
    resid = (y - v).astype(np.float32)
    pk_nr = _peakiness(resid)
    clean = pk_nr < pk_bad - 6.0
    print(f"[{'OK' if clean else 'FAIL'}] NR residual not tonal "
          f"({pk_nr:.1f}dB vs bad {pk_bad:.1f}dB)")
    return bool(sensitive and clean)


def test_p99() -> bool:
    nr = NarrowbandNr("SSB")
    rng = np.random.default_rng(9)
    blk = (rng.standard_normal(2752) * 0.2).astype(np.float32)
    nr.process(blk)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        nr.process(blk)
        ts.append((time.perf_counter() - t0) * 1000.0)
    p50, p99 = float(np.median(ts)), float(np.partition(ts, 29)[29])
    ok = p99 < 15.0
    print(f"[{'OK' if ok else 'FAIL'}] p50={p50:.2f}ms p99={p99:.2f}ms "
          f"(need p99<15ms; NFM/SSB path budget)")
    return bool(ok)


def test_wiring() -> bool:
    d = SdrDspPipeline(1152000, 48000)
    ok = (d.nbm_nr_enabled is False and d.nbm_nr is None)
    print(f"[{'OK' if ok else 'FAIL'}] default OFF, lazy instance")
    # OFF時は経路不変 (コード到達不能のため同一性は自明だが、例外なく通ることを確認)
    rng = np.random.default_rng(4)
    iq = (0.5 * np.exp(1j * np.linspace(0, 200 * np.pi, 16512 * 2))
          + 0.01 * (rng.standard_normal(16512 * 2)
                    + 1j * rng.standard_normal(16512 * 2))).astype(np.complex64)
    try:
        a = d.demodulate_nfm(iq[:16512])
        d2 = SdrDspPipeline(1152000, 48000)
        d2.nbm_nr_enabled = True
        b = d2.demodulate_nfm(iq[:16512])
        ok2 = len(a) == len(b) and bool(np.all(np.isfinite(b)))
        print(f"[{'OK' if ok2 else 'FAIL'}] NFM ON path runs (len {len(a)}->{len(b)})")
    except Exception as e:
        print(f"[FAIL] NFM wiring raised: {e}")
        ok2 = False
    try:
        iq48 = (0.3 * np.exp(1j * 2 * np.pi * 1500.0 * np.arange(2208 * 4) / 48000)
                ).astype(np.complex64)
        d3 = SdrDspPipeline(1152000, 48000)
        d3.nbm_nr_enabled = True
        c = d3.demodulate_ssb(iq48, "USB")
        ok3 = len(c) > 0 and bool(np.all(np.isfinite(c)))
        print(f"[{'OK' if ok3 else 'FAIL'}] SSB ON path runs (len {len(c)})")
    except Exception as e:
        print(f"[FAIL] SSB wiring raised: {e}")
        ok3 = False
    return bool(ok and ok2 and ok3)


def main() -> int:
    ok = test_lsa_sane()
    ok &= test_clean_bypass()
    ok &= test_effect()
    ok &= test_musical()
    ok &= test_p99()
    ok &= test_wiring()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
