"""黒魔法三点セットの合成ベンチマーク (実機不要)。

7条件×OFF/ONで以下を比較する:
- pilot detection rate / false detection rate (cyclo present率)
- squelch chatter count (stereo_blendの0.5閾値 crossing回数)
- audio SNR (1kHzトーン vs 噪音床) / RMS差 / 高域エネルギー差
- ステレオセパレーション / THD相当値
- CPU時間 / ブロック処理遅延 (最大)

Usage: python benchmark_black_magic.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests"))

from dsp import SdrDspPipeline
from rmt_denoiser import SafeRmtDenoiser

RF = 1152000
BLOCK = 132096
NBLK = 20  # S-meter収束 (τ0.3s) 後の定常で測るため20ブロック
N = (BLOCK // 2) * NBLK


def make_iq(stereo=True, snr_db=None, seed=7):
    t = np.arange(N) / RF
    left = np.sin(2 * np.pi * 1000.0 * t)
    right = np.sin(2 * np.pi * 5000.0 * t)
    if stereo:
        mpx = (0.45 * (left + right)
               + 0.45 * (left - right) * np.sin(2 * np.pi * 38000.0 * t)
               + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    else:
        mpx = left + right
    mpx = mpx / (np.max(np.abs(mpx)) + 1e-9)
    phase = 2 * np.pi * 30000.0 * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * phase)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p_noise = 0.36 / (10 ** (snr_db / 10.0))
        iq = iq + np.sqrt(p_noise / 2.0) * (
            rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return iq


def to_raw(iq):
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    return raw


def decode(raw, enable_bm):
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.afc_enabled = False
    dsp.cognitive_enabled = False
    dsp.slow_agc_enabled = False
    if enable_bm:
        dsp.black_magic_enabled = True
        dsp.bm_cyclo_enabled = True
        dsp.bm_rmt_enabled = True
        dsp.bm_sr_enabled = True
    chunks = []
    blends = []
    presents = []
    confs = []
    lat = []
    t_all = time.perf_counter()
    for k in range(NBLK):
        t0 = time.perf_counter()
        audio, _ = dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode="WFM")
        lat.append((time.perf_counter() - t0) * 1000.0)
        a = np.asarray(audio)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        chunks.append(a)
        blends.append(float(dsp.stereo_blend))
        cyc = getattr(dsp, "cyclo_detector", None)
        presents.append(bool(getattr(cyc, "pilot_present", False)) if cyc else False)
        confs.append(float(getattr(dsp, "bm_cyclo_confidence", 0.0)))
    total_ms = (time.perf_counter() - t_all) * 1000.0
    cat = np.concatenate(chunks, axis=0)
    return {"audio": cat, "blends": np.array(blends),
            "presents": np.array(presents), "confs": np.array(confs),
            "lat": np.array(lat), "total_ms": total_ms}


def amp(x, freq, sr=48000):
    # 定常部で評価する (先頭はS-meter収束・適応立上の過渡を含む)。
    # liveでは選局直後の一過性であり、定常ON==OFFが正しい比較になる。
    seg = x[3 * len(x) // 4: 3 * len(x) // 4 + sr]
    if len(seg) < 256:
        seg = x[len(x) // 2:]
    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs > freq - 60) & (freqs < freq + 60)
    return float(np.max(spectrum[mask]) + 1e-12)


def band_energy(x, lo, hi, sr=48000):
    # 定常部で評価する (ampと同じ理由)。短い入力では全体を使う。
    seg = x[3 * len(x) // 4:] if len(x) > 4 * sr else x[-2 * sr:]
    # 信号端の不連続が広帯域漏れとして高域を水増しするためHann窓を掛ける
    w = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * w))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    m = (freqs >= lo) & (freqs < hi)
    return float(np.sum(spectrum[m] ** 2) + 1e-18)


def metrics(audio):
    L = audio[:, 0].astype(np.float64)
    R = audio[:, 1].astype(np.float64)
    tone = amp(L, 1000.0)
    floor = np.median([band_energy(L, 6000, 7000), band_energy(L, 9000, 10000)])
    snr = 20.0 * np.log10(tone / (np.sqrt(floor) + 1e-12))
    sep = 20.0 * np.log10((amp(L, 1000.0) + amp(R, 5000.0) + 1e-12)
                          / (amp(L, 5000.0) + amp(R, 1000.0) + 1e-12))
    harm = sum(amp(L, f) ** 2 for f in (2000.0, 3000.0, 4000.0, 5000.0))
    thd = 10.0 * np.log10((harm + 1e-18) / (tone ** 2 + 1e-18))
    hf = 10.0 * np.log10(band_energy(L, 10000, 15000) + 1e-18)
    # 高域はトーン基準の絶対値でも評価する (無音床との比は発散して
    # 実害を見誤る。RMT境界バーストはトーン比-56dBで可聴限界以下)
    hf_rel_tone = 10.0 * float(np.log10((band_energy(L, 10000, 15000) + 1e-18)
                                        / (tone ** 2 + 1e-18)))
    rms = float(np.sqrt(np.mean(L ** 2)) + 1e-18)
    return {"snr": snr, "sep": sep, "thd": thd, "hf": hf,
            "hf_rel_tone": hf_rel_tone, "rms": rms}


def chatter(blends, thr=0.5):
    s = (blends > thr).astype(int)
    return int(np.sum(np.abs(np.diff(s))))


def run_case(name, iq):
    raw = to_raw(iq)
    off = decode(raw, False)
    on = decode(raw, True)
    mo, mn = metrics(off["audio"]), metrics(on["audio"])
    print(f"--- {name} ---")
    print(f"  pilot検出率 OFF n/a ON {np.mean(on['presents']):.2f} "
          f"(conf {np.mean(on['confs']):.2f})")
    print(f"  chatter OFF {chatter(off['blends'])} ON {chatter(on['blends'])} "
          f"(blend終値 {off['blends'][-1]:.2f}/{on['blends'][-1]:.2f})")
    print(f"  audioSNR OFF {mo['snr']:.1f} ON {mn['snr']:.1f} dB "
          f"(Δ {mn['snr'] - mo['snr']:+.1f})")
    print(f"  RMS比 {20 * np.log10(mn['rms'] / mo['rms']):+.2f} dB, "
          f"高域差 {mn['hf'] - mo['hf']:+.2f} dB "
          f"(トーン比 {mn['hf_rel_tone']:.1f}/{mo['hf_rel_tone']:.1f}dB)")
    print(f"  分離度 OFF {mo['sep']:.1f} ON {mn['sep']:.1f} dB, "
          f"THD OFF {mo['thd']:.1f} ON {mn['thd']:.1f} dB")
    print(f"  CPU OFF {off['total_ms']:.0f}ms ON {on['total_ms']:.0f}ms "
          f"(最大ブロック {np.max(off['lat']):.1f}/{np.max(on['lat']):.1f}ms)")
    return {"off": mo, "on": mn}


def main():
    rng = np.random.default_rng(11)
    # 1. FMステレオ＋白色雑音
    run_case("1. stereo+white SNR10", make_iq(True, 10.0))
    # 1b. 弱パイロット (RF SNR3dB: ネイティブPLLはロックするが余裕なし。
    # cycloの検出収束と非介入を確認する条件)
    run_case("1b. WEAK pilot RF-SNR3", make_iq(True, 3.0))
    # 2. ＋インパルスノイズ (0.1%スパイク)
    iq = make_iq(True, 15.0)
    idx = rng.choice(N, N // 1000, replace=False)
    iq[idx] *= 5.0
    run_case("2. stereo+impulse", iq)
    # 3. ＋周波数オフセット (搬送波+3kHz)
    t = np.arange(N) / RF
    run_case("3. carrier-offset +3kHz", make_iq(True, 15.0) * np.exp(2j * np.pi * 3000.0 * t))
    # 4. ＋マルチパス (8.7μsエコー×0.5)
    iq = make_iq(True, 15.0)
    d = 10
    iq4 = iq.copy()
    iq4[d:] += 0.5 * iq[:-d]
    run_case("4. multipath echo", iq4)
    # 5. パイロットなしノイズ (誤検出評価)
    iq5 = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.05
    r = run_case("5. noise-only (false detect)", iq5)
    # 6. 隣接強信号 (+200kHzに+20dBのFM)
    iq6 = make_iq(True, 15.0) + 10.0 * make_iq(True, None, seed=3) * np.exp(
        2j * np.pi * 200000.0 * t)
    run_case("6. adjacent +20dB", iq6)
    # 7. 音声＋定常ヒス (RMT直接評価)
    print("--- 7. audio+hiss (RMT direct) ---")
    dn = SafeRmtDenoiser()
    ta = np.arange(48000) / 48000.0
    clean = 0.05 * np.sin(2 * np.pi * 1000.0 * ta)
    x = (clean + 0.05 * rng.standard_normal(48000)).astype(np.float32)
    outs = []
    for k in range(0, 48000, 2752):
        y, info = dn.process_mono(x[k:k + 2752], s_meter_dbfs=-40.0, snr_db=12.0)
        outs.append(y)
    y = np.concatenate(outs)
    # ラッパー出力はlookahead整合で15サンプル遅延するため整合して評価する
    # (未整合だと1kHzの位相ずれでSNRが-2dB台に悪化表示される実測あり)
    D = 15
    ya = y[D:]
    ca = clean[:len(ya)]
    xa = x[:len(ya)]
    err_in = float(np.mean((xa.astype(np.float64) - ca) ** 2))
    err_out = float(np.mean((ya.astype(np.float64) - ca) ** 2))
    snr_in = 10 * np.log10(np.mean(ca ** 2) / err_in)
    snr_out = 10 * np.log10(np.mean(ca ** 2) / (err_out + 1e-18))
    print(f"  RMT SNR {snr_in:.1f} -> {snr_out:.1f} dB (Δ {snr_out - snr_in:+.1f})")
    print(f"  RMS差 {20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) / np.sqrt(np.mean(x.astype(np.float64) ** 2))):+.2f} dB, "
          f"rank {info['retained_rank']}, noise {info['estimated_noise_power']:.2e}, "
          f"{info['processing_ms']:.2f}ms/block")


if __name__ == "__main__":
    main()
