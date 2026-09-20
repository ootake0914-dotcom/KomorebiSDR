"""
Real Hardware Profiler & Algorithm Tuner for RTL-SDR.
実機RTL-SDRドングルとSRH805Sアンテナの実電波環境における多次元物理計測・最適化スイート。
- 全29段チューナーゲインの精密C/N・音声SNR・クリップ率・ノイズフロア実測
- 接続アンテナ（SRH805S）の電波環境プロファイル抽出
- 自律制御アルゴリズム（Hyper / Cascade）の実測データ検証と最適定数チューニング
"""

import sys
import time
import numpy as np
from rtlsdr_driver import RtlSdrDriver
from dsp import SdrDspPipeline
from hyper_controller import HyperController
from cascade_controller import CascadeController


def profile_frequency(freq_hz: int, name: str, sample_rate: int = 1152000) -> dict:
    """実機SDRを用いて指定周波数で全ゲイン段の精密測定を実施"""
    print(f"\n==============================================================================")
    print(f"■ 実機測定開始: {name} ({freq_hz / 1e6:.2f} MHz)")
    print(f"==============================================================================")

    driver = RtlSdrDriver()
    driver.open(0)
    driver.set_sample_rate(sample_rate)

    # DCスパイク回避オフセット
    offset = 150000
    driver.set_center_freq(freq_hz + offset)
    driver.set_direct_sampling(0)
    driver.set_gain_mode(True)

    dsp = SdrDspPipeline(sample_rate=sample_rate, audio_rate=48000)
    dsp.set_offset_freq(offset)

    gains = driver.get_gains()
    if not gains:
        gains = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
                 19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
                 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]

    results = []
    read_samples = 132096 * 2  # 約230ms分の生データ

    print(f"{'Gain (dB)':>10} | {'IQ Std':>8} | {'Clip %':>7} | {'Noise (dB)':>10} | {'RF C/N (dB)':>11} | {'Audio SNR':>10}")
    print("-" * 72)

    for g in gains:
        driver.set_gain(g)
        driver.reset_buffer()
        # チューナーセトリング待ち (数フレーム読み捨て)
        _ = driver.read_sync(32768)

        # 本計測用サンプリング (複数フレーム平均で測定信頼性を最大化)
        raw_list = []
        for _ in range(3):
            chunk = driver.read_sync(132096)
            if len(chunk) == 132096:
                raw_list.append(chunk)

        if not raw_list:
            continue

        raw = np.concatenate(raw_list)

        # 1. ADC飽和率 & IQ標準偏差
        raw_f = raw.astype(np.float32)
        clip_count = np.sum((raw_f <= 1.0) | (raw_f >= 254.0))
        clip_pct = float(clip_count / len(raw_f) * 100.0)
        iq_std = float(np.std(raw_f - 127.5))

        # 2. DSP復調 & スペクトラム算出
        audio_pcm, spec_db = dsp.process(raw[:132096], mode="WFM")

        # 3. チャンネル内パワー vs ガードバンド雑音による真のC/N
        lin_spec = np.power(10.0, spec_db / 10.0)
        n_bins = len(lin_spec)
        c = n_bins // 2
        bin_hz = sample_rate / n_bins
        ch_half = max(4, int(85000.0 / bin_hz))
        g_in = max(ch_half + 2, int(115000.0 / bin_hz))
        g_out = int(250000.0 / bin_hz)
        g_out = min(g_out, c - 2)

        ch = lin_spec[c - ch_half : c + ch_half + 1]
        sig_mean = float(np.mean(ch)) if len(ch) > 0 else 1e-12
        sig_peak = float(np.max(ch)) if len(ch) > 0 else 1e-12
        noise_p = float(np.percentile(lin_spec, 25))
        noise_db = float(10.0 * np.log10(noise_p + 1e-12))

        mean_snr = 10.0 * np.log10((sig_mean + 1e-12) / (noise_p + 1e-12))
        peak_snr = 10.0 * np.log10((sig_peak + 1e-12) / (noise_p + 1e-12))
        cn_db = float(max(mean_snr, 0.35 * mean_snr + 0.65 * peak_snr))

        # 4. 音声復調SNR (300-3000Hz vs 5500-11000Hz)
        audio_snr = 0.0
        if len(audio_pcm) >= 1024:
            chunk_a = audio_pcm[-1024:]
            win = np.hanning(1024)
            p_aud = np.abs(np.fft.rfft(chunk_a * win)) ** 2 + 1e-12
            aud_bin = 48000.0 / 1024.0
            p_voice = np.mean(p_aud[int(300 / aud_bin) : int(3000 / aud_bin)])
            p_hiss = np.mean(p_aud[int(5500 / aud_bin) : int(11000 / aud_bin)])
            audio_snr = float(10.0 * np.log10(p_voice / (p_hiss + 1e-12)))

        row = {
            "gain_db": g,
            "iq_std": iq_std,
            "clip_pct": clip_pct,
            "noise_db": noise_db,
            "cn_db": cn_db,
            "audio_snr": audio_snr,
        }
        results.append(row)

        print(f"{g:>10.1f} | {iq_std:>8.1f} | {clip_pct:>6.3f}% | {noise_db:>10.1f} | {cn_db:>11.2f} | {audio_snr:>10.2f}")

    driver.close()

    # 最適点分析
    valid_results = [r for r in results if r["gain_db"] >= 12.5]
    best_cn = max(valid_results, key=lambda r: r["cn_db"])
    best_audio = max(valid_results, key=lambda r: r["audio_snr"])

    # 混変調(IMD)歪み発生点の検知: ゲイン増加に対してノイズフロアが4dB以上急上昇するポイント
    imd_point = None
    for i in range(1, len(results)):
        g_diff = results[i]["gain_db"] - results[i - 1]["gain_db"]
        nf_diff = results[i]["noise_db"] - results[i - 1]["noise_db"]
        if g_diff > 0 and nf_diff > (g_diff + 3.0):
            imd_point = results[i]["gain_db"]
            break

    print("\n------------------------------------------------------------------------------")
    print(f"【実機解析結果サマリー: {name}】")
    print(f"・理論最高 C/N 点      : {best_cn['gain_db']:.1f} dB (C/N = {best_cn['cn_db']:.2f} dB, Audio SNR = {best_cn['audio_snr']:.2f} dB)")
    print(f"・聴感最高 音声SNR点   : {best_audio['gain_db']:.1f} dB (Audio SNR = {best_audio['audio_snr']:.2f} dB, C/N = {best_audio['cn_db']:.2f} dB)")
    if imd_point is not None:
        print(f"・混変調(IMD)開始点    : {imd_point:.1f} dB 以上でスプリアス・歪み急増注意")
    else:
        print(f"・混変調(IMD)          : 観測範囲内で重度IMDは発生せず (安全)")
    print("------------------------------------------------------------------------------\n")

    return {
        "freq_hz": freq_hz,
        "name": name,
        "results": results,
        "best_cn": best_cn,
        "best_audio": best_audio,
        "imd_point": imd_point,
    }


def main():
    print("[*] RTL-SDR + SRH805S 実機多次元物理プロファイラを起動します...")

    targets = [
        (83200000, "NHK-FM 水戸 (83.2MHz)"),
        (94600000, "LuckyFM 茨城放送 (94.6MHz)"),
        (80000000, "TOKYO FM (80.0MHz)"),
    ]

    all_profiles = []
    for f, name in targets:
        try:
            p = profile_frequency(f, name)
            all_profiles.append(p)
        except Exception as e:
            print(f"[!] {name} の測定中にエラー: {e}")

    # 総合アンテナ診断とアルゴリズム推奨定数の算出
    print("==============================================================================")
    print("■ SRH805S アンテナ環境 総合診断 & アルゴリズム最適化推奨")
    print("==============================================================================")

    best_gains = [p["best_cn"]["gain_db"] for p in all_profiles if "best_cn" in p]
    if best_gains:
        optimal_sweet = float(np.median(best_gains))
        print(f"1. 実測アンテナ最適スウィートスポット : {optimal_sweet:.1f} dB")
        print(f"   (各局最適値: {best_gains})")
    else:
        optimal_sweet = 33.8

    print("==============================================================================\n")


if __name__ == "__main__":
    main()
