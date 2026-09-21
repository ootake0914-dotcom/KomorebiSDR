"""
PC直挿し状態での内部スプリアス・干渉波測定スクリプト
"""
import sys
import time
import numpy as np
from rtlsdr_driver import RtlSdrDriver

def measure_spurious(center_freq=80.0e6, sample_rate=1.152e6, gain="auto", duration_sec=1.0):
    drv = RtlSdrDriver()
    if drv.get_device_count() == 0:
        print("RTL-SDRデバイスが見つかりません。")
        return

    print(f"RTL-SDRオープン中... (中心周波数: {center_freq/1e6:.2f} MHz, サンプリングレート: {sample_rate/1e6:.3f} MSps)")
    drv.open(0)
    drv.set_sample_rate(int(sample_rate))
    drv.set_center_freq(int(center_freq))
    if gain == "auto":
        drv.set_tuner_gain_mode(False)
    else:
        drv.set_tuner_gain_mode(True)
        drv.set_tuner_gain(int(gain * 10))

    # ウォームアップ読み込み (過渡特性安定化)
    time.sleep(0.1)
    _ = drv.read_samples(32768)

    # 本測定
    num_samples = int(sample_rate * duration_sec)
    # バッチサイズに合わせて複数回読み込み
    batch_size = 65536
    all_samples = []
    collected = 0
    while collected < num_samples:
        n = min(batch_size, num_samples - collected)
        buf = drv.read_samples(n)
        if len(buf) == 0:
            break
        all_samples.append(buf)
        collected += len(buf)

    drv.close()
    if not all_samples:
        print("サンプルの取得に失敗しました。")
        return

    iq = np.concatenate(all_samples)
    print(f"取得サンプル数: {len(iq)} ({len(iq)/sample_rate:.2f} 秒間)")

    # FFT パワースペクトラム解析 (ウェルチ法風 平均)
    nfft = 4096
    step = nfft // 2
    num_segments = (len(iq) - nfft) // step
    psd_accum = np.zeros(nfft, dtype=np.float64)
    win = np.hanning(nfft)

    for i in range(num_segments):
        seg = iq[i * step : i * step + nfft] * win
        fft_v = np.fft.fftshift(np.fft.fft(seg))
        psd_accum += np.abs(fft_v) ** 2

    psd = psd_accum / max(1, num_segments)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))

    med_floor = float(np.median(psd_db))
    p90_floor = float(np.percentile(psd_db, 75))
    noise_floor_est = med_floor

    print("\n==========================================")
    print(f" [PC直挿し内部ノイズ・スプリアス診断結果]")
    print(f" 中心周波数: {center_freq/1e6:.3f} MHz")
    print(f" 推定ノイズフロア: {noise_floor_est:.1f} dB (相対値)")
    print("==========================================")

    # 局所ピーク検出 (ノイズフロアから+10dB以上突出)
    peaks = []
    for i in range(2, nfft - 2):
        val = psd_db[i]
        # DC近傍 (中央 ±3ビン) はチューナーLOリークなのでPCスプリアスとは区別
        if abs(i - nfft // 2) <= 3:
            continue
        if val > noise_floor_est + 10.0:
            if val > psd_db[i-1] and val > psd_db[i+1] and val > psd_db[i-2] and val > psd_db[i+2]:
                diff_db = val - noise_floor_est
                f_rel_khz = freqs[i] / 1e3
                f_abs_mhz = (center_freq + freqs[i]) / 1e6
                peaks.append((diff_db, f_rel_khz, f_abs_mhz))

    peaks.sort(key=lambda x: x[0], reverse=True)

    if not peaks:
        print("突出したスプリアスは検出されませんでした（良好、または微弱）。")
    else:
        print(f"検出された突出スプリアス・クロックビート: {len(peaks)} 件")
        print(f"{'順位':<4} {'オフセット(kHz)':<16} {'絶対周波数(MHz)':<18} {'フロア比(dB)':<12}")
        print("-" * 54)
        for rank, (diff, f_rel, f_abs) in enumerate(peaks[:10], 1):
            print(f"{rank:<4} {f_rel:>+10.2f} kHz   {f_abs:>12.4f} MHz   {diff:>+8.1f} dB")

    return peaks

if __name__ == "__main__":
    freq = 80.0e6
    if len(sys.argv) > 1:
        freq = float(sys.argv[1]) * 1e6
    measure_spurious(center_freq=freq)
