"""
Noise and Audio Spectral Analyzer.
復調された音声信号の周波数分布、19kHzパイロットトーン、高域ノイズ比率、周波数ズレ(PPM)を精密分析。
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wave

def analyze_audio_file(wav_path: str, out_img: str = "audio_spectrum.png"):
    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        fs = wf.getframerate()
        data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"[*] 音声ファイル解析: {wav_path}")
    print(f"• サンプル数: {len(samples)} ({len(samples)/fs:.2f}秒, {fs}Hz)")

    # FFT解析
    n_fft = 4096
    window = np.hanning(n_fft)
    n_chunks = len(samples) // n_fft
    if n_chunks == 0:
        print("[!] サンプル数が不足しています")
        return

    spec_accum = np.zeros(n_fft // 2, dtype=np.float64)
    for i in range(n_chunks):
        chunk = samples[i * n_fft : (i + 1) * n_fft] * window
        fft_res = np.abs(np.fft.rfft(chunk)[:-1]) ** 2
        spec_accum += fft_res

    spec_avg = spec_accum / n_chunks + 1e-12
    spec_db = 10.0 * np.log10(spec_avg)
    freqs = np.linspace(0, fs / 2, len(spec_db))

    # 帯域別のエネルギー比率
    band_voice = np.mean(spec_avg[(freqs >= 300) & (freqs <= 3500)]) # 人の声 (300Hz-3.5kHz)
    band_high = np.mean(spec_avg[(freqs >= 5000) & (freqs <= 15000)]) # 高域ノイズ (5kHz-15kHz)
    band_pilot = np.mean(spec_avg[(freqs >= 18500) & (freqs <= 19500)]) # 19kHzパイロット

    print(f"• 音声帯域 (300Hz-3.5kHz) 平均パワー: {10*np.log10(band_voice):.1f} dB")
    print(f"• 高域ノイズ帯域 (5kHz-15kHz) 平均パワー: {10*np.log10(band_high):.1f} dB")
    print(f"• 19kHzパイロットトーン帯域 平均パワー: {10*np.log10(band_pilot):.1f} dB")
    high_to_voice_ratio = 10 * np.log10(band_high / band_voice)
    print(f"• 音声に対する高域ノイズ比: {high_to_voice_ratio:+.1f} dB")

    # スペクトル画像を生成
    width, height = 900, 450
    img = Image.new("RGB", (width, height), color=(18, 22, 30))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_label = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = font_label = ImageFont.load_default()

    draw.text((30, 15), f"AUDIO SPECTRUM ANALYSIS - High/Voice Noise Ratio: {high_to_voice_ratio:+.1f} dB", fill=(0, 230, 180), font=font_title)

    x0, y0, x1, y1 = 60, 50, 840, 380
    draw.rectangle([x0, y0, x1, y1], fill=(24, 29, 40), outline=(50, 60, 80))

    db_min, db_max = -80.0, 0.0
    for db in range(-80, 1, 10):
        r = (db - db_min) / (db_max - db_min)
        gy = y1 - r * (y1 - y0)
        draw.line([(x0, gy), (x1, gy)], fill=(38, 45, 62), width=1)
        draw.text((x0 - 45, gy - 6), f"{db}dB", fill=(130, 140, 160), font=font_label)

    # 折れ線
    norm_y = np.clip((spec_db - db_min) / (db_max - db_min), 0.0, 1.0)
    x_step = (x1 - x0) / (len(norm_y) - 1)
    pts = []
    for idx, y_val in enumerate(norm_y):
        px = x0 + idx * x_step
        py = y1 - y_val * (y1 - y0)
        pts.append((px, py))
    draw.line(pts, fill=(0, 255, 170), width=2)

    # 周波数ラベル
    for f_khz in [0, 5, 10, 15, 19, 24]:
        px = x0 + (f_khz * 1000 / (fs / 2)) * (x1 - x0)
        draw.line([(px, y1), (px, y1 + 5)], fill=(130, 140, 160), width=1)
        draw.text((px - 15, y1 + 8), f"{f_khz}kHz", fill=(130, 140, 160), font=font_label)
        if f_khz == 19:
            draw.line([(px, y0), (px, y1)], fill=(255, 100, 100), width=1)
            draw.text((px - 20, y0 + 10), "19k Pilot", fill=(255, 100, 100), font=font_label)

    img.save(out_img)
    print(f"[*] 音声スペクトル画像を保存しました: {out_img}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "lucky_hifi.wav"
    analyze_audio_file(path)
