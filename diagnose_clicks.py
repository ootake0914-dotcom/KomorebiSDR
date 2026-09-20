"""
Click/Pop Noise Diagnostic Tool.
プツプツ・ブツブツ音の原因（ADC飽和、バッファ境界不連続、アンダーラン）を精密診断。
"""

import time
import numpy as np
import wave
from PIL import Image, ImageDraw, ImageFont

from rtlsdr_driver import RtlSdrDriver
from dsp import SdrDspPipeline


def run_diagnostics(freq_hz=94600000, duration=2.0, out_img="clicks_analysis.png"):
    print(f"[*] ブツブツ音の診断開始: {freq_hz / 1e6:.4f} MHz, 計測時間: {duration}秒")
    sample_rate = 1152000
    audio_rate = 48000
    offset = 150000.0

    driver = RtlSdrDriver()
    driver.open(0)
    driver.set_sample_rate(sample_rate)
    driver.set_direct_sampling(0)
    driver.set_center_freq(int(freq_hz + offset))
    driver.set_gain_mode(True)
    driver.set_gain(29.7)

    dsp = SdrDspPipeline(sample_rate, audio_rate)
    dsp.set_offset_freq(offset)

    chunk_size = 131072
    num_chunks = 0
    raw_iq_clipped_count = 0
    total_iq_samples = 0

    audio_chunks = []
    chunk_lengths = []

    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            raw = driver.read_sync(chunk_size)
            if len(raw) == 0:
                continue

            # ADC飽和判定 (0または255のサンプル数)
            clipped_iq = np.sum((raw <= 1) | (raw >= 254))
            raw_iq_clipped_count += clipped_iq
            total_iq_samples += len(raw)

            pcm, _ = dsp.process(raw, mode="WFM")
            audio_chunks.append(pcm)
            chunk_lengths.append(len(pcm))
            num_chunks += 1

    finally:
        driver.close()

    all_audio = np.concatenate(audio_chunks)
    total_audio_len = len(all_audio)

    # 1. ADC飽和率
    iq_clip_ratio = (raw_iq_clipped_count / total_iq_samples) * 100.0
    print(f"• ADC飽和サンプル率 (0/255張り付き): {iq_clip_ratio:.4f}% ({raw_iq_clipped_count}/{total_iq_samples})")

    # 2. チャンク境界での不連続点解析
    # チャンク境界インデックス
    boundaries = np.cumsum(chunk_lengths)[:-1]
    print(f"• 処理チャンク数: {num_chunks} 個, 1チャンクあたりのオーディオサンプル数: ~{chunk_lengths[0]} (約{chunk_lengths[0]/audio_rate*1000:.1f}ms周期)")

    # 差分（微分）によるクリック検出
    diffs = np.abs(np.diff(all_audio))
    threshold_click = 0.15  # 1サンプルで0.15以上の跳躍
    click_indices = np.where(diffs > threshold_click)[0]

    # チャンク境界付近（±10サンプル以内）で起きたクリックの割合
    boundary_clicks = 0
    for c_idx in click_indices:
        if np.any(np.abs(boundaries - c_idx) < 15):
            boundary_clicks += 1

    print(f"• 検出されたクリックノイズ数 (跳躍 > {threshold_click}): {len(click_indices)} 箇所")
    if len(click_indices) > 0:
        print(f"• そのうちバッファ境界で発生した割合: {boundary_clicks}/{len(click_indices)} ({boundary_clicks/len(click_indices)*100:.1f}%)")

    # 3. グラフ生成
    width, height = 1000, 500
    img = Image.new("RGB", (width, height), color=(18, 22, 30))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_label = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = font_label = ImageFont.load_default()

    draw.text((30, 15), f"CLICK / POP NOISE DIAGNOSTICS - Boundary Clicks: {boundary_clicks}/{len(click_indices)} | ADC Clip: {iq_clip_ratio:.2f}%", fill=(0, 230, 180), font=font_title)

    # 最初の約100ms（約2〜3チャンク分）の波形をズームして可視化
    zoom_len = min(len(all_audio), 6000)
    zoom_wave = all_audio[:zoom_len]

    x0, y0, x1, y1 = 60, 60, 940, 420
    draw.rectangle([x0, y0, x1, y1], fill=(24, 29, 40), outline=(50, 60, 80))

    # 中心線 (0V)
    cy = (y0 + y1) // 2
    draw.line([(x0, cy), (x1, cy)], fill=(50, 60, 80), width=1)

    # バッファ境界の垂直線
    for b in boundaries:
        if b < zoom_len:
            bx = x0 + (b / zoom_len) * (x1 - x0)
            draw.line([(bx, y0), (bx, y1)], fill=(255, 60, 80), width=1)
            draw.text((bx + 3, y0 + 5), "Chunk Boundary", fill=(255, 80, 100), font=font_label)

    # 波形描画
    pts = []
    for idx, val in enumerate(zoom_wave):
        px = x0 + (idx / (zoom_len - 1)) * (x1 - x0)
        # -1.0..1.0 を y1..y0 に写像
        py = cy - val * (y1 - y0) * 0.45
        pts.append((px, py))
    draw.line(pts, fill=(0, 255, 170), width=2)

    # クリック箇所を赤い丸で強調
    for c in click_indices:
        if c < zoom_len:
            cx_pt = x0 + (c / (zoom_len - 1)) * (x1 - x0)
            cy_pt = cy - zoom_wave[c] * (y1 - y0) * 0.45
            draw.ellipse([cx_pt - 4, cy_pt - 4, cx_pt + 4, cy_pt + 4], fill=(255, 0, 0), outline=(255, 255, 255))

    img.save(out_img)
    print(f"[*] 診断グラフを保存しました: {out_img}")

    return {
        "iq_clip_ratio": iq_clip_ratio,
        "num_chunks": num_chunks,
        "click_count": len(click_indices),
        "boundary_click_ratio": boundary_clicks / max(1, len(click_indices)),
    }

if __name__ == "__main__":
    run_diagnostics()
