"""
GUIの美しいスクリーンショットを生成するスクリプト (実機占有なし)
"""
import os
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()

from gui import SdrGui

def generate_screenshot(output_path="assets/screenshot.png"):
    gui = SdrGui()
    gui.current_freq = 80000000
    gui.station_name = "TOKYO FM"
    gui.scan_status_text = "Receiving: 80.00 MHz - TOKYO FM (Stereo Hi-Fi)"
    gui.telemetry_text = "SNR: 28.5dB | IQ: 48 | DSP: OK | Gain: Auto | Lock: Fixed | SIC: -18.2dB"

    # 美しいFM放送スペクトラムのシミュレーション
    freqs = np.linspace(-576, 576, 1024)
    # ベースのノイズフロア (-65dB) + 緩やかなロールオフ
    floor = -65.0 + 2.5 * np.sin(freqs / 80.0) + np.random.randn(1024) * 1.2
    # 80.0MHz TOKYO FM (中央 0kHz に ±75kHz のFM変調スペクトル)
    fm_peak = 48.0 * np.exp(-0.5 * (freqs / 45.0) ** 2)
    # 隣接局 (±200kHz に微弱局)
    adj1 = 22.0 * np.exp(-0.5 * ((freqs - 200.0) / 30.0) ** 2)
    adj2 = 18.0 * np.exp(-0.5 * ((freqs + 200.0) / 30.0) ** 2)
    spec_base = floor + fm_peak + adj1 + adj2

    # オーディオ波形 (美しい音楽正弦波の合成)
    t = np.arange(4800) / 48000.0
    pcm = (0.55 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t + 0.4) + 0.15 * np.sin(2 * np.pi * 1760 * t)).astype(np.float32)

    # ウォーターフォールを綺麗に埋めるため 60 フレーム描画
    for frame in range(70):
        # 音楽・音声の揺らぎを付加
        wobble = np.random.randn(1024) * 0.8
        instant_spec = (spec_base + wobble).astype(np.float32)
        gui.render(instant_spec, pcm)

    pygame.image.save(gui.screen, output_path)
    print(f"Screenshot saved successfully to {output_path}")

if __name__ == "__main__":
    generate_screenshot()
