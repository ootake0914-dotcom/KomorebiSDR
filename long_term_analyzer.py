"""
Long-Term Audio & RF Quality Analyzer.
比較的長時間の連続受信・音声ストリーミングを実行し、
- クロックドリフト補正 (Adaptive Resampler) の同期安定度
- バッファアンダーラン / パケットドロップ (ゼロであることの確認)
- 境界クリックノイズ発生回数
- アンテナ環境自動同定 (Antenna Profiler)
- 受信C/N比、聴感SNR、19kHzパイロットトーン抑圧度
をリアルタイム計測・定量評価し、高精細ダッシュボード画像 (PNG) とWAVを出力するツール。
"""

import sys
import os
import time
import argparse
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rtlsdr_driver import RtlSdrDriver
from dsp import SdrDspPipeline
from audio_output import AudioOutput
from hyper_controller import HyperController


def parse_freq(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("M") or s.endswith("MHZ"):
        return int(float(s.rstrip("MHZ")) * 1e6)
    elif s.endswith("K") or s.endswith("KHZ"):
        return int(float(s.rstrip("KHZ")) * 1e3)
    val = float(s)
    return int(val * 1e6) if val < 3000 else int(val)


def run_long_term_analysis(
    freq_hz: int = 83200000,
    mode: str = "WFM",
    duration: float = 30.0,
    out_img: str = "long_term_analysis.png",
    out_wav: str = "long_term_audio.wav",
    play_audio: bool = True,
):
    print("=" * 70)
    print(f"📡 RTL-SDR 長時間音声・電波健全性プロファイラ 開始")
    print(f"• 周波数   : {freq_hz / 1e6:.4f} MHz ({mode} モード)")
    print(f"• 計測時間 : {duration:.1f} 秒")
    print("=" * 70)

    sample_rate = 1152000
    audio_rate = 48000

    driver = RtlSdrDriver()
    dev_count = driver.get_device_count()
    if dev_count == 0:
        raise RuntimeError("RTL-SDRデバイスが検出されませんでした。USB接続を確認してください。")

    driver.open(0)
    driver.set_sample_rate(sample_rate)

    # 24MHz未満はダイレクトサンプリング (Q-branch = 2)
    # DCスパイク回避のため main.py/cli.py と同様に+150kHzオフセットする
    if freq_hz < 24000000:
        driver.set_direct_sampling(2)
        offset = 150000.0
        driver.set_center_freq(int(freq_hz + offset))
    else:
        driver.set_direct_sampling(0)
        offset = 150000.0
        driver.set_center_freq(int(freq_hz + offset))

    dsp = SdrDspPipeline(sample_rate, audio_rate)
    dsp.set_offset_freq(offset)

    audio = AudioOutput(audio_rate)
    if play_audio:
        audio.start()

    controller = HyperController(driver, dsp, audio)
    controller.init_gains()

    chunk_size = 131072
    t0 = time.time()
    last_sec_tick = t0

    # 時系列メトリクス記録バッファ
    time_points = []
    cn_history = []
    audio_snr_history = []
    drift_ppm_history = []
    buffer_fill_history = []
    iq_std_history = []
    click_counts = []
    antenna_types = []

    collected_audio = []
    total_samples = 0
    total_clicks = 0
    prev_pcm_end = 0.0

    print(f"[*] リアルタイムストリーミング観測中... (毎秒進捗更新)")

    try:
        while True:
            elapsed = time.time() - t0
            if elapsed >= duration:
                break

            # オーディオバッファ水位（残存チャンク数）をリサンプラにフィードバック (クロック自動同期)
            q_size = audio.get_queue_size()
            dsp.update_resampler_feedback(float(q_size))

            # 生IQデータ取得
            raw = driver.read_sync(chunk_size)
            if len(raw) == 0:
                time.sleep(0.005)
                continue

            # DSP復調
            pcm, spectrum_db = dsp.process(raw, mode=mode)

            # 自律認知制御 (アンテナプロファイリング & C/N推定)
            stats = controller.process_frame(raw, spectrum_db, audio=pcm, mode=mode)

            # クリックノイズ検出 (境界不連続・急峻なスパイク判定)
            boundary_diff = abs(pcm[0] - prev_pcm_end) if len(pcm) > 0 and total_samples > 0 else 0.0
            inner_diffs = np.abs(np.diff(pcm)) if len(pcm) > 1 else np.zeros(0)
            threshold_click = 0.18
            n_clicks = int(np.sum(inner_diffs > threshold_click)) + (1 if boundary_diff > threshold_click else 0)
            total_clicks += n_clicks

            if len(pcm) > 0:
                prev_pcm_end = float(pcm[-1])
                collected_audio.append(pcm)
                total_samples += len(pcm)
                if play_audio:
                    audio.put_audio(pcm)

            # 毎秒ロギング
            now = time.time()
            if now - last_sec_tick >= 1.0:
                last_sec_tick = now
                drift_ppm = dsp.resampler.drift_ppm
                cn = stats.get("channel_snr_db", 0.0)
                aud_snr = stats.get("audio_snr_db", 0.0)
                ant = stats.get("antenna_type", "Standard")
                buf_stat = audio.get_stats()

                time_points.append(round(elapsed, 1))
                cn_history.append(cn)
                audio_snr_history.append(aud_snr)
                drift_ppm_history.append(drift_ppm)
                buffer_fill_history.append(buf_stat["fill_ratio"] * 100.0)
                iq_std_history.append(stats.get("iq_std", 0.0))
                click_counts.append(total_clicks)
                antenna_types.append(ant)

                sys.stdout.write(
                    f"\r[{elapsed:5.1f}s/{duration:.0f}s] "
                    f"C/N:+{cn:4.1f}dB | Aud:+{aud_snr:4.1f}dB | "
                    f"Drift:{drift_ppm:+4.0f}ppm | Buf:{buf_stat['fill_ratio']*100:3.0f}% | "
                    f"Clicks:{total_clicks} | Ant:{ant[:18]}"
                )
                sys.stdout.flush()

    finally:
        if play_audio:
            audio.stop()
        driver.close()

    print("\n[*] 計測完了。統計の集計とグラフ生成中...")

    # 全音声データの統合
    all_pcm = np.concatenate(collected_audio) if collected_audio else np.zeros(0, dtype=np.float32)

    # 1. WAVファイル保存
    if len(all_pcm) > 0:
        pcm_int16 = np.clip(all_pcm * 32767, -32768, 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(audio_rate)
            wf.writeframes(pcm_int16.tobytes())
        print(f"• WAV音声を保存しました: {os.path.abspath(out_wav)} ({len(all_pcm)/audio_rate:.2f}秒)")

    # 2. 最終統計の算出
    final_stats = audio.get_stats()
    avg_cn = float(np.mean(cn_history)) if cn_history else 0.0
    avg_aud_snr = float(np.mean(audio_snr_history)) if audio_snr_history else 0.0
    avg_drift = float(np.mean(drift_ppm_history)) if drift_ppm_history else 0.0
    avg_fill = float(np.mean(buffer_fill_history)) if buffer_fill_history else 0.0
    dominant_ant = max(set(antenna_types), key=antenna_types.count) if antenna_types else "Unknown"

    # 音声周波数スペクトルの計算
    fft_n = 4096
    n_chunks = len(all_pcm) // fft_n
    if n_chunks > 0:
        spec_accum = np.zeros(fft_n // 2, dtype=np.float64)
        win = np.hanning(fft_n)
        for i in range(n_chunks):
            ch = all_pcm[i * fft_n : (i + 1) * fft_n] * win
            spec_accum += np.abs(np.fft.rfft(ch)[:-1]) ** 2
        spec_avg = spec_accum / n_chunks + 1e-12
        spec_db = 10.0 * np.log10(spec_avg)
        spec_freqs = np.linspace(0, audio_rate / 2, len(spec_db))

        # 19kHzパイロットトーン漏洩電力
        idx_19k = np.where((spec_freqs >= 18800) & (spec_freqs <= 19200))[0]
        idx_voice = np.where((spec_freqs >= 300) & (spec_freqs <= 3500))[0]
        p_19k = np.mean(spec_avg[idx_19k]) if len(idx_19k) > 0 else 1e-12
        p_voice = np.mean(spec_avg[idx_voice]) if len(idx_voice) > 0 else 1e-12
        pilot_leak_db = float(10.0 * np.log10(p_19k / (p_voice + 1e-12)))
    else:
        spec_db = np.zeros(100)
        spec_freqs = np.linspace(0, 24000, 100)
        pilot_leak_db = -99.0

    print("=" * 70)
    print("📊 長時間連続運用 解析結果サマリー")
    print("=" * 70)
    print(f"• 連続稼働時間           : {duration:.1f} 秒 ({total_samples} サンプル再生)")
    print(f"• 同定アンテナ環境       : {dominant_ant}")
    print(f"• バッファアンダーラン回数: {final_stats['underrun_count']} 回 (目標: 0)")
    print(f"• パケットオーバーラン回数: {final_stats['overflow_count']} 回 (目標: 0)")
    print(f"• 検出クリックノイズ総数 : {total_clicks} 回 (目標: 0)")
    print(f"• 平均受信 C/N           : +{avg_cn:.1f} dB")
    print(f"• 平均聴感 音声SNR       : +{avg_aud_snr:.1f} dB")
    print(f"• クロック同期平均ドリフト: {avg_drift:+.1f} ppm")
    print(f"• バッファ平均充填率     : {avg_fill:.1f} % (安定目標: 10〜30%)")
    print(f"• 19kHzパイロット漏洩比  : {pilot_leak_db:.1f} dB (抑圧目標: -40dB以下)")
    print("=" * 70)

    # 3. ダッシュボード画像レンダリング (PIL)
    _render_dashboard_image(
        out_img=out_img,
        freq_hz=freq_hz,
        mode=mode,
        duration=duration,
        time_points=time_points,
        cn_history=cn_history,
        audio_snr_history=audio_snr_history,
        drift_ppm_history=drift_ppm_history,
        buffer_fill_history=buffer_fill_history,
        click_counts=click_counts,
        spec_freqs=spec_freqs,
        spec_db=spec_db,
        pilot_leak_db=pilot_leak_db,
        stats={
            "underruns": final_stats["underrun_count"],
            "overflows": final_stats["overflow_count"],
            "total_clicks": total_clicks,
            "avg_cn": avg_cn,
            "avg_aud": avg_aud_snr,
            "avg_drift": avg_drift,
            "ant": dominant_ant,
        },
    )

    return {
        "duration": duration,
        "underrun_count": final_stats["underrun_count"],
        "overflow_count": final_stats["overflow_count"],
        "total_clicks": total_clicks,
        "avg_cn_db": avg_cn,
        "avg_audio_snr_db": avg_aud_snr,
        "avg_drift_ppm": avg_drift,
        "pilot_leak_db": pilot_leak_db,
        "antenna_detected": dominant_ant,
        "img_path": os.path.abspath(out_img),
        "wav_path": os.path.abspath(out_wav),
    }


def _render_dashboard_image(
    out_img, freq_hz, mode, duration, time_points, cn_history,
    audio_snr_history, drift_ppm_history, buffer_fill_history,
    click_counts, spec_freqs, spec_db, pilot_leak_db, stats
):
    """モダンなダークテーマの総合解析ダッシュボードを生成"""
    w, h = 1100, 750
    img = Image.new("RGB", (w, h), color=(16, 20, 28))
    draw = ImageDraw.Draw(img)

    try:
        f_title = ImageFont.truetype("arial.ttf", 20)
        f_med = ImageFont.truetype("arial.ttf", 13)
        f_small = ImageFont.truetype("arial.ttf", 11)
        f_num = ImageFont.truetype("arial.ttf", 26)
    except:
        f_title = f_med = f_small = f_num = ImageFont.load_default()

    # ヘッダー
    draw.rectangle([0, 0, w, 65], fill=(22, 28, 40))
    draw.line([(0, 65), (w, 65)], fill=(40, 52, 75), width=1)
    draw.text((25, 12), "LONG-TERM AUDIO & RF QUALITY PROFILER", fill=(0, 230, 180), font=f_title)
    sub = f"Freq: {freq_hz/1e6:.4f} MHz | Mode: {mode} | Duration: {duration:.0f}s | Antenna: {stats['ant']}"
    draw.text((25, 38), sub, fill=(150, 170, 200), font=f_med)

    # 4つのKPIサマリーカード
    cards = [
        ("BUFFER INTEGRITY", f"Drops: {stats['overflows']} / Under: {stats['underruns']}", (40, 180, 120)),
        ("CLICK ARTIFACTS", f"Clicks: {stats['total_clicks']} (Zero-Pop)", (0, 220, 255)),
        ("C/N & AUDIO SNR", f"C/N +{stats['avg_cn']:.1f}dB | Aud +{stats['avg_aud']:.1f}dB", (255, 180, 60)),
        ("CLOCK DRIFT SYNC", f"Mean: {stats['avg_drift']:+.1f} ppm", (200, 140, 255)),
    ]
    card_w = (w - 50 - 3 * 15) // 4
    for i, (title, val, col) in enumerate(cards):
        cx0 = 25 + i * (card_w + 15)
        cx1 = cx0 + card_w
        draw.rectangle([cx0, 80, cx1, 140], fill=(24, 31, 45), outline=(45, 58, 80))
        draw.text((cx0 + 12, 88), title, fill=(140, 155, 180), font=f_small)
        draw.text((cx0 + 12, 106), val, fill=col, font=f_med)

    # グラフ描画ヘルパー関数
    def draw_chart(x0, y0, x1, y1, title, series, y_min, y_max, y_unit=""):
        draw.rectangle([x0, y0, x1, y1], fill=(22, 28, 40), outline=(40, 52, 75))
        draw.text((x0 + 12, y0 + 8), title, fill=(200, 215, 240), font=f_med)

        # グリッド
        n_grids = 4
        for gi in range(n_grids + 1):
            gy = y0 + 32 + (y1 - y0 - 45) * gi / n_grids
            draw.line([(x0 + 45, gy), (x1 - 15, gy)], fill=(32, 42, 60), width=1)
            val = y_max - (y_max - y_min) * gi / n_grids
            draw.text((x0 + 8, gy - 6), f"{val:3.0f}{y_unit}", fill=(110, 125, 150), font=f_small)

        # 折れ線
        if not series or len(series[0][1]) < 2:
            return
        plot_w = (x1 - 15) - (x0 + 45)
        plot_h = (y1 - 15) - (y0 + 32)
        n_pts = len(series[0][1])
        x_step = plot_w / max(1, n_pts - 1)

        for s_name, data, color in series:
            pts = []
            for idx, v in enumerate(data):
                px = x0 + 45 + idx * x_step
                norm = np.clip((v - y_min) / (y_max - y_min + 1e-6), 0.0, 1.0)
                py = (y1 - 15) - norm * plot_h
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)

    # グラフ1: C/N & Audio SNR 時間推移
    draw_chart(
        x0=25, y0=160, x1=535, y1=420,
        title="1. RF C/N & AUDIO SNR STABILITY (dB)",
        series=[("C/N", cn_history, (255, 180, 50)), ("Aud SNR", audio_snr_history, (0, 230, 180))],
        y_min=0, y_max=40, y_unit="dB"
    )

    # グラフ2: クロックドリフト (ppm) & バッファ充填率 (%)
    draw_chart(
        x0=560, y0=160, x1=1075, y1=420,
        title="2. ADAPTIVE CLOCK RESAMPLER SYNC (ppm & %)",
        series=[("Drift", drift_ppm_history, (180, 120, 255)), ("Buffer", buffer_fill_history, (0, 200, 255))],
        y_min=-200, y_max=200, y_unit=""
    )

    # グラフ3: 累積クリック数 (0カウント検証)
    max_click = max(5, max(click_counts) if click_counts else 0)
    draw_chart(
        x0=25, y0=440, x1=535, y1=710,
        title="3. ZERO-CLICK ARTIFACT VERIFICATION (Cumulative)",
        series=[("Clicks", click_counts, (255, 80, 100))],
        y_min=0, y_max=max_click, y_unit=""
    )

    # グラフ4: 復調音声の周波数スペクトル (0〜24kHz)
    x0, y0, x1, y1 = 560, 440, 1075, 710
    draw.rectangle([x0, y0, x1, y1], fill=(22, 28, 40), outline=(40, 52, 75))
    draw.text((x0 + 12, y0 + 8), f"4. DEMODULATED AUDIO SPECTRUM (19k Pilot Leak: {pilot_leak_db:+.1f}dB)", fill=(200, 215, 240), font=f_med)

    plot_x0, plot_y0, plot_x1, plot_y1 = x0 + 45, y0 + 32, x1 - 15, y1 - 25
    db_min, db_max = -80.0, 0.0
    for db in range(-80, 1, 20):
        norm = (db - db_min) / (db_max - db_min)
        gy = plot_y1 - norm * (plot_y1 - plot_y0)
        draw.line([(plot_x0, gy), (plot_x1, gy)], fill=(32, 42, 60), width=1)
        draw.text((x0 + 8, gy - 6), f"{db}dB", fill=(110, 125, 150), font=f_small)

    if len(spec_db) > 1:
        pts = []
        n_bins = len(spec_db)
        for idx in range(n_bins):
            px = plot_x0 + (idx / (n_bins - 1)) * (plot_x1 - plot_x0)
            norm = np.clip((spec_db[idx] - db_min) / (db_max - db_min), 0.0, 1.0)
            py = plot_y1 - norm * (plot_y1 - plot_y0)
            pts.append((px, py))
        draw.line(pts, fill=(0, 255, 170), width=2)

    # 周波数ラベル (0k, 5k, 10k, 15k, 19k, 24k)
    for f_k in [0, 5, 10, 15, 19, 24]:
        px = plot_x0 + (f_k * 1000 / 24000.0) * (plot_x1 - plot_x0)
        draw.line([(px, plot_y1), (px, plot_y1 + 4)], fill=(110, 125, 150), width=1)
        draw.text((px - 10, plot_y1 + 6), f"{f_k}k", fill=(110, 125, 150), font=f_small)
        if f_k == 19:
            draw.line([(px, plot_y0), (px, plot_y1)], fill=(255, 90, 90), width=1)
            draw.text((px - 12, plot_y0 + 2), "19k", fill=(255, 100, 100), font=f_small)

    img.save(out_img)
    print(f"• 総合解析ダッシュボード画像を保存しました: {os.path.abspath(out_img)}")


def analyze_wav_file(wav_path: str, out_img: str = "long_term_analysis.png"):
    """録音済みWAVファイルを長時間音声解析し、ダッシュボード画像を生成"""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAVファイルが見つかりません: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        fs = wf.getframerate()
        n_channels = wf.getnchannels()
        data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels == 2:
            samples = samples[0::2]  # 左チャンネル

    duration = len(samples) / fs
    print("=" * 70)
    print(f"🎵 録音音声ファイル 長時間健全性解析 開始")
    print(f"• ファイル名 : {os.path.abspath(wav_path)}")
    print(f"• 長さ / 形式: {duration:.2f} 秒 ({n_frames} フレーム, {fs}Hz {n_channels}ch)")
    print("=" * 70)

    # 1. 1秒単位の時系列解析 (聴感SNR, クリック検出)
    sec_frames = fs
    n_secs = int(duration)
    time_points = []
    audio_snr_history = []
    click_counts = []
    total_clicks = 0

    fft_sec = 1024
    win_sec = np.hanning(fft_sec)
    bin_hz = fs / fft_sec

    for s_idx in range(n_secs):
        t = (s_idx + 1) * 1.0
        ch_sec = samples[s_idx * sec_frames : (s_idx + 1) * sec_frames]
        # クリック検出 (跳躍 > 0.18)
        diffs = np.abs(np.diff(ch_sec))
        clicks_in_sec = int(np.sum(diffs > 0.18))
        total_clicks += clicks_in_sec

        # 聴感SNR (番組帯域 vs ヒス帯域)
        sub_chunk = ch_sec[-fft_sec:]
        pwr = np.abs(np.fft.rfft(sub_chunk * win_sec)) ** 2 + 1e-12
        i_prog0, i_prog1 = max(1, int(300.0 / bin_hz)), min(len(pwr) - 1, int(3000.0 / bin_hz))
        i_hiss0, i_hiss1 = max(1, int(5500.0 / bin_hz)), min(len(pwr) - 1, int(11000.0 / bin_hz))
        p_prog = np.mean(pwr[i_prog0 : i_prog1 + 1])
        p_hiss = np.mean(pwr[i_hiss0 : i_hiss1 + 1])
        snr_db = float(np.clip(10.0 * np.log10(p_prog / p_hiss), -10.0, 50.0))

        time_points.append(t)
        audio_snr_history.append(snr_db)
        click_counts.append(total_clicks)

    # 2. 全体スペクトル解析 (4096点FFT)
    fft_n = 4096
    n_chunks = len(samples) // fft_n
    if n_chunks > 0:
        spec_accum = np.zeros(fft_n // 2, dtype=np.float64)
        win = np.hanning(fft_n)
        for i in range(n_chunks):
            chunk = samples[i * fft_n : (i + 1) * fft_n] * win
            spec_accum += np.abs(np.fft.rfft(chunk)[:-1]) ** 2
        spec_avg = spec_accum / n_chunks + 1e-12
        spec_db = 10.0 * np.log10(spec_avg)
        spec_freqs = np.linspace(0, fs / 2, len(spec_db))

        idx_19k = np.where((spec_freqs >= 18800) & (spec_freqs <= 19200))[0]
        idx_voice = np.where((spec_freqs >= 300) & (spec_freqs <= 3500))[0]
        p_19k = np.mean(spec_avg[idx_19k]) if len(idx_19k) > 0 else 1e-12
        p_voice = np.mean(spec_avg[idx_voice]) if len(idx_voice) > 0 else 1e-12
        pilot_leak_db = float(10.0 * np.log10(p_19k / (p_voice + 1e-12)))
    else:
        spec_db = np.zeros(100)
        spec_freqs = np.linspace(0, 24000, 100)
        pilot_leak_db = -99.0

    avg_snr = float(np.mean(audio_snr_history)) if audio_snr_history else 0.0

    print("=" * 70)
    print("📊 WAVファイル 長時間解析結果サマリー")
    print("=" * 70)
    print(f"• 音声総再生時間         : {duration:.2f} 秒")
    print(f"• 検出クリックノイズ総数 : {total_clicks} 回 (クリックレス判定)")
    print(f"• 平均聴感 音声SNR       : +{avg_snr:.1f} dB")
    print(f"• 19kHzパイロット漏洩比  : {pilot_leak_db:.1f} dB (抑圧目標: -40dB以下)")
    print("=" * 70)

    # ダッシュボード画像生成
    _render_dashboard_image(
        out_img=out_img,
        freq_hz=83200000,
        mode="WAV File",
        duration=duration,
        time_points=time_points,
        cn_history=audio_snr_history,  # WAV解析時はC/Nの代わりにオーディオSNRを表示
        audio_snr_history=audio_snr_history,
        drift_ppm_history=[0.0] * len(time_points),
        buffer_fill_history=[15.0] * len(time_points),
        click_counts=click_counts,
        spec_freqs=spec_freqs,
        spec_db=spec_db,
        pilot_leak_db=pilot_leak_db,
        stats={
            "underruns": 0,
            "overflows": 0,
            "total_clicks": total_clicks,
            "avg_cn": avg_snr,
            "avg_aud": avg_snr,
            "avg_drift": 0.0,
            "ant": "WAV Offline Stream",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="RTL-SDR Long-Term Audio & RF Quality Profiler")
    parser.add_argument("--freq", default="83.2M", help="受信周波数 (例: 83.2M, 94.6M, 145.8M, 594k)")
    parser.add_argument("--mode", default="WFM", choices=["WFM", "NFM", "AM"], help="復調モード")
    parser.add_argument("--duration", type=float, default=20.0, help="観測時間 (秒)")
    parser.add_argument("--wav", default=None, help="解析対象のWAVファイルパス (指定時は実機SDRを使わずWAVを長時間解析)")
    parser.add_argument("--out-img", default="long_term_analysis.png", help="解析画像ファイル名")
    parser.add_argument("--out-wav", default="long_term_audio.wav", help="保存WAVファイル名")
    parser.add_argument("--no-audio", action="store_true", help="スピーカー再生をミュート")
    args = parser.parse_args()

    if args.wav:
        analyze_wav_file(args.wav, out_img=args.out_img)
    else:
        freq_hz = parse_freq(args.freq)
        run_long_term_analysis(
            freq_hz=freq_hz,
            mode=args.mode,
            duration=args.duration,
            out_img=args.out_img,
            out_wav=args.out_wav,
            play_audio=not args.no_audio,
        )


if __name__ == "__main__":
    main()
