"""
CLI-based Lightweight SDR Radio - Smart DX & Auto-Tuning Edition.
- 帯域全自動スキャン (FM 76〜95MHz)
- 通常聞こえない微弱局（DX局）の自動発掘＆超高感度受信
- AFC（自動周波数追従）＆ 狭帯域IFノイズカット
- キーボード操作による次局シーク (n: 次局 / p: 前局 / d: DXモード / q: 終了)
"""

import sys
import os
import time
import argparse
import signal
import threading
import queue
import numpy as np

# Windows環境での非ブロッキングキー入力
try:
    import msvcrt
except ImportError:
    msvcrt = None

from rtlsdr_driver import RtlSdrDriver
from dsp import SdrDspPipeline
from audio_output import AudioOutput
from cascade_controller import CascadeController
from hyper_controller import HyperController
from auto_tuner import AutoTuner, match_station_name
from config import detect_country, region_profile


def fm_scan_range() -> tuple[int, int]:
    """地域プロファイルのFM帯域 (JP以外では76〜95MHz直書きをしない)。"""
    prof = region_profile(detect_country())
    return int(prof["fm_start_hz"]), int(prof["fm_end_hz"])


def parse_freq_str(s: str) -> int:
    """'94.6M'/'594K'/'80000K'/'94600000'をHzへ。不正入力はSystemExitで clean 終了。
    rstrip("MHZ")は文字集合除去(例:'80MM'→'80')のため接尾辞除去に置換。
    接尾辞なしで3000未満のみMHz扱い (594HZ は 594Hz のまま、594MHz にはしない)。"""
    t = s.strip().upper()
    mult = 1.0
    matched = False
    for suffix, m in (("MHZ", 1e6), ("KHZ", 1e3), ("M", 1e6), ("K", 1e3), ("HZ", 1.0)):
        if t.endswith(suffix) and len(t) > len(suffix):
            t = t[: -len(suffix)].strip()
            mult = m
            matched = True
            break
    try:
        val = float(t)
    except ValueError:
        raise SystemExit(f"周波数の形式が不正です: {s!r} (例: 94.6M, 594K)")
    hz = val * mult
    if not matched and val < 3000:
        # 後方互換: 単位無しで3000未満はMHz扱い
        hz = val * 1e6
    return int(hz)


def check_freq_range(f_hz: int) -> int:
    """RTL-SDRの実用範囲外は clean 終了 (負値・99GHzのHW直行を防止)。"""
    if not (100000 <= f_hz <= 1750000000):
        raise SystemExit(f"周波数が範囲外です: {f_hz} Hz (0.1MHz〜1750MHz)")
    return f_hz


def print_scan_table(stations: list[dict]):
    """検出された局一覧を整形表示"""
    print("\n" + "=" * 75)
    print(f"📡 FM帯域自動探査結果 (検出数: {len(stations)} 局)")
    print("=" * 75)
    print(f"| {'周波数':<9} | {'SNR':<7} | {'信号品質':<11} | {'AFC補正':<9} | {'推定放送局名':<26} |")
    print("|" + "-"*11 + "|" + "-"*9 + "|" + "-"*13 + "|" + "-"*11 + "|" + "-"*28 + "|")
    for s in stations:
        print(f"| {s['freq_mhz']:6.2f}MHz | +{s['snr_db']:4.1f}dB | {s['quality']:11} | {s['afc_offset_hz']:>+6.0f}Hz | {s['name']:26} |")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="KomorebiSDR (Smart Auto-Tuner & DX Edition)")
    parser.add_argument("freq", nargs="?", default="94.6M", help="周波数 (例: 94.6M, 83.2M) またはモード ('scan', 'auto', 'dx')")
    parser.add_argument("--mode", default="WFM", choices=["WFM", "AM"], help="復調方式")
    parser.add_argument("--gain", default="auto", help="ゲイン (auto/hyper [自律最適化] / cascade [従来版] / 数値dB)")
    parser.add_argument("--controller", default="hyper", choices=["hyper", "cascade"], help="自律最適化エンジン種別")
    parser.add_argument("--vol", type=float, default=0.7, help="音量 (0.0〜1.0)")
    parser.add_argument("--filter", default="clean", choices=["clean", "wide", "narrow"], help="ノイズフィルタ (wide / clean / narrow)")
    parser.add_argument("--dx", action="store_true", help="DX超高感度モード（微弱局用：狭帯域IF＋最大ゲイン＋音声ノイズカット）")
    args = parser.parse_args()

    cmd_mode = args.freq.strip().lower()

    # 1. ハードウェア初期化
    driver = RtlSdrDriver()
    try:
        driver.open(0)
    except RuntimeError as e:
        raise SystemExit(f"デバイスが見つかりません: {e} (ZadigでWinUSB導入を確認)")

    tuner = AutoTuner(driver)

    # モード判定: スキャンのみ
    if cmd_mode == "scan":
        lo, hi = fm_scan_range()
        print(f"[*] FM全帯域 ({lo/1e6:.1f}〜{hi/1e6:.1f}MHz) の高精度自動スキャンを実行中...")
        stations = tuner.scan_band(lo, hi, step_hz=1800000, snr_threshold=4.2)
        print_scan_table(stations)
        driver.close()
        return

    # モード判定: 自動選局 (auto / dx)
    target_freq_hz = 94600000
    is_dx_mode = args.dx

    if cmd_mode in ["auto", "seek"]:
        lo, hi = fm_scan_range()
        print("[*] 全自動チューニング: 最強局を探査中...")
        stations = tuner.scan_band(lo, hi, step_hz=1800000, snr_threshold=4.2)
        print_scan_table(stations)
        best = tuner.get_strongest_station()
        if best:
            target_freq_hz = best["freq_hz"]
            print(f"[*] 自動選局完了: {best['name']} ({best['freq_mhz']:.2f}MHz, SNR: +{best['snr_db']}dB)")
        else:
            print("[!] 検出局がありません。デフォルト周波数 (94.6MHz) を使用します。")
    elif cmd_mode == "dx":
        lo, hi = fm_scan_range()
        print("[*] DX超高感度チューニング: 通常聞こえない微弱局を探査中...")
        stations = tuner.scan_band(lo, hi, step_hz=1800000, snr_threshold=3.8)
        print_scan_table(stations)
        dx_stations = tuner.get_dx_stations()
        if dx_stations:
            # 最も微弱な局を選択
            target = min(dx_stations, key=lambda s: s["snr_db"])
            target_freq_hz = target["freq_hz"]
            is_dx_mode = True
            print(f"[*] 微弱DX局を発見: {target['name']} ({target['freq_mhz']:.2f}MHz, SNR: +{target['snr_db']}dB)")
        else:
            print("[!] 微弱局が見つかりませんでした。デフォルト周波数を使用します。")
    else:
        target_freq_hz = check_freq_range(parse_freq_str(args.freq))

    sample_rate = 1152000
    audio_rate = 48000

    driver.set_sample_rate(sample_rate)

    def apply_frequency(f_hz):
        if f_hz < 24000000:
            driver.set_direct_sampling(2)
            # ダイレクトサンプリングはDC付近に巨大スパイクがあるため+150kHzずらす
            # (main.pyと同一方式。旧実装はoffset=0でスパイクが乗っていた)
            offset = 150000.0
            driver.set_center_freq(int(f_hz + offset))
        else:
            driver.set_direct_sampling(0)
            offset = 150000.0  # +150kHz DCスパイク回避
            driver.set_center_freq(int(f_hz + offset))
        return offset

    current_freq = target_freq_hz
    current_offset = apply_frequency(current_freq)

    dsp = SdrDspPipeline(sample_rate, audio_rate)
    dsp.set_offset_freq(current_offset)
    if is_dx_mode:
        dsp.filter_mode = "narrow"

    audio = AudioOutput(audio_rate)
    if not (0.0 <= args.vol <= 1.0):
        print(f"[!] --vol は0.0〜1.0に丸めます: {args.vol}")
    audio.set_volume(args.vol)

    if args.controller == "cascade" or args.gain.lower() == "cascade":
        controller = CascadeController(driver, dsp, audio)
    else:
        controller = HyperController(driver, dsp, audio)
    controller.dx_mode = is_dx_mode
    use_cascade = (args.gain.lower() in ["auto", "hyper", "cascade"])
    ctrl_name = "Hyper" if isinstance(controller, HyperController) else "Cascade"
    if use_cascade:
        controller.init_gains()
    else:
        controller.enabled = False
        driver.set_gain_mode(True)
        try:
            driver.set_gain(float(args.gain))
        except (ValueError, TypeError):
            raise SystemExit(f"--gain の数値が不正です: {args.gain!r}")
        dsp.filter_mode = args.filter

    st_name = match_station_name(current_freq)
    print("=" * 65)
    print(f"📻 KOMOREBISDR RADIO (Smart Auto-Tuner & DX Edition)")
    print("=" * 65)
    print(f"[*] 受信周波数: {current_freq / 1e6:.4f} MHz ({st_name})")
    print(f"[*] 復調モード: {args.mode} | DX超高感度: {'ON' if is_dx_mode else 'OFF'}")
    print(f"[*] 制御エンジン: {ctrl_name}自律最適化 ({'自動' if use_cascade else '手動ゲイン'})")
    print(f"[*] 操作キー: [n]次局シーク | [p]前局シーク | [d]DXモード切替 | [+/-]音量 | [q]終了")

    raw_queue = queue.Queue(maxsize=30)
    running = True
    usb_enabled = threading.Event()
    usb_enabled.set()

    def on_usb_data(raw_bytes):
        if not running:
            return
        if raw_queue.full():
            try:
                raw_queue.get_nowait()
            except queue.Empty:
                pass
        raw_queue.put(raw_bytes)

    def async_usb_thread():
        while running:
            if not usb_enabled.is_set():
                time.sleep(0.05)
                continue
            try:
                driver.read_async(on_usb_data, num_buffers=16, buffer_len=132096)
            except Exception:
                time.sleep(0.01)

    def dsp_worker():
        last_log_time = 0.0
        while running:
            try:
                raw = raw_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                pcm, spec_db = dsp.process(raw, mode=args.mode)
                audio.put_audio(pcm)

                if use_cascade:
                    if ctrl_name == "Hyper":
                        stats = controller.process_frame(raw, spec_db, audio=pcm, mode=args.mode)
                    else:
                        stats = controller.process_frame(raw, spec_db)
                    now = time.time()
                    if now - last_log_time >= 0.5:
                        last_log_time = now
                        status_tag = "★収束完了(最適)" if stats["converged"] else ">> 追従調整中..."
                        dx_tag = " [DX:ON]" if stats.get("dx_mode") else ""
                        afc_tag = f" AFC:{dsp.afc_offset_hz:>+5.0f}Hz" if dsp.afc_enabled else ""
                        if ctrl_name == "Hyper":
                            eng_tag = (f" C/N:{stats.get('channel_snr_db', 0.0):4.1f}dB"
                                       f" 聴感:{stats.get('audio_snr_db', 0.0):4.1f}dB"
                                       f" Cut:{stats.get('cutoff_hz', 0) / 1000.0:4.1f}k"
                                       f" IF:{stats.get('if_bw_hz', 0) / 1000.0:5.1f}k")
                        else:
                            eng_tag = f" SNR:+{stats['estimated_snr']:4.1f}dB"
                        print(f"\r[{ctrl_name}] ゲイン:{stats['gain_db']:4.1f}dB | クリップ:{stats['adc_clip_pct']:4.2f}% | IQ分散:{stats['iq_std']:4.1f} |{eng_tag} | フィルタ:{stats['filter_mode'].upper()}{dx_tag}{afc_tag} | {status_tag}   ", end="", flush=True)

            except Exception:
                if running:
                    time.sleep(0.005)

    # 音声ストリーム起動 (内部のis_prerolled判定により3チャンク蓄積後に自動再生開始)
    audio.start()

    t_usb = threading.Thread(target=async_usb_thread, daemon=True)
    t_dsp = threading.Thread(target=dsp_worker, daemon=True)
    t_usb.start()
    t_dsp.start()

    print("[*] 放送受信・再生開始！ (音声ストリーム稼働中)")
    print("-" * 65)

    def retune_to(new_freq_hz):
        nonlocal current_freq, current_offset
        current_freq = new_freq_hz
        current_offset = apply_frequency(current_freq)
        dsp.set_offset_freq(current_offset)
        if use_cascade:
            controller.reset_tracking()
        name = match_station_name(current_freq)
        print(f"\n[*] 同調変更: {current_freq / 1e6:.2f} MHz ({name})")

    def do_seek(direction):
        """局リスト未取得時はUSBを停止して安全にスキャン→復元してからシーク。
        (非同期ストリーム動作中にread_syncすると競合・ハングするため)"""
        nonlocal current_freq
        if not tuner.discovered_stations:
            print("\n[*] 局リスト未取得のためFM帯域スキャンを実行中...")
            usb_enabled.clear()
            driver.cancel_async(timeout=3.0)
            try:
                lo, hi = fm_scan_range()
                tuner.scan_band(lo, hi, step_hz=1800000, snr_threshold=4.2)
                # スキャン後の復元 (サンプルレート・ゲイン・受信周波数)
                driver.set_sample_rate(sample_rate)
                apply_frequency(current_freq)
                if use_cascade:
                    controller.init_gains()
                else:
                    driver.set_gain_mode(True)
                    driver.set_gain(float(args.gain))
            finally:
                # cancel_asyncで_setした_async_stopを必ず解除する。
                # 解除しないとread_asyncが即returnを繰り返し、以降USB受信が
                # 永久停止する (main.pyのstart_usb_streamと同じ理由)。
                driver.resume_async()
                usb_enabled.set()
        st = tuner.seek_next(current_freq, direction=direction)
        if st:
            retune_to(st["freq_hz"])

    try:
        while running:
            # キーボード入力チェック
            if msvcrt and msvcrt.kbhit():
                ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if ch == "q":
                    break
                elif ch == "n":
                    # 次局シーク
                    do_seek(+1)
                elif ch == "p":
                    # 前局シーク
                    do_seek(-1)
                elif ch == "d":
                    # DXモード切り替え
                    controller.dx_mode = not controller.dx_mode
                    if use_cascade:
                        controller.reset_tracking()
                    print(f"\n[*] DX超高感度モード: {'ON (狭帯域IF + 最大ゲイン)' if controller.dx_mode else 'OFF'}")
                elif ch in ["+", "="]:
                    audio.set_volume(audio.volume + 0.05)
                    print(f"\n[*] 音量: {int(audio.volume * 100)}%")
                elif ch in ["-", "_"]:
                    audio.set_volume(audio.volume - 0.05)
                    print(f"\n[*] 音量: {int(audio.volume * 100)}%")

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] 終了処理中...")
        running = False
        driver.cancel_async()
        t_usb.join(timeout=0.5)
        t_dsp.join(timeout=0.5)
        audio.stop()
        driver.close()
        print("[*] 正常に終了しました。")


if __name__ == "__main__":
    main()
