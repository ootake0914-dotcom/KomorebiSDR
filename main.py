"""
KomorebiSDR - Smart Auto-Tuner & DX Edition.
メイン実行スクリプト。
RTL-SDR受信スレッド、DSP復調パイプライン、オーディオ出力、Pygame GUI、
および全自動帯域探査＆DX微弱局発掘エンジン (AutoTuner) を統合。
"""

import sys
import atexit
import time
import threading
import queue
import argparse
import numpy as np

from rtlsdr_driver import RtlSdrDriver
from dsp import SdrDspPipeline
from audio_output import AudioOutput
from gui import SdrGui, show_message_screen
from cascade_controller import CascadeController
from hyper_controller import HyperController
from auto_tuner import AutoTuner, match_station_name
from ppm_cal import PpmCalibrator
from signal_logger import SignalLogger, is_settled as _sig_settled
import sw_schedule
from rt_profile import RtProfile
from config import (load_config, save_config, detect_country, detect_language,
                    region_profile, LOG_PATH, shortwave_band_name)
from i18n import set_language as i18n_set_language, get_language as i18n_language, t


class NoDeviceFoundError(RuntimeError):
    """RTL-SDRデバイスが1台も検出されなかった場合"""


class SdrApp:
    """SDRアプリケーション統合コントローラ (AutoTuner & DX機能搭載)"""

    def __init__(self, initial_freq=None, initial_mode=None, controller_type="hyper",
                 config=None, country=None, stereo=None, stereo_nr=None, sic=None):
        # ---- 設定・地域プロファイル ----
        # configはコピーして使う (--mono等のセッション限り上書きを元cfgに
        # 書き込まず、終了時のsave_configで永続化させないため)
        self.config = dict(config) if config is not None else load_config()
        # CLI一時上書き (--mono / --no-stereo-nr / --no-sic はこのセッションのみ有効・保存しない)
        if stereo is not None:
            self.config["stereo"] = bool(stereo)
        if stereo_nr is not None:
            self.config["stereo_nr"] = bool(stereo_nr)
        if sic is not None:
            self.config["sic"] = bool(sic)
        self.country = (country or self.config.get("country") or detect_country()).upper()
        self.profile = region_profile(self.country)
        if initial_freq is None:
            initial_freq = int(self.profile["default_freq_hz"])
        if initial_mode is None:
            initial_mode = self.profile["default_mode"]

        self.freq = initial_freq
        self.mode = initial_mode
        self.sample_rate = 1152000
        self.audio_rate = 48000
        self.controller_type = controller_type

        # RT締切プロファイル (1ブロック=66048IQ@1.152MHz ≒ 57.3ms)
        self.rt_profile = RtProfile(budget_ms=57.3)
        self._rt_summary_cache = ""

        # ハードウェアドライバ
        self.driver = RtlSdrDriver()
        # DSPパイプライン
        self.dsp = SdrDspPipeline(self.sample_rate, self.audio_rate)
        # 地域規格 (ディエンファシス 50/75μs) を適用。
        self.dsp.set_deemphasis(self.profile["deemphasis_us"])
        # ステレオ・NRは常時ON固定 (保存済み設定に依らず上書き)。
        self.dsp.sic_enabled = bool(self.config.get("sic", True))
        self._apply_black_magic_config()
        # オーディオ出力
        self.audio = AudioOutput(self.audio_rate)
        self.audio.set_volume(float(self.config.get("volume", 0.7)))
        # GUI (音量は起動時config固定＋システム音量。GUI側に音量概念なし)
        self.gui = SdrGui()
        # 地域表示・スキャン帯域・プリセット
        self.gui.set_region(self.profile["label"], self.profile["fm_start"], self.profile["fm_end"])
        self.gui.scan_status_text = t("idle")
        self.gui.set_presets(*self._initial_presets())
        # 自律最適化コントローラ (hyper: Cascade全超越エンジン / cascade: 従来版)
        if controller_type == "cascade":
            self.controller = CascadeController(self.driver, self.dsp, self.audio)
        else:
            self.controller = HyperController(self.driver, self.dsp, self.audio)
        self.use_controller = True
        # 自動選局・スキャンエンジン
        self.tuner = AutoTuner(self.driver)
        # ドングルPPM自動較正器 (放送搬送波を基準に背景収集)
        self.ppm_cal = PpmCalibrator()
        self._ppm_last_tick = 0.0
        # 信号健康ロガー (特定局の揺れ切り分け用)
        self.sig_logger = SignalLogger()
        self._siglog_last = 0.0
        self._ppm_dwell_freq = None
        self._ppm_dwell_since = 0.0
        self._ppm_prev_afc = 0.0

        # スレッド間通信用
        self.cmd_queue = queue.Queue()
        self.spectrum_lock = threading.Lock()
        self.latest_spectrum = np.zeros(self.dsp.fft_size, dtype=np.float32)
        self.latest_audio = np.zeros(1024, dtype=np.float32)
        self.running = False
        self.sdr_thread = None
        # ゲインはHyper自動に一本化 (手動操作削除・木漏れ日整理)。

        # GUIコールバック登録
        self.gui.on_freq_change = self.set_frequency
        self.gui.on_mode_change = self.set_mode
        # 音量ボタン廃止 (起動時config＋システム音量)。
        # Filterはclean固定 (GUIボタン削除・木漏れ日整理)。
        self.gui.on_seek_change = lambda d: self.cmd_queue.put(("SEEK", d))
        self.gui.on_scan_request = lambda: self.cmd_queue.put(("SCAN", None))
        self.gui.on_sw_scan_request = lambda: self.cmd_queue.put(("SW_SCAN", None))
        self.gui.on_ham_scan_request = lambda: self.cmd_queue.put(("HAM_SCAN", None))
        self.gui.on_bfo_change = lambda d: self.cmd_queue.put(("BFO", d))
        # DXはC/N連動の自動絞りに一本化 (ボタン削除・木漏れ日整理)。
        # AFCは常時ON固定 (GUIボタン削除・木漏れ日整理)。
        # ステレオ・NRは常時ON固定 (自動ブレンドに一本化)。保存済みOFF設定も上書きする。

        self.gui.center_freq = self.freq
        self.gui.mode = self.mode
        self.gui._sync_bfo_visibility()
        self.gui.sample_rate = self.sample_rate

    def _region_defaults(self) -> tuple:
        """地域既定プリセット (JPは既知局、他地域は空)"""
        fm = []
        am = []
        if self.profile["region"] == "JP":
            fm = [
                {"name": "NHK水戸 83.2", "freq_hz": 83200000, "mode": "WFM"},
                {"name": "NHK東京 82.5", "freq_hz": 82500000, "mode": "WFM"},
                {"name": "NHK前橋 86.4", "freq_hz": 86400000, "mode": "WFM"},
                {"name": "FM GUNMA 92.8", "freq_hz": 92800000, "mode": "WFM"},
                {"name": "LuckyFM 94.6", "freq_hz": 94600000, "mode": "WFM"},
                {"name": "TOKYO 80.0", "freq_hz": 80000000, "mode": "WFM"},
                {"name": "J-WAVE 81.3", "freq_hz": 81300000, "mode": "WFM"},
                {"name": "TBS 90.5", "freq_hz": 90500000, "mode": "WFM"},
                {"name": "ニッポン 93.0", "freq_hz": 93000000, "mode": "WFM"},
                {"name": "ISS 145.8", "freq_hz": 145800000, "mode": "NFM"},
            ]
            am = [
                {"name": "LuckyFM 1197k", "freq_hz": 1197000, "mode": "AM"},
                {"name": "NHK第1 594k", "freq_hz": 594000, "mode": "AM"},
                {"name": "NHK第2 693k", "freq_hz": 693000, "mode": "AM"},
                {"name": "AFN 810k", "freq_hz": 810000, "mode": "AM"},
                {"name": "TBS 954k", "freq_hz": 954000, "mode": "AM"},
                {"name": "ニッポン 1242k", "freq_hz": 1242000, "mode": "AM"},
                {"name": "短波NIKKEI 6.055M", "freq_hz": 6055000, "mode": "AM"},
            ]
        return fm, am

    def _initial_presets(self):
        """設定保存されたプリセット、無ければ地域既定 (日本は既知局) を返す。
        側面ごとに補完: 保存済みが空の側は地域既定へフォールバック (FMスキャンで
        AM側が空に上書き保存されてもJP既定が復活する)。地域不一致時は既定へ戻す。"""
        dfm, dam = self._region_defaults()
        if self.config.get("presets_region") != self.profile["region"]:
            return dfm, dam
        fm = self.config.get("presets_fm")
        am = self.config.get("presets_am")
        return (fm if fm else dfm), (am if am else dam)

    def _update_presets_from_scan(self, stations):
        """スキャン結果の強力なFM局からプリセットを自動生成し、設定へ保存"""
        fm = [s for s in stations if s["freq_hz"] >= 24000000]
        fm = sorted(fm, key=lambda s: s["snr_db"], reverse=True)[:10]
        if not fm:
            return
        fm_presets = [
            {"name": f"{s['freq_mhz']:.1f}", "freq_hz": int(s["freq_hz"]),
             "mode": s.get("mode", "WFM") if isinstance(s.get("mode"), str) else "WFM"}
            for s in fm
        ]
        fm_presets.sort(key=lambda p: p["freq_hz"])
        self.config["presets_fm"] = fm_presets
        self.config["presets_region"] = self.profile["region"]
        save_config(self.config)
        # AM側はNoneで維持 (空リストを渡すとAMプリセットが消えるため)
        self.gui.set_presets(fm_presets, None)

    def _update_presets_from_sw_scan(self, stations):
        """短波スキャン結果からAMプリセットを生成し、設定へ保存"""
        if not stations:
            return
        top = sorted(stations, key=lambda s: s["snr_db"], reverse=True)[:7]
        am_presets = []
        for s in top:
            name = ""
            try:
                hit = sw_schedule.lookup(s["freq_hz"], tolerance_hz=2500)
                if hit and hit["station"]:
                    name = hit["station"][:13]
            except Exception:
                pass
            if not name:
                name = f"{shortwave_band_name(s['freq_hz'])} {s['freq_mhz']:.2f}"
            am_mode = s.get("mode", "AM") if isinstance(s.get("mode"), str) else "AM"
            am_presets.append({"name": name, "freq_hz": int(s["freq_hz"]), "mode": am_mode})
        am_presets.sort(key=lambda p: p["freq_hz"])
        self.config["presets_am"] = am_presets
        self.config["presets_region"] = self.profile["region"]
        save_config(self.config)
        # FM側はNoneで維持 (空リストを渡すとFMプリセットが消えるため)
        self.gui.set_presets(None, am_presets)

    def init_hardware(self):
        """RTL-SDRデバイスの初期化"""
        dev_count = self.driver.get_device_count()
        if dev_count == 0:
            raise NoDeviceFoundError()

        print(f"[*] Found {dev_count} RTL-SDR device(s)")
        self.driver.open(0)
        print(f"[*] Device opened: {self.driver.get_device_name(0)}")

        self.driver.set_sample_rate(self.sample_rate)
        self._apply_frequency_and_mode(self.freq, self.mode)

        # 保存済みPPM較正値をドングルへ適用 (HW不可時はSWフォールバック)
        # 旧式ドライバ (set_ppm_correction欠落) では何もしない
        try:
            saved_ppm = self.config.get("ppm")
            apply = getattr(self.driver, "set_ppm_correction", None)
            if callable(apply) and isinstance(saved_ppm, (int, float)) and saved_ppm != 0:
                hw = apply(int(saved_ppm))
                print(f"[*] PPM correction {int(saved_ppm)} applied ({'HW' if hw else 'SW fallback'})")
        except Exception as e:
            print(f"[WARN] PPM restore failed: {e}", file=sys.stderr)

        # コントローラ初期化 (ゲインは常に自動。状態チップ廃止のため表示更新なし)
        self.controller.init_gains()
        print(f"[*] {self._ctrl_label()} autonomous optimization engine enabled")

        # 起動直後の音声途切れ(2秒後のスキャン停止)を防止するため、
        # 帯域スキャンはGUI上の [全帯域スキャン] ボタン押下時に実行する設計に変更
        # threading.Thread(target=self._initial_bg_scan, daemon=True).start()

    def _ctrl_label(self) -> str:
        return "Hyper: ON" if self.controller_type == "hyper" else "Cascade: ON"

    def _initial_bg_scan(self):
        """起動直後の初回バックグラウンドFMスキャン"""
        # 受信が安定するまで少し待機
        time.sleep(2.0)
        if not self.running:
            return
        print("[*] Background full-band scan started...")
        self.cmd_queue.put(("SCAN", "auto_best"))

    # 合体可能コマンド: 滞留中の同種は最新のみ適用 (スキャン中のスライダ連打対策)。
    # BFOは差分(±50Hz)コマンドのため合体すると押した分が消える→除外。
    _COALESCE_CMDS = ("FREQ", "MODE", "FILTER")

    def _drain_commands(self):
        """キューを全排出して陳腐化コマンドを合体する。
        SCAN/SW_SCAN/SEEK/TOGGLE類は回数・順序が意味を持つため全件維持。"""
        pending = []
        while True:
            try:
                pending.append(self.cmd_queue.get_nowait())
            except queue.Empty:
                break
        if len(pending) < 2:
            return pending
        keep = set()
        out = []
        for cmd, val in reversed(pending):
            if cmd in self._COALESCE_CMDS:
                if cmd in keep:
                    continue
                keep.add(cmd)
            out.append((cmd, val))
        out.reverse()
        return out

    def _apply_black_magic_config(self):
        """configのblack_magic節をdsp属性へ反映 (起動時1回。既定は全OFF)。

        パラメータ本体はdsp.bm_cfg経由で遅延生成インスタンスへ渡る。
        破損値はconfig側の検証＋dsp側の_min/maxで二重に弾く。
        """
        try:
            bm = self.config.get("black_magic", None) or {}
            if not isinstance(bm, dict):
                bm = {}
            dsp = self.dsp
            dsp.black_magic_enabled = bool(bm.get("enabled", False))
            dsp.bm_cfg = {k: v for k, v in bm.items() if isinstance(v, dict)}
            c = bm.get("cyclostationary", None) or {}
            dsp.bm_cyclo_enabled = bool(c.get("enabled", True)) if isinstance(c, dict) else True
            try:
                dsp.bm_cyclo_min_conf = float(c.get("min_confidence", 0.55))
            except (TypeError, ValueError, AttributeError):
                dsp.bm_cyclo_min_conf = 0.55
            r = bm.get("rmt_denoiser", None) or {}
            dsp.bm_rmt_enabled = bool(r.get("enabled", False)) if isinstance(r, dict) else False
            s = bm.get("stochastic_resonance", None) or {}
            dsp.bm_sr_enabled = bool(s.get("enabled", False)) if isinstance(s, dict) else False
            a = bm.get("adaptive_notch", None) or {}
            dsp.bm_notch_enabled = bool(a.get("enabled", False)) if isinstance(a, dict) else False
            q = bm.get("squelch_assist", None) or {}
            if isinstance(q, dict):
                dsp.bm_sq_assist_enabled = bool(q.get("enabled", False))
                for attr, key, lo, hi, default in (
                        ("bm_sq_open_conf", "open_conf", 0.0, 1.0, 0.75),
                        ("bm_sq_close_conf", "close_conf", 0.0, 1.0, 0.55),
                        ("bm_sq_close_smeter_db", "close_smeter_db", -120.0, 0.0, -25.0),
                        ("bm_sq_open_smeter_db", "open_smeter_db", -120.0, 0.0, -40.0),
                        ("bm_sq_min_close_blocks", "min_close_blocks", 0, 200, 20)):
                    try:
                        v = q.get(key, default)
                        if isinstance(v, bool):
                            v = default
                        v = min(max(float(v), lo), hi)
                        if attr == "bm_sq_min_close_blocks":
                            v = int(round(v))
                        setattr(dsp, attr, v)
                    except (TypeError, ValueError):
                        pass
            else:
                dsp.bm_sq_assist_enabled = False
            sk = bm.get("seeking", None) or {}
            dsp.bm_seek_enabled = bool(sk.get("enabled", False)) if isinstance(sk, dict) else False
        except Exception:
            pass

    def _apply_frequency_and_mode(self, freq: int, mode: str):
        """周波数と復調モードをハードウェア・DSPに適用"""
        self.freq = freq
        self.mode = mode

        # AM中波帯 (24MHz未満) の場合はダイレクトサンプリング (Q-branch = 2) を自動有効化
        if freq < 24000000:
            self.driver.set_direct_sampling(2)
            # ダイレクトサンプリングはDC付近に巨大なスパイクがあるため、キャリアを
            # +150kHzずらして受信しDSP側で戻す (スパイクがAM復調を汚染しない)
            offset = 150000
            self.driver.set_center_freq(freq + offset)
            self.dsp.set_offset_freq(offset)
        else:
            self.driver.set_direct_sampling(0)
            # DCスパイクを避けるため +150kHz オフセットしてチューニング
            offset = 150000
            self.driver.set_center_freq(freq + offset)
            self.dsp.set_offset_freq(offset)

        # ノイズ推定履歴をリセット (選局先の電界強度へ素早く追従)
        self.dsp.reset_stereo_nr()

        # 短波(HF)は番組表DBから実局名を引く
        st_name = ""
        sw_info = ""
        if freq < 24000000:
            try:
                hit = sw_schedule.lookup(freq, tolerance_hz=2500)
                if hit:
                    st_name = hit["name"]
                    target = hit.get("target") or ""
                    tm = hit.get("time") or ""
                    extra = f" [{target}] {tm}UTC" if (target or tm) else ""
                    sw_info = f"{hit['name']}{extra}"
            except Exception:
                pass
        self.gui.sw_info = sw_info
        if not st_name:
            st_name = match_station_name(freq)
        if st_name == "Unknown FM Station":
            st_name = t("station_unknown")
        self.gui.station_name = "" if st_name == t("station_unknown") else st_name
        self.gui.scan_status_text = t("receiving", freq=f"{freq / 1e6:.2f}") + f" - {st_name}"

        # 選局変更時は探索状態をリセットして新局へ即座に適応
        if hasattr(self, "controller") and self.use_controller and self.controller.available_gains:
            self.controller.reset_tracking()

    def set_frequency(self, freq_hz: int):
        self.cmd_queue.put(("FREQ", freq_hz))

    def set_mode(self, mode: str):
        self.cmd_queue.put(("MODE", mode))

    def _ppm_background_tick(self, now: float):
        """PPM背景収集 (ワーカーから2秒毎)。強力FMステレオ局のAFC定常残差を
        標本化し、3局以上たまれば中央値PPMを推定・適用・保存する。
        選局・スキャン中は標本化しない (AFC収束前の残差は無効)。"""
        try:
            dsp = self.dsp
            # 参照に使えるのはWFM放送帯の強力ステレオ局のみ
            if self.mode != "WFM" or self.freq < 24000000:
                self._ppm_dwell_freq = None
                return
            if not bool(getattr(dsp, "afc_enabled", False)):
                self._ppm_dwell_freq = None
                return
            if not bool(getattr(dsp, "is_stereo", False)):
                self._ppm_dwell_freq = None
                return
            if abs(float(getattr(dsp, "stereo_pilot_lock", 0.0))) < 0.6:
                self._ppm_dwell_freq = None
                return
            if float(getattr(dsp, "s_units", 0.0)) < 6.0:
                self._ppm_dwell_freq = None
                return
            afc = float(getattr(dsp, "afc_offset_hz", 0.0))
            # 同一周波数への滞留・AFC定常を要求 (収束前残差の混入防止)
            if self._ppm_dwell_freq != self.freq:
                self._ppm_dwell_freq = self.freq
                self._ppm_dwell_since = now
                self._ppm_prev_afc = afc
                return
            if now - self._ppm_dwell_since < 10.0:
                self._ppm_prev_afc = afc
                return
            if abs(afc - self._ppm_prev_afc) > 30.0:
                # まだ収束途中: 今回は見送り、次tickへ
                self._ppm_prev_afc = afc
                return
            self._ppm_prev_afc = afc
            n = self.ppm_cal.collect(self.freq, afc)
            self._ppm_try_apply(n, "background")
        except Exception as e:
            print(f"[WARN] PPM estimate failed: {e}", file=sys.stderr)

    def _ppm_try_apply(self, n: int, source: str) -> bool:
        """推定→適用→保存の共通後段。適用したらTrue (標本破棄・集め直し)。"""
        ppm, _, confident = self.ppm_cal.estimate()
        if not confident or ppm is None:
            return False
        get_ppm = getattr(self.driver, "get_ppm_correction", None)
        set_ppm = getattr(self.driver, "set_ppm_correction", None)
        if not callable(get_ppm) or not callable(set_ppm):
            return False  # 旧式ドライバでは較正しない
        cur = int(get_ppm())
        new = int(round(ppm))
        if abs(new - cur) < 1 or abs(new) > PpmCalibrator.MAX_PPM:
            return False
        hw = set_ppm(new)
        self.config["ppm"] = new
        save_config(self.config)
        print(f"[*] PPM auto-calibrated ({source}): {cur} -> {new} "
              f"({n} stations, {'HW' if hw else 'SW fallback'})")
        self.gui.scan_status_text = t("ppm_done", ppm=new, n=n)
        # 補正後は残差基準が変わるため標本を破棄して集め直す
        self.ppm_cal.clear()
        self._ppm_dwell_freq = None
        return True

    def _sdr_worker(self):
        """SDRデータ受信 & DSP処理ワーカースレッド"""
        # FTZ/DAZ (denormalジッタ対策) はスレッド単位の設定のため、
        # ワーカースレッドでも明示的に有効化する (import時はメインスレッドにのみ効く)
        try:
            from dsp import _NATIVE
            if _NATIVE is not None and hasattr(_NATIVE, "sdr_fast_fpu"):
                _NATIVE.sdr_fast_fpu()
        except Exception:
            pass
        # GUI描画(GIL)による一時的な処理落ちを吸収する深いバッファ (約6.9秒分)。
        # 浅いバッファだと溢れた生IQが捨てられ、音声が時間圧縮(早回し+飛び)になる。
        raw_queue = queue.Queue(maxsize=120)
        usb_running = threading.Event()

        def on_async_data(raw_bytes):
            if not self.running or not usb_running.is_set():
                return
            if raw_queue.full():
                try:
                    raw_queue.get_nowait()
                except queue.Empty:
                    pass
            raw_queue.put(raw_bytes)

        # 非同期USB受信スレッド
        # 抜き差し検出: read_asyncが例外なく即時復帰し続けたらデバイス喪失とみなし、
        # タイトループ(100%CPU)を避けて待機＋警告する。例外もカウント＋間引きログ。
        usb_err_count = 0

        def async_usb_loop():
            nonlocal usb_err_count
            while self.running and usb_running.is_set():
                try:
                    t0 = time.monotonic()
                    self.driver.read_async(on_async_data, num_buffers=16, buffer_len=132096)
                    dt = time.monotonic() - t0
                    if dt < 0.05 and self.running and usb_running.is_set():
                        usb_err_count += 1
                        if usb_err_count == 1 or usb_err_count % 50 == 0:
                            print(f"[WARN] USB stream returned immediately "
                                  f"x{usb_err_count} (device unplugged?)", file=sys.stderr)
                        time.sleep(0.2)
                    else:
                        usb_err_count = 0
                except Exception as e:
                    usb_err_count += 1
                    if usb_err_count % 20 == 1:
                        print(f"[WARN] USB async error x{usb_err_count}: {e}", file=sys.stderr)
                    time.sleep(0.05)

        def start_usb_stream():
            self.driver.resume_async()
            usb_running.set()
            t = threading.Thread(target=async_usb_loop, daemon=True)
            t.start()
            return t

        def stop_usb_stream(t) -> bool:
            """Cループが実際に抜けるまで待機してから同期読み込みを行う (競合・ハング防止)。
            戻り値: 旧スレッドが抜けた(True)/残留(False)。残留時は二重read_asyncを
            避けるため呼び出し側はスキャンを中止しなければならない。"""
            usb_running.clear()
            self.driver.cancel_async(timeout=3.0)
            t.join(timeout=3.0)
            if t.is_alive():
                print("[ERROR] USB thread stuck; scan aborted "
                      "(keep old stream, no double read_async)", file=sys.stderr)
                return False
            return True

        def restore_after_scan():
            """帯域スキャンで変更された受信パラメータを通常受信状態へ復元
            各ステップをベストエフォートで復元し、一部の失敗で全体を諦めない。"""
            try:
                self.driver.set_sample_rate(self.sample_rate)
            except Exception as e:
                print(f"[ERROR] restore sample_rate failed: {e}", file=sys.stderr)
            try:
                self._apply_frequency_and_mode(self.freq, self.mode)
            except Exception as e:
                print(f"[ERROR] restore freq/mode failed: {e}", file=sys.stderr)
            try:
                if self.use_controller:
                    # 自動モード: コントローラの探索へ戻す (手動ゲイン廃止のため常に自動)
                    self.controller.init_gains()
            except Exception as e:
                print(f"[ERROR] restore gains failed: {e}", file=sys.stderr)

        def _drain_raw_queue():
            """raw_queueに滞留したスキャン前周波数の古いIQを破棄する。
            破棄しないと復帰後に数秒の遅延音声・誤スペクトルが出る。"""
            drained = 0
            try:
                while True:
                    raw_queue.get_nowait()
                    drained += 1
            except queue.Empty:
                pass
            return drained

        def safe_band_scan(status_label: str, auto_best: bool = False):
            """USBストリームを安全に停止して帯域スキャンを実行 (失敗しても必ず復帰)"""
            nonlocal t_usb
            self.gui.scan_status_text = status_label
            if not stop_usb_stream(t_usb):
                # 旧スレッドがjoin直後に抜けた可能性がある: 生死を再確認し、
                # 死んでいれば立て直す。生きていれば回復に委ねる。
                if not t_usb.is_alive():
                    try:
                        t_usb = start_usb_stream()
                    except Exception as e:
                        print(f"[ERROR] USB restart failed: {e}", file=sys.stderr)
                else:
                    # cancel_asyncで_setした _async_stop を必ず解除しないと、
                    # async_usb_loopが即時復帰を繰り返し永久に受信が止まる。
                    self.driver.resume_async()
                    usb_running.set()
                self.gui.scan_status_text = "USB busy - scan skipped"
                return []
            stations = []
            try:
                stations = self.tuner.scan_band(
                    self.profile["fm_start_hz"], self.profile["fm_end_hz"],
                    step_hz=1800000, snr_threshold=4.2)
                self.gui.detected_stations = stations
                self._update_presets_from_scan(stations)
            except Exception as e:
                print(f"[ERROR] Band scan failed: {e}", file=sys.stderr)
            finally:
                try:
                    restore_after_scan()
                except Exception as e:
                    print(f"[ERROR] Failed to restore receiver after scan: {e}", file=sys.stderr)
                _drain_raw_queue()
                t_usb = start_usb_stream()

            self.gui.scan_status_text = t("scan_done", n=len(stations))

            best_station = None
            if auto_best and stations:
                best = max(stations, key=lambda s: s["snr_db"])
                cur_snr = -99.0
                for s in stations:
                    if abs(s["freq_hz"] - self.freq) <= 50000:
                        cur_snr = s["snr_db"]
                        break
                if best["snr_db"] >= cur_snr + 4.0:
                    best_station = best
                    self.freq = best["freq_hz"]

            if best_station:
                self._apply_frequency_and_mode(self.freq, self.mode)
                self.gui.center_freq = best_station["freq_hz"]
                self.gui.scan_status_text = t("auto_tuned", freq=f"{best_station['freq_mhz']:.2f}", snr=f"{best_station['snr_db']:+.1f}")
            return stations

        def safe_hf_scan(status_label: str, auto_tune_best: bool = True):
            """短波(HF)放送バンドをスキャンし、AMプリセットを更新する。
            auto_tune_best=False はシーク用 (最良局への自動同調・AM強制をせず、
            呼び出し元の周波数/モードを保つ)。"""
            nonlocal t_usb
            self.gui.scan_status_text = status_label
            if not stop_usb_stream(t_usb):
                if not t_usb.is_alive():
                    try:
                        t_usb = start_usb_stream()
                    except Exception as e:
                        print(f"[ERROR] USB restart failed: {e}", file=sys.stderr)
                else:
                    self.driver.resume_async()
                    usb_running.set()
                self.gui.scan_status_text = "USB busy - scan skipped"
                return []
            stations = []
            try:
                stations = self.tuner.scan_band_hf(snr_threshold=6.5)
                self.gui.detected_stations = stations
                self._update_presets_from_sw_scan(stations)
            except Exception as e:
                print(f"[ERROR] Shortwave scan failed: {e}", file=sys.stderr)
            finally:
                try:
                    restore_after_scan()
                except Exception as e:
                    print(f"[ERROR] Failed to restore receiver after scan: {e}", file=sys.stderr)
                _drain_raw_queue()
                t_usb = start_usb_stream()

            if not auto_tune_best:
                # シーク用: 最良局へ同調せず件数だけ通知 (起点/モードを保つ)
                self.gui.scan_status_text = (t("scan_done", n=len(stations))
                                             if stations else t("sw_scan_none"))
                return stations
            if stations:
                best = max(stations, key=lambda s: s["snr_db"])
                self._apply_frequency_and_mode(best["freq_hz"], "AM")
                self.gui.center_freq = best["freq_hz"]
                self.gui.mode = "AM"
                self.gui._sync_bfo_visibility()
                self.gui.scan_status_text = t(
                    "sw_scan_done", n=len(stations),
                    freq=f"{best['freq_mhz']:.3f}", snr=f"{best['snr_db']:+.1f}")
            else:
                self.gui.scan_status_text = t("sw_scan_none")
            return stations

        def safe_ham_scan(status_label: str, auto_tune_best: bool = True):
            """アマチュア無線HFバンド (80/40/20m) をスキャンする。
            LSB/USBは周波数から自動判定。プリセット更新なし (検出局リストのみ)。
            CWはSSB検出後に手動切替 (BFO±)。"""
            nonlocal t_usb
            self.gui.scan_status_text = status_label
            if not stop_usb_stream(t_usb):
                if not t_usb.is_alive():
                    try:
                        t_usb = start_usb_stream()
                    except Exception as e:
                        print(f"[ERROR] USB restart failed: {e}", file=sys.stderr)
                else:
                    self.driver.resume_async()
                    usb_running.set()
                self.gui.scan_status_text = "USB busy - scan skipped"
                return []
            stations = []
            try:
                stations = self.tuner.scan_band_ham(snr_threshold=5.0)
                self.gui.detected_stations = stations
            except Exception as e:
                print(f"[ERROR] Ham scan failed: {e}", file=sys.stderr)
            finally:
                try:
                    restore_after_scan()
                except Exception as e:
                    print(f"[ERROR] Failed to restore receiver after scan: {e}", file=sys.stderr)
                _drain_raw_queue()
                t_usb = start_usb_stream()

            if not auto_tune_best:
                self.gui.scan_status_text = (t("ham_scan_done", n=len(stations))
                                             if stations else t("ham_scan_none"))
                return stations
            if stations:
                best = max(stations, key=lambda s: s["snr_db"])
                best_mode = best.get("mode", "USB")
                if best_mode not in ("USB", "LSB", "CW"):
                    best_mode = "USB"
                self._apply_frequency_and_mode(best["freq_hz"], best_mode)
                self.gui.center_freq = best["freq_hz"]
                self.gui.mode = best_mode
                self.gui._sync_bfo_visibility()
                self.gui.scan_status_text = t(
                    "ham_scan_done", n=len(stations))
            else:
                self.gui.scan_status_text = t("ham_scan_none")
            return stations

        t_usb = start_usb_stream()
        self._worker_t0 = time.monotonic()

        while self.running:
            # コマンドキューの処理 (陳腐化合体: 連続FREQ等は最新のみ適用)
            for cmd, val in self._drain_commands():
                try:
                    if cmd == "FREQ":
                        self._apply_frequency_and_mode(val, self.mode)
                    elif cmd == "MODE":
                        self._apply_frequency_and_mode(self.freq, val)
                    elif cmd == "SEEK":
                        # 次局/前局シーク (AM/短波ではSW局リスト、FM/その他ではFM局リスト)
                        direction = val
                        use_sw = (self.freq < 24000000)
                        # 初回シークは局リストが無いため帯域スキャンが必要
                        # (USBストリームを安全に停止しないとread_syncが競合・ハングする)
                        is_sw = (self.mode in ("AM", "USB", "LSB", "CW")) or (self.freq < 24000000)
                        if self.mode in ("USB", "LSB", "CW"):
                            # SSB/CW時はアマチュア無線リストから探し、局のモードで同調
                            if not self.tuner.discovered_ham:
                                safe_ham_scan(t("first_seek_scan"), auto_tune_best=False)
                            ham = getattr(self.tuner, "discovered_ham", [])
                            st = None
                            if ham:
                                margin = 500
                                if direction > 0:
                                    cands = [s for s in ham if s["freq_hz"] > self.freq + margin]
                                    st = cands[0] if cands else None
                                else:
                                    cands = [s for s in ham if s["freq_hz"] < self.freq - margin]
                                    st = cands[-1] if cands else None
                            if st:
                                next_mode = st.get("mode", "USB")
                                if next_mode not in ("USB", "LSB", "CW"):
                                    next_mode = "USB"
                                self._apply_frequency_and_mode(st["freq_hz"], next_mode)
                                self.gui.center_freq = self.freq
                                self.gui.scan_status_text = t("tuned", freq=f"{st['freq_mhz']:.3f}", snr=f"{st['snr_db']:+.1f}")
                        elif is_sw:
                            if not self.tuner.discovered_sw:
                                safe_hf_scan(t("first_seek_scan"), auto_tune_best=False)
                            st = self.tuner.seek_next(self.freq, direction=direction, use_sw=True)
                            if st:
                                # WFM/NFMのまま短波へ飛ぶと無音/誤復調のためAMへ補正
                                next_mode = self.mode if self.mode in ("AM", "USB", "LSB", "CW") else "AM"
                                self._apply_frequency_and_mode(st["freq_hz"], next_mode)
                                self.gui.center_freq = self.freq
                                self.gui.scan_status_text = t("tuned", freq=f"{st.get('freq_mhz', st['freq_hz'] / 1e6):.3f}", snr=f"{st['snr_db']:+.1f}")
                        else:
                            # 初回シークは局リストが無いため帯域スキャンが必要
                            # (USBストリームを安全に停止しないとread_syncが競合・ハングする)
                            if not self.tuner.discovered_stations:
                                safe_band_scan(t("first_seek_scan"))
                            st = self.tuner.seek_next(self.freq, direction=direction, use_sw=False)
                            if st:
                                # AM系のままFM帯へ飛ぶと誤復調のためWFMへ補正
                                next_mode = self.mode if self.mode in ("WFM", "NFM") else "WFM"
                                self._apply_frequency_and_mode(st["freq_hz"], next_mode)
                                self.gui.center_freq = self.freq
                                self.gui.scan_status_text = t("tuned", freq=f"{st['freq_mhz']:.2f}", snr=f"{st['snr_db']:+.1f}")
                    elif cmd == "SCAN":
                        # 全帯域スキャン (USBストリームを安全に停止してスイープ)
                        safe_band_scan(t("scanning", start=f"{self.profile['fm_start']:g}", end=f"{self.profile['fm_end']:g}"), auto_best=(val == "auto_best"))
                    elif cmd == "SW_SCAN":
                        # 短波(HF)放送バンドスキャン (ダイレクトサンプリング)
                        safe_hf_scan(t("sw_scanning"))
                    elif cmd == "HAM_SCAN":
                        # アマチュア無線HFバンドスキャン (80/40/20m、LSB/USB自動)
                        safe_ham_scan(t("ham_scanning"))
                    elif cmd == "BFO":
                        bfo = float(np.clip(self.dsp.bfo_offset_hz + float(val), -2000.0, 2000.0))
                        self.dsp.bfo_offset_hz = bfo
                        self.gui.scan_status_text = t("bfo_msg", hz=f"{bfo:+.0f}")
                except Exception as e:
                    # 1つのコマンド失敗で受信ワーカー全体を落とさない
                    print(f"[ERROR] Command '{cmd}' failed: {e}", file=sys.stderr)

            try:
                # 生IQデータをキューから取得
                try:
                    raw_bytes = raw_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # オーディオバッファ水位（残存チャンク数）を適応リサンプラにフィードバック (クロック自動同期)
                q_size = self.audio.get_queue_size()
                self.dsp.update_resampler_feedback(float(q_size))

                # DSP復調処理 (適応リサンプラによる欠落を抑えたストリーミング)
                t_dsp = time.perf_counter()
                audio_pcm, spectrum_db = self.dsp.process(raw_bytes, mode=self.mode)
                self.rt_profile.add((time.perf_counter() - t_dsp) * 1000.0)
                # ステレオ/モノラル状態とSメーターをGUIへ反映
                self.gui.is_stereo = bool(getattr(self.dsp, "is_stereo", False))
                self.gui.stereo_status = getattr(self.dsp, "stereo_status", "MONO")
                self.gui.s_units = float(getattr(self.dsp, "s_units", 0.0))

                # RDS PS名・RadioText(楽曲名/番組名)が取れたらGUIへ反映
                if self.mode == "WFM":
                    rds_ps = getattr(self.dsp, "rds_ps", "")
                    if rds_ps and rds_ps != self.gui.station_name:
                        self.gui.station_name = rds_ps
                    rds_rt = getattr(self.dsp, "rds_rt", "")
                    if rds_rt != self.gui.rds_text:
                        self.gui.rds_text = rds_rt
                else:
                    self.gui.rds_text = ""

                # 自律最適化ループ (Hyperは復調音声そのものを聴感評価に使用)
                if self.use_controller:
                    if self.controller_type == "hyper":
                        stats = self.controller.process_frame(
                            raw_bytes, spectrum_db, audio=audio_pcm, mode=self.mode
                        )
                    else:
                        stats = self.controller.process_frame(raw_bytes, spectrum_db)
                    self.gui.gain_val = stats["gain_db"]
                    # ゲイン状態チップ廃止 (自動固定)。gain_valは信号ログ用に維持。

                    if self.controller_type == "hyper":
                        ant_tag = stats.get("antenna_profile", "BALANCED").replace("LOW_GAIN_", "LOW:").replace("HIGH_GAIN_", "HI:").replace("SATELLITE_", "SAT:")
                        drift = self.dsp.resampler.drift_ppm
                        sync_tag = "LOCK" if abs(drift) < 0.5 else f"{drift:+.0f}ppm"
                        afc_val = self.dsp.nfm_afc_offset_hz if self.mode == "NFM" else self.dsp.afc_offset_hz
                        afc_str = f"AFC:{afc_val:+.0f}Hz" if abs(afc_val) >= 1.0 else "AFC:0Hz"
                        cn_val = stats.get('channel_snr_db', stats['estimated_snr'])
                        aud_val = stats.get('audio_snr_db', 0.0)
                        dsp_tag = self._rt_summary_cache
                        if dsp_tag.startswith("DSP "):
                            dsp_parts = dsp_tag[4:].split()
                            dsp_tag = f"DSP: {dsp_parts[0]}" if dsp_parts else "DSP: OK"
                        txt = (
                            f"C/N: {cn_val:+.1f}dB | "
                            f"Aud: {aud_val:+.1f}dB | "
                            f"{dsp_tag} | "
                            f"Ant: {ant_tag} | "
                            f"Sync: {sync_tag} | "
                            f"{afc_str}"
                        )
                    else:
                        lock_str = t("lock_fixed") if hard_locked else (t("lock_converged") if stats["converged"] else t("lock_searching"))
                        txt = (
                            f"SNR: {stats['estimated_snr']:.1f}dB | "
                            f"IQ: {stats['iq_std']:.0f} | "
                            f"DSP: {self._rt_summary_cache} | "
                            f"Gain: {stats['gain_db']:.1f}dB | "
                            f"Lock: {lock_str}"
                        )
                    # テレメトリ表示は5Hzに間引き (GUI描画/GIL競合の低減)。
                    # summary()のpartition 4発もここでのみ実行する。
                    now_t = time.time()
                    if now_t - getattr(self, "_last_telemetry_time", 0.0) >= 0.2:
                        self._last_telemetry_time = now_t
                        self._rt_summary_cache = self.rt_profile.summary()
                        self.gui.telemetry_text = txt

                # 音声キューへ転送
                self.audio.put_audio(audio_pcm)

                # 信号健康ログ (1Hz。特定局の揺れ切り分け用 signal_log.csv)
                # 起動・選局の過渡 (AFC収束前) はブランキングして記録しない
                try:
                    now_sig = time.time()
                    if now_sig - getattr(self, "_siglog_last", 0.0) >= 1.0:
                        self._siglog_last = now_sig
                        d = self.dsp
                        if _sig_settled(time.monotonic(),
                                        getattr(d, "_tune_monotonic", 0.0),
                                        getattr(self, "_worker_t0", 0.0)):
                            snr_val = float(stats.get("channel_snr_db", stats.get("estimated_snr", 0.0))) if "stats" in locals() and isinstance(stats, dict) else 0.0
                            aud_snr = float(stats.get("audio_snr_db", 0.0)) if "stats" in locals() and isinstance(stats, dict) else 0.0
                            self.sig_logger.log({
                                "freq_hz": self.freq,
                                "mode": self.mode,
                                "gain_db": getattr(self.gui, "gain_val", 0.0),
                                "pilot_lock": getattr(d, "stereo_pilot_lock", 0.0),
                                "blend": getattr(d, "stereo_blend", 0.0),
                                "nr_gain": getattr(d, "stereo_nr_gain", 1.0),
                                "cut_hz": getattr(d, "stereo_cut_hz", 0.0),
                                "wiener_gain": getattr(d, "stereo_wiener_gain", 1.0),
                                "multipath_gain": getattr(d, "multipath_gain", 1.0),
                                "afc_hz": getattr(d, "afc_offset_hz", 0.0),
                                "s_units": getattr(d, "s_units", 0.0),
                                "snr_db": snr_val,
                                "audio_snr_db": aud_snr,
                            })
                except Exception:
                    pass

                # PPM自動較正の背景収集 (2秒間隔。強力FM局のAFC残差を標本化)
                try:
                    now_ppm = time.time()
                    if now_ppm - getattr(self, "_ppm_last_tick", 0.0) >= 2.0:
                        self._ppm_last_tick = now_ppm
                        self._ppm_background_tick(now_ppm)
                except Exception as e:
                    print(f"[WARN] PPM tick failed: {e}", file=sys.stderr)

                # スペクトラム・波形データの更新
                with self.spectrum_lock:
                    self.latest_spectrum = spectrum_db
                    self.latest_audio = audio_pcm

            except Exception as e:
                # 受信ループは継続しつつ、原因を可視化 (2秒に1回だけ表示)
                now = time.time()
                if now - getattr(self, "_last_worker_error_time", 0.0) > 2.0:
                    self._last_worker_error_time = now
                    import traceback
                    tb_lines = [line.strip() for line in traceback.format_exc().strip().splitlines()[-3:]]
                    tb_info = " | ".join(tb_lines)
                    print(f"[ERROR] Receiver exception: {type(e).__name__}: {e} [{tb_info}]", file=sys.stderr)
                time.sleep(0.005)

        stop_usb_stream(t_usb)

    def run(self):
        """アプリケーションのメインループ"""
        print("[*] Starting KomorebiSDR...")
        # 短波番組表をバックグラウンドで取得 (オフラインでも動作継続)
        threading.Thread(target=sw_schedule.load_schedule, daemon=True).start()
        print(f"[*] Region: {self.profile['region']} ({self.profile['label']}), "
              f"language: {i18n_language()}, stereo: {self.config.get('stereo', True)}")
        try:
            self.init_hardware()
        except NoDeviceFoundError:
            print("[ERROR] No RTL-SDR device found.")
            show_message_screen(t("no_device_title"), t("no_device_body"))
            return
        except Exception as e:
            print(f"[ERROR] Hardware initialization failed: {e}")
            show_message_screen(t("device_error_title"), [str(e), "", t("quit_hint")])
            return

        self.running = True
        self.sdr_thread = threading.Thread(target=self._sdr_worker, daemon=True)
        self.sdr_thread.start()

        # 音声ストリーム起動 (内部のis_prerolled判定により自動プレロール)
        self.audio.start()
        print("[*] Receiver running. Use the GUI window to control.")

        # GUIループ (メインスレッド)
        try:
            while self.gui.running:
                self.gui.handle_events()

                with self.spectrum_lock:
                    spec_copy = self.latest_spectrum.copy()
                    audio_copy = self.latest_audio.copy()

                self.gui.render(spec_copy, audio_copy)

        except KeyboardInterrupt:
            pass
        finally:
            print("[*] Shutting down...")
            self.running = False
            if self.sdr_thread:
                self.sdr_thread.join(timeout=5.0)
                if self.sdr_thread.is_alive():
                    # スキャン中のread_sync等で残留: 生存ワーカーがあるまま
                    # audio/driverを破棄すると競合するため警告して待機を1回延長
                    print("[WARN] SDR worker still alive; waiting once more...",
                          file=sys.stderr)
                    self.sdr_thread.join(timeout=5.0)
                    if self.sdr_thread.is_alive():
                        print("[WARN] SDR worker did not exit; skipping driver.close "
                              "to avoid use-after-close (retry on next start)",
                              file=sys.stderr)
                        try:
                            self.audio.stop()
                        except Exception:
                            pass
                        try:
                            self.gui.close()
                        except Exception:
                            pass
                        print("[*] Exited (driver left open for stuck worker).")
                        return
            self.audio.stop()
            self.driver.close()
            self.gui.close()
            print("[*] Exited cleanly.")


def _setup_stdout_log():
    """--windowed exe では stdout が無いため、ログファイルへリダイレクト"""
    try:
        if sys.stdout is None or sys.stderr is None:
            logf = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
            atexit.register(logf.close)
            sys.stdout = logf
            sys.stderr = logf
    except Exception:
        pass


def main():
    _setup_stdout_log()
    parser = argparse.ArgumentParser(description="KomorebiSDR - Modern Zero-Configuration SDR Radio")
    parser.add_argument("--freq", type=float, default=None,
                        help="Initial frequency in MHz (default: region profile)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["WFM", "AM", "NFM", "USB", "LSB", "CW"],
                        help="Demodulation mode (WFM / AM / NFM / USB / LSB / CW)")
    parser.add_argument("--controller", type=str, default="hyper", choices=["hyper", "cascade"],
                        help="Autonomous optimization engine (hyper / cascade)")
    parser.add_argument("--country", type=str, default=None,
                        help="Country code override (e.g. JP, US, DE). Default: auto-detect")
    parser.add_argument("--lang", type=str, default=None, choices=["ja", "en"],
                        help="UI language override. Default: auto-detect")
    parser.add_argument("--mono", action="store_true", help="Force monaural FM (disable stereo)")
    parser.add_argument("--no-stereo-nr", action="store_true",
                        help="Disable stereo noise reduction (keep full stereo even when noisy)")
    parser.add_argument("--no-sic", action="store_true",
                        help="Disable digital self-interference cancellation (SIC)")
    args = parser.parse_args()

    cfg = load_config()
    detected_country = detect_country()
    country = (args.country or cfg.get("country") or detected_country or "CCIR").upper()
    language = args.lang or cfg.get("language") or detect_language()
    i18n_set_language(language)

    freq_hz = int(args.freq * 1e6) if args.freq is not None else None
    app = SdrApp(initial_freq=freq_hz, initial_mode=args.mode,
                 controller_type=args.controller, config=cfg, country=country,
                 stereo=(False if args.mono else None),
                 stereo_nr=(False if args.no_stereo_nr else None),
                 sic=(False if args.no_sic else None))

    try:
        app.run()
    finally:
        # 音量・言語・地域など次回起動に引き継ぐ設定を保存
        # (国は明示指定 or 初回自動判定できた場合のみ固定し、判定不能な "CCIR" は保存しない)
        app.config["volume"] = app.audio.volume
        if args.country:
            app.config["country"] = args.country.upper()
        elif detected_country and not cfg.get("country"):
            app.config["country"] = detected_country
        app.config["language"] = language
        save_config(app.config)


if __name__ == "__main__":
    main()
