"""
Auto-Tuner & Smart DX Station Search Engine for SDR.
帯域全体を電光石火の速さで高速スキャンし、ノイズに埋もれた微弱局（DX局）から
強力なローカル局までを100Hz精度で自動探査・分析・選局する自律チューニングエンジン。
"""

import time
import numpy as np
from rtlsdr_driver import RtlSdrDriver
from config import SHORTWAVE_BANDS, SW_MAX_HZ, shortwave_scan_centers


# 日本の主要FM局データベース (関東・茨城・東京・広域)
KNOWN_STATIONS = {
    # 茨城・水戸周辺
    94600000: "LuckyFM 茨城放送 (水戸)",
    88100000: "LuckyFM 茨城放送 (日立)",
    83200000: "NHK-FM 水戸",
    # 東京・広域民放
    80000000: "TOKYO FM (東京)",
    81300000: "J-WAVE (東京)",
    82500000: "NHK-FM 東京",
    89700000: "InterFM897 (東京)",
    # ワイドFM (東京)
    90500000: "TBSラジオ (ワイドFM)",
    91600000: "文化放送 (ワイドFM)",
    93000000: "ニッポン放送 (ワイドFM)",
    # 関東近郊FM
    78000000: "bayfm78 (千葉)",
    79500000: "NACK5 (埼玉)",
    80300000: "NHK-FM 宇都宮",
    84700000: "Fm yokohama (神奈川)",
    86300000: "FM GUNMA (群馬)",
    86400000: "NHK-FM 前橋",
    92800000: "FM GUNMA (ワイドFM)",
    94100000: "栃木放送 CRT (ワイドFM)",
    76400000: "RADIO BERRY (栃木)",
    # コミュニティFM (茨城)
    76200000: "FMぱるるん (水戸)",
    84200000: "FMラヂオつくば",
}


def match_station_name(freq_hz: int) -> str:
    """周波数から既知の放送局名を取得 (±50kHz以内でマッチング)"""
    for f, name in KNOWN_STATIONS.items():
        if abs(f - freq_hz) <= 50000:
            return name
    return "Unknown FM Station"


class AutoTuner:
    """全自動帯域探査＆微弱局（DX）発掘エンジン"""

    SCAN_TTL_SEC = 600.0  # 局リスト鮮度TTL (10分超過で再スキャン)

    def __init__(self, driver: RtlSdrDriver = None):
        self.driver = driver if driver is not None else RtlSdrDriver()
        self.owns_driver = driver is None
        self.discovered_stations = []  # スキャンで発見された局リスト (FM)
        self.discovered_sw = []        # 短波(HF)スキャン結果
        self.last_scan_time = 0.0

    def scan_band(
        self,
        start_hz: int = 76000000,
        end_hz: int = 95000000,
        step_hz: int = 1500000,
        snr_threshold: float = 7.5,
    ) -> list[dict]:
        """
        帯域全体を高速スイープし、本物のFM放送局のみを確実に抽出 (偽局・ノイズスプリアス完全排除)。
        :param start_hz: スキャン開始周波数 (デフォルト 76.0MHz)
        :param end_hz: スキャン終了周波数 (デフォルト 95.0MHz)
        :param step_hz: チューナーステップ幅 (デフォルト 1.5MHz)
        :param snr_threshold: ピーク検知しきい値 (ノイズフロアの微細な山を排除するため 7.5dB に設定)
        :return: 発見された局のリスト（周波数、SNR、信号強度、局名、AFC補正値）
        """
        if step_hz <= 0 or end_hz <= start_hz:
            return []
        was_open = self.driver.is_open
        if not was_open:
            self.driver.open(0)

        # スキャン用の最適サンプリングレート
        scan_rate = 2048000
        self.driver.set_sample_rate(scan_rate)
        self.driver.set_direct_sampling(0)
        self.driver.set_gain_mode(True)
        self.driver.set_gain(33.8)  # 低利得アンテナでの実測SNR最大スウィートスポット (33.8dB)

        stations = []
        fft_size = 1024
        window = np.hamming(fft_size).astype(np.float32)
        win_power = np.sum(window**2) / fft_size

        freq_centers = list(range(start_hz + scan_rate // 2, end_hz, step_hz))
        if not freq_centers or freq_centers[-1] < end_hz - scan_rate // 2:
            freq_centers.append(max(end_hz - scan_rate // 2, start_hz))

        for fc in freq_centers:
            self.driver.set_center_freq(fc)
            self.driver.reset_buffer()
            # 安定化のため少量を読み捨て
            _ = self.driver.read_sync(32768)

            # FFT用のIQデータを取得 (複数フレーム平均でノイズを抑圧)
            num_avg = 8
            read_len = fft_size * 2 * num_avg
            raw = self.driver.read_sync(read_len)
            if len(raw) < fft_size * 2:
                continue

            raw_f = raw.astype(np.float32)
            iq = (raw_f[0::2] - 127.5) * (1.0 / 128.0) + 1j * (raw_f[1::2] - 127.5) * (1.0 / 128.0)

            accum_power = np.zeros(fft_size, dtype=np.float64)
            actual_frames = len(iq) // fft_size
            for f_idx in range(actual_frames):
                chunk = iq[f_idx * fft_size : (f_idx + 1) * fft_size] * window
                fft_data = np.fft.fftshift(np.fft.fft(chunk)) / fft_size
                accum_power += (np.abs(fft_data) ** 2) / win_power

            avg_power = accum_power / max(1, actual_frames) + 1e-12
            spec_db = 10.0 * np.log10(avg_power)

            # ノイズフロア算出 (下位30パーセンタイル)
            noise_floor = float(np.percentile(spec_db, 30))

            # 周波数軸 (FFTビン中心: bin i は fc-half + i*binwidth。
            # linspace(endpoint=True) は末尾が fc+half になり半ビン〜2ビンずれる)
            binw = scan_rate / fft_size
            freq_axis = fc - scan_rate / 2 + np.arange(fft_size) * binw

            # DC中心スパイク領域 (±5kHzのみ。±25kHzだと100kHzグリッド局が
            # 掃引中心近傍に来るたびマスクされ、18ch毎に1局見逃す構造だった)
            dc_center_idx = fft_size // 2
            dc_guard = max(1, int(5000 / binw))
            spec_db[dc_center_idx - dc_guard : dc_center_idx + dc_guard + 1] = noise_floor

            # ピーク検出 (帯域幅チェック付き)
            peaks = self._find_spectral_peaks(spec_db, freq_axis, noise_floor, snr_threshold)
            for p in peaks:
                # 日本のFMグリッド (100kHz単位) にスナップ
                snapped_hz = int(round(p["peak_freq"] / 100000.0) * 100000)
                if snapped_hz < start_hz or snapped_hz > end_hz:
                    continue

                afc_offset = p["peak_freq"] - snapped_hz  # ドングルや送信機のズレ
                name = match_station_name(snapped_hz)

                # 本物のFM放送局判定:
                # 1. 既知局DBにある場合は SNR 5.0dB でも許容
                # 2. 未知局の場合は SNR >= 8.5dB かつ FMグリッド偏差 ±12kHz 以内であることを要求
                is_known = (name != "Unknown FM Station")
                if not is_known:
                    if p["snr_db"] < 8.5 or abs(afc_offset) > 12000:
                        continue
                else:
                    if p["snr_db"] < 5.0:
                        continue

                # 重複登録防止（最もSNRが高い観測値を採用）
                existing = [s for s in stations if s["freq_hz"] == snapped_hz]
                if existing:
                    if p["snr_db"] > existing[0]["snr_db"]:
                        existing[0]["snr_db"] = p["snr_db"]
                        existing[0]["peak_power_db"] = p["peak_power"]
                        existing[0]["afc_offset_hz"] = afc_offset
                else:
                    quality = (
                        "STRONG" if p["snr_db"] >= 20.0
                        else "MEDIUM" if p["snr_db"] >= 10.0
                        else "WEAK (DX)"
                    )
                    stations.append({
                        "freq_hz": snapped_hz,
                        "freq_mhz": snapped_hz / 1e6,
                        "name": name,
                        "snr_db": round(p["snr_db"], 1),
                        "peak_power_db": round(p["peak_power"], 1),
                        "afc_offset_hz": round(afc_offset, 0),
                        "quality": quality,
                    })

        if not was_open and self.owns_driver:
            self.driver.close()

        # 周波数順にソート
        stations.sort(key=lambda s: s["freq_hz"])

        # 隣接チャンネルスプラッター（強力局の裾野によるゴースト局）の除去フィルタ
        filtered_stations = []
        for i, s in enumerate(stations):
            is_ghost = False
            # 周囲 ±200kHz 以内に自身より強力な局が存在するか判定
            for other in stations:
                if other["freq_hz"] == s["freq_hz"]:
                    continue
                dist_hz = abs(other["freq_hz"] - s["freq_hz"])
                if dist_hz <= 100000:
                    # 隣接 ±100kHz で相手のほうがSNRが高い場合、未知局は相手の帯域外漏洩スプラッター
                    if s["name"] == "Unknown FM Station" and other["snr_db"] > s["snr_db"] + 1.5:
                        is_ghost = True
                        break
                elif dist_hz <= 200000 and other["snr_db"] >= 16.0:
                    # 強力局の±200kHzスカート
                    if s["name"] == "Unknown FM Station" and (other["snr_db"] - s["snr_db"] >= 7.0):
                        is_ghost = True
                        break
            if not is_ghost:
                filtered_stations.append(s)

        self.discovered_stations = filtered_stations
        self.last_scan_time = time.time()
        return filtered_stations

    def scan_band_hf(
        self,
        snr_threshold: float = 6.5,
        rate_hz: int = 1152000,
        max_results: int = 40,
    ) -> list[dict]:
        """短波(HF)放送バンドをダイレクトサンプリング(Qブランチ)でスキャンする。

        2.0〜14.4MHzのSW放送バンドを5kHzグリッドで走査し、受信可能な局を返す。
        15MHz以上はRTL-SDRのダイレクトサンプリング上限を超えるため対象外
        (アップコンバータ使用時は config.SW_MAX_HZ を変更)。
        """
        was_open = self.driver.is_open
        if not was_open:
            self.driver.open(0)

        self.driver.set_sample_rate(int(rate_hz))
        self.driver.set_direct_sampling(2)   # Qブランチ (HF)
        self.driver.set_gain_mode(True)
        self.driver.set_gain(33.8)           # ダイレクトサンプリングでは実質無効

        centers = shortwave_scan_centers(rate_hz)
        fft_size = 2048
        window = np.hamming(fft_size).astype(np.float32)
        win_power = np.sum(window ** 2) / fft_size
        band_ranges = [
            (int(lo * 1000), int(min(hi * 1000, SW_MAX_HZ)))
            for _name, lo, hi in SHORTWAVE_BANDS
        ]
        stations = []

        for fc in centers:
            self.driver.set_center_freq(fc)
            self.driver.reset_buffer()
            _ = self.driver.read_sync(32768)

            num_avg = 6
            raw = self.driver.read_sync(fft_size * 2 * num_avg)
            if len(raw) < fft_size * 2:
                continue
            raw_f = raw.astype(np.float32)
            iq = (raw_f[0::2] - 127.5) * (1.0 / 128.0) + \
                 1j * (raw_f[1::2] - 127.5) * (1.0 / 128.0)

            accum_power = np.zeros(fft_size, dtype=np.float64)
            frames = len(iq) // fft_size
            for fi in range(frames):
                chunk = iq[fi * fft_size:(fi + 1) * fft_size] * window
                fft_data = np.fft.fftshift(np.fft.fft(chunk)) / fft_size
                accum_power += (np.abs(fft_data) ** 2) / win_power
            avg_power = accum_power / max(1, frames) + 1e-12
            spec_db = 10.0 * np.log10(avg_power)

            noise_floor = float(np.percentile(spec_db, 40))
            binw2 = rate_hz / fft_size
            freq_axis = fc - rate_hz / 2 + np.arange(fft_size) * binw2

            # DCスパイク除去 (±5kHz。±30kHzは短波5kHzグリッド局を多数マスクする)
            dc = fft_size // 2
            guard = max(1, int(5000 / binw2))
            spec_db[dc - guard: dc + guard + 1] = noise_floor

            peaks = self._find_spectral_peaks(
                spec_db, freq_axis, noise_floor, snr_threshold, min_width_bins=3)
            for p in peaks:
                # SW放送は5kHzグリッド
                snapped = int(round(p["peak_freq"] / 5000.0) * 5000)
                if not any(lo <= snapped <= hi for lo, hi in band_ranges):
                    continue
                existing = next(
                    (s for s in stations if abs(s["freq_hz"] - snapped) < 4000), None)
                if existing:
                    if p["snr_db"] > existing["snr_db"]:
                        existing["snr_db"] = round(p["snr_db"], 1)
                        existing["peak_power_db"] = round(p["peak_power"], 1)
                    continue
                stations.append({
                    "freq_hz": snapped,
                    "freq_mhz": snapped / 1e6,
                    "name": "Unknown FM Station",
                    "snr_db": round(p["snr_db"], 1),
                    "peak_power_db": round(p["peak_power"], 1),
                    "afc_offset_hz": round(p["peak_freq"] - snapped, 0),
                    "quality": "STRONG" if p["snr_db"] >= 20.0 else (
                        "MEDIUM" if p["snr_db"] >= 10.0 else "WEAK (DX)"),
                })

        if not was_open and self.owns_driver:
            self.driver.close()

        # 上位をSNR順で選んでから周波数順に並べ直す
        # (周波数順で切ると低帯域の弱局で枠が埋まり、22m等の強局が漏れる)
        stations.sort(key=lambda s: s["snr_db"], reverse=True)
        stations = stations[:max_results]
        stations.sort(key=lambda s: s["freq_hz"])
        self.discovered_sw = stations
        self.last_scan_time = time.time()
        return stations

    def _find_spectral_peaks(
        self, spec_db: np.ndarray, freq_axis: np.ndarray, noise_floor: float,
        threshold_snr: float, min_width_bins: int = 5,
    ) -> list[dict]:
        """スペクトラムから凸形状かつFM帯域幅エネルギーを持つ本物の放送ピークを抽出"""
        peaks = []
        n = len(spec_db)
        margin = 15

        for i in range(margin, n - margin):
            val = spec_db[i]
            snr = val - noise_floor
            if snr < threshold_snr:
                continue

            # 局所極大判定 (±min_width_bins の中で最大か)
            surrounding = spec_db[i - min_width_bins : i + min_width_bins + 1]
            if val == np.max(surrounding):
                # プロミネンス判定: 周囲の谷に対して少なくとも 2.5dB 以上の明瞭な盛り上がりがあるか
                prominence = val - np.min(surrounding)
                if prominence < 2.5:
                    continue

                # 占有帯域幅エネルギー検証:
                # 本物のFM放送は ±30kHz (約15ビン) にわたってエネルギーが台形状に広がる。
                # 針状の孤立クロックスパイク (幅 < 15kHz) を除外
                left_shoulder = spec_db[max(0, i - 12)] - noise_floor
                right_shoulder = spec_db[min(n - 1, i + 12)] - noise_floor
                # ショルダー部がノイズフロア以下かつプロミネンスが異常に尖っている(針状)場合はスプリアスと判定
                if left_shoulder < 0.5 and right_shoulder < 0.5 and prominence > 15.0:
                    continue

                # 周波数の重心計算（サブビン精度でピーク中心を算出）
                region = spec_db[i - 2 : i + 3]
                weights = 10.0 ** (region / 10.0)  # リニアパワーで重み付け
                freq_region = freq_axis[i - 2 : i + 3]
                sub_peak_freq = float(np.sum(freq_region * weights) / np.sum(weights))

                peaks.append({
                    "peak_freq": sub_peak_freq,
                    "snr_db": float(snr),
                    "peak_power": float(val),
                    "prominence": float(prominence),
                })

        return peaks

    def seek_next(self, current_freq_hz: int, direction: int = 1, use_sw: bool = False) -> dict | None:
        """
        現在周波数から次の局へ自動ジャンプ (direction: +1 で上へ、-1 で下へ)。
        use_sw=True のときは短波(HF)スキャン結果から探す (AM/短波モード用)。
        局リストがTTL超過で古い場合は再スキャンする。
        """
        stations = self.discovered_sw if use_sw else self.discovered_stations
        if not stations:
            if use_sw:
                self.scan_band_hf()
                stations = self.discovered_sw
            else:
                self.scan_band()
                stations = self.discovered_stations
        elif float(self.last_scan_time) != 0.0 and (time.time() - float(self.last_scan_time)) > self.SCAN_TTL_SEC:
            # 鮮度切れ: バンドプラン/地域変更後も古い局へ飛ぶのを防止
            # (last_scan_time==0 はテスト用fake等で時刻未設定のため再スキャンしない)
            try:
                if use_sw:
                    self.scan_band_hf()
                    stations = self.discovered_sw
                else:
                    # デフォルト帯域で再スキャン (呼出側が帯域指定済みの場合は上書きされる)
                    self.scan_band()
                    stations = self.discovered_stations
            except Exception:
                pass
        if not stations:
            return None

        freqs = [s["freq_hz"] for s in stations]
        # 短波は5kHzグリッドのためFM用±50kHz窓では隣接局を飛ばす/自局ラップする。
        # SW時は±2kHz窓に狭め、ラップ時は自局を除外する。
        margin_hz = 2000 if use_sw else 50000
        if direction > 0:
            for s in stations:
                if s["freq_hz"] > current_freq_hz + margin_hz:
                    return s
            for s in stations:
                if abs(s["freq_hz"] - current_freq_hz) > margin_hz:
                    return s
            return None  # 自局しかない場合は動かない
        else:
            for s in reversed(stations):
                if s["freq_hz"] < current_freq_hz - margin_hz:
                    return s
            for s in reversed(stations):
                if abs(s["freq_hz"] - current_freq_hz) > margin_hz:
                    return s
            return None

    def get_dx_stations(self) -> list[dict]:
        """通常聞こえない微弱なDX局のみを抽出"""
        return [s for s in self.discovered_stations if s["quality"] == "WEAK (DX)"]

    def get_strongest_station(self) -> dict | None:
        """最も電波が強い局を取得"""
        if not self.discovered_stations:
            return None
        return max(self.discovered_stations, key=lambda s: s["snr_db"])
