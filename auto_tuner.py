"""
Auto-Tuner & Smart DX Station Search Engine for SDR.
帯域全体を電光石火の速さで高速スキャンし、ノイズに埋もれた微弱局（DX局）から
強力なローカル局までを100Hz精度で自動探査・分析・選局する自律チューニングエンジン。
"""

import time
import numpy as np
from rtlsdr_driver import RtlSdrDriver


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

    def __init__(self, driver: RtlSdrDriver = None):
        self.driver = driver if driver is not None else RtlSdrDriver()
        self.owns_driver = driver is None
        self.discovered_stations = []  # スキャンで発見された局リスト
        self.last_scan_time = 0.0

    def scan_band(
        self,
        start_hz: int = 76000000,
        end_hz: int = 95000000,
        step_hz: int = 1500000,
        snr_threshold: float = 4.2,
    ) -> list[dict]:
        """
        帯域全体を高速スイープし、微弱局(SNR >= 3.5dB)から強力局までを全て抽出。
        :param start_hz: スキャン開始周波数 (デフォルト 76.0MHz)
        :param end_hz: スキャン終了周波数 (デフォルト 95.0MHz)
        :param step_hz: チューナーステップ幅 (デフォルト 1.5MHz)
        :param snr_threshold: ピーク検知しきい値 (微弱局を拾うため 3.5dB に設定)
        :return: 発見された局のリスト（周波数、SNR、信号強度、局名、AFC補正値）
        """
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
            freq_centers.append(end_hz - scan_rate // 2)

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

            # 周波数軸
            freq_axis = np.linspace(fc - scan_rate / 2, fc + scan_rate / 2, fft_size)

            # DC中心スパイク領域 (±25kHz) は除外
            dc_center_idx = fft_size // 2
            dc_guard = int(25000 / (scan_rate / fft_size))
            spec_db[dc_center_idx - dc_guard : dc_center_idx + dc_guard + 1] = noise_floor

            # ピーク検出
            peaks = self._find_spectral_peaks(spec_db, freq_axis, noise_floor, snr_threshold)
            for p in peaks:
                # 日本のFMグリッド (100kHz単位) にスナップ
                snapped_hz = int(round(p["peak_freq"] / 100000.0) * 100000)
                if snapped_hz < start_hz or snapped_hz > end_hz:
                    continue

                afc_offset = p["peak_freq"] - snapped_hz  # ドングルや送信機のズレ
                name = match_station_name(snapped_hz)

                # 日本のFM規格アライメント判定:
                # 偏差が ±15kHz を超えるものはFMグリッド外のランダムノイズ突起と判定して除外
                # （既知局名がある場合、またはSNRが12dB以上の強局は保護）
                if abs(afc_offset) > 15000 and name == "Unknown FM Station" and p["snr_db"] < 12.0:
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
                        "STRONG" if p["snr_db"] >= 22.0
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

    def _find_spectral_peaks(
        self, spec_db: np.ndarray, freq_axis: np.ndarray, noise_floor: float, threshold_snr: float
    ) -> list[dict]:
        """スペクトラムから凸形状のピークを抽出"""
        peaks = []
        n = len(spec_db)
        # FM信号の帯域幅（約100〜150kHz）に相当するビン幅
        min_width_bins = 5
        margin = 10

        for i in range(margin, n - margin):
            val = spec_db[i]
            snr = val - noise_floor
            if snr < threshold_snr:
                continue

            # 局所極大判定 (±min_width_bins の中で最大か)
            surrounding = spec_db[i - min_width_bins : i + min_width_bins + 1]
            if val == np.max(surrounding):
                # プロミネンス判定: 周囲の谷に対して少なくとも 1.5dB 以上の盛り上がりがあるか
                prominence = val - np.min(surrounding)
                if prominence < 1.5:
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

    def seek_next(self, current_freq_hz: int, direction: int = 1) -> dict | None:
        """
        現在周波数から次の局へ自動ジャンプ (direction: +1 で上へ、-1 で下へ)
        """
        if not self.discovered_stations:
            self.scan_band()
        if not self.discovered_stations:
            return None

        freqs = [s["freq_hz"] for s in self.discovered_stations]
        if direction > 0:
            for s in self.discovered_stations:
                if s["freq_hz"] > current_freq_hz + 50000:
                    return s
            return self.discovered_stations[0]  # 先頭へループ
        else:
            for s in reversed(self.discovered_stations):
                if s["freq_hz"] < current_freq_hz - 50000:
                    return s
            return self.discovered_stations[-1]  # 末尾へループ

    def get_dx_stations(self) -> list[dict]:
        """通常聞こえない微弱なDX局のみを抽出"""
        return [s for s in self.discovered_stations if s["quality"] == "WEAK (DX)"]

    def get_strongest_station(self) -> dict | None:
        """最も電波が強い局を取得"""
        if not self.discovered_stations:
            return None
        return max(self.discovered_stations, key=lambda s: s["snr_db"])
