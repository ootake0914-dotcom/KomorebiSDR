"""
Cascade Autonomous Controller Module for SDR - Maximum SNR & Weak Antenna Edition.
強電界から極微弱電界（低利得・クソアンテナ）まで、受信環境に応じて自律的に
「ADC飽和防止」「分散最大化」「実効SNR極大探索(Hill Climbing)」を切り替える次世代最適化エンジン。
"""

import time
import numpy as np


class CascadeController:
    """SDRカスケード自律最適化コントローラ"""

    MIN_SAFE_GAIN_DB = 19.7  # 実用最低安全ゲイン: 19.7dB未満(熱雑音沈没)への転落を完全防止
    FLOOR_GAIN_DB = 12.5  # 持続クリップ時の非常用下限

    def _min_safe_idx(self) -> int:
        if not self.available_gains:
            return 0
        return int(min(range(len(self.available_gains)), key=lambda i: abs(self.available_gains[i] - self.MIN_SAFE_GAIN_DB)))

    def _floor_idx(self) -> int:
        if not self.available_gains:
            return 0
        return int(min(range(len(self.available_gains)), key=lambda i: abs(self.available_gains[i] - self.FLOOR_GAIN_DB)))

    def __init__(self, driver, dsp, audio):
        self.driver = driver
        self.dsp = dsp
        self.audio = audio

        self.enabled = True
        self.dx_mode = False  # DX微弱局超高感度モード
        self.available_gains = []
        self.current_gain_idx = 0

        # 第1層（RFゲイン制御）パラメータ
        self.target_std = 32.0
        self.std_margin = 6.0
        self.last_gain_adjust_time = 0.0
        self.gain_adjust_interval = 0.12  # 120msごとに適応更新 (レスポンス向上)

        # 弱電界（クソアンテナ）用 Maximum SNR 探索パラメータ
        self.snr_smooth = 10.0
        self.snr_alpha = 0.25
        self.noise_floor_smooth = -70.0
        self.gain_snr_history = {}  # {gain_idx: smoothed_snr}
        self.search_direction = +1  # +1: ゲイン上げ探索, -1: ゲイン下げ探索
        self.weak_converged_count = 0
        self.hard_lock = False  # 収束後の完全決め打ち固定 (フェージング・無音での誤再探索を完全防止)
        self.filter_override = None  # 手動フィルタ固定 (hyper互換: None=自動)
        self._filter_hold_ticks = 0

        # 内部統計モニタ用
        self.last_stats = {
            "adc_clip_pct": 0.0,
            "iq_std": 0.0,
            "gain_db": 0.0,
            "estimated_snr": 0.0,
            "filter_mode": "clean",
            "converged": False,
            "hard_lock": False,
            "dx_mode": False,
        }

    def init_gains(self):
        """利用可能なチューナーゲイン一覧を取得して初期化"""
        gains = self.driver.get_gains()
        if not gains:
            # 取得できない場合の標準的なR820Tゲインテーブル
            gains = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
                     19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
                     38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
        self.available_gains = sorted(gains)

        # 初期値は実機SRH805S実測から得られた高感度・安全な最適値 (43.4dB) 付近
        default_val = 43.4 if 43.4 in self.available_gains else (33.8 if 33.8 in self.available_gains else self.available_gains[len(self.available_gains) // 2])
        self.current_gain_idx = self.available_gains.index(default_val)
        self.driver.set_gain_mode(True)
        self.driver.set_gain(default_val)
        self.gain_snr_history.clear()
        self.weak_converged_count = 0
        self.hard_lock = False

    def reset_tracking(self, initial_gain: float = None):
        """
        周波数変更・選局時に探索状態を初期化し、新局の電界強度へ即座に適応させる
        :param initial_gain: リセット時に設定するゲイン (Noneの場合はスウィートスポット43.4dB)
        """
        self.gain_snr_history.clear()
        self.weak_converged_count = 0
        self.search_direction = +1
        self.last_gain_adjust_time = 0.0
        self.snr_smooth = 10.0
        self.hard_lock = False

        if self.available_gains:
            if initial_gain is not None:
                # 指定ゲインに最も近いゲインを選択
                idx = min(range(len(self.available_gains)), key=lambda i: abs(self.available_gains[i] - initial_gain))
            else:
                # 実機実測スウィートスポット (43.4dB)
                target = 43.4 if 43.4 in self.available_gains else self.available_gains[len(self.available_gains) // 2]
                idx = self.available_gains.index(target)
            self.current_gain_idx = idx
            self.driver.set_gain(self.available_gains[self.current_gain_idx])

    def set_hard_lock(self, locked: bool):
        """ユーザーまたは収束による完全決め打ち固定 (True: 固定, False: 再探索開始)"""
        self.hard_lock = locked
        if not locked:
            # ロック解除時は探索カウンタをリセットして即座に再評価
            self.weak_converged_count = 0
            self.gain_snr_history.clear()

    def set_filter_override(self, mode):
        """手動フィルタ固定 (hyper互換)。Noneで自動へ復帰。"""
        self.filter_override = mode
        # 数tickは自動切替を抑止して手動設定を定着させる
        self._filter_hold_ticks = 10

    def process_frame(self, raw_bytes: np.ndarray, spectrum_db: np.ndarray = None):
        """
        毎フレームの生データとスペクトルから多段カスケードフィードバックを実行
        """
        if not self.enabled or len(raw_bytes) < 100:
            return self.last_stats

        now = time.time()

        # ========================================================
        # [計測部] 信号品質指標のリアルタイム算出
        # ========================================================
        # 1. ADC飽和率 (0 または 255)
        clip_count = np.sum((raw_bytes <= 1) | (raw_bytes >= 254))
        clip_pct = (clip_count / len(raw_bytes)) * 100.0

        # 2. 生IQの標準偏差 (ゼロ中心での分散度合い)
        iq_centered = raw_bytes.astype(np.float32) - 127.5
        iq_std = float(np.std(iq_centered))

        # 3. 推定SNRの算出
        if spectrum_db is not None and len(spectrum_db) > 0:
            noise_floor = float(np.percentile(spectrum_db, 20))
            peak_val = float(np.max(spectrum_db))
            inst_snr = max(0.0, peak_val - noise_floor)
            self.snr_smooth = self.snr_alpha * inst_snr + (1.0 - self.snr_alpha) * self.snr_smooth
            self.noise_floor_smooth = self.snr_alpha * noise_floor + (1.0 - self.snr_alpha) * self.noise_floor_smooth

        # ========================================================
        # [第1層] ハイブリッド・カスケードゲイン制御 (外側ループ)
        # ========================================================
        converged = self.hard_lock
        if self.hard_lock:
            converged = True
        elif now - self.last_gain_adjust_time >= self.gain_adjust_interval:
            self.last_gain_adjust_time = now

            # 現在のゲイン段での平滑化SNRを記憶
            current_idx = self.current_gain_idx
            min_idx = self._min_safe_idx()
            self.gain_snr_history[current_idx] = self.snr_smooth

            # --- A. 緊急サチュレーション回避 (本物のクリップ 1.2% 以上時のみ安全に減衰) ---
            # 持続する強クリップ(>2.5%)ではFLOOR(12.5dB)までの非常減衰を許可し永久クリップを解消
            if clip_pct > 1.2:
                # クリップ時は即座に1〜2段下げる (通常は安全最低ゲイン未満には下げない)
                step_down = 2 if clip_pct > 2.5 else 1
                floor_idx = self._floor_idx() if clip_pct > 2.5 else min_idx
                new_idx = max(floor_idx, current_idx - step_down)
                if new_idx != current_idx:
                    self.current_gain_idx = new_idx
                    self.driver.set_gain(self.available_gains[self.current_gain_idx])
                self.weak_converged_count = 0

            # --- B. 強電界モード: IQ分散によるダイナミックレンジ最大化 ---
            elif iq_std >= 18.0:
                self.weak_converged_count = 0
                if iq_std > (self.target_std + self.std_margin):
                    if current_idx > min_idx:
                        self.current_gain_idx -= 1
                        self.driver.set_gain(self.available_gains[self.current_gain_idx])
                elif clip_pct == 0.0 and iq_std < (self.target_std - self.std_margin):
                    if current_idx < len(self.available_gains) - 1:
                        self.current_gain_idx += 1
                        self.driver.set_gain(self.available_gains[self.current_gain_idx])
                else:
                    converged = True
                    self.hard_lock = True  # 強電界最適点で決め打ち固定

            # --- C. 弱電界（クソアンテナ / DX）モード: 実効SNR極大探索 (Hill Climbing) ---
            else:
                # 目標分散32に届かないため、分散ではなく「SNRが最大になるゲイン段」を探す！
                # 1. ゲインが低すぎる場合 (IQ分散 < 1.0) は即座にジャンプアップ
                if iq_std < 1.0 and current_idx < len(self.available_gains) - 3:
                    self.current_gain_idx += 3
                    self.driver.set_gain(self.available_gains[self.current_gain_idx])
                    self.weak_converged_count = 0
                elif iq_std < 1.8 and current_idx < len(self.available_gains) - 2:
                    self.current_gain_idx += 2
                    self.driver.set_gain(self.available_gains[self.current_gain_idx])
                    self.weak_converged_count = 0
                else:
                    # 前後のSNRを比較して極大点を探索
                    next_idx = max(min_idx, min(len(self.available_gains) - 1, current_idx + self.search_direction))
                    prev_idx = max(min_idx, min(len(self.available_gains) - 1, current_idx - self.search_direction))

                    # ゲインを上げた結果、SNRが改善したか悪化したかを判定
                    if prev_idx in self.gain_snr_history and prev_idx != current_idx:
                        prev_snr = self.gain_snr_history[prev_idx]
                        if self.snr_smooth > prev_snr + 0.3:
                            # ゲイン変更でSNRが改善！同じ方向へ進む
                            if min_idx <= next_idx < len(self.available_gains) and next_idx != current_idx:
                                self.current_gain_idx = next_idx
                                self.driver.set_gain(self.available_gains[self.current_gain_idx])
                                self.weak_converged_count = 0
                            else:
                                converged = True
                                self.hard_lock = True
                        elif self.snr_smooth < prev_snr - 0.3:
                            # ゲインを上げたらLNA内部熱雑音が増えてSNRが悪化した！
                            # 直前のゲイン段が最良だったので戻して探索終了
                            self.current_gain_idx = prev_idx
                            self.driver.set_gain(self.available_gains[self.current_gain_idx])
                            self.search_direction = -self.search_direction
                            self.weak_converged_count += 1
                        else:
                            # 差が微小 (ほぼ極大付近)
                            self.weak_converged_count += 1
                    else:
                        # 比較対象がない場合はまず上を探索
                        if current_idx < len(self.available_gains) - 1:
                            self.current_gain_idx += 1
                            self.driver.set_gain(self.available_gains[self.current_gain_idx])
                        else:
                            converged = True
                            self.hard_lock = True

                    if self.weak_converged_count >= 2:
                        converged = True
                        self.hard_lock = True  # 収束完了: 自動的に決め打ち固定

        current_gain = self.available_gains[self.current_gain_idx] if self.available_gains else 0.0

        # ========================================================
        # [第2層] 推定SNRに応じた適応型フィルタ帯域制御 (中間ループ)
        # ========================================================
        # 強電界 (SNR > 22dB): 音楽用ワイド(14kHz)を開放
        # 中電界 (10dB <= SNR <= 22dB): ヒスノイズ除去クリーン(8.5kHz)
        # 微弱電界 (SNR < 10dB) または DXモード: DX超高感度狭帯域(5.5kHz + ±60kHz狭帯域IF)
        if self.dx_mode or self.snr_smooth < 9.0:
            target_filter = "narrow"
        elif self.snr_smooth > 22.0:
            target_filter = "wide"
        else:
            target_filter = "clean"

        # 手動フィルタ固定の尊重: GUIからの手動設定は次tickで上書きしない
        # (main.py FILTERハンドラがdsp.filter_modeへ直書きするため、ここでは
        #  自動切替を1tick見送ることで定着させる。hyperのfilter_override相当の簡易版)
        if self.filter_override is not None:
            target_filter = self.filter_override
        if getattr(self, "_filter_hold_ticks", 0) > 0:
            self._filter_hold_ticks -= 1
            target_filter = self.dsp.filter_mode if hasattr(self.dsp, "filter_mode") else target_filter
        if hasattr(self.dsp, "filter_mode") and self.dsp.filter_mode != target_filter:
            self.dsp.filter_mode = target_filter

        # ========================================================
        # 統計情報の更新
        # ========================================================
        self.last_stats = {
            "adc_clip_pct": round(clip_pct, 3),
            "iq_std": round(iq_std, 1),
            "gain_db": current_gain,
            "estimated_snr": round(self.snr_smooth, 1),
            "filter_mode": target_filter,
            "converged": converged,
            "hard_lock": self.hard_lock,
            "dx_mode": self.dx_mode,
        }
        return self.last_stats
