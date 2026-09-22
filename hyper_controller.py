"""
HyperController - Cascade-Transcending Autonomous SDR Control Engine.

従来の CascadeController を全次元で超越する次世代統合認知制御エンジン。

【Cascadeの限界と本エンジンの突破】
  1. ゲイン制御:
     Cascade: 120ms毎に±1段のヒルクライミング（局所解・発振・収束遅延）
     Hyper  : 全ゲイン軸の粗探索→放物線近傍精密化→ロック→定期再検証という
              モデルベース・グローバル探索。クリップ履歴を記憶し数プローブで最適点を特定。
  2. SNR評価:
     Cascade: スペクトラム全帯域の最大ピーク−ノイズフロア（隣接強局に汚染される）
     Hyper  : 受信チャンネル帯域内パワー vs ガードバンド中央値による真のC/N測定
  3. フィルタ制御:
     Cascade: narrow/clean/wide の3段離散切替（境界で音質が不連続に跳躍）
     Hyper  : カットオフ・IF帯域幅を無段階モーフィング（クリック・ポップ抑制）
  4. 品質評価:
     Cascade: 電波スペクトル情報のみ（聴感と乖離）
     Hyper  : 復調後オーディオの番組帯域(300-3kHz)対ヒス帯域(5.5-11kHz)比を直接測定し
              カルマン推定器で真の聴感SNRを追跡
  5. 雑音抑制:
     Cascade: 離散エキスパンダー
     Hyper  : 線形位相クロスオーバー・ハイシェルフによる心理音響的連続ヒス抑圧
  6. 外乱追従:
     Cascade: 固定時定数
     Hyper  : イノベーション分散に応じた適応制御周期＋劣化検出時の自動再取得
"""

import time
import numpy as np


class HyperController:
    """ゲイン自動調整の統合制御エンジン"""

    SWEET_SPOT_DB = 33.8  # 実機R820T実測: 36.4dB以上はADCクリップ多発(1.3%→18%→35%)、33.8dBがクリップフリー上限
    MIN_SAFE_GAIN_DB = 19.7  # 実用最低安全ゲイン: 19.7dB未満への転落を防ぐ目安
    FLOOR_GAIN_DB = 12.5  # 過大入力時の非常用下限 (0.0dBまで沈めると音声がほぼ無音化するため)
    LOCK_MARGIN_DB = 1.5  # ロック時、最高品質からこの範囲内で最も高いゲインを選ぶ (測定誤差対策)
    _GAIN_FRACTIONS = (0.25, 0.55, 0.75, 0.90, 1.0)  # 実用高C/N帯(28.0〜49.6dB)に重点配置

    def _min_safe_idx(self) -> int:
        """
        無音化を防ぐ安全最低ゲインのインデックスを取得。
        通常時は19.7dB未満への転落を防止するが、強電界・過大入力環境
        (HIGH_GAIN_HEAVY判定、またはADCクリッピング多発時)では
        混変調歪みとADCサチュレーションを解消するためFLOOR_GAIN_DB(12.5dB)までの減衰を許可する。
        (0.0dBまで落とすと受信音が実用不能レベルまで沈むため、非常時でも下限を設ける)
        """
        if not self.available_gains:
            return 0
        if getattr(self, "antenna_profile", None) == "HIGH_GAIN_HEAVY" or getattr(self, "state_clip_pct", 0.0) > 0.5:
            return self._nearest_idx(self.FLOOR_GAIN_DB)
        return self._nearest_idx(self.MIN_SAFE_GAIN_DB)

    def __init__(self, driver, dsp, audio=None):
        self.driver = driver
        self.dsp = dsp
        self.audio = audio

        self.enabled = True
        # DX手動モードは廃止 (木漏れ日整理)。弱電界では下のC/N連動マップが
        # 自動でDX相当 (4300Hz/0.14/122kHz) まで絞る。cascade流の snr<9 判定と同等。
        self.available_gains = []
        self.current_gain_idx = 0

        # --- カルマン状態推定器 ---
        self._kf_p = 1.0
        self._kf_q = 0.10
        self._kf_r = 0.35
        self.state_channel_snr = 6.0
        self.state_audio_snr = 6.0
        self.state_quality = 6.0  # RF C/N + 聴感SNR融合の制御目的関数
        self.state_iq_std = 5.0
        self.state_noise_floor = -70.0
        self.state_clip_pct = 0.0
        self.audio_clip_pct = 0.0

        # --- アンテナ適応型プロファイラ (どんなアンテナにも自動最適化) ---
        self.antenna_profile = "BALANCED"  # "LOW_GAIN_MICRO" / "BALANCED" / "HIGH_GAIN_HEAVY" / "SATELLITE_NFM"
        self.antenna_type_detected = "Auto-Detecting..."
        self.noise_floor_history = []
        self.intermod_warning = False

        # --- ゲイン曲線モデル探索 ---
        self.gain_curve = {}        # gain_db -> 最高観測C/N
        self.gain_observed_at = {}  # gain_db -> 最終観測時刻
        self.best_gain_db = None
        self.best_snr = -1e9
        self.locked_gain_db = None  # ロック中の確定ゲイン (ヒステリシス基準)
        self.search_phase = "coarse"  # coarse -> fine -> done
        self._coarse_plan = []
        self.locked = False
        self.hard_lock = False        # 収束後の決め打ち固定 (フェージング・無音での誤再探索を抑える)
        self._verify_dir = 1
        self._regret_frames = 0

        # --- クリップ境界の二分探索 ---
        self.clip_upper_idx = None  # クリップが確認された最低インデックス
        self.clip_free_idx = None   # 非クリップが確認された最高インデックス

        # --- 制御周期 ---
        self.probe_interval = 0.16
        self.control_interval = 0.10
        self.verify_interval = 8.0
        self.verify_cooldown = 3.0
        self.verify_drift_db = 1.2
        self.stale_seconds = 20.0
        self.reacquire_drop_db = 3.5
        self.switch_margin_db = 0.6
        self.settle_frames = 0
        self.last_action_time = 0.0
        self.last_verify_time = 0.0

        # --- 連続DSPパラメータ ---
        self.target_cutoff_hz = 8500.0
        self.target_hf_gain = 1.0
        self.target_if_bw_hz = 190000.0
        self.filter_override = None  # None / "wide" / "clean" / "narrow"

        # --- 統計 ---
        self.frames = 0
        self.gain_changes = 0
        self.reacquires = 0
        self.last_stats = {
            "adc_clip_pct": 0.0,
            "iq_std": 0.0,
            "gain_db": 0.0,
            "estimated_snr": 0.0,
            "filter_mode": "clean",
            "converged": False,
            "dx_auto": True,
        }

        self.rf_rate = float(getattr(dsp, "rf_rate", 1152000))
        self.audio_rate = float(getattr(dsp, "audio_rate", 48000))

    # ================================================================
    # 初期化 / 再同調
    # ================================================================
    def init_gains(self):
        """利用可能なチューナーゲイン一覧を取得して探索状態を初期化"""
        gains = self.driver.get_gains()
        if not gains:
            gains = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
                     19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
                     38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
        self.available_gains = sorted(gains)
        self.reset_tracking()

    def reset_tracking(self, initial_gain: float = None):
        """選局変更時にベイズ的スウィートスポットから探索を再開"""
        self.gain_curve.clear()
        self.gain_observed_at.clear()
        self.best_gain_db = None
        self.best_snr = -1e9
        self.locked_gain_db = None
        self.clip_upper_idx = None
        self.clip_free_idx = None
        self.search_phase = "coarse"
        self._coarse_plan = []
        self.locked = False
        self.hard_lock = False  # 新局選局時は自動探索を許可
        self._regret_frames = 0
        self._kf_p = 1.0
        self.last_action_time = 0.0
        self.last_verify_time = 0.0
        self.settle_frames = 1

        if not self.available_gains:
            return

        if initial_gain is not None:
            idx = self._nearest_idx(initial_gain)
        else:
            idx = self._nearest_idx(self.SWEET_SPOT_DB)
        self.current_gain_idx = idx
        self.driver.set_gain_mode(True)
        self.driver.set_gain(self.available_gains[idx])

    def set_hard_lock(self, locked: bool):
        """ユーザーまたは収束イベントによる決め打ち固定 (True: 固定, False: 再探索開始)"""
        self.hard_lock = locked
        if locked:
            self.locked = True
            if self.available_gains:
                self.locked_gain_db = self.available_gains[self.current_gain_idx]
        else:
            # ロック解除時は再探索へ移行
            self._unlock("manual_unlock")

    def set_filter_override(self, mode: str):
        """GUIの手動フィルタ操作: None で適応制御へ復帰"""
        if mode in (None, "wide", "clean", "narrow"):
            self.filter_override = mode

    # ================================================================
    # 計測部
    # ================================================================
    def _measure_raw(self, raw_bytes: np.ndarray):
        raw_f = raw_bytes.astype(np.float32)
        clip_count = np.sum((raw_f <= 1.0) | (raw_f >= 254.0))
        clip_pct = float(clip_count / len(raw_f) * 100.0)
        iq_std = float(np.std(raw_f - 127.5))
        return clip_pct, iq_std

    def _measure_channel_snr(self, spectrum_db: np.ndarray, mode: str):
        """チャンネル内パワー vs ガードバンド雑音中央値による真のC/N測定"""
        if spectrum_db is None or len(spectrum_db) < 128:
            return None, None
        spec = np.asarray(spectrum_db, dtype=np.float64)
        # 非有限スペクトル (NaN混入等) は計測不能扱い。NaNをC/Nに混ぜると
        # カルマン状態が永久汚染され int(round(nan)) でワーカーが死ぬ。
        spec = np.nan_to_num(spec, nan=-120.0, posinf=0.0, neginf=-120.0)
        lin = np.power(10.0, spec / 10.0)
        n = len(lin)
        c = n // 2
        bin_hz = self.rf_rate / n

        if mode == "NFM":
            # ISS / アマチュア無線 NFM: ±8kHz (16kHz IF帯域)
            ch_half = max(2, int(8000.0 / bin_hz))
            g_in = max(ch_half + 1, int(14000.0 / bin_hz))
            g_out = int(40000.0 / bin_hz)
        elif mode in ("AM", "AM_NARROW"):
            ch_half = max(2, int(5000.0 / bin_hz))
            g_in = max(ch_half + 1, int(10000.0 / bin_hz))
            g_out = int(35000.0 / bin_hz)
        else:
            ch_half = max(4, int(85000.0 / bin_hz))
            g_in = max(ch_half + 2, int(115000.0 / bin_hz))
            g_out = int(250000.0 / bin_hz)

        g_out = min(g_out, c - 2)
        g_in = min(g_in, g_out - 1)
        if g_out <= g_in or c - g_out < 0:
            return None, None

        ch = lin[c - ch_half: c + ch_half + 1]
        if len(ch) == 0:
            return None, None

        sig_mean = float(np.mean(ch))
        sig_peak = float(np.max(ch))
        if mode in ("AM", "AM_NARROW", "NFM"):
            guards = np.concatenate((lin[c - g_out: c - g_in + 1], lin[c + g_in: c + g_out + 1]))
            if len(guards) < 4:
                return None, None
            noise_p = float(np.median(guards))
        else:
            # 実機FM: 強局自身の側波帯/隣接局がガード帯を汚染するため、
            # 帯域全体の下位25%タイルを雑音床に採用 (頑健推定)
            noise_p = float(np.percentile(lin, 25))
        noise_db = 10.0 * np.log10(noise_p + 1e-12)
        mean_snr = 10.0 * np.log10((sig_mean + 1e-12) / (noise_p + 1e-12))
        peak_snr = 10.0 * np.log10((sig_peak + 1e-12) / (noise_p + 1e-12))
        # 弱電界で平均法が潰れる実機特性に合わせ、搬送波尖頭優勢で融合
        snr_db = max(mean_snr, 0.35 * mean_snr + 0.65 * peak_snr)
        if not (np.isfinite(snr_db) and np.isfinite(noise_db)):
            return None, None
        return float(snr_db), float(noise_db)

    def _profile_antenna(self, clip_pct: float, iq_std: float, noise_floor: float, mode: str):
        """
        接続されたアンテナの環境を自動判別 (どんなアンテナでも最適化)
        - LOW_GAIN_MICRO: 簡易小型ロッド/室内アンテナ (高ゲイン積極探索 & ノイズ抑圧優先)
        - BALANCED      : 標準アンテナ
        - HIGH_GAIN_HEAVY: 屋外大型アンテナ / 強電界 (過大入力・相互変調歪みIMD防止)
        - SATELLITE_NFM : ISS等衛星用 (急激なフェージング高速追従)
        """
        if mode == "NFM":
            self.antenna_profile = "SATELLITE_NFM"
            self.antenna_type_detected = "Satellite / Narrowband VHF"
            return

        # ノイズフロア履歴は統計用に保持 (ゲイン上昇でノイズフロアも上昇するのは
        # 増幅として正常であり、それ自体を相互変調と誤判定しない)
        if noise_floor is not None and self.available_gains:
            cur_g = self.available_gains[self.current_gain_idx]
            self.noise_floor_history.append((cur_g, noise_floor))
            if len(self.noise_floor_history) > 6:
                self.noise_floor_history.pop(0)

        # 真の過大入力判定は「実際のADCクリッピング」のみを根拠にする。
        # (IQ Stdはアンテナ・利得に依らず大きく変動するため判定に使わない)
        if clip_pct > 0.5:
            self.antenna_profile = "HIGH_GAIN_HEAVY"
            self.antenna_type_detected = "High-Gain / Outdoor / Overload-Risk"
        elif iq_std < 6.5:
            self.antenna_profile = "LOW_GAIN_MICRO"
            self.antenna_type_detected = "Compact / SRH805S / Indoor"
        else:
            self.antenna_profile = "BALANCED"
            self.antenna_type_detected = "Standard / Tuned Antenna"

    def _measure_audio(self, audio: np.ndarray):
        """復調後オーディオの番組帯域 vs ヒス帯域パワー比（聴感品質の直接指標）"""
        # 音声が無いフレームでは直前のオーディオクリップを残さない。
        self.audio_clip_pct = 0.0
        if audio is None or len(audio) < 256:
            return None
        fft_n = 1024
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim == 2:
            # ステレオはモノラル換算して評価
            chunk = chunk.mean(axis=1)
        if len(chunk) >= fft_n:
            chunk = chunk[-fft_n:]
        else:
            pad = np.zeros(fft_n, dtype=np.float32)
            pad[-len(chunk):] = chunk
            chunk = pad

        self.audio_clip_pct = float(np.mean(np.abs(chunk) > 0.97))
        win = np.hanning(fft_n)
        power = np.abs(np.fft.rfft(chunk * win)) ** 2 + 1e-12
        bin_hz = self.audio_rate / fft_n

        def band_mean(lo, hi):
            i0 = max(1, int(lo / bin_hz))
            i1 = min(len(power) - 1, int(hi / bin_hz))
            if i1 <= i0:
                return 1e-12
            return float(np.mean(power[i0:i1 + 1]))

        prog = band_mean(300.0, 3000.0)
        hiss = band_mean(5500.0, 11000.0)
        snr_db = 10.0 * np.log10((prog + 1e-12) / (hiss + 1e-12))
        if not np.isfinite(snr_db):
            return None
        return float(np.clip(snr_db, -20.0, 60.0))

    @staticmethod
    def _inband_gain_objective(chan_q: float, audio_q: float, have_audio: bool,
                               adc_clip_pct: float, audio_clip_frac: float) -> float:
        """ゲイン探索用の帯域内SNR目的関数。

        RFチャンネル内C/Nを主軸にし、復調後オーディオの番組/ヒス比で補正する。
        ADC飽和や復調後クリップはSNR比較を無効化するため強いペナルティを与える。
        """
        chan = float(np.clip(float(chan_q), -30.0, 80.0))
        if have_audio and audio_q is not None and np.isfinite(audio_q):
            aud = float(np.clip(float(audio_q), -30.0, 60.0))
            base = 0.62 * chan + 0.38 * aud
        else:
            base = chan
        adc = float(adc_clip_pct) if np.isfinite(adc_clip_pct) else 0.0
        aud_clip = float(audio_clip_frac) if np.isfinite(audio_clip_frac) else 0.0
        if adc >= 4.0 or aud_clip >= 0.05:
            penalty = 20.0
        elif adc >= 1.2 or aud_clip >= 0.01:
            penalty = 10.0
        else:
            penalty = 0.0
        return float(base - penalty)

    # ================================================================
    # ゲイン探索 (モデルベース・グローバル探索)
    # ================================================================
    def _nearest_idx(self, gain_db: float) -> int:
        return int(min(range(len(self.available_gains)),
                       key=lambda i: abs(self.available_gains[i] - gain_db)))

    def _is_stale(self, gain_db: float) -> bool:
        t = self.gain_observed_at.get(gain_db)
        return t is None or (time.time() - t) > self.stale_seconds

    def _refresh_clip_bounds(self):
        """古くなったクリップ境界情報を破棄（フェージング・帯域変化への追従）"""
        now = time.time()
        for attr in ("clip_upper_idx", "clip_free_idx"):
            idx = getattr(self, attr)
            if idx is None:
                continue
            g = self.available_gains[idx]
            t = self.gain_observed_at.get(g)
            if t is None or (now - t) > self.stale_seconds:
                setattr(self, attr, None)

    def _build_coarse_plan(self):
        n = len(self.available_gains)
        if n == 0:
            self._coarse_plan = []
            return
        self._refresh_clip_bounds()
        sweet = self._nearest_idx(self.SWEET_SPOT_DB)
        candidates = {sweet}
        min_idx = self._min_safe_idx()
        for f in self._GAIN_FRACTIONS:
            candidates.add(int(round(min_idx + f * (n - 1 - min_idx))))
        # クリップ境界の直下は最適点候補として優先的に調べる (安全下限未満へは下げない)
        if self.clip_upper_idx is not None and self.clip_upper_idx > min_idx:
            candidates.add(max(min_idx, self.clip_upper_idx - 1))
            candidates.add(max(min_idx, self.clip_upper_idx - 2))
        if self.clip_free_idx is not None and self.clip_free_idx >= min_idx:
            candidates.add(self.clip_free_idx)

        def allowed(i):
            if i < min_idx:
                return False  # 安全下限未満 (0.0dBなど) は候補から除外
            if i == self.current_gain_idx:
                return False  # 現在地点は毎tick観測済み
            if self.clip_upper_idx is not None and i >= self.clip_upper_idx:
                return False  # 既知のクリップ領域は再探査しない
            return self._is_stale(self.available_gains[i])

        ordered = sorted(candidates, key=lambda i: abs(i - sweet))
        self._coarse_plan = [self.available_gains[i] for i in ordered if allowed(i)]

    def _next_candidate(self):
        if self.search_phase == "coarse":
            if not self._coarse_plan:
                self._build_coarse_plan()
            while self._coarse_plan:
                g = self._coarse_plan.pop(0)
                if self.clip_upper_idx is not None and self._nearest_idx(g) >= self.clip_upper_idx:
                    continue
                if self._is_stale(g):
                    return g
            self.search_phase = "fine"

        if self.search_phase == "fine":
            if self.best_gain_db is not None:
                base = self._nearest_idx(self.best_gain_db)
            else:
                base = self.current_gain_idx
            min_idx = self._min_safe_idx()
            for d in (1, -1, 2, -2):
                i = base + d
                if 0 <= i < len(self.available_gains):
                    if i < min_idx:
                        continue  # 安全下限未満は候補から除外 (coarseと同等)
                    if self.clip_upper_idx is not None and i >= self.clip_upper_idx:
                        continue  # 既知のクリップ領域はスキップ
                    g = self.available_gains[i]
                    if self._is_stale(g):
                        return g
            self.search_phase = "done"

        return None

    def _model_best(self):
        if not self.gain_curve:
            return None, None
        return max(self.gain_curve.items(), key=lambda kv: kv[1])

    def _record_observation(self, clipped: bool = False):
        if not self.available_gains:
            return
        g = self.available_gains[self.current_gain_idx]
        now = time.time()
        if clipped:
            self.gain_curve[g] = self.state_quality - 15.0
        else:
            prev = self.gain_curve.get(g)
            if prev is None or self.state_quality > prev:
                self.gain_curve[g] = self.state_quality
            idx = self.current_gain_idx
            self.clip_free_idx = idx if self.clip_free_idx is None else max(self.clip_free_idx, idx)
        self.gain_observed_at[g] = now

        if self.gain_curve:
            bg, bs = max(self.gain_curve.items(), key=lambda kv: kv[1])
            self.best_gain_db, self.best_snr = bg, bs

    def _apply_gain(self, idx: int):
        min_idx = self._min_safe_idx()
        idx = max(min_idx, min(len(self.available_gains) - 1, idx))
        if self.clip_upper_idx is not None and self.clip_upper_idx > min_idx:
            idx = min(idx, max(min_idx, self.clip_upper_idx - 1))  # 既知クリップ域へは戻らないが安全下限は維持
        if idx == self.current_gain_idx:
            self.settle_frames = max(self.settle_frames, 1)
            return
        self.current_gain_idx = idx
        self.driver.set_gain(self.available_gains[idx])
        self.gain_changes += 1
        self.settle_frames = 2  # カルマン状態が新動作点へ収束するまで2tick待機
        self._kf_p = max(self._kf_p, 1.5)  # 動作点変更で推定器を再活性化

    def _unlock(self, _reason: str):
        self.locked = False
        self.search_phase = "coarse"
        self._coarse_plan = []
        self.last_verify_time = time.time()
        self.settle_frames = max(self.settle_frames, 1)

    # ================================================================
    # 制御ループ
    # ================================================================
    def _control_tick(self, now: float, clip_pct: float):
        self._refresh_clip_bounds()
        if self.settle_frames > 0:
            self.settle_frames -= 1
            if self.settle_frames > 0:
                return
            # セトリング完了: このtickで計測反映と次操作を同時に行う

        self._record_observation()
        cur_idx = self.current_gain_idx
        n = len(self.available_gains)

        # --- ハードロック（決め打ち固定）中: ゲイン再探索・変更を止める ---
        if self.hard_lock:
            return

        # --- 極微弱電界: 探索プランを待たず即時ジャンプアップ ---
        if self.state_iq_std < 1.0 and cur_idx < n - 3:
            self._apply_gain(cur_idx + 3)
            return
        if self.state_iq_std < 1.8 and cur_idx < n - 2:
            self._apply_gain(cur_idx + 2)
            return

        # --- 探索フェーズ ---
        if not self.locked:
            candidate = self._next_candidate()
            if candidate is not None:
                self._apply_gain(self._nearest_idx(candidate))
                return
            # 全プローブ完了 -> 観測モデルの最大点にロック (ヒステリシス付き)
            self.locked = True
            target = self.best_gain_db
            # 測定誤差に強い選択: 最高品質から LOCK_MARGIN_DB 以内で最も高いゲインを採用。
            # (低ゲイン側の見かけ上のピークにロックして音量が沈むのを防ぐ)
            if self.gain_curve:
                best_q = max(self.gain_curve.values())
                within = [g for g, q in self.gain_curve.items() if q >= best_q - self.LOCK_MARGIN_DB]
                if within:
                    target = max(within)
            if target is None or target < self.available_gains[self._min_safe_idx()]:
                target = self.SWEET_SPOT_DB
            if self.locked_gain_db is not None and self.locked_gain_db >= self.available_gains[self._min_safe_idx()]:
                ref = self.gain_curve.get(self.locked_gain_db, -1e9)
                if self.best_snr <= ref + self.switch_margin_db:
                    target = self.locked_gain_db
            tidx = self._nearest_idx(target)
            # 既知のクリップ境界より上のゲインへはロックしない (過大入力の決め打ち防止)
            if self.clip_upper_idx is not None and tidx >= self.clip_upper_idx:
                tidx = max(self._min_safe_idx(), self.clip_upper_idx - 1)
                target = self.available_gains[tidx]
            elif self.clip_free_idx is not None and tidx > self.clip_free_idx:
                # 非クリップが確認できている上限を超える場合はその上限まで引き下げる
                tidx = max(self._min_safe_idx(), self.clip_free_idx)
                target = self.available_gains[tidx]
            # 2段探索の後半でクリップ多発だった局は安全域に留める
            if self.state_clip_pct > 0.5:
                tidx = max(self._min_safe_idx(), tidx - 1)
                target = self.available_gains[tidx]
            self.locked_gain_db = target
            if tidx != cur_idx:
                self._apply_gain(tidx)
            # 収束完了: 決め打ち固定 (以後、自動再探索を止める)
            self.hard_lock = True
            self.search_phase = "locked"
            return

        # --- ロック中: 別段が有意差で優位なら乗り換え ---
        locked_g = self.locked_gain_db if self.locked_gain_db is not None else self.available_gains[cur_idx]
        locked_ref = self.gain_curve.get(locked_g, self.state_quality)
        bg, bs = self._model_best()
        if bg is not None and abs(bg - locked_g) > 1e-9 and bs > locked_ref + self.switch_margin_db:
            self.locked_gain_db = bg
            bidx = self._nearest_idx(bg)
            if bidx != cur_idx:
                self._apply_gain(bidx)
            return

        # --- ロック中: 劣化検出で自動再取得 (選局変更や大フェージング時のみ) ---
        if self.state_quality < locked_ref - 7.0:  # 7dB以上の確実な電波喪失時のみ
            self._regret_frames += 1
            if self._regret_frames >= 10:  # 1秒以上持続した場合のみ
                self.locked_gain_db = self.available_gains[cur_idx]
                self.best_gain_db = self.available_gains[cur_idx]
                self.best_snr = self.state_quality
                self._regret_frames = 0
                self.reacquires += 1
                self._unlock("degraded")
        else:
            self._regret_frames = 0

        # --- ロック中: 安定聴取維持 (リスニング中の無駄なゲイン揺さぶり・クリック音の抑制) ---
        # 一度最適ゲインにロックされたら、電波が喪失しない限りゲインを動かさず静かに維持

    def _map_continuous_parameters(self, mode: str):
        if not hasattr(self.dsp, "set_cognitive_parameters"):
            return

        s = self.state_channel_snr
        # 実機実測(弱電界C/N 1〜6dB)に基づく再校正: 弱電界では狭帯域へ、十分強ければ開放。
        # 弱端は旧DX相当 (4300Hz/hf0.14/122kHz) に寄せ、手動切替なしで自動到達する。
        sig_c = 1.0 / (1.0 + np.exp(-0.28 * (s - 10.0)))
        sig_a = 1.0 / (1.0 + np.exp(-0.30 * (self.state_audio_snr - 13.0)))
        cutoff = 4300.0 + 10200.0 * (0.55 * sig_c + 0.45 * sig_a)
        # 実機A/B試聴の結果、中〜強電界ではハイシェルフを早めに全開放し
        # 5-8kHzの存在感を保持する特性を採用 (弱電界のみ抑圧)
        hf = float(np.clip((s - 2.0) / 8.0, 0.14, 1.0))
        if_bw = 122000.0 + 68000.0 / (1.0 + np.exp(-0.25 * (s - 10.0)))

        if mode == "NFM":
            # ISS / アマチュア無線 NFM: 通信音声帯域 (3000Hz) & 16kHz IF帯域
            cutoff = 3000.0
            hf = 1.0
            if_bw = 16000.0
        elif mode in ("AM", "AM_NARROW"):
            cutoff = min(cutoff, 3500.0 if mode == "AM_NARROW" else 8000.0)
            hf = 1.0  # AM経路はシェルフ非適用 (透明なまま帯域のみ適応)
        else:
            # アンテナ環境別の微小バイアス
            if self.antenna_profile == "LOW_GAIN_MICRO":
                # 簡易アンテナ: 熱雑音フロアを抑えるためIF帯域を少し狭窄
                if_bw = min(if_bw, 145000.0)

        if self.filter_override == "wide":
            cutoff = 14000.0
            hf = 1.0
            if_bw = 190000.0
        elif self.filter_override == "clean":
            cutoff = 8500.0
            hf = 1.0
            if_bw = 150000.0
        elif self.filter_override == "narrow":
            cutoff = 5500.0
            hf = 0.3
            if_bw = 130000.0
        # DXモード中に手動overrideがある場合は表示と音の乖離を避けるため
        # DX狭帯域を維持しつつcutoffのみ上書き済みである旨を統計側で扱う
        # (hf/ifはoverrideプリセットへ連動させ、表示=wideでも音がDXのままにならない)

        self.target_cutoff_hz = cutoff
        self.target_hf_gain = hf
        self.target_if_bw_hz = if_bw
        self.dsp.set_cognitive_parameters(
            cutoff_hz=cutoff,
            hf_gain=hf,
            if_bw_hz=if_bw,
        )

    # ================================================================
    # メインエントリ
    # ================================================================
    def process_frame(self, raw_bytes: np.ndarray, spectrum_db: np.ndarray = None,
                      audio: np.ndarray = None, mode: str = "WFM") -> dict:
        """毎フレームの生IQ・スペクトル・復調音声から統合認知制御を実行"""
        if not self.enabled or raw_bytes is None or len(raw_bytes) < 100:
            return self.last_stats

        now = time.time()
        self.frames += 1

        # ---- 計測 ----
        clip_pct, iq_std = self._measure_raw(raw_bytes)
        chan_snr, noise_floor = self._measure_channel_snr(spectrum_db, mode)
        audio_snr = self._measure_audio(audio)

        # アンテナ環境の自動同定 (どんなアンテナでも最適化)
        self._profile_antenna(clip_pct, iq_std, noise_floor, mode)

        # ハイシェルフ適用時はヒス抑圧分を補償し、真の聴感SNRを復元
        hf_applied = float(getattr(self.dsp, "hf_gain_applied", 1.0))
        if audio_snr is not None and 0.01 < hf_applied < 0.999:
            audio_snr += 20.0 * np.log10(hf_applied)
        # 実機アーティファクト対策: カットオフがヒス帯域(5.5k〜)より下だと
        # ヒス帯はフィルタ遮断域となり聴感SNRが飽和するため計測を無効化
        applied_cut = float(getattr(self.dsp, "applied_cutoff_hz", 99999.0))
        if applied_cut < 5500.0:
            audio_snr = None

        # ---- カルマン状態推定 ----
        # 非有限計測は捨てる (混ぜると状態が永久NaN汚染される)
        if chan_snr is not None and not np.isfinite(chan_snr):
            chan_snr = None
        if audio_snr is not None and not np.isfinite(audio_snr):
            audio_snr = None
        if noise_floor is not None and not np.isfinite(noise_floor):
            noise_floor = None
        if chan_snr is None and audio_snr is not None:
            chan_snr = audio_snr + 6.0
        if chan_snr is not None:
            # 衛星(ISS)等の高速フェージング時はカルマンゲインを一時的に高めて追従加速
            q_var = 0.25 if self.antenna_profile == "SATELLITE_NFM" else self._kf_q
            p_pred = self._kf_p + q_var
            k_gain = p_pred / (p_pred + self._kf_r)
            self.state_channel_snr += k_gain * (chan_snr - self.state_channel_snr)
            self._kf_p = (1.0 - k_gain) * p_pred
        if audio_snr is not None:
            self.state_audio_snr = 0.25 * audio_snr + 0.75 * self.state_audio_snr
        self.state_iq_std = 0.3 * iq_std + 0.7 * self.state_iq_std
        self.state_clip_pct = 0.5 * clip_pct + 0.5 * self.state_clip_pct
        if noise_floor is not None:
            self.state_noise_floor = 0.2 * noise_floor + 0.8 * self.state_noise_floor
        # 制御目的関数: 帯域内SNR（RF C/N主軸＋聴感オーディオ比）から算出し、
        # ADC/オーディオ飽和をペナルティで除外する。IF/カットオフ側の独立状態は温存する。
        self.state_quality = self._inband_gain_objective(
            self.state_channel_snr,
            self.state_audio_snr,
            audio_snr is not None,
            clip_pct,
            self.audio_clip_pct,
        )

        # ---- ゲイン制御 ----
        if self.available_gains:
            cur_idx = self.current_gain_idx
            min_idx = self._min_safe_idx()

            # ハードロック（ユーザー手動固定）中: 真の重度サチュレーション(>2.5%)時のみ安全段へスライド(ロックは維持)
            if self.hard_lock:
                if clip_pct >= 2.5:
                    step = 2 if clip_pct > 4.0 else 1
                    new_idx = max(min_idx, cur_idx - step)
                    if new_idx != cur_idx:
                        self.locked_gain_db = self.available_gains[new_idx]
                        self._apply_gain(new_idx)
            elif clip_pct > 1.2 or self.state_clip_pct > 0.8:
                # 自動制御時の緊急サチュレーション回避: 本物の過大入力時のみ安全に減衰
                self._record_observation(clipped=True)
                self.clip_upper_idx = cur_idx if self.clip_upper_idx is None else min(self.clip_upper_idx, cur_idx)
                if self.clip_free_idx is not None and self.clip_free_idx < self.clip_upper_idx:
                    new_idx = max(min_idx, self.clip_free_idx)
                else:
                    step = 3 if clip_pct > 4.0 else (2 if clip_pct > 2.0 else 1)
                    new_idx = max(min_idx, cur_idx - step)
                # 既知クリップ領域への再探索を防ぐため、安全領域で即時収束
                self.locked = False
                self.search_phase = "fine"
                self._coarse_plan = []
                self.last_verify_time = now
                self._apply_gain(new_idx)
            else:
                interval = self.control_interval if self.locked else self.probe_interval
                if now - self.last_action_time >= interval:
                    self.last_action_time = now
                    self._control_tick(now, clip_pct)

        # ---- 連続DSPパラメータ更新 ----
        self._map_continuous_parameters(mode)

        # ---- 統計 ----
        gain_db = self.available_gains[self.current_gain_idx] if self.available_gains else 0.0
        if self.target_cutoff_hz >= 12000.0:
            label = "wide"
        elif self.target_cutoff_hz >= 7000.0:
            label = "clean"
        else:
            label = "narrow"
        if self.filter_override:
            label = self.filter_override

        self.last_stats = {
            "adc_clip_pct": round(self.state_clip_pct, 3),
            "iq_std": round(self.state_iq_std, 1),
            "gain_db": gain_db,
            "estimated_snr": round(self.state_channel_snr, 1),
            "filter_mode": label,
            "converged": self.locked,
            "hard_lock": self.hard_lock,
            "dx_auto": True,
            "channel_snr_db": round(self.state_channel_snr, 1),
            "audio_snr_db": round(self.state_audio_snr, 1),
            "quality": round(self.state_quality, 1),
            "cutoff_hz": int(round(self.target_cutoff_hz)),
            "if_bw_hz": int(round(self.target_if_bw_hz)),
            "hf_gain": round(self.target_hf_gain, 2),
            "search_phase": "locked" if self.locked else self.search_phase,
            "probes": self.gain_changes,
            "reacquires": self.reacquires,
            "antenna_profile": self.antenna_profile,
            "antenna_type": self.antenna_type_detected,
        }
        return self.last_stats
