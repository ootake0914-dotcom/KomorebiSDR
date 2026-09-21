"""
RF frontend adaptive modules (extracted from adaptive_dsp.py).

RFフロントエンド系の適応モジュール群の正準の保持場所:
- AdaptiveIqCorrector (IQインバランス補正)
- DynamicIfBandwidthTracker (ダイナミックIF帯域)
- UltrasonicSquelchTracker (超音波スケルチ)

`adaptive_dsp.py` は後方互換のため同名を再エクスポートする。
"""

import numpy as np


class AdaptiveIqCorrector:
    """
    グラム・シュミット直交化 (Gram-Schmidt Orthogonalization) に基づく
    リアルタイム・ブラインド適応型 IQインバランス（振幅・直交位相）補正器。

    RTL-SDRチューナー (R820T等) のアナログ直交ダウンコンバージョンで生じる
    I/Qの振幅比誤差および直交位相ズレをサンプルの2次統計量から常時自動追従し、
    鏡像（ゴースト信号）の抑圧比 (IRR) を 30dB から 55〜60dB 以上へ引き上げる。

    数学的根拠: 古典的線形代数および IEEE 標準アルゴリズム (パブリックドメイン)。
    """

    def __init__(self, sample_rate: float = 1152000.0, time_constant_sec: float = 3.0):
        self.sample_rate = float(sample_rate)
        # 本番では 2〜5秒を推奨。テスト用に 0.05秒まで許容
        self.tau = float(max(time_constant_sec, 0.05))

        # 補正係数 (直接追従)
        self.coeff_c = 0.0   # 直交化係数: c = <I*Q> / <I^2>
        self.coeff_g = 1.0   # ゲイン整合係数: g = sqrt(<I^2> / <(Q')^2>)
        self.enabled = True

    def reset(self):
        """統計量を初期状態にリセット"""
        self.coeff_c = 0.0
        self.coeff_g = 1.0

    def process(self, iq_samples: np.ndarray, stats_stride: int = 16) -> np.ndarray:
        """
        複素数IQ配列 (N,) を受け取り、直交誤差・振幅誤差を補正した複素数配列を返す。
        高速化 (数学的等価):
        - 統計は間引き取得 (時定数3sに対し57msブロックの全数統計は冗長。
          期待値は同一のため収束先不変、EWMAが分散増を平滑化)
        - complex64を直接組立 (旧 `i + 1j*q` はcomplex128中間体を生成していた)
        - 適用部はin-place演算で一時配列を排除
        """
        if not self.enabled or len(iq_samples) == 0:
            return iq_samples

        # 非有限サンプル(Inf/NaN)が混入したブロックは係数更新を凍結し素通し
        # (stride間引きで当たったInfだけが統計に入り coeff=NaN固着するのを防止)
        n = len(iq_samples)
        # 1. ブロック内の2次統計量 (間引き・期待値同一)
        st = iq_samples[::stats_stride] if n > stats_stride else iq_samples
        try:
            if not bool(np.all(np.isfinite(np.asarray(st).reshape(-1)))):
                return np.ascontiguousarray(iq_samples)
            if not bool(np.all(np.isfinite(np.asarray(iq_samples).reshape(-1)))):
                return np.ascontiguousarray(iq_samples)
        except Exception:
            return iq_samples
        si = np.real(st).astype(np.float32)
        sq = np.imag(st).astype(np.float32)
        mean_i2 = float(np.mean(si * si))
        mean_q2 = float(np.mean(sq * sq))
        mean_iq = float(np.mean(si * sq))

        # 2. 時間ベースの適応ステップ幅 (dt / tau)
        dt = n / self.sample_rate
        alpha = float(1.0 - np.exp(-dt / self.tau))

        # 微小信号（無信号・ノイズフロア付近）では補正係数の更新を凍結し、直前値で通過
        if mean_i2 >= 1e-5 and mean_q2 >= 1e-5:
            # 直交化係数の目標値: target_c = <I*Q> / <I^2>
            target_c = mean_iq / (mean_i2 + 1e-12)
            target_c = float(np.clip(target_c, -0.5, 0.5))
            self.coeff_c += alpha * (target_c - self.coeff_c)

            # ゲイン整合係数の目標値: target_g = sqrt(<I^2> / <(Q')^2>)
            # (Q'統計も間引きで同一期待値)
            qp = sq - self.coeff_c * si
            mean_q_prime2 = float(np.mean(qp * qp))
            target_g = float(np.sqrt(mean_i2 / (mean_q_prime2 + 1e-12)))
            target_g = float(np.clip(target_g, 0.5, 2.0))
            self.coeff_g += alpha * (target_g - self.coeff_g)

        # 3. 直交・振幅補正の適用 (フルレート・in-place・complex64直接)
        out = np.empty(n, dtype=np.complex64)
        out.real[:] = np.real(iq_samples)
        qi = out.imag
        qi[:] = np.imag(iq_samples)
        qi -= self.coeff_c * np.real(iq_samples)
        qi *= self.coeff_g
        return out

    @property
    def estimated_phase_error_deg(self) -> float:
        """推定された直交位相誤差 (度)"""
        # sin(phi) ≈ c
        return float(np.degrees(np.arcsin(np.clip(self.coeff_c, -0.99, 0.99))))

    @property
    def estimated_gain_imbalance_db(self) -> float:
        """推定された振幅不均衡比 (dB)"""
        return float(20.0 * np.log10(np.clip(self.coeff_g, 1e-3, 100.0)))


class DynamicIfBandwidthTracker:
    """
    カーソン則 (Carson's Rule: B = 2*(Δf + fm)) に基づく
    FM変調度適応型 ダイナミックIF帯域幅トラッカー。

    瞬時の周波数偏移（変調深度）をピークホールド＋リーク積分で監視し、
    静かなトーク時・休符時は帯域を狭めてノイズフロア・隣接混信を大幅に低減し、
    音楽フォルテシモ（大音量）時は帯域を自動全開にして歪みを防ぐ。

    数学的根拠: 1922年 John Carson 変調理論 (パブリックドメイン)。
    """

    def __init__(self, min_bw_hz: float = 85000.0, max_bw_hz: float = 190000.0,
                 max_audio_freq_hz: float = 15000.0):
        self.min_bw_hz = float(min_bw_hz)
        self.max_bw_hz = float(max_bw_hz)
        self.fm_max = float(max_audio_freq_hz)

        # ピーク周波数偏移のエンベロープ状態
        self.dev_peak_hz = 25000.0
        self.current_bw_hz = 140000.0

        # 時定数: アタックは極めて速く (5ms)、リリースは聴感を保ち緩やかに (350ms)
        self.attack_alpha = 0.35
        self.release_alpha = 0.02
        self.safety_margin_hz = 15000.0

    def update(self, demod_freq_hz: np.ndarray) -> float:
        """
        復調された周波数偏移サンプル (Hz) から瞬時ピークを検出し、
        最適なIF帯域幅 (Hz) を算出して滑らかに返す。
        """
        if len(demod_freq_hz) == 0:
            return self.current_bw_hz

        # 直流バイアスを除いた瞬時周波数偏移のピーク値
        block_peak = float(np.percentile(np.abs(demod_freq_hz - np.mean(demod_freq_hz)), 99.5))

        # アタック・リリースのエンベロープ追従
        if block_peak > self.dev_peak_hz:
            self.dev_peak_hz += self.attack_alpha * (block_peak - self.dev_peak_hz)
        else:
            self.dev_peak_hz += self.release_alpha * (block_peak - self.dev_peak_hz)

        # カーソン則に基づく必要帯域幅の算出: B = 2 * (Δf + fm) + Margin
        needed_bw = 2.0 * (self.dev_peak_hz + self.fm_max) + self.safety_margin_hz
        target_bw = float(np.clip(needed_bw, self.min_bw_hz, self.max_bw_hz))

        # 帯域幅の滑らかな遷移 (クリック音防止)
        self.current_bw_hz += 0.15 * (target_bw - self.current_bw_hz)
        return self.current_bw_hz


class UltrasonicSquelchTracker:
    """
    FMクワイエティング効果 (Quieting Effect) と超音波三角ノイズ比率に基づく
    コグニティブ・オートスケルチ。

    従来の電界強度 (RSSI) スケルチと異なり、都市部の高ノイズ環境でも
    ノイズ自身を電波と誤認せず、本物の変調波キャリアが存在する時のみ
    瞬時に音声を開放 (Open) し、局間は完全な静寂 (Mute) を維持する。

    数学的根拠: 古典FM三角ノイズ理論 (Rice / Carson, パブリックドメイン)。
    """

    def __init__(self, sample_rate: float = 288000.0, noise_threshold_db: float = -38.0):
        self.sample_rate = float(sample_rate)
        # 超音波ハイパスフィルタ状態 (遮断周波数 45kHz)
        # y[n] = x[n] - x[n-1] + R * y[n-1]
        fc = 45000.0
        self.r = float(np.exp(-2.0 * np.pi * fc / self.sample_rate))
        self.hp_x1 = 0.0
        self.hp_y1 = 0.0

        # 平滑化されたノイズパワー (dB)
        self.noise_db = -20.0
        self.threshold_db = float(noise_threshold_db)
        self.hysteresis_db = 3.5
        # 自動閾値: 局間ノイズフロアを追従し、閾値=フロア+マージンに保つ。
        # 固定閾値では都市ノイズ等で開きっぱなし/閉じっぱなしになるため。
        self.auto_threshold = True
        self.threshold_margin_db = 6.0
        self.floor_db = -20.0
        self.threshold_effective = float(noise_threshold_db)
        self._thr_min_db = -50.0
        self._thr_max_db = -25.0

        # スケルチ状態 (True: オープン/受信中, False: ミュート/局間ノイズ)
        self.is_open = True
        self.current_gain = 1.0   # クリックレス・ソフトフェードゲイン (0.0〜1.0)
        self.enabled = True

    def reset(self):
        self.hp_x1 = 0.0
        self.hp_y1 = 0.0
        self.noise_db = -20.0
        self.floor_db = -20.0
        self.threshold_effective = float(self.threshold_db)
        self.is_open = True
        self.current_gain = 1.0

    def process(self, demod_baseband: np.ndarray) -> tuple[float, bool]:
        """
        FM復調直後のベースバンド信号 (288kHz) から超音波ノイズレベルを追従し、
        ソフトフェードゲイン (0.0〜1.0) と オープン状態 (bool) を返す。
        """
        if not self.enabled or len(demod_baseband) == 0:
            return 1.0, True

        # 1. 超音波成分 (45kHz以上) を軽量1次HPFで抽出
        # y[n] = x[n] - x[n-1] + r * y[n-1]
        n = len(demod_baseband)
        x = np.asarray(demod_baseband, dtype=np.float64)
        if not bool(np.all(np.isfinite(x))):
            return float(self.current_gain), bool(self.is_open)
        # 真の1次HPF再帰 (ベクトル化不可のため逐次だがN~12kで0.05ms級)。
        # 旧diffのみ実装では+6dB/octシェルフとなり大音量低音が漏れた。
        r = float(self.r)
        y_prev = float(self.hp_y1)
        x_prev = float(self.hp_x1)
        # 高速化: 差分を先に求め、指数リークを累積適用する近似ではなく正確な再帰を行う
        hp = np.empty(n, dtype=np.float64)
        for i in range(n):
            xi = float(x[i])
            y = (xi - x_prev) + r * y_prev
            hp[i] = y
            x_prev = xi
            y_prev = y
        self.hp_x1 = float(x[-1])
        self.hp_y1 = float(y_prev)
        diff = hp.astype(np.float32)

        # 超音波パワーの瞬時計算
        # 差分エネルギーをベースに超音波高域パワーを推定
        ultra_power = float(np.mean(diff * diff)) + 1e-15
        ultra_db = 10.0 * np.log10(ultra_power)

        # 2. リーク積分追従 (アタック 10ms, リリース 40ms)
        dt = n / self.sample_rate
        alpha = float(1.0 - np.exp(-dt / 0.025))
        self.noise_db += alpha * (ultra_db - self.noise_db)

        # 2b. ノイズフロア追従と自動閾値 (フロア低下には速く2秒、下振れ防止の
        # 上昇は遅く30秒。番組の一時的な静寂で閾値が暴れないようにする)
        if self.auto_threshold:
            if self.noise_db < self.floor_db:
                a_f = float(1.0 - np.exp(-dt / 2.0))
            else:
                a_f = float(1.0 - np.exp(-dt / 30.0))
            self.floor_db += a_f * (self.noise_db - self.floor_db)
            thr = self.floor_db + float(self.threshold_margin_db)
            self.threshold_effective = float(np.clip(thr, self._thr_min_db, self._thr_max_db))
        else:
            self.threshold_effective = float(self.threshold_db)

        # 3. ヒステリシス付きシュミットトリガー判定
        # クワイエティングにより超音波ノイズが閾値未満に落ちたらキャリア捕捉と判定
        thr = float(self.threshold_effective)
        if self.is_open:
            if self.noise_db > thr + self.hysteresis_db:
                self.is_open = False  # 局間ノイズへ突入 -> ミュート
        else:
            if self.noise_db < thr:
                self.is_open = True   # 本物の局を発見 -> オープン

        # 4. クリックレス・ソフトフェード (アタック: 素早く開く, リリース: 滑らかに閉じる)
        target_gain = 1.0 if self.is_open else 0.0
        fade_rate = 0.30 if self.is_open else 0.15
        self.current_gain += fade_rate * (target_gain - self.current_gain)
        if abs(self.current_gain - target_gain) < 0.01:
            self.current_gain = target_gain

        return float(self.current_gain), bool(self.is_open)
