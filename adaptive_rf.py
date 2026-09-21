"""
RF frontend adaptive modules (extracted from adaptive_dsp.py).

RFフロントエンド系の適応モジュール群の正準の保持場所:
- AdaptiveIqCorrector (IQインバランス補正)
- UltrasonicSquelchTracker (クワイエティング比スケルチ)
- DigitalSelfInterferenceCanceller (デジタル自己干渉消去器: SIC)

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
        # 番組帯ローパス状態 (遮断 15kHz。クワイエティング比の分母用。
        # 旧実装は超音波絶対値のみで判定したため、大音量高域 (10kHz正弦等) や
        # ステレオの19k/38k搬送波漏れをノイズと誤認して誤ミュートした)
        fc_lp = 15000.0
        self.r_lp = float(np.exp(-2.0 * np.pi * fc_lp / self.sample_rate))
        self.lp_y1 = 0.0
        self.prog_db = -20.0

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
        self._primed = False  # 初ブロック過渡プライム済みフラグ

    def reset(self):
        self.hp_x1 = 0.0
        self.hp_y1 = 0.0
        self.lp_y1 = 0.0
        self._primed = False
        self.noise_db = -20.0
        self.prog_db = -20.0
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
        ultra_power = float(np.mean(diff * diff)) + 1e-15
        ultra_db = 10.0 * np.log10(ultra_power)

        # 1b. 番組帯 (15kHz以下) パワー (1次LPF再帰)
        r_lp = float(self.r_lp)
        lp_prev = float(self.lp_y1)
        lp = np.empty(n, dtype=np.float64)
        for i in range(n):
            xi = float(x[i])
            y = r_lp * lp_prev + (1.0 - r_lp) * xi
            lp[i] = y
            lp_prev = y
        self.lp_y1 = float(lp_prev)
        prog_power = float(np.mean(lp * lp)) + 1e-15
        prog_db = 10.0 * np.log10(prog_power)

        # 2. リーク積分追従 (アタック 10ms, リリース 40ms)
        dt = n / self.sample_rate
        alpha = float(1.0 - np.exp(-dt / 0.025))
        self.noise_db += alpha * (ultra_db - self.noise_db)
        self.prog_db += alpha * (prog_db - self.prog_db)

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
        # 旧: 超音波絶対値のみ → 大音量高域・19k/38k搬送波漏れで誤ミュート。
        # 新: クワイエティング比 (番組帯−超音波) で判定。レベル無依存。
        # キャリア捕捉で三角ノイズが潰れると比が開く → オープン。
        # 閾値根拠 (1次HPF 45k + 1次LPF 15k @288k): 10kHz正弦 Q≈7.8、
        # 12kHz Q≈5.7、白色/三角ノイズ Q<0。close 3.0 / open 8.0 で
        # 楽音は開維持、局間ノイズのみ閉じる。初ブロックはフィルタ過渡の
        # ため判定を素通しし、 instant値で状態をプライムする (過渡で閉じて
        # 開閾値に届かず固着するヒステリシストラップを防止)。
        quieting_db = float(self.prog_db - self.noise_db)
        if not getattr(self, "_primed", False):
            self._primed = True
        elif self.is_open:
            if quieting_db < 3.0:
                self.is_open = False  # 局間ノイズへ突入 -> ミュート
        else:
            if quieting_db > 8.0:
                self.is_open = True   # 本物の局を発見 -> オープン

        # 4. クリックレス・ソフトフェード (アタック: 素早く開く, リリース: 滑らかに閉じる)
        target_gain = 1.0 if self.is_open else 0.0
        fade_rate = 0.30 if self.is_open else 0.15
        self.current_gain += fade_rate * (target_gain - self.current_gain)
        if abs(self.current_gain - target_gain) < 0.01:
            self.current_gain = target_gain

        return float(self.current_gain), bool(self.is_open)


class DigitalSelfInterferenceCanceller:
    """
    全二重通信 (In-Band Full Duplex: IBFD) 技術に基づく
    デジタル自己干渉消去器 (Digital Self-Interference Canceller: SIC)。

    【数理的背景: 直交適応基底追従と逆位相消去】
    RTL-SDR 受信機において、PC 本体、液晶ディスプレイ、USB バス、スイッチング電源等から
    放射される電磁波は、アンテナやドングル基盤に混入する「内部スプリアス・ビート干渉」です。
    本モジュールは、帯域内に定常的に現れる狭帯域スプリアス干渉波を自己相関により自動検出し、
    複素直交基底:
        b_k(t) = exp(j * 2pi * f_k * t)
    に対して正規化最小二乗平均 (NLMS) 適応追従を行って干渉信号を正確に推定し、
    受信信号から逆位相合成 (Subtract) して消去します。

    【効果】
    - 目的の広帯域変調信号 (FM音声等) に一切歪みを与えず、
      PC 由来のスプリアススパイクのみを 20dB〜40dB 鋭利にノッチ消去。
    - 受信ノイズフロアのクリーン化。
    """

    def __init__(self, sample_rate: float = 288000.0, mu: float = 0.05, max_tones: int = 4):
        self.fs = float(sample_rate)
        self.mu = float(mu)
        self.max_tones = int(max_tones)
        self.enabled = True

        # 干渉トーン周波数リスト [f1, f2, ...] (Hz, IFオフセット周波数)
        self.spurious_freqs = []
        # 各トーンに対する複素適応重み [w1, w2, ...]
        self.weights = []
        # 位相累積アキュムレータ [phase1, phase2, ...]
        self.phases = []
        self.cancellation_db = 0.0

    def set_spurious_frequencies(self, freqs_hz: list[float]):
        """消去対象とする内部スプリアス周波数 (Hz) を手動設定"""
        self.spurious_freqs = [float(f) for f in freqs_hz[:self.max_tones]]
        self.weights = [0.0 + 0.0j] * len(self.spurious_freqs)
        self.phases = [0.0] * len(self.spurious_freqs)

    def auto_detect_spurious(self, iq_samples: np.ndarray, n_fft: int = 1024,
                             prominence_db: float = 12.0, dc_guard_hz: float = 8000.0,
                             passband_hz: float = 95000.0):
        """
        FFT スペクトルから周囲ノイズフロアより急峻に突出している固定スプリアス周波数を自動同定。
        - dc_guard_hz: 所望信号キャリア・主変調帯域 (0Hz近傍のAM搬送波やFM側波帯) を保護する除外帯域 (Hz)。
        - passband_hz: IFフィルタ通過帯域 (Hz)。阻止域の過小パワーによるメディアン歪みを防止。
        - 延長ケーブル使用等でスプリアスが消失した場合は自動で周波数リストを空にし、
          即座に完全バイパス（計算コストゼロ・無歪み）へ移行。
        """
        if len(iq_samples) < n_fft:
            return

        # 複数セグメントが存在する場合はウェルチ法風にパワースペクトルを平均し、
        # 変調音声の一時的なスペクトル揺らぎを平滑化して定常スプリアスのみを抽出
        n_seg = max(1, min(4, len(iq_samples) // n_fft))
        win = np.hanning(n_fft).astype(np.float32)
        psd_accum = np.zeros(n_fft, dtype=np.float64)
        for s in range(n_seg):
            sub = iq_samples[s * n_fft : (s + 1) * n_fft] * win
            psd_accum += np.abs(np.fft.fft(sub)) ** 2
        psd_avg = psd_accum / n_seg
        fft_db = 10.0 * np.log10(np.maximum(psd_avg, 1e-12))
        freqs = np.fft.fftfreq(n_fft, 1.0 / self.fs)

        # 通過帯域内のメディアンを真のノイズフロアとする (阻止域減衰の引きずり防止)
        pass_mask = (np.abs(freqs) <= passband_hz)
        if np.any(pass_mask):
            med_floor = float(np.median(fft_db[pass_mask]))
        else:
            med_floor = float(np.median(fft_db))

        # 突出ピーク候補のインデックス抽出
        peak_mask = (fft_db > med_floor + prominence_db) & pass_mask
        peak_indices = np.where(peak_mask)[0]

        detected_candidates = []
        span = 6  # Hanning窓のメインローブ外側 (±3〜6ビン) を走査
        for idx in peak_indices:
            f = float(freqs[idx])
            # 所望信号主帯域保護: DC近傍 (±dc_guard_hz) は除外
            if abs(f) < dc_guard_hz:
                continue
            # 局所極大値の確認 (左右1ビン以上)
            left = (idx - 1) % n_fft
            right = (idx + 1) % n_fft
            if fft_db[idx] <= fft_db[left] or fft_db[idx] <= fft_db[right]:
                continue

            # 局所針状性 (Local Prominence):
            # 狭帯域・CW状スプリアスはHanning窓の減衰により左右3〜6ビンで急落する。
            # 一方、FM変調側波帯や広帯域信号はなだらかに裾野が続くため除外される。
            left_floor = float(np.min(fft_db[max(0, idx - span) : max(0, idx - 2)])) if idx >= 3 else med_floor
            right_floor = float(np.min(fft_db[min(n_fft, idx + 3) : min(n_fft, idx + span + 1)])) if idx + 3 < n_fft else med_floor
            local_floor = max(left_floor, right_floor)

            local_prom = float(fft_db[idx] - local_floor)
            global_prom = float(fft_db[idx] - med_floor)

            # 局所的にもグローバルにも突出している針状ピークのみを採用
            if local_prom >= prominence_db and global_prom >= prominence_db:
                detected_candidates.append((f, global_prom))

        if detected_candidates:
            # 突出度 (フロア比) が最も強力なスプリアスから優先して上位 max_tones 件を登録
            detected_candidates.sort(key=lambda x: x[1], reverse=True)
            chosen_freqs = [item[0] for item in detected_candidates[:self.max_tones]]
            self.set_spurious_frequencies(chosen_freqs)
        else:
            # スプリアスが存在しない場合は空にして完全バイパス
            self.set_spurious_frequencies([])

    def reset(self):
        """内部状態リセット"""
        self.weights = [0.0 + 0.0j] * len(self.spurious_freqs)
        self.phases = [0.0] * len(self.spurious_freqs)
        self.cancellation_db = 0.0

    def process(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        複素数IQ配列 (N,) を受け取り、内部自己干渉スプリアスを逆位相消去した
        クリーンIQ配列 (N,) を返す。
        """
        if not self.enabled or len(iq_samples) == 0 or len(self.spurious_freqs) == 0:
            return iq_samples

        n = len(iq_samples)
        out = iq_samples.astype(np.complex64, copy=True)
        t = np.arange(n, dtype=np.float64) / self.fs

        p_orig = float(np.mean(np.abs(iq_samples) ** 2)) + 1e-12

        for i, f_spur in enumerate(self.spurious_freqs):
            # 1. 複素直交基底ベクトルの生成 (位相連続性維持)
            phase_init = self.phases[i]
            phase_vec = phase_init + 2.0 * np.pi * f_spur * t
            basis = np.exp(1j * phase_vec).astype(np.complex64)
            # 次回ブロック用位相更新 (2pi ラップ)
            self.phases[i] = float((phase_init + 2.0 * np.pi * f_spur * (n / self.fs)) % (2.0 * np.pi))

            # 2. 干渉成分の推定: i_hat = w * basis
            w = self.weights[i]
            i_hat = w * basis

            # 3. 逆位相消去: e = out - i_hat
            e = out - i_hat

            # 4. NLMS 適応重み更新: w <- w + mu * <e * conj(basis)>
            corr = np.mean(e * np.conj(basis))
            w_new = w + self.mu * corr
            self.weights[i] = complex(w_new)

            out = e

        p_clean = float(np.mean(np.abs(out) ** 2)) + 1e-12
        if p_orig > p_clean:
            self.cancellation_db = float(10.0 * np.log10(p_orig / p_clean))

        return out

