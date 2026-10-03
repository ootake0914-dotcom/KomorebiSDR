"""AM demodulation for SdrDspPipeline (extracted from dsp.py).

包絡線＋同期検波・AGC・適応帯域・ブランカ。SdrDspPipeline の mixin として
動作する (純粋移動・動作同一)。
"""

import ctypes

import numpy as np

from dsp_filters import design_fir_kaiser
from dsp_native import (
    _NATIVE,
    NATIVE_AM_SYNC,
    _fptr,
)

try:
    from adaptive_notch import AdaptiveNotchCanceller
except ImportError:
    AdaptiveNotchCanceller = None


class DspAmMixin:
    """AM復調メソッド群 (SdrDspPipeline に mixin される)"""

    def _blank_impulses_iq(self, iq_if: np.ndarray) -> np.ndarray:
        """AM/短波用インパルスノイズブランカ (電源・イグニッション雑音対策)。
        正準実装は dsp_filters.blank_impulses_iq (モード共通)。
        ここはAM既定パラメータの薄いラッパー (動作同一・ビット等価)。"""
        if not self.impulse_blanker_enabled:
            return iq_if
        try:
            from dsp_filters import blank_impulses_iq as _blank
        except ImportError:
            return iq_if
        return _blank(iq_if)

    def _am_sideband_combine(self, iq_if: np.ndarray, sig: np.ndarray) -> np.ndarray:
        """AM側波帯ダイバーシティ合成 (DSBのUSB/LSBを分離・評価・重み付け合成)。

        手順:
          1. iq_if (搬送波≈DC) をFFT brickwallでU/L枝に相補分離
             (DCビンは両枝へ半分ずつ。U+L==iqが厳密に成り立つため、
             クリーン時の加算パスは全帯域検波と一致する)。
          2. ブロック平均位相を搬送波基準に各枝を実オーディオへ落とす
             (a_U, a_L)。DSBでは両枝が同一番組＋独立雑音になる。
          3. 汚染判定は側波帯ACエネルギー不平衡 (dB) を見る。DSBの両枝は
             鏡像のためクリーンなら不平衡≈0dB。片側汚染で膨らむ。
             ヒステリシス (入2dB/出1dB)＋保持で呼吸を防ぐ。
          4. クリーン (w→0) では既存sigをそのまま返す (透過)。
             汚染 (w→1) では各枝と中央値の相関で重み付け合成する。
        重みはヒステリシス＋保持カウンタ＋20msスルーレートで滑らかに
        (切替クリック防止)。非有限・無搬送波時は既存パスへ退避する。
        """
        n = len(iq_if)
        try:
            w_cur = float(getattr(self, "_am_sb_w", 0.0))
        except Exception:
            w_cur = 0.0
        if n < 256 or len(sig) != n:
            return sig
        # 搬送波の権威はPLL (_am_sync_detect済みの_mix)。ブロック平均位相は
        # 微小オフセット回転で減衰するためゲートには使わない (15Hz×57msで
        # 約0.86回転し平均が潰れる)。mix<0.35では判定不能として透過する。
        try:
            _mix = float(getattr(self, "_am_sync_mix", 0.0))
        except Exception:
            _mix = 0.0
        if _mix < 0.35:
            self._am_sb_w = float(w_cur * 0.5)
            self._am_sb_hold = 0
            return sig
        try:
            iq = np.asarray(iq_if, dtype=np.complex128).reshape(-1)
            if not bool(np.all(np.isfinite(iq.real))) or not bool(np.all(np.isfinite(iq.imag))):
                raise ValueError("non-finite iq")
            # 0. 微小オフセット回転の除去 (ブロック平均は15Hz×57ms≈0.86回転で
            #    潰れるため、そのままでは搬送波基準が取れない)。
            #    搬送波ピークピック: DC±3kHzで最大スペクトルを探す。
            #    Kay推定子 (電力重み平均) は片側強妨害に引っ張られて搬送波を
            #    見失うため不採用。PLLロック中 (mix gate通過) は搬送波が最強
            #    ピークであることが保証される。放物線補間で細分化し、
            #    位相はブロック跨ぎで連続させる (SSBの_ssb_bp_phaseと同型)。
            fs = float(self.if_rate)
            Xs = np.fft.fft(iq)
            fr = np.fft.fftfreq(n, 1.0 / fs)
            win = np.flatnonzero(np.abs(fr) <= 3000.0)
            mags = np.abs(Xs[win])
            k = int(np.argmax(mags))
            kk = int(win[k])
            if 0 < k < len(mags) - 1:
                a, b, c = float(mags[k - 1]), float(mags[k]), float(mags[k + 1])
                dd = a - 2.0 * b + c
                sh = 0.5 * (a - c) / dd if abs(dd) > 1e-18 else 0.0
                sh = min(max(sh, -1.0), 1.0)
            else:
                sh = 0.0
            bin_hz = fs / n
            fest_raw = float(fr[kk] + sh * bin_hz)
            if not np.isfinite(fest_raw):
                raise ValueError("non-finite fest")
            fest = float(getattr(self, "_am_sb_fest", 0.0))
            if not np.isfinite(fest):
                fest = 0.0
            # 初回はスナップ (0初期値からの指数収束遅れを避ける)、以降EMA
            if not bool(getattr(self, "_am_sb_fest_ok", False)):
                fest = fest_raw
                self._am_sb_fest_ok = True
            else:
                fest += 0.2 * (fest_raw - fest)
            self._am_sb_fest = fest
            ph0 = float(getattr(self, "_am_sb_phase", 0.0))
            if not np.isfinite(ph0):
                ph0 = 0.0
            tt = np.arange(n, dtype=np.float64) / fs
            rot = np.exp(-1j * (ph0 + 2.0 * np.pi * fest * tt))
            self._am_sb_phase = float((ph0 + 2.0 * np.pi * fest * n / fs) % (2.0 * np.pi))
            derot = iq * rot
            m = complex(np.mean(derot))
            mag = abs(m)
            env_mean = float(np.mean(np.abs(derot))) + 1e-18
            # 搬送波が立っていない (選択性フェードの谷・無信号) 側波帯判定は無意味
            if mag < 0.35 * env_mean:
                self._am_sb_w = float(w_cur * 0.5)
                self._am_sb_hold = 0
                return sig
            # 相補分離: U+L==derot (DC共有)。1FFT+1IFFTのみ。
            X = np.fft.fft(derot)
            freqs = np.fft.fftfreq(n, 1.0 / fs)
            wpos = (freqs > 0.0).astype(np.float64)
            wpos[freqs == 0.0] = 0.5
            U = np.fft.ifft(X * wpos)
            L = derot - U
            # 搬送波基準は枝ごとに取る (両枝とも搬送波半分を含む。
            # 番組・妨害はブロック内でほぼ整数回転し平均に残らない)。
            mU, mL = complex(np.mean(U)), complex(np.mean(L))
            magU, magL = abs(mU), abs(mL)
            eU = float(np.mean(np.abs(U))) + 1e-18
            eL = float(np.mean(np.abs(L))) + 1e-18
            if magU < 0.2 * eU or magL < 0.2 * eL:
                raise ValueError("no carrier in branch")
            aU = 2.0 * np.real(U * np.conj(mU / magU))
            aL = 2.0 * np.real(L * np.conj(mL / magL))
            # 汚染判定は側波帯エネルギー不平衡 (DSBではUSB/LSBが鏡像のため
            # ACエネルギーは等しいはず。不平衡＝片側妨害・選択性フェージング)。
            # d=a_U-a_L電力比も見たが、枝雑音が独立のためcleanでも0.27止まりで
            # 分離が悪い (dirty 0.43)。不平衡はclean -0.4dB/dirty +2.5dBと
            # 分離良好 (合成条件での実測)。符号が汚染側を示す。
            # プログラム自体は鏡像対称のため、音楽・音声でも不平衡は出ない。
            eUac = float(np.mean(np.abs(U - mU) ** 2)) + 1e-24
            eLac = float(np.mean(np.abs(L - mL) ** 2)) + 1e-24
            imb_db = 10.0 * float(np.log10(eUac / eLac))
            if not np.isfinite(imb_db):
                raise ValueError("non-finite imb")
            gap = abs(imb_db)
            # ヒステリシス＋保持 (呼吸防止): 汚染確定は|不平衡|>2dB、復帰は
            # <1dBが8ブロック連続してから。しきい値の間は現状態を維持する。
            dirty = bool(getattr(self, "_am_sb_dirty", False))
            hold = int(getattr(self, "_am_sb_hold", 0))
            if gap > 2.0:
                dirty, hold = True, 8
            elif gap < 1.0:
                if hold > 0:
                    hold -= 1
                else:
                    dirty = False
            else:
                hold = max(hold, 2)
            self._am_sb_dirty = dirty
            self._am_sb_hold = hold
            # 枝エネルギー slow baseline: 非汚染ロック時のみ5秒時定数で更新
            # (妨害への適応を防ぐ)。クリーン時に学んだ正常値を、汚染時の
            # 乖離判定の物差しにする。選局直後の初回は現在値で初期化する。
            # w==0の透過時も更新する (汚染開始前に正常値を学ぶ必要があるため、
            # 重み計算より前・早期リターンより前で行う)。
            dt = n / fs
            try:
                baseU = getattr(self, "_am_sb_baseU", None)
                baseL = getattr(self, "_am_sb_baseL", None)
                if baseU is None or baseL is None or not (
                        np.isfinite(float(baseU)) and np.isfinite(float(baseL))):
                    baseU, baseL = eUac, eLac
                elif not dirty:
                    k = 1.0 - np.exp(-dt / 5.0)
                    baseU = float(baseU) + k * (eUac - float(baseU))
                    baseL = float(baseL) + k * (eLac - float(baseL))
                self._am_sb_baseU, self._am_sb_baseL = float(baseU), float(baseL)
            except Exception:
                baseU, baseL = eUac, eLac
            target = 1.0 if dirty else 0.0
            # 20msスルーレート (モード切替クロスフェードと同尺。rf_healthの20ms窓に準拠)
            step = dt / 0.02
            if target > w_cur:
                w_new = min(target, w_cur + step)
            else:
                w_new = max(target, w_cur - step)
            self._am_sb_w = float(w_new)
            if w_new <= 0.0:
                return sig
            # 枝重み: baselineからの乖離で絞る (相関重みは逆に働くため不採用:
            # 汚染枝ほどmidと妨害を共有し相関が上がる)。干渉 (上昇) も
            # フェージング (低下) も「逸脱した枝が使えない枝」で向きは一致。
            # 下限0.2 (最大5:1) で誤判定時の被害を有界化する。
            try:
                wU = min(eUac, float(baseU)) / max(eUac, float(baseU))
                wL = min(eLac, float(baseL)) / max(eLac, float(baseL))
            except Exception:
                wU, wL = 0.5, 0.5
            wU = float(min(max(wU, 0.2), 1.0))
            wL = float(min(max(wL, 0.2), 1.0))
            out_sb = (wU * aU + wL * aL) / (wU + wL)
            if not bool(np.all(np.isfinite(out_sb))):
                raise ValueError("non-finite out_sb")
            out = ((1.0 - w_new) * np.asarray(sig, dtype=np.float64)
                   + w_new * out_sb)
            return out.astype(np.float32)
        except Exception:
            try:
                self._am_sb_w = float(w_cur * 0.5)
            except Exception:
                pass
            return sig

    def _am_agc_normalize(self, sig: np.ndarray) -> np.ndarray:
        """AM搬送波レベルAGC (demodulate_am から純粋移動・動作同一)。

        信号レベルやダイレクトサンプリングの低入力でも一定音量にする
        (時定数 ~0.2sアタック / ~1sリリース。無信号時の過剰増幅は3000倍で制限)。
        無信号フロア: レベル極小でAGC=0張り付き→gain3000→AMは-0.6のDC定数出力や
        ノイズ爆音になるため、フロア以下では出力を滑らかにミュートする。
        ハングタイマ: 単語間の息継ぎでゲインが跳ね上がる呼吸を防ぐため、
        レベル低下後は約400ms(7ブロック)だけ減衰を保持してからリリースする。
        """
        level = float(np.mean(np.abs(sig)))
        if not np.isfinite(level) or not bool(np.all(np.isfinite(sig))):
            # 非有限混入時は状態を汚さず無音で通過 (NaN固着の防止)。
            # 比較系が全てFalseになりgain/fadeへNaNが拡散するのを断つ。
            return np.zeros(len(sig), dtype=np.float32)
        if self.am_agc_level <= 0.0:
            self.am_agc_level = max(level, 2e-4)
            self._am_agc_hang = 0
        elif level > self.am_agc_level:
            self.am_agc_level += 0.1 * (level - self.am_agc_level)
            # 20→28blkへ延長 (語間の息継ぎでのゲイン跳ね上がり呼吸を抑制)
            self._am_agc_hang = 28
        elif self._am_agc_hang > 0:
            self._am_agc_hang -= 1
        else:
            self.am_agc_level += 0.005 * (level - self.am_agc_level)
        gain = min(1.0 / (self.am_agc_level + 1e-9), 3000.0)
        fade = min(1.0, level / 2e-4)
        return np.clip((sig * gain - 1.0) * 0.6, -1.0, 1.0) * fade

    def demodulate_am(self, iq_if: np.ndarray, mode: str = "AM") -> np.ndarray:
        if len(iq_if) == 0:
            return np.zeros(0, dtype=np.float32)

        iq_if = self._blank_impulses_iq(iq_if)
        env = np.abs(iq_if)
        power_db = 10.0 * np.log10(np.mean(env**2) + 1e-12)
        if self.squelch_enabled and power_db < self.squelch_threshold:
            return np.zeros(len(iq_if) // self.audio_decim, dtype=np.float32)

        # 同期検波と包絡線検波をロック状態に応じて混合 (未ロック時は包絡線へ自動復帰)
        sig = env
        if self.am_sync_enabled:
            coherent = self._am_sync_detect(iq_if)
            w = self._am_sync_mix
            if w > 0.01:
                sig = w * coherent + (1.0 - w) * env

        # 側波帯ダイバーシティ合成 (既定ON。非対称2dB超で自動発動)。
        # 方針は「選択」でなく「合成」: クリーン時は加算≒全帯域検波と一致するため
        # 既存パスへ戻し (ビット等価級の透過)、片側汚染時だけ汚れた枝を絞る。
        # DSBでは両側波帯が同一番組を運ぶため、加算で片側単独より約+3dB得する。
        if bool(getattr(self, "am_sideband_enabled", False)):
            try:
                sig = self._am_sideband_combine(iq_if, sig)
            except Exception:
                pass

        # 搬送波レベルAGC: 信号強度やダイレクトサンプリングの低入力でも一定音量にする
        audio_raw = self._am_agc_normalize(sig)
        if self.cognitive_enabled:
            fir_final = self._get_dynamic_filter("audio", min(self.applied_cutoff_hz, 8000.0))
        elif self.filter_mode == "wide":
            fir_final = self.fir_audio_wide
        else:
            fir_final = self.fir_am_audio  # AMは4kHz (8.5kHzでは短波のヒスが酷い)
        audio = self.decimate_with_history(audio_raw, self.fir_if_audio, self.audio_decim, "history_if_audio")

        audio = self.decimate_with_history(audio, fir_final, 1, "history_final")
        # ActiveDcServo有効時は30Hz HPFをバイパス (低域位相の一本化。WFM側と同一理由)
        _servo_am = getattr(self, "dc_servo", None)
        if _servo_am is None or not bool(getattr(_servo_am, "enabled", False)):
            audio = self._apply_dc_highpass(audio)
        # 下限2.5k→3.2kHzへ (強〜中電界のこもりを軽減。ヒス時は依然狭窄する)
        # AM_NARROW時は上限も3500Hzに絞り、クリーン時の4000Hzへの抜けを防ぐ
        if mode == "AM_NARROW":
            audio = self._voice_bandwidth(audio, 3500.0, 3200.0)
        else:
            audio = self._voice_bandwidth(audio, 4000.0, 3200.0)
        # AM経路の適応ハムノッチ (既定OFF。短波の電源ハム・ヘテロダイン対策)。
        # WFM側とは履歴を共有しない (chキー分離。帯域・レベルが異なるため)。
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_notch_enabled", False)):
            try:
                if self.bm_notch is None and AdaptiveNotchCanceller is not None:
                    self.bm_notch = self._bm_make_notch()
                if self.bm_notch is not None:
                    audio, _ = self.bm_notch.process_mono(
                        audio, ch="bm_am",
                        clip=bool(getattr(self, "adc_clipped", False)))
                    audio = np.asarray(audio, dtype=np.float32)
            except Exception:
                pass
        # AM経路のRMTは不採用 (狭帯域誤作動。verdict参照)。
        # SR検出プローブのみ残す (既定OFF。confidence公開のみ)
        try:
            _lock = float(getattr(self, "am_sync_lock", 0.0))

            def _am_base(v, _lk=_lock):
                vv = np.asarray(v, dtype=np.float64).reshape(-1)
                return bool(_lk > 0.35) and bool(np.mean(vv ** 2) > 1e-8)

            self._bm_sr_probe(audio, _am_base, _lock)
        except Exception:
            pass
        return audio.astype(np.float32)

    def _voice_bandwidth(self, audio: np.ndarray, f_max: float = 4000.0,
                         f_min: float = 2500.0) -> np.ndarray:
        """AM/SSB適応帯域: 3.2-4.5kHzのヒス量に応じて音声帯域を連続的に狭める"""
        if not self.voice_auto_bw or len(audio) < 256:
            return audio
        n = 1024
        if len(audio) >= n:
            seg = audio[-n:].astype(np.float32)
        else:
            seg = np.zeros(n, dtype=np.float32)
            seg[-len(audio):] = audio
        seg = seg - float(np.mean(seg))
        power = np.abs(np.fft.rfft(seg * self._nr_window)) ** 2 + 1e-12
        bin_hz = self.audio_rate / n

        def band(lo: float, hi: float) -> float:
            i0 = max(1, int(lo / bin_hz))
            i1 = min(len(power) - 1, int(hi / bin_hz))
            return float(np.mean(power[i0:i1 + 1])) if i1 > i0 else 1e-12

        ratio_db = 10.0 * np.log10(band(3200.0, 4500.0) / band(300.0, 3000.0))
        dt = len(audio) / self.audio_rate
        # 歯擦音 (100〜200ms) での帯域チラつき防止に0.5→0.8sへ鈍化
        self._vc_ratio_db += (1.0 - np.exp(-dt / 0.8)) * (ratio_db - self._vc_ratio_db)
        x = float(np.clip((self._vc_ratio_db + 32.0) / 20.0, 0.0, 1.0))
        s = x * x * (3.0 - 2.0 * x)
        self.voice_cut_hz = f_max * (f_min / f_max) ** s
        fir = self._get_dynamic_filter("audio", self.voice_cut_hz)
        return self.decimate_with_history(audio, fir, 1, "history_am_audio")

    def _am_sync_detect(self, iq_if: np.ndarray) -> np.ndarray:
        """キャリア再生PLLによるAM同期検波 (Cコア) とロック度の平滑化"""
        if len(iq_if) == 0:
            return np.zeros(0, dtype=np.float32)
        if _NATIVE is not None and NATIVE_AM_SYNC:
            # PLLゲインは振幅に比例するため、先に正規化 (AGC) して感度を一定化。
            # これが無いとダイレクトサンプリングの微小入力 (1e-4) でPLLが実質停止する。
            scale = float(np.mean(np.abs(iq_if))) + 1e-12
            work = np.ascontiguousarray(iq_if / scale, dtype=np.complex64)
            out = np.empty(len(work), dtype=np.float32)
            th = ctypes.c_double(self._am_th)
            ig = ctypes.c_double(self._am_ig)
            ef = ctypes.c_double(self._am_ef)
            lock = ctypes.c_float(0.0)
            _NATIVE.sdr_am_sync(_fptr(work), _fptr(out), len(work),
                                ctypes.byref(th), ctypes.byref(ig), ctypes.byref(ef),
                                self.am_kp, self.am_ki, self._am_alpha, ctypes.byref(lock))
            self._am_th = th.value
            self._am_ig = ig.value
            self._am_ef = ef.value
            out *= scale  # 正規化を戻し包絡線とスケールを一致させる
            lock_v = float(lock.value)
        else:
            # フォールバック: ブロック平均位相によるコヒーレント検波
            m = complex(np.mean(iq_if))
            theta = float(np.angle(m))
            out = np.real(iq_if * np.exp(-1j * theta)).astype(np.float32)
            lock_v = float(abs(m) / (np.mean(np.abs(iq_if)) + 1e-12))

        # 180°逆相ロックガード: 同期出力が包絡線と逆符号なら反転させる。
        # 未対策だとDC反転→-1.0クリップの爆音歪みになる。
        try:
            if float(np.mean(out * np.abs(iq_if).astype(np.float32))) < 0.0:
                out = -out
        except Exception:
            pass

        self.am_sync_lock = lock_v
        x = float(np.clip((lock_v - 0.35) / 0.30, 0.0, 1.0))
        target = x * x * (3.0 - 2.0 * x)
        dt = len(iq_if) / self.if_rate
        tau = 0.5 if target > self._am_sync_mix else 2.5
        self._am_sync_mix += (1.0 - np.exp(-dt / tau)) * (target - self._am_sync_mix)
        return out

    def _init_am_state(self):
        """AM用FIR・AGC・同期検波等の状態初期化 (__init__ から純粋移動)。"""
        # AM用ローパス (±6kHz: 中波放送用)
        cutoff_am = 6000.0 / self.rf_rate
        self.fir_am = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_am, beta=6.5)

        # 短波放送用 狭帯域AMローパス (±3.5kHz: 短波HF帯の混信をカット)
        cutoff_am_narrow = 3500.0 / self.rf_rate
        self.fir_am_narrow = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_am_narrow, beta=7.0)
        # AM/SSB通信用 4kHzローパス (8.5kHzでは短波のヒスを通しすぎるため)
        cutoff_am_audio = 4000.0 / self.audio_rate
        self.fir_am_audio = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_am_audio, beta=7.0)
        self.am_agc_level = 0.0  # AM搬送波レベルAGC状態
        self._am_agc_hang = 0  # AGCハングタイマ (残ブロック数)
        self.impulse_blanker_enabled = True  # AMインパルスノイズブランカ
        # ===== AM同期検波 (キャリア再生PLL) =====
        # 選択性フェージング時のひずみを避けるため、包絡線検波ではなく
        # キャリアに同期した同相検波を使う。ロックできない時は包絡線へ自動復帰。
        self.am_sync_enabled = True
        self.am_sync_lock = 0.0
        # ===== AM側波帯ダイバーシティ合成 (既定ON。非対称2dB超で自動発動) =====
        # クリーン時は分岐内でw=0のままsigを返しビット等価 (透過テストで保証)。
        # 実測: 片側妨害と利得の境界は不平衡2dB (利得ゼロ群は≤1.35dB、+3.6dB群は
        # ≥2.4dB) で、しきい値2dB/1dBは適正。両側妨害・クリーンでは発動しない。
        self.am_sideband_enabled = True
        self._am_sb_w = 0.0       # 側波帯パスへの混合重み (0=既存透過)
        self._am_sb_dirty = False  # 片側汚染状態 (ヒステリシス付き)
        self._am_sb_hold = 0       # 復帰保持カウンタ (呼吸防止)
        self._am_sb_fest = 0.0     # 搬送波オフセット推定 (ピークピック＋EMA)
        self._am_sb_fest_ok = False
        self._am_sb_phase = 0.0    # デローテーション位相 (ブロック跨ぎ連続)
        self._am_sb_baseU = None   # 枝ACエネルギーのslow baseline (非汚染時のみ更新)
        self._am_sb_baseL = None
        # ===== AM/SSB 適応音声帯域 =====
        # ヒスが多い時は音声帯域を狭めて了解度を上げる (自動トーンコントロール)
        self.voice_auto_bw = True
        self.voice_cut_hz = 4000.0
        self._vc_ratio_db = -40.0
        self._am_th = 0.0
        self._am_ig = 0.0
        self._am_ef = 0.0
        self._am_sync_mix = 0.0
        # ループ帯域 ~20Hz (搬送波のドリフトに追従しつつ変調側波帯は追わない)
        self.am_kp = 3.0e-4
        self.am_ki = 5.0e-8
        # ループ帯域を約10Hz級へ狭帯域化 (旧100Hzでは低音音声がVCOを変調し
        # 混変調歪みを誘発)。ロックが遅くなってもmixが包絡線へ自動復帰する。
        self._am_alpha = 2.0 * np.pi * 30.0 / self.if_rate
