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
        変調包絡の中央値/MAD基準で孤立パルスだけを検出し、端点線形補間で消去する。
        変調ピーク (最大2倍) や選択性フェージングの谷には触れない。
        検出率2%超のブロックは信号とみなして無処理 (安全装置)。"""
        n = len(iq_if)
        if n < 64 or not self.impulse_blanker_enabled:
            return iq_if
        mag = np.abs(iq_if).astype(np.float32)
        # 統計は間引き＋partition直取り (np.medianはNaN検査経路で遅い。
        # 期待値同一のため検出性能不変。順序統計量単点で十分)
        sm = mag[::16] if n > 256 else mag
        k = len(sm) // 2
        med = float(np.partition(sm, k)[k])
        if med < 1e-9:
            return iq_if
        dev = np.abs(sm - med)
        mad = float(np.partition(dev, k)[k]) + 1e-12
        thr = max(med + 6.0 * mad, med * 2.5)
        mask = mag > thr
        if float(np.mean(mask)) > 0.02:
            return iq_if
        edges = np.diff(mask.astype(np.int8))
        starts = list(np.flatnonzero(edges == 1) + 1)
        ends = list(np.flatnonzero(edges == -1) + 1)
        if mask[0]:
            starts.insert(0, 0)
        if mask[-1]:
            ends.append(n)
        if not starts:
            return iq_if
        out = iq_if.copy()
        for s, e in zip(starts[:512], ends[:512]):
            if e - s > 48:
                continue  # 長い区間は信号として残す
            l = out[s - 1] if s > 0 else out[e]
            r = out[e] if e < n else l
            k = (e - s)
            # Smooth cosine interpolation prevents phase/envelope kinks
            w = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, k + 1, dtype=np.float32) / (k + 1.0)))
            out[s:e] = (l * (1.0 - w) + r * w).astype(out.dtype)
        return out

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
            self._am_agc_hang = 20
        elif self._am_agc_hang > 0:
            self._am_agc_hang -= 1
        else:
            self.am_agc_level += 0.005 * (level - self.am_agc_level)
        gain = min(1.0 / (self.am_agc_level + 1e-9), 3000.0)
        fade = min(1.0, level / 2e-4)
        return np.clip((sig * gain - 1.0) * 0.6, -1.0, 1.0) * fade

    def demodulate_am(self, iq_if: np.ndarray) -> np.ndarray:
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
        audio = self._voice_bandwidth(audio, 4000.0, 2500.0)
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
        self._vc_ratio_db += (1.0 - np.exp(-dt / 0.5)) * (ratio_db - self._vc_ratio_db)
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
