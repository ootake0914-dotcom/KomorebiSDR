"""NFM/SSB demodulation for SdrDspPipeline (extracted from dsp.py).

ナローバンドFM・SSB/CW復調。SdrDspPipeline の mixin として動作する
(純粋移動・動作同一)。_voice_bandwidth は dsp_am.py にあり self 経由で使う。
"""

import numpy as np

from dsp_filters import design_fir_kaiser
from dsp_native import (
    _NATIVE,
    _fptr,
)


class DspNfmMixin:
    """NFM/SSB復調メソッド群 (SdrDspPipeline に mixin される)"""

    def _apply_voice_highpass(self, audio: np.ndarray) -> np.ndarray:
        """通信音声用の300Hzハイパスフィルタ (Cコア: GIL解放で並行実行)"""
        if len(audio) == 0:
            return audio
        if _NATIVE is not None:
            x = np.ascontiguousarray(audio, dtype=np.float32)
            y = np.empty_like(x)
            _NATIVE.sdr_one_pole_highpass(_fptr(x), _fptr(y), len(x),
                                          float(self.voice_hp_r), _fptr(self._voice_hp_state))
            return y
        y = np.empty_like(audio)
        r = self.voice_hp_r
        x1 = self.voice_hp_x1
        y1 = self.voice_hp_y1
        for i in range(len(audio)):
            x = audio[i]
            curr_y = x - x1 + r * y1
            y[i] = curr_y
            x1 = x
            y1 = curr_y
        self.voice_hp_x1 = x1
        self.voice_hp_y1 = y1
        return y

    def demodulate_nfm(self, iq_if: np.ndarray) -> np.ndarray:
        """
        ISS（国際宇宙ステーション）やアマチュア無線用ナローバンドFM（NFM）復調器。
        - Carson帯域幅 12〜16kHz
        - ドップラーシフト（±3.5kHz）追従AFC
        - 通信音声用帯域通過フィルタ（300Hz〜3000Hz）
        - 最適化された通信オーディオゲイン補償
        """
        if len(iq_if) < 2:
            return np.zeros(0, dtype=np.float32)

        # スケルチ判定 (通信用NFMは無信号時の突発ノイズを防止するため、スケルチ必須)
        power_db = 10.0 * np.log10(np.mean(np.abs(iq_if) ** 2) + 1e-12)
        effective_threshold = self.squelch_threshold if self.squelch_enabled else -74.0
        if power_db < effective_threshold:
            return np.zeros(len(iq_if) // self.audio_decim, dtype=np.float32)

        # 1. ハードリミッター適用
        limited = self._apply_hard_limiter(iq_if)

        # 2. 瞬時位相差分法 (FM復調)
        if _NATIVE is not None:
            work = np.ascontiguousarray(limited, dtype=np.complex64)
            demod = np.empty(len(work), dtype=np.float32)
            last = np.array([self.nfm_last_sample.real, self.nfm_last_sample.imag], dtype=np.float32)
            _NATIVE.sdr_fm_demod(_fptr(work), _fptr(demod), len(work), _fptr(last))
            self.nfm_last_sample = complex(float(last[0]), float(last[1]))
        else:
            s = np.concatenate(([self.nfm_last_sample], limited))
            self.nfm_last_sample = limited[-1]
            diff = s[1:] * np.conj(s[:-1])
            demod = np.angle(diff)

        # 超音波三角ノイズ比追従型 コグニティブ・オートスケルチ
        ultra_gain = 1.0
        if getattr(self, "ultra_squelch", None) is not None and self.ultra_squelch.enabled:
            ultra_gain, _ = self.ultra_squelch.process(demod)

        # 3. NFMドップラー自動周波数追従 (ISSが飛翔する際の ±3.5kHz 移動に自動ロック)
        if self.afc_enabled and len(demod) > 0:
            mean_dc = float(np.mean(demod))
            freq_error_hz = mean_dc * (self.if_rate / (2.0 * np.pi))
            if abs(freq_error_hz) < 8000.0:  # ±8kHz以内のドップラー偏移に追従
                # 15Hz未満の微小ジッターは補正を休止しロックを維持
                if abs(freq_error_hz) > 15.0:
                    self.nfm_afc_offset_hz = float(np.clip(
                        self.nfm_afc_offset_hz - self.nfm_afc_alpha * freq_error_hz,
                        -8000.0,
                        8000.0
                    ))

        # 4. オーディオゲイン補償 (NFMの微小周波数偏移を標準通信音量に最適化)
        demod_scaled = demod * 8.5

        # 5. IFオーディオデシメーション (288kHz -> 48kHz)
        audio = self.decimate_with_history(demod_scaled, self.fir_if_audio, self.audio_decim, "history_if_audio")

        # 6. 通信用3.0kHzハイカットフィルタ
        audio = self.decimate_with_history(audio, self.fir_nfm_audio, 1, "history_nfm_audio")

        # 7. 通信用300Hz音声ハイパスフィルタ
        audio = self._apply_voice_highpass(audio)

        if ultra_gain < 0.999:
            audio = audio * ultra_gain

        return audio.astype(np.float32)

    def demodulate_ssb(self, iq_48: np.ndarray, mode: str) -> np.ndarray:
        """SSB/CW復調 (48kHz複素IF)。
        シフト→実LPF→逆シフトで非対称バンドパスを構成し、選択側波帯のみを取り出す。
        USB=+1.5kHz帯, LSB=-1.5kHz帯, CW=+650Hz±350Hz (BFOで微調整可能)。"""
        if len(iq_48) == 0:
            return np.zeros(0, dtype=np.float32)

        if mode == "USB":
            center, taps, attr = 1500.0, self.fir_ssb_lp, "history_ssb_lp"
        elif mode == "LSB":
            center, taps, attr = -1500.0, self.fir_ssb_lp, "history_ssb_lp"
        else:  # CW
            center, taps, attr = 650.0, self.fir_cw_lp, "history_cw_lp"

        n = len(iq_48)
        omega = 2.0 * np.pi * center / self.audio_rate
        ph = self._ssb_bp_phase + omega * np.arange(n, dtype=np.float64)
        self._ssb_bp_phase = float((ph[-1] + omega) % (2.0 * np.pi))
        # BFO位相 (音声ドメインで再シフト位相へ加算。USB/LSB/CW共通で正=ピッチ上昇)
        wb = 2.0 * np.pi * float(self.bfo_offset_hz) / self.audio_rate
        if mode == "LSB":
            wb = -wb
        phb = self._ssb_bfo_phase + wb * np.arange(n, dtype=np.float64)
        self._ssb_bfo_phase = float((phb[-1] + wb) % (2.0 * np.pi))
        shifted = (iq_48 * np.exp(-1j * ph)).astype(np.complex64)
        lp = self.decimate_with_history(shifted, taps, 1, attr)
        dly = (len(taps) - 1) // 2
        audio = np.real(lp * np.exp(1j * ((ph - omega * dly) + (phb - wb * dly)))).astype(np.float32)

        # AGC (SSBは搬送波が無いため平均振幅で正規化。無信号時の過剰増幅は3000倍で制限)
        # 無信号フロア (AM側と同型): ノイズ2400倍の爆音化を防ぐため滑らかにミュート。
        # ハングタイマ (AM側と同型、約400ms保持)。
        level = float(np.mean(np.abs(audio)))
        if self.ssb_agc_level <= 0.0:
            self.ssb_agc_level = max(level, 2e-4)
            self._ssb_agc_hang = 0
        elif level > self.ssb_agc_level:
            self.ssb_agc_level += 0.1 * (level - self.ssb_agc_level)
            self._ssb_agc_hang = 20
        elif self._ssb_agc_hang > 0:
            self._ssb_agc_hang -= 1
        else:
            self.ssb_agc_level += 0.005 * (level - self.ssb_agc_level)
        gain = min(1.0 / (self.ssb_agc_level + 1e-9), 3000.0)
        fade = min(1.0, level / 2e-4)
        audio = np.clip(audio * gain * 0.8, -1.0, 1.0) * fade

        # 音声帯域整形 (3kHz LPF + 300Hz HP)
        if self.cognitive_enabled:
            fir_final = self._get_dynamic_filter("audio", min(self.applied_cutoff_hz, 3500.0))
        else:
            fir_final = self.fir_audio_narrow if self.filter_mode == "narrow" else self.fir_nfm_audio
        audio = self.decimate_with_history(audio, fir_final, 1, "history_ssb_audio")
        audio = self._apply_voice_highpass(audio)
        audio = self._voice_bandwidth(audio, 3000.0, 2200.0)
        return audio.astype(np.float32)

    def _init_nfm_state(self):
        """NFM/SSB用FIR等の状態初期化 (__init__ から純粋移動)。"""
        # ISS / アマチュア無線専用 NFM (ナローバンドFM) IFローパス (±8kHz Carson帯域幅)
        cutoff_nfm_if = 8000.0 / self.rf_rate
        self.fir_nfm = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_nfm_if, beta=7.0)

        # NFM用 通信音声帯域ハイカットフィルタ (3.0kHz, 48kHzレート)
        cutoff_nfm_audio = 3000.0 / self.audio_rate
        self.fir_nfm_audio = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_nfm_audio, beta=7.0)
        # SSB用 複素バンドパスを構成する実LPF (±1.5kHz通過, 48kHzレート)
        # シフト→LPF→逆シフトで非対称バンドパスを作り、反対側波帯を除去する
        # 反対側波帯は±1.8kHz以遠にあるため、急峻な401タップで十分な阻止特性を確保
        cutoff_ssb_lp = 1350.0 / self.audio_rate
        self.fir_ssb_lp = design_fir_kaiser(num_taps=401, cutoff_norm=cutoff_ssb_lp, beta=7.5)

        # CW用 狭帯域LPF (±350Hz)
        cutoff_cw_lp = 350.0 / self.audio_rate
        self.fir_cw_lp = design_fir_kaiser(num_taps=481, cutoff_norm=cutoff_cw_lp, beta=8.0)
        self.nfm_last_sample = 0.0 + 0.0j
        self.nfm_afc_offset_hz = 0.0
        self.nfm_afc_alpha = 0.08  # ISSドップラー追従用時定数
        self._voice_hp_state = np.zeros(2, dtype=np.float32)
        # ===== SSB / CW =====
        self.bfo_offset_hz = 0.0     # BFO微調整 (SSB/CWのみ)
        self.ssb_agc_level = 0.0
        self._ssb_agc_hang = 0
        self._ssb_bp_phase = 0.0
        self._ssb_bfo_phase = 0.0
