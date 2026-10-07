"""WFM demodulation for SdrDspPipeline (extracted from dsp.py).

FM復調・ステレオMPXデコード・パイロットPLL・CMA・NR・ポスト処理。
SdrDspPipeline の mixin として動作する (純粋移動・動作同一)。
"""

import ctypes
import math
from collections import deque

import numpy as np

from adaptive_dsp import (
    QuadratureMpxCanceller,
    DeepSpaceEkfDemodulator,
    KalmanPilotTracker,
    HolographicAudioEnhancer,
    RiemannianTopologicalDemodulator,
    TopologicalClickSuppressor,
    SuperSpatialBssStereoSeparator,
    RmtHankelDenoiser,
    MonoNoiseSuppressor,
)
from dsp_filters import (
    design_fir_kaiser,
    design_fir_highpass,
    design_inverse_sinc,
    _deemph_sections,
)
from dsp_native import (
    _NATIVE,
    NATIVE_CMA,
    NATIVE_PLLFM,
    NATIVE_PLL3,
    _fptr,
)

try:
    from rmt_denoiser import SafeRmtDenoiser
except ImportError:
    SafeRmtDenoiser = None
try:
    from stochastic_resonance import StochasticResonanceDetector
except ImportError:
    StochasticResonanceDetector = None
try:
    from adaptive_notch import AdaptiveNotchCanceller
except ImportError:
    AdaptiveNotchCanceller = None
try:
    from cyclostationary_detector import CyclostationaryPilotDetector
except ImportError:
    CyclostationaryPilotDetector = None


class DspWfmMixin:
    """WFM復調メソッド群 (SdrDspPipeline に mixin される)"""

    def _apply_cma(self, iq_if: np.ndarray) -> np.ndarray:
        """CMAブラインド等化器 (history前置でブロック連続性を保つ)。"""
        n_out = len(iq_if)
        if n_out == 0:
            return iq_if
        taps = self._cma_taps
        if len(self._cma_hist) != taps - 1:
            self._cma_hist = np.zeros(taps - 1, dtype=np.complex64)
        x_ext = np.concatenate((self._cma_hist, np.ascontiguousarray(iq_if)))
        self._cma_hist = x_ext[-(taps - 1):].copy()

        xa = np.ascontiguousarray(x_ext, dtype=np.complex64)
        y = np.empty(n_out, dtype=np.complex64)
        _NATIVE.sdr_cma_equalize(_fptr(xa), _fptr(y), n_out,
                                 _fptr(self._cma_w), taps, float(self._cma_mu))
        # フェイルセーフ: 万一 NaN/Inf が出力されたら重みと履歴を即座にリセットし
        # 原信号をサニタイズして通過 (NaN履歴が数ブロック再発するのを防ぐ)
        if not np.all(np.isfinite(y)):
            self._cma_w[:] = 0.0
            self._cma_w[2 * (taps // 2)] = 1.0
            self._cma_hist[:] = 0.0
            return np.nan_to_num(iq_if, nan=0.0, posinf=1.0, neginf=-1.0)
        return y

    def _update_cma_auto_gate(self) -> bool:
        """CMA自動介入ゲート（ヒステリシス＋信号存在条件）。

        強い反射波でのみ作動し、弱まったら速やかに戻す。手動設定は常に尊重する。
        cognitive無効時は自動介入しない（従来テスト/非認知経路の挙動を保護）。
        介入条件はパイロットlock>0.2が必須。S-meterだけでの作動
        (旧OR条件) は、lock≈0の深フェードでCMAが入りっぱなしになり
        blendを下げる (合成deep-fadeでblend 1.00→0.73を実測) ため廃止。
        S-meterは-60dBFSのノイズ床 veto としてのみ使う。
        """
        if not self.multipath_auto_cancel or not self.cognitive_enabled:
            self._cma_auto = bool(self.multipath_cancel_enabled)
            return self._cma_auto
        try:
            lock = abs(float(self.stereo_pilot_lock))
            s_db = float(self.s_meter_dbfs)
        except (TypeError, ValueError):
            lock, s_db = 0.0, -90.0
        if not (math.isfinite(lock) and math.isfinite(s_db)):
            lock, s_db = 0.0, -90.0
        present = lock > 0.2 and s_db > -60.0
        amt = float(self.multipath_amount)
        if self._cma_auto:
            hold = amt >= 0.18 and present
        else:
            hold = amt >= 0.30 and present
        self._cma_auto = bool(self.multipath_cancel_enabled or hold)
        return self._cma_auto

    def _decode_stereo_pair(self, demod_scaled: np.ndarray, mono: np.ndarray,
                              ultra_gain: float):
        """ステレオMPXデコード (38k検波〜L/Rスタック)。

        demodulate_wfm から純粋移動 (動作同一)。デコード不可時は None を返し、
        呼び出し側はモノラル経路へ継続する。
        """
        stereo_diff = None
        _carrier_ok = (self._last_sin2 is not None
                       and len(self._last_sin2) == len(demod_scaled))
        if _carrier_ok and self._stereo_blend > 0.02:
            # BS.450準拠: 38kHz副搬送波はsin(2wt)。
            # PLLがth = wt - pi/2でロックしているため、sin(2*th) = sin(2wt - pi) = -sin(2wt)。
            # したがって同相復調キャリアは -self._last_sin2。
            carrier = -self._last_sin2
            if abs(self.stereo_phase_offset) > 1e-6 and self._last_cos2 is not None:
                co = np.cos(self.stereo_phase_offset)
                si = np.sin(self.stereo_phase_offset)
                carrier = carrier * co - self._last_cos2 * si
            lpr = demod_scaled * carrier
            diff_raw = self.decimate_with_history(lpr, self.fir_if_audio,
                                                  self.audio_decim, "history_lpr") * 2.0

            # 38kHz 直交副搬送波マルチパス適応キャンセラ (Quadrature MPX Decoupler)
            if (getattr(self, "mpx_canceller", None) is not None
                    and self.mpx_canceller.enabled and self._last_cos2 is not None):
                carrier_q = -self._last_cos2
                if abs(self.stereo_phase_offset) > 1e-6:
                    carrier_q = carrier_q * co + self._last_sin2 * si
                lpr_q = demod_scaled * carrier_q
                diff_q = self.decimate_with_history(lpr_q, self.fir_if_audio,
                                                    self.audio_decim, "history_lpr_q") * 2.0
                if len(diff_q) == len(diff_raw):
                    self._last_diff_q = diff_q
                    diff_raw = self.mpx_canceller.process(diff_raw, diff_q)
                else:
                    self._last_diff_q = None

            if len(diff_raw) == len(mono):
                # 常に実測 (診断・A/B用)。NR無効時はフィルタ素通し・フルステレオ固定
                self._update_stereo_nr(diff_raw, mono)
                # 38kHz位相オートトリム (L-R電力最大化。NR推定と同タイミングで観測)
                self._update_stereo_trim(diff_raw, mono)
                if self.stereo_nr_enabled:
                    cut = self._nr_cut_eff
                    blend = self._stereo_blend * self.stereo_nr_gain
                    blend *= self.multipath_gain * self.aci_gain
                    diff = self._diff_lowpass(diff_raw, cut)
                    diff = self._wiener_diff(diff)
                    # 周波数依存ブレンド: 弱電界で高域から先にモノラル化
                    # (低域のステレオ感を残す。blend=1時は旧スカラー動作と一致)
                    stereo_diff = self._freq_dependent_blend(diff, blend)
                    # IF切落としによる差信号振幅不足を校正 (NR推定は無倍率で観測)
                    stereo_diff = (stereo_diff
                                   * float(self.stereo_diff_gain)).astype(np.float32)
                    # 差信号FIRの群遅延を補償 (mono/diffの位相ズレによる分離度劣化を防止)
                    mono = self._delay_mono(mono)
                else:
                    # NR無効時はLPF/STFT往復をせず生差信号へブレンドのみ
                    # (15kHz LPF＋COLAリップルが可聴域を変える問題とCPU浪費を回避)。
                    # mono遅延履歴だけは更新し、再有効時の継ぎ目を無くす
                    # (出力は遅延させない。遅延させると未遅延diffと2msずれる)。
                    stereo_diff = (diff_raw
                                   * (self._stereo_blend * self.multipath_gain
                                      * self.aci_gain
                                      * float(self.stereo_diff_gain))).astype(np.float32)
                    self._delay_mono(mono)
                    self.stereo_wiener_gain = 1.0

        # 6. チャンネル別ポスト処理 (ディエンファシス・ハイカット・DCカット・シェルフ)
        if stereo_diff is not None:
            # 注: L/Rのスレッド並列化は実測で逆効果 (CPython GIL + C呼び出しが短く
            # オーバーヘッドが上回る)。逐次実行が最速。
            # 単一ch NRはMid/Sideで共有するためここではスキップする (L/R独立
            # ゲインは音像を揺らす。後段でMidのみ処理しSideは同量遅延で素通し)。
            use_ms_nr = (getattr(self, "mono_nr", None) is not None
                         and self.mono_nr.enabled
                         and getattr(self, "mono_nr_enabled", True)
                         and (self.cognitive_enabled
                              or getattr(self, "mono_nr_always", False)))
            left = self._post_process_wfm(mono + stereo_diff, "_l",
                                          skip_mono_nr=use_ms_nr)
            right = self._post_process_wfm(mono - stereo_diff, "_r",
                                           skip_mono_nr=use_ms_nr)
            if use_ms_nr:
                left, right = self._mid_side_mono_nr(left, right)

            # BSSステレオ分離器 (FastICA。差信号中の逆相寄りヒスノイズ低減)
            if (getattr(self, "bss_separator", None) is not None
                    and self.bss_separator.enabled
                    and (self.cognitive_enabled or getattr(self, "bss_always", False))):
                left, right = self.bss_separator.process(left, right, stereo_blend=self._stereo_blend)

            if ultra_gain < 0.999:
                left = left * ultra_gain
                right = right * ultra_gain
            # 黒魔法2-1: 適応ハムノッチ (既定OFF)。RMTより前段に置く
            # (ハム除去後のクリーンな信号をNRへ渡す)。
            if (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_notch_enabled", False)):
                try:
                    if self.bm_notch is None and AdaptiveNotchCanceller is not None:
                        self.bm_notch = self._bm_make_notch()
                    if self.bm_notch is not None and len(left) == len(right):
                        (left, right), _ = self.bm_notch.process_stereo(
                            left, right,
                            clip=bool(getattr(self, "adc_clipped", False)))
                        left = np.asarray(left, dtype=np.float32)
                        right = np.asarray(right, dtype=np.float32)
                except Exception:
                    pass
            # 黒魔法B: ステレオ対のMid/Side安全RMT (既定OFF)。
            # L/R独立処理は分離度を落とすため、Mid通常・Side低強度で処理する。
            # 既存rmt_denoiser側との二重処理は避けること (docs参照)。
            if (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_rmt_enabled", False)):
                try:
                    if self.bm_rmt is None and SafeRmtDenoiser is not None:
                        self.bm_rmt = self._bm_make_rmt()
                    if self.bm_rmt is not None and len(left) == len(right):
                        self.bm_rmt.max_strength = self._bm_rmt_strength_cap()
                        (left, right), _bm_info = self.bm_rmt.process_stereo(
                            left, right,
                            s_meter_dbfs=float(getattr(self, "s_meter_dbfs",
                                                       -20.0)),
                            snr_db=None,
                            clip=bool(getattr(self, "adc_clipped", False)))
                        left = np.asarray(left, dtype=np.float32)
                        right = np.asarray(right, dtype=np.float32)
                        try:
                            _bm_mid = (_bm_info.get("mid", None) or {})
                            self._bm_last_rmt_info = {
                                "hf_loss_db": float(_bm_mid.get("hf_loss_db", 0.0)),
                                "rms_diff_db": float(_bm_mid.get("rms_diff_db", 0.0)),
                            }
                        except Exception:
                            pass
                except Exception:
                    pass
            blend = self._stereo_blend * (self.stereo_nr_gain if self.stereo_nr_enabled else 1.0) \
                * self.multipath_gain * self.aci_gain
            self.is_stereo = blend > 0.5
            if blend > 0.85:
                self.stereo_status = "STEREO"
            elif blend > 0.02:
                self.stereo_status = "BLEND"
            else:
                self.stereo_status = "MONO"
            # NaN/Infサニタイズと暴走時のみの安全クランプ (±4)。ピーク処理は
            # 最終段ルックアヘッドリミッタに任せる (±1で切るとスローAGC前の
            # 過変調・スパイク歪みが残り、AGC後も消えない)。
            out = np.nan_to_num(np.stack([left, right], axis=1), nan=0.0,
                                posinf=4.0, neginf=-4.0)
            return np.clip(out, -4.0, 4.0).astype(np.float32)
        return None

    def demodulate_wfm(self, iq_if: np.ndarray) -> np.ndarray:
        if len(iq_if) < 2:
            return np.zeros(0, dtype=np.float32)

        # スケルチ判定
        power_db = 10.0 * np.log10(np.mean(np.abs(iq_if) ** 2) + 1e-12)
        if self.squelch_enabled and power_db < self.squelch_threshold:
            return np.zeros(len(iq_if) // self.audio_decim, dtype=np.float32)

        # 0. マルチパス検出 (包絡線の変動 = PM→AM変換量)
        # 平滑は鈍め (測定1.0秒・反映1.5秒): 速すぎると番組の包絡変動や
        # フェージングの瞬時値にステレオ幅が呼吸してしまう (市街地局で実測)
        if self.multipath_enabled:
            env = np.abs(iq_if)
            var = float(np.std(env) / (np.mean(env) + 1e-12))
            dt_mp = len(iq_if) / self.if_rate
            self._mp_var += (1.0 - np.exp(-dt_mp / 1.0)) * (var - self._mp_var)
            x = float(np.clip((self._mp_var - self.mp_lo) / (self.mp_hi - self.mp_lo), 0.0, 1.0))
            s = x * x * (3.0 - 2.0 * x)
            tau = 1.5 if s > self.multipath_amount else 2.5
            self.multipath_amount += (1.0 - np.exp(-dt_mp / tau)) * (s - self.multipath_amount)
            self.multipath_gain = 1.0 - self.mp_depth * self.multipath_amount

        # 0b. 隣接妨害 (ACI) ガード: 左右どちらかのD/Uが12dB未満なら、L-R側を
        # 最大60%絞る (周波数依存ブレンドと併用で高域から先にモノラル化)。
        # ACIは38kHz副搬送波を先に汚すため、モノラル音声よりL-Rを守る方が効く。
        # 隣接帯がノイズ床+6dB未満の側は「局ではなく床」とみなし発動しない。
        _du_l = float(self._aci_l_db) if self._aci_l_above_db > 6.0 else 99.0
        _du_r = float(self._aci_r_db) if self._aci_r_above_db > 6.0 else 99.0
        _min_du = min(_du_l, _du_r)
        _x = float(np.clip((12.0 - _min_du) / 12.0, 0.0, 1.0))
        _aci_target = 1.0 - self.aci_depth * _x
        dt_a = len(iq_if) / float(self.if_rate)
        tau_a = 0.3 if _aci_target < self.aci_gain else 1.5
        self.aci_gain += (1.0 - float(np.exp(-dt_a / tau_a))) * (_aci_target - self.aci_gain)

        # 1. CMA等化 (マルチパス・キャンセル) + ハードリミッター適用
        use_cma = self._update_cma_auto_gate()
        if (use_cma and _NATIVE is not None and NATIVE_CMA
                and self.multipath_amount > 0.15
                and abs(self.stereo_pilot_lock) > 0.2
                and self.s_meter_dbfs > -60.0):
            # CMA等化 (ハードリミット前。リミット後は包絡線一定で誤差が出ない)。
            # 信号存在ゲート: lock必須＋ノイズ床veto (S-meterだけでの作動は
            # 深フェードでblendを下げるため廃止。ゲート側と条件を一致させる)。
            # C/Nクロスフェード: 比較的クリーンな条件では等化人工物が
            # 逆効果のため、重み0では等化自体を呼ばず素通し (適応も凍結)。
            _cn = float(getattr(self, "_if_snr_db", 25.0))
            _chi = float(getattr(self, "cma_cn_hi", 30.0))
            _clo = float(getattr(self, "cma_cn_lo", 25.0))
            w_cma = float(np.clip((_chi - _cn) / max(_chi - _clo, 1e-6),
                                  0.0, float(getattr(self, "cma_w_max", 1.0))))
            if w_cma > 0.01:
                iq_eq = self._apply_cma(iq_if)
                iq_if = ((1.0 - w_cma) * iq_if + w_cma * iq_eq)
                self.cma_active = True
            else:
                self.cma_active = False
        else:
            self.cma_active = False

        limited = self._apply_hard_limiter(iq_if)

        # 2. FM復調: ハイブリッド復調エンジン
        # - 通常 (強信号) : 瞬時位相差分法 (分離度-42dB・高変調域も歪みにくい広帯域復調)
        # - 弱信号 (S7相当以下) : 拡張カルマンフィルタ (EKF) による確率論的MMSE復調
        #   (FM閾値拡張を狙う。低CNR下での2π位相スリップ・クリックスパイクノイズの抑止)
        # NOTE: 旧 use_pll (NATIVE_PLLFM) 経路は未使用デッドコードだったため削除。
        # PLL-FMが必要になった場合は fm_pll_enabled とは独立に有効化すること。
        if _NATIVE is not None:
            work = np.ascontiguousarray(limited, dtype=np.complex64)
            demod = np.empty(len(work), dtype=np.float32)
            last = np.array([self.fm_last_sample.real, self.fm_last_sample.imag],
                            dtype=np.float32)
            _NATIVE.sdr_fm_demod(_fptr(work), _fptr(demod), len(work), _fptr(last))
            self.fm_last_sample = complex(float(last[0]), float(last[1]))
        else:
            s = np.concatenate(([self.fm_last_sample], limited))
            self.fm_last_sample = limited[-1]
            diff = s[1:] * np.conj(s[:-1])
            demod = np.angle(diff)

        # 位相スリップ (ライスクリック) 補修: 弱電界のみ。クリック検出時だけ
        # 該当サンプルを補間値へ差し替える (クリーン・強電界ではビット等価)。
        if (getattr(self, "tda_click", None) is not None
                and self.tda_click.enabled
                and float(getattr(self, "_if_snr_db", 0.0)) < self.tda_click_gate_db):
            tda_demod, tda_mask = self.tda_click.process_with_mask(iq_if)
            if len(tda_demod) == len(demod) and bool(np.any(tda_mask)):
                demod = demod.copy()
                demod[tda_mask] = tda_demod[tda_mask]
                self.tda_clicks += int(np.count_nonzero(tda_mask))

        # 弱電界・モノラル時におけるEKFのシームレス・クロスフェード
        # (ステレオ時は38kHz副搬送波の広帯域通過のため広帯域差分法を維持し、
        # 弱電界・モノラル局でEKFを併用してクリックスパイクと三角雑音を抑える)
        # NOTE: 旧条件は fm_pll_enabled(False既定) とのANDで常時OFFになっていた。
        # EKFは ekf_enabled 単独で制御する (PLLとは独立)。
        if (getattr(self, "ekf_enabled", False)
                and getattr(self, "ekf_demod", None) is not None
                and self._stereo_blend <= 0.05):
            # 切替はレベル(dBFS)でなくC/N (_if_snr_db) で行う。dBFSはHyperの
            # ゲインで動き、弱局を拾おうとゲインを上げると補助が外れる逆動作
            # になる。推定値は合成chSNR+約14dB (実測): 26dBで0、20dBで1。
            _cn = float(getattr(self, "_if_snr_db", 26.0))
            _ehi = float(getattr(self, "ekf_cn_hi", 26.0))
            _elo = float(getattr(self, "ekf_cn_lo", 20.0))
            w_ekf = float(np.clip((_ehi - _cn) / max(_ehi - _elo, 1e-6),
                                  0.0, float(getattr(self, "ekf_w_max", 1.0))))
            if w_ekf > 0.01:
                demod_ekf = self.ekf_demod.demodulate(limited)
                if len(demod_ekf) == len(demod):
                    demod = ((1.0 - w_ekf) * demod + w_ekf * demod_ekf).astype(np.float32)

        # 位相スリップ防止FM復調 (弱電界フェージング時のクリック雑音抑制)
        if (getattr(self, "riemann_demodulator", None) is not None
                and self.riemann_demodulator.enabled
                and (self.cognitive_enabled or getattr(self, "riemann_always", False))):
            # EKFと同じくC/N基準 (dBFS条件はゲイン依存のため廃止)。
            # 28dBで0、22dB以下で0.75上限。
            _cn = float(getattr(self, "_if_snr_db", 28.0))
            _rhi = float(getattr(self, "riemann_cn_hi", 28.0))
            _rlo = float(getattr(self, "riemann_cn_lo", 22.0))
            w_riemann = float(np.clip(
                (_rhi - _cn) / max(_rhi - _rlo, 1e-6),
                0.0, float(getattr(self, "riemann_w_max", 0.75))))
            if w_riemann > 0.01:
                demod_riemann = self.riemann_demodulator.demodulate(iq_if)
                if len(demod_riemann) == len(demod):
                    demod = ((1.0 - w_riemann) * demod
                             + w_riemann * demod_riemann).astype(np.float32)

        # 超音波三角ノイズ比追従型 オートスケルチ
        ultra_gain = 1.0
        if getattr(self, "ultra_squelch", None) is not None and self.ultra_squelch.enabled:
            ultra_gain, _ = self.ultra_squelch.process(demod)

        # 黒魔法C: 確率共鳴は検出用副経路のみ。メインのdemod配列には一切触らない。
        # confidenceを属性 (bm_sr_confidence) として公開するだけで、
        # スケルチ判定への自動反映は既存動作との競合回避のため見送る。
        # (評価はbenchmark_black_magic.pyの検出率比較で行う)
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_sr_enabled", False)):
            try:
                _bmp2 = getattr(self, "_bm_params", None) or {}
                # コントローラがSR非活性と判断したら試行自体を休止する
                _sr_active = bool(_bmp2.get("sr_active", True))
                if self.bm_sr is None and StochasticResonanceDetector is not None:
                    self.bm_sr = self._bm_make_sr()
                if (self.bm_sr is not None and _sr_active
                        and len(demod) >= 1024):
                    n_sr = len(demod)

                    def _bm_base(v, _n=n_sr):
                        vv = np.asarray(v, dtype=np.float64).reshape(-1)
                        if len(vv) != _n:
                            return False
                        try:
                            from dsp_native import dft_bins
                            r = dft_bins(vv, [19000.0], float(self.if_rate))
                            return bool(abs(complex(float(r[0, 0]),
                                                    float(r[0, 1]))) > 0.02)
                        except Exception:
                            return False

                    _uq = getattr(self, "ultra_squelch", None)
                    floor = 10.0 ** (float(getattr(_uq, "noise_db", -40.0)) / 20.0)
                    snr = float(getattr(self, "s_meter_dbfs", -45.0)) + 45.0
                    base_conf = min(max(float(getattr(self, "stereo_pilot_lock",
                                                      0.0)), 0.0), 1.0)
                    res = self.bm_sr.assess(
                        demod, _bm_base, floor, snr, base_conf,
                        clip=bool(getattr(self, "adc_clipped", False)),
                        candidate_present=bool(self.stereo_enabled))
                    if res.get("enabled"):
                        self.bm_sr_confidence = float(res.get("sr_confidence",
                                                              base_conf))
                    else:
                        self.bm_sr_confidence = base_conf
            except Exception:
                pass

        # 黒魔法①: cyclo→スケルチ統合 (既定OFF)。1ブロック遅れのconfidenceと
        # S-meterで開閉を決め、ソフトフェードゲインに反映する。音声への適用は
        # process()終端 (スローAGC後) で行う。音声自体は変えない。
        self._bm_update_squelch_assist()

        # AFC (Automatic Frequency Control): 復調信号のDCバイアスから周波数偏差を推定してフィードバック
        if self.afc_enabled and len(demod) > 0:
            mean_dc = float(np.mean(demod))
            freq_error_hz = mean_dc * (self.if_rate / (2.0 * np.pi))
            afc_limit = float(getattr(self, "afc_limit_hz", 42000.0))
            if abs(freq_error_hz) < afc_limit:  # 許容偏差範囲に自動追従
                # 20Hz未満の微小ジッターは補正を休止しロックを維持
                if abs(freq_error_hz) > 20.0:
                    self.afc_offset_hz = float(np.clip(
                        self.afc_offset_hz - self.afc_alpha * freq_error_hz,
                        -afc_limit,
                        afc_limit
                    ))

        # 3. 適切な名目オーディオゲインにスケーリング
        # 日本規格の最大周波数偏移(±75kHz)でも振幅0.95以内に収め、過変調時のソフトリミッターポンピング歪みを抑制
        demod_scaled = demod * 0.58

        # 3b. 差分検波アパーチャ逆補正 (M/D共通前段。mono/差/RDSすべて
        # 同一遅延で通るため分離度の時間整合は不変)
        if getattr(self, "inv_sinc_enabled", False) and len(demod_scaled) > 0:
            demod_scaled = self.decimate_with_history(
                np.asarray(demod_scaled, dtype=np.float32),
                self.fir_inv_sinc, 1, "history_inv_sinc")

        # 4. モノラル (L+R) を48kHzへデシメーション
        #    (19kHzパイロット・38kHz副搬送波はアンチエイリアスLPFで除去)
        #    ※同一 history を2回呼ぶと mono だけ履歴が二重送りになり、差信号との
        #      位相がずれて分離度・モノラル特性が劣化するため、呼び出しは1回のみ。
        mono = self.decimate_with_history(demod_scaled, self.fir_if_audio,
                                          self.audio_decim, "history_if_audio")

        # 5a. 副搬送波清浄化は廃止 (固定LPに劣る適応軟しきい値だった。
        # 既知周波数の搬送波に適応は不要という結論。素通し)
        demod_mpx = demod_scaled

        # 5b. ステレオMPXデコード (19kHzパイロットPLL + 38kHz同期検波)
        self._update_stereo_pilot(demod_mpx)

        # 5c. RDS復調 (57kHz = 3θ, ステレオ状態と独立して常時動作)
        # パイロットロック連動ゲート: パイロット不在時 (無信号・モノラル) は
        # 57kHzに信号が存在し得ないためfeedを省略する。ノイズ入力でデコーダの
        # 同期探索 (syndrome全探索) が約4ms/block浪費していた問題の抑制。
        # デコーダ状態は保持されるため、ロック復帰時は即時再開する。
        if (self.rds_enabled and self._last_cos3 is not None
                and abs(self.stereo_pilot_lock) > 0.2):
            try:
                if self.rds is None:
                    import rds as rds_mod
                    self.rds = rds_mod.RdsDecoder(12000.0)
                # BS.450/EN 50067準拠: パイロットsin(wt)に対して57kHz副搬送波はsin(3wt)。
                # PLLがth = wt - pi/2でロックしているため、cos(3*th) = cos(3wt - 3pi/2) = -sin(3wt)。
                # したがって、同相復調キャリアは -self._last_cos3。
                carrier57 = -self._last_cos3
                if abs(self.rds_phase_offset) > 1e-6 and self._last_sin3 is not None:
                    co = np.cos(self.rds_phase_offset)
                    si = np.sin(self.rds_phase_offset)
                    carrier57 = carrier57 * co - self._last_sin3 * si
                rds_mix = demod_scaled * carrier57
                rds_base = self.decimate_with_history(rds_mix, self.fir_rds, 24, "history_rds")
                self.rds.feed(rds_base)
                self.rds_ps = self.rds.ps_name
                self.rds_rt = self.rds.radio_text
                self.rds_pi = self.rds.pi
                self.rds_pty = self.rds.pty
                self.rds_groups = self.rds.groups
                # BLER早期警報用に失敗計装も公開 (制御への使用は相関確認までは行わない)
                self.rds_bler = float(getattr(self.rds, "bler", 0.0))
                self.rds_groups_checked = int(getattr(self.rds, "groups_checked", 0))
                self.rds_groups_failed = int(getattr(self.rds, "groups_failed", 0))
                self.rds_sync_losses = int(getattr(self.rds, "sync_losses", 0))
            except Exception:
                pass

        stereo_out = self._decode_stereo_pair(demod_scaled, mono, ultra_gain)
        if stereo_out is not None:
            return stereo_out

        self.is_stereo = False
        self.stereo_status = "MONO"
        # モノラル中も遅延履歴を進めておく (ステレオ復帰時に履歴が古い/ゼロだと
        # 先頭_nr_delayサンプルが無音になり2msの欠落クリックになる)。
        # また、NR有効時はモノラル出力も同量遅延させることで、ステレオ/モノラル自動切替時の
        # タイムワープ (2ms音飛び・重複・クリック) を抑止しタイムラインを連続化する。
        mono_delayed = self._delay_mono(mono)
        mono_out = mono_delayed if self.stereo_nr_enabled else mono
        # モノラル信号はステレオのセンター定位 (L=mono, R=mono) と同一レベル (0dB差) で出力
        out_mono = self._post_process_wfm(mono_out, "")
        if ultra_gain < 0.999:
            out_mono = out_mono * ultra_gain
        # 黒魔法2-1: モノラル経路の適応ハムノッチ (既定OFF)
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_notch_enabled", False)):
            try:
                if self.bm_notch is None and AdaptiveNotchCanceller is not None:
                    self.bm_notch = self._bm_make_notch()
                if self.bm_notch is not None:
                    out_mono, _ = self.bm_notch.process_mono(
                        out_mono, ch="bm",
                        clip=bool(getattr(self, "adc_clipped", False)))
                    out_mono = np.asarray(out_mono, dtype=np.float32)
            except Exception:
                pass
        # 黒魔法B: モノラル経路の安全RMT (既定OFF)
        if (getattr(self, "black_magic_enabled", False)
                and getattr(self, "bm_rmt_enabled", False)):
            try:
                if self.bm_rmt is None and SafeRmtDenoiser is not None:
                    self.bm_rmt = self._bm_make_rmt()
                if self.bm_rmt is not None:
                    self.bm_rmt.max_strength = self._bm_rmt_strength_cap()
                    out_mono, _bm_info_m = self.bm_rmt.process_mono(
                        out_mono,
                        s_meter_dbfs=float(getattr(self, "s_meter_dbfs", -20.0)),
                        snr_db=None, ch="bm",
                        clip=bool(getattr(self, "adc_clipped", False)))
                    out_mono = np.asarray(out_mono, dtype=np.float32)
                    try:
                        self._bm_last_rmt_info = {
                            "hf_loss_db": float(_bm_info_m.get("hf_loss_db", 0.0)),
                            "rms_diff_db": float(_bm_info_m.get("rms_diff_db", 0.0)),
                        }
                    except Exception:
                        pass
            except Exception:
                pass
        out_mono = np.nan_to_num(out_mono, nan=0.0, posinf=4.0, neginf=-4.0)
        return np.clip(out_mono, -4.0, 4.0)

    def _delay_mono(self, mono: np.ndarray) -> np.ndarray:
        """monoを群遅延分だけ遅延させ、NRフィルタ通過後の差信号と時間整合を取る。
        NR経路の実遅延 (_nr_delay + STFTゼロ埋め分) に動的に一致させる。"""
        d = self._nr_delay + int(getattr(self, "_wf_extra_delay", 0))
        if d <= 0 or len(mono) == 0:
            return mono
        # 履歴長が不足する場合はゼロで延長 (遅延量を厳密に保つ)
        if len(self.history_mono_delay) < d:
            pad = np.zeros(d - len(self.history_mono_delay), dtype=np.float32)
            self.history_mono_delay = np.concatenate((pad, self.history_mono_delay))
        y = np.concatenate((self.history_mono_delay, mono))[:len(mono)]
        if len(mono) >= d:
            self.history_mono_delay = mono[-d:].copy()
        else:
            self.history_mono_delay = np.concatenate((self.history_mono_delay, mono))[-d:]
        return y.astype(np.float32)

    def _update_stereo_nr(self, diff: np.ndarray, mono: np.ndarray):
        """(L-R)高域ノイズを(L+R)中域プログラムレベルで正規化してヒス量を推定する"""
        # NOTE: 間引きは行わない。適応速度(τ0.35〜2.5s)がテストで厳密に検証され、
        # 間引きは収束を遅らせてtest_nr_program_decouplingを破る (0.138<0.15)。
        # 0.3msの節約より適応特性の保存を優先する。
        n = 1024
        if len(diff) < 128 or len(mono) < 128:
            return
        bin_hz = self.audio_rate / n

        def band_power(sig: np.ndarray, lo: float, hi: float) -> float:
            if len(sig) >= n:
                seg = sig[-n:].astype(np.float32)
            else:
                seg = np.zeros(n, dtype=np.float32)
                seg[-len(sig):] = sig
            seg = seg - float(np.mean(seg))
            power = np.abs(np.fft.rfft(seg * self._nr_window)) ** 2 + 1e-12
            i0 = max(1, int(lo / bin_hz))
            i1 = min(len(power) - 1, int(hi / bin_hz))
            if i1 <= i0:
                return 1e-12
            return float(np.mean(power[i0:i1 + 1]))

        hf = band_power(diff, 6000.0, 15000.0)
        mf = band_power(mono, 300.0, 3000.0)
        # 番組パワーの平滑化 (τ2.5秒対称): 瞬時mfのままだと音楽の緩急が
        # そのままヒス比に載ってNRゲイン・カットオフがポンピングする
        # (強音楽局の実測で発覚)。フロア側は10秒履歴なので分母だけ鈍らせる。
        dt = len(diff) / self.audio_rate
        if not self._nr_primed or self._nr_mf_smooth <= 0.0:
            self._nr_mf_smooth = mf
        else:
            a_mf = 1.0 - np.exp(-dt / 2.5)
            self._nr_mf_smooth += a_mf * (mf - self._nr_mf_smooth)
        mf = self._nr_mf_smooth
        # 分母飽和ガード: 300-3kHz番組が前回ノイズ床-20dB未満なら比は不定。
        # 高域のみの番組 (管楽器高音・シンバル・拍手) でヒス推定が非物理値
        # (+40dB級) に発散しNRが全閉する実害の修正。推定全体を凍結し
        # (前回値保持)、履歴への学習も止めて番組HFを床と誤学習しない。
        # 初回 (床未確定) は素通しする。
        if (self._nr_primed and self._nr_floor_pow > 0.0
                and mf < self._nr_floor_pow * 0.01):
            return
        # ノイズフロア推定: ~10秒履歴の下位10%を使い、番組自身の高域成分ではなく
        # 定常的に存在するとヒス成分のみを検出する (明るい音楽での過剰なNRを防止)
        self._nr_hist.append(hf)
        # 下位10%タイル: percentile(ソート)よりpartition(O(n))で高速化
        if len(self._nr_hist) >= 4:
            arr = np.asarray(self._nr_hist, dtype=np.float32)
            k = int(0.10 * (len(arr) - 1))
            floor = float(np.partition(arr, k)[k])
        else:
            floor = hf
        # 物理SNRクロスチェック (radiko_labのオラクル実験由来): 強電界なのに
        # 音声ドメインの床が高い場合、床はヒスでなく番組HF (ギャップの無い
        # 連続番組) の疑いが強い。実測: 合成連続音声はIF37-47dBで床-8dB→
        # 全モノラル化 (nr_gain 0.15)、実録LuckyはIF26dBで床-27dB (崩壊なし)。
        # 強電界側だけ床を割り引き、幅崩壊と過剰Wienerを防ぐ (IF30dB以下の
        # 弱電界は不変)。
        try:
            _snr = float(getattr(self, "_if_snr_db", 10.0))
        except Exception:
            _snr = 10.0
        if np.isfinite(_snr):
            _rel = float(np.clip(
                (float(self._nr_snr_gate_hi) - _snr)
                / (float(self._nr_snr_gate_hi) - float(self._nr_snr_gate_lo)),
                0.0, 1.0))
        else:
            _rel = 1.0
        if _rel < 1.0:
            floor = float(floor) * 10.0 ** (
                -float(self._nr_snr_penalty_db) * (1.0 - _rel) / 10.0)
        self._nr_floor_pow = floor
        ratio_db = 10.0 * np.log10((floor + 1e-12) / (mf + 1e-12))
        # ベルト兼用クランプ: 白色雑音でも+5dB程度が上限のため+12dB、
        # 下側は実測クリーン (-78dB) に余裕を見て-80dB。
        ratio_db = float(np.clip(ratio_db, -80.0, 12.0))

        dt = len(diff) / self.audio_rate
        if not self._nr_primed:
            # 初回は実測値で即座に初期化 (起動直後のランプを排除)。
            # ただし初回から不定比 (高域のみ信号) の場合はクリーン既定へ。
            self._nr_primed = True
            if floor > 0.0 and mf < floor * 0.01:
                self.stereo_hiss_db = -60.0
            else:
                self.stereo_hiss_db = ratio_db
        else:
            a = 1.0 - np.exp(-dt / 0.35)
            self.stereo_hiss_db += a * (ratio_db - self.stereo_hiss_db)

        def amount(value_db: float, lo: float, hi: float) -> float:
            x = float(np.clip((value_db - lo) / (hi - lo), 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)  # smoothstep

        # マッピング用の超低速ヒス推定 (τ12s)。番組構成で動く速い値はゲート
        # (s_b=ブレンド) と表示にのみ使い、Wiener適用度 (s_w) は定常値で駆動
        # して「幅の呼吸」を防ぐ。初期化は速い値から (選局直後の立ち上がり維持)
        if self._nr_hiss_slow is None:
            self._nr_hiss_slow = float(self.stereo_hiss_db)
        else:
            a_hs = 1.0 - np.exp(-dt / 12.0)
            self._nr_hiss_slow += a_hs * (float(self.stereo_hiss_db)
                                          - self._nr_hiss_slow)
        s_b = amount(self.stereo_hiss_db, self._nr_lo_db, self._nr_hi_db)
        s_w = amount(self._nr_hiss_slow, self._nr_wiener_lo_db,
                     self._nr_wiener_hi_db)
        # 非対称スムージング: ノイズ増加時は速く、回復はゆっくり
        tau = 0.3 if s_b > self._nr_s else 1.5
        self._nr_s += (1.0 - np.exp(-dt / tau)) * (s_b - self._nr_s)
        tau_w = 0.25 if s_w > self._nr_s_w else 2.5
        self._nr_s_w += (1.0 - np.exp(-dt / tau_w)) * (s_w - self._nr_s_w)
        self.stereo_nr_gain = 1.0 - self._nr_s
        self.stereo_cut_hz = self._nr_cut_max_hz * (
            (self._nr_cut_min_hz / self._nr_cut_max_hz) ** self._nr_s
        )
        # 固定高域ブレンドの適応化: クリーン時は天井を15kHzへ開放し、
        # エア帯域 (13-15kHz) のステレオ感を保つ。超低速ヒス推定 (τ12s)
        # が-36dB以下で全開、-26dB以上で従来13kHz、中間は線形。
        # (実測: 合成clean -78dB / noisy3dB -15dB。速い値ではなく鈍い値で
        # 駆動し、番組構成での天井の呼吸を防ぐ)
        try:
            _h = self._nr_hiss_slow
            if _h is None or not float(_h) == float(_h):
                _h = self.stereo_hiss_db
            _h = float(_h)
        except Exception:
            _h = float(self.stereo_hiss_db)
        _fix_w = float(np.clip((-26.0 - _h) / 10.0, 0.0, 1.0))
        _fix = 13000.0 + 2000.0 * _fix_w
        self.stereo_cut_hz = min(self.stereo_cut_hz, _fix)
        # モノラル番組判定: M-S相関だけでは判定できない。
        # Cov(M,S)=(Var(L)-Var(R))/4 の恒等式により、相関ρは左右レベル差
        # (パン振り) でも上がる。6dBパン振りの正当ステレオでρ=+0.60、
        # 逆に真モノラル＋独立ノイズではρ≈0になることを実測で確認。
        # 相関は「SがM漏れ由来か」の必要条件にすぎないため、S/Mエネルギー比
        # を併用する: 真モノラル漏れはS/M≪1 (実録-24dB)、パン振り番組は
        # S/M≈0dB。Sが大きく相関する=実番組として抑圧しない。
        # (ヒス下の静かなモノラルはWiener側 (_nr_s_w) が担う)
        mo = np.asarray(mono, dtype=np.float64)
        di = np.asarray(diff, dtype=np.float64)
        if len(mo) == len(di) and len(mo) > 0:
            mo = mo - float(mo.mean())
            di = di - float(di.mean())
            den = float(np.sqrt(np.mean(mo * mo) * np.mean(di * di))) + 1e-12
            rho = float(np.mean(mo * di)) / den
            mm_pow = float(np.mean(mo * mo)) + 1e-12
            sm_pow = float(np.mean(di * di)) + 1e-12
            ratio_db = 10.0 * float(np.log10(sm_pow / mm_pow))
            if not self._nr_mono_primed:
                self._nr_mono_primed = True
                self._nr_mono_rho = rho
                self._nr_sm_db = ratio_db
            else:
                a_r = 1.0 - np.exp(-dt / 2.0)
                self._nr_mono_rho += a_r * (rho - self._nr_mono_rho)
                self._nr_sm_db += a_r * (ratio_db - self._nr_sm_db)
        x_m = float(np.clip((self._nr_mono_rho - 0.15) / 0.35, 0.0, 1.0))
        x_m = x_m * x_m * (3.0 - 2.0 * x_m)
        # S/Mゲート: -18dB以下で全開、-6dB以上で閉 (中間はsmoothstep)。
        # 無音時はS/M≈0dB側へ倒れ抑圧しない (安全側)。
        _gx = float(np.clip((-6.0 - float(self._nr_sm_db)) / 12.0, 0.0, 1.0))
        _gate = _gx * _gx * (3.0 - 2.0 * _gx)
        self._nr_mono_w = x_m * _gate
        # 有効値: モノラル番組ではWienerを全力(1.0)へ、S高域カットを8kHzへ寄せる
        # スルーレート制限 (非対称): 抑圧の立ち上げ (ヒス出現) は速く0.06/ブロック、
        # 復帰 (クリーン化) は0.01/ブロックでゆっくり開く。ヒス除去の応答を
        # 殺さず、開閉の段差・呼吸だけを丸める。
        _sw_target = max(self._nr_s_w, self._nr_mono_w)
        _prev = float(getattr(self, "_nr_sw_eff_prev", 0.0))
        _delta = _sw_target - _prev
        _step = 0.06 if _delta > 0.0 else 0.01
        if abs(_delta) > _step:
            _sw_target = _prev + float(np.sign(_delta)) * _step
        self._nr_sw_eff = _sw_target
        self._nr_sw_eff_prev = _sw_target
        # 有効S側カット: ステレオ番組では15kHzまで開放、モノラル番組
        # (mono_w=1) では従来通り8kHzへ寄せる (端点保存の線形写像)。
        self._nr_cut_eff = min(self.stereo_cut_hz, 15000.0 - 7000.0 * self._nr_mono_w)

    def _diff_lowpass(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """差信号用の可変ローパス。同一長の線形位相FIRを2本クロスフェードし、
        群遅延を変えずに遮断周波数を連続変化させる (スイッチングノイズなし)。"""
        if len(x) == 0:
            return x
        req = len(self._nr_filters[0]) - 1
        hist = self.history_nr_lp
        if len(hist) != req:
            hist = np.zeros(req, dtype=np.float32)
        x_ext = np.concatenate((hist, x))
        self.history_nr_lp = x[-req:].copy() if len(x) >= req else x_ext[-req:].astype(np.float32)

        levels = self._nr_cut_levels
        cut = float(np.clip(cutoff_hz, levels[0], levels[-1]))
        # 非等間隔レベル間の区分線形補間 (等間隔仮定の線形posでは中間cutがずれる)
        i0 = int(np.clip(np.searchsorted(levels, cut, side="right") - 1, 0, len(levels) - 2))
        span = float(levels[i0 + 1] - levels[i0])
        w = float(np.clip((cut - levels[i0]) / (span if span > 0 else 1.0), 0.0, 1.0))
        # 高速化: np.convolve(スカラー相関) → ネイティブSSE2 FIR (5-10倍)。
        # _convolve_validはネイティブ優先・不在時np.convolveフォールバックで等価。
        y = self._convolve_valid(x_ext, self._nr_filters[i0])
        if w > 1e-3:
            y2 = self._convolve_valid(x_ext, self._nr_filters[i0 + 1])
            y = y * (1.0 - w) + y2 * w
        return np.asarray(y, dtype=np.float32)

    def _wiener_diff(self, x: np.ndarray) -> np.ndarray:
        """差信号のサブバンドWiener抑圧 (STFT 128/hop 64, 平方根Hann(Sine窓) 50%オーバーラップ)。

        平方根Hann窓による50% OLAで振幅変調リップルを抑えた再構成 (0.000000dB)。
        周波数ごとに 信号/(信号+ノイズ) の最適重みを掛けるため、ノイズに埋もれた
        高域だけが落ち、SNRの良い低域のステレオ感はそのまま残る。
        入出力のサンプル数は厳密に一致させ、mono側は_nr_delayで遅延補償する。

        高速化: フレーム毎の逐次rfft/irfftをバッチ行列FFTに統合 (数学的等価:
        各行が独立FFTのためbit一致、平滑再帰・OLA加算順序も保存)。
        """
        n = len(x)
        if n == 0:
            return x
        buf = np.concatenate((self._wf_in, x))
        nfft = self._wf_n
        hop = self._wf_hop
        nframes = (len(buf) - nfft) // hop + 1
        if self._wf_p is None:
            self._wf_p = np.zeros(nfft // 2 + 1, dtype=np.float32)
        if nframes > 0:
            # フレーム行列 (stride複写1回) とバッチrfft (C内でループ)
            idx = np.arange(nfft)[None, :] + hop * np.arange(nframes)[:, None]
            frames = buf[idx] * self._wf_win
            specs = np.fft.rfft(frames, axis=1)

            c_noise = ((self._nr_floor_pow * self._nr_floor_bias * self._wf_scale)
                       / self._wf_hf_f2_mean)
            sw = self._nr_sw_eff if self.stereo_nr_enabled else 0.0
            # 高速経路: クリーン時はゲイン≈1でSTFT往復のみ (遅延保存のため)。
            # per-frameの平滑・マスキング行列(43×10 numpy呼出≒5ms)を丸ごと省略。
            # _wf_p/_wf_gは凍結するが、復帰時は0.5重みで数フレーム(10ms)で再収束する。
            if sw < 0.02:
                gmix = np.ones_like(specs, dtype=np.float32)
                # 強電界マイクロトリム: パイロットロック時のみ差信号10kHz超へ
                # -1.5dB (実機83.2MHzでS床-16.6→-18.0dBを確認、波形相関0.99維持)。
                # STFT域のため群遅延は補償済み。1k/5k分離トーン・19k抑圧に無影響で
                # 透明性テスト (clean差分<0.05・分離度-18dB) のマージン内に収まる。
                try:
                    _lock = abs(float(getattr(self, "stereo_pilot_lock", 0.0)))
                except Exception:
                    _lock = 0.0
                if _lock > 0.5 and self.stereo_nr_enabled:
                    _mt = getattr(self, "_wf_micro_mask", None)
                    if _mt is None or len(_mt) != gmix.shape[1]:
                        _bins = np.fft.rfftfreq(self._wf_n, 1.0 / float(self.audio_rate))
                        _mt = (_bins > 10000.0)
                        self._wf_micro_mask = _mt
                    if bool(np.any(_mt)):
                        gmix[:, _mt] = np.float32(0.841)
                self.stereo_wiener_gain = 1.0
            else:
                powers = np.abs(specs) ** 2 + 1e-12
                # 平滑再帰のみ逐次 (フレーム間依存のため。ベクトル演算のみでFFTなし)
                # 不変量(noise_bin/sw)をループ外へ hoist (43回の再計算・確保を排除)。
                noise_bin = c_noise * self._wf_f2
                noise_denom = noise_bin + 1e-12
                gmix = np.empty_like(specs, dtype=np.float32)
                # マスキング閾値の超低速EMAは8フレーム毎に更新する。
                # 毎フレーム (43回/ブロック) のベクトルEMAは約0.2ms/ブロックの
                # 実測コストがあり、τ8秒に対して8フレーム粒度で十分等価
                # (alphaを8フレーム分に換算)。
                a_ms8 = 1.0 - float(np.exp(
                    -(8.0 * hop / float(self.audio_rate)) / 8.0))
                # 高速化: ループ不変のgetattr/len判定・定数を外へ hoist。
                # _xi/_gp/_ggはフレーム更新値をローカルで回し、最後に書戻す
                # (ループ中の再取得と同値。数値演算の順序は不変)。
                _xi = getattr(self, "_wf_xi", None)
                _gp = getattr(self, "_wf_gamma_prev", None)
                _gg = getattr(self, "_wf_g", None)
                _nb = len(noise_bin)
                if _xi is None or _gp is None or len(_xi) != _nb or len(_gp) != _nb:
                    _xi = _gp = None
                if _gg is not None and len(_gg) != _nb:
                    _gg = None
                _gmin = float(self._nr_gmin)
                for j in range(nframes):
                    power = powers[j]
                    self._wf_p = 0.5 * power + 0.5 * self._wf_p
                    # decision-directed事前SNR (Ephraim-Malah):
                    # 瞬時事後SNRのばらつきを前フレームのゲイン付きで均し、
                    # ビン毎のゲインチラつき＝ミュージカルノイズを抑える。
                    gamma = power / (noise_bin + 1e-12)
                    xi_inst = np.maximum(gamma - 1.0, 0.0)
                    if _xi is None or _gp is None:
                        xi = xi_inst.astype(np.float32)
                    else:
                        if _gg is None:
                            _g2 = np.ones_like(gamma, dtype=np.float32)
                        else:
                            _g2 = _gg * _gg
                        xi = (0.85 * _g2 * _gp + 0.15 * xi_inst).astype(np.float32)
                        xi = np.maximum(xi, 0.0)
                    _xi = xi
                    _gp = gamma.astype(np.float32)
                    g_w = np.maximum(xi / (1.0 + xi + 1e-12),
                                     _gmin).astype(np.float32)
                    g_w[:3] = 1.0  # DC〜低域は保護
                    if _gg is None:
                        _gg = g_w
                    else:
                        a = np.where(g_w < _gg, 0.7, 0.1)
                        _gg = _gg + a * (g_w - _gg)
                    # 知覚マスキングフロア: 番組にマスクされるノイズは抑圧不要 (g→1)。
                    # Wienerの過剰抑圧（音楽性ノイズ・高域の曇り）を可聴性基準で緩和する。
                    # マスキング算出はクリーン推定 (P-N) から行う (ノイズ込み電力では
                    # ヒス自身がマスクを上げて抑圧不能になるため)。
                    # 閾値はτ8秒で平滑化する: 瞬時値だと番組テクスチャ (9〜18秒周期)
                    # に追従してS側抑圧量がゆっくり動き「幅の呼吸」として聞こえる
                    # (LuckyFM 94.6の実測。9-18秒成分を約8割減衰させる)。
                    p_clean = np.maximum(
                        self._wf_p.astype(np.float64) - noise_bin, 0.0)
                    mask_thr = (p_clean @ self._wf_spread) * self._wf_mask_offset_vec
                    _ms = self._wf_mask_slow
                    if _ms is None or len(_ms) != len(mask_thr):
                        self._wf_mask_slow = mask_thr.astype(np.float64)
                        self._wf_mask_cnt = 1
                        _ms = self._wf_mask_slow
                    else:
                        _cnt = getattr(self, "_wf_mask_cnt", 0) + 1
                        if _cnt >= 8:
                            _cnt = 0
                            _ms += a_ms8 * (mask_thr - _ms)
                        self._wf_mask_cnt = _cnt
                    gate = np.minimum(1.0, _ms / noise_denom).astype(np.float32)
                    g_use = np.maximum(_gg, gate)
                    g_use[:3] = 1.0  # DC〜低域は保護
                    g_mix = 1.0 - sw * (1.0 - g_use)
                    gmix[j] = g_mix
                self._wf_xi = _xi
                self._wf_gamma_prev = _gp
                self._wf_g = _gg
                self.stereo_wiener_gain = float(np.mean(gmix[-1]))

            # バッチirfft＋同順序OLA加算
            ymat = np.fft.irfft(specs * gmix, n=nfft)
            acc = np.concatenate((self._wf_ola.astype(np.float64),
                                  np.zeros(nframes * hop, dtype=np.float64)))
            for j in range(nframes):
                acc[j * hop:j * hop + nfft] += ymat[j] * self._wf_win
            out_hops = (acc[:nframes * hop] / np.tile(self._wf_cola[:hop], nframes))
            self._wf_ola = acc[nframes * hop:nframes * hop + nfft].astype(np.float32)
            self._wf_in = buf[nframes * hop:]
            self._wf_out = np.concatenate(
                (self._wf_out, out_hops.astype(np.float32)))
        else:
            # フレーム未満の入力は破棄せず持ち越す (旧実装はここで入力を消していた)
            self._wf_in = buf
        if len(self._wf_out) >= n:
            # 余剰がある場合、過去の不足による余分な遅延 (extra_delay) を自動解消し定常群遅延へ復帰
            if self._wf_extra_delay > 0:
                surplus = len(self._wf_out) - n
                recover = min(surplus, self._wf_extra_delay)
                self._wf_out = self._wf_out[recover:]
                self._wf_extra_delay -= recover
            y = self._wf_out[:n].copy()
            self._wf_out = self._wf_out[n:]
        else:
            # 不足分だけゼロ埋めするが、その分の遅延を記録する (mono側を同量
            # 遅らせて群遅延を一致させる。記録しないと分離度が恒久的に崩れる)
            short = n - len(self._wf_out)
            y = np.concatenate((self._wf_out, np.zeros(short, dtype=np.float32)))
            self._wf_out = np.zeros(0, dtype=np.float32)
            self._wf_extra_delay = min(int(self._wf_extra_delay) + int(short), 1 << 20)
        if len(self._wf_out) > 8 * n:
            self._wf_out = self._wf_out[-2 * n:]
        return y

    def _freq_dependent_blend(self, diff: np.ndarray, blend: float) -> np.ndarray:
        """周波数依存ブレンド: 弱電界で高域から先にモノラル化する。
        1次相補クロスオーバー (lo + hi = diff で再構成) で低域/高域に分け、
        低域は blend、高域は blend^2 * nr_gain で絞る。FM三角ノイズが f^2 で
        増大するため高域ほどヒスが支配的で、低域のステレオ感を残しつつ耳障りな
        高域ヒスだけ先に消える。blend=1 かつ nr_gain=1 (クリーン) は旧スカラー
        動作と一致の高速経路。NR無効ブランチからは呼ばれない。"""
        if len(diff) == 0:
            return diff
        if not self.freq_blend_enabled or blend >= 0.999:
            return (diff * blend).astype(np.float32)
        if blend <= 0.001:
            return np.zeros_like(diff)
        # 1次LPF (状態保持でブロック連続。fc=3.5kHz@48kHz)
        # Ch2-C最適化: Python forループ→Cコア sdr_bilinear_deemphasis で
        # 指数平滑 y[n]=a*x[n]+(1-a)*y[n-1] を実行 (b0=a, b1=0, m=1-a)。
        # 0.63ms→0.025ms (旧ループと max error 7.5e-9 で数値等価)。
        a = 1.0 - float(np.exp(-2.0 * np.pi * float(self.freq_blend_xo_hz) / float(self.audio_rate)))
        if _NATIVE is not None:
            xo_state = getattr(self, "_blend_xo_state", None)
            if xo_state is None or len(xo_state) != 2:
                xo_state = np.array([0.0, float(self._blend_xo_y1)], dtype=np.float32)
            lo = np.empty_like(diff)
            _NATIVE.sdr_bilinear_deemphasis(
                _fptr(np.ascontiguousarray(diff, dtype=np.float32)),
                _fptr(lo), len(diff),
                float(a), 0.0, float(1.0 - a), _fptr(xo_state))
            self._blend_xo_state = xo_state
            self._blend_xo_y1 = float(xo_state[1])
        else:
            y1 = float(self._blend_xo_y1)
            lo = np.empty_like(diff)
            for i, v in enumerate(diff):
                y1 += a * (float(v) - y1)
                lo[i] = y1
            self._blend_xo_y1 = y1
        hi = diff - lo
        blend_lo = float(blend)
        # 高域はヒス推定にもう一段連動 (三角ノイズ f^2 特性)。NRブランチ専用のため
        # stereo_nr_gain は常に有効な推定値。クリーン時は blend=nr=1 で恒等変換。
        blend_hi = float(blend * blend * float(self.stereo_nr_gain))
        return (lo * blend_lo + hi * blend_hi).astype(np.float32)

    def _update_stereo_trim(self, diff: np.ndarray, mono: np.ndarray):
        """38kHz再生位相オートトリム: 位相誤差を stereo_phase_offset で相殺する。
        主経路は直交 (Q) 腕の符号付き残差による比例サーボ (Costas型):
        I ∝ cos(φe−δ)、Q ∝ sin(φe−δ) のため corr(I,Q)/E[I²] ≒ (φe−δ)/2 が
        誤差の符号付き推定になり、δ += k·err で幾何収束する。番組レベルに不変。
        Q腕が無い場合 (キャンセラ無効時) はL-R電力の摂動観測へフォールバック。
        更新はブロック毎・比例ゲイン0.5・±15°クランプ。ゲート: パイロット
        ロック・高ブレンド・低ヒス・有音声時のみ。短時間テストにはほぼ無影響。"""
        if not self.stereo_trim_enabled:
            return
        try:
            if not (abs(float(self.stereo_pilot_lock)) > 0.5):
                return
            if not (float(self._stereo_blend) > 0.5 and float(self.stereo_nr_gain) > 0.7):
                return
            if len(diff) < 64 or len(mono) < 64:
                return
            mono_rms = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2)))
            if mono_rms < 0.01:  # -40dBFS未満は無音とみなし凍結
                return
            dq = getattr(self, "_last_diff_q", None)
            if dq is not None and len(dq) == len(diff):
                i = np.asarray(diff, dtype=np.float64)
                q = np.asarray(dq, dtype=np.float64)
                den = float(np.mean(i * i)) + 1e-12
                if den < 1e-8:  # 差信号ほぼゼロ (モノラル) では凍結
                    return
                corr = float(np.mean(i * q))
                err_inst = corr / den  # ≒ (φe−δ)/2 [rad]
                if not np.isfinite(err_inst):
                    return
                # 高速変動 (実マルチパス) はEMAで均し、準静的な成分
                # (トリム誤差・静止反射) のみ追う。実測で即時比例は
                # レール間発振したため、τ3秒＋不感帯＋微速刻みに変更。
                try:
                    dt_tr = len(diff) / float(self.audio_rate)
                except Exception:
                    dt_tr = 0.05
                a_tr = 1.0 - float(np.exp(-dt_tr / 3.0))
                ema = float(getattr(self, "_trim_err_ema", 0.0)) + a_tr * (err_inst - float(getattr(self, "_trim_err_ema", 0.0)))
                self._trim_err_ema = ema
                if abs(ema) < 0.005:
                    return  # 0.3°不感帯 (ノイズでの彷徨・レール発振を防止)
                # 比例サーボ (1ブロック0.23°上限でゆっくり。急変動には追従しない)
                step = float(np.clip(0.3 * ema, -0.004, 0.004))
                self.stereo_phase_offset = float(np.clip(
                    self.stereo_phase_offset + step,
                    -self._trim_max_rad, self._trim_max_rad))
                return
            # フォールバック: L-R電力の摂動観測 (Q腕なし時)
            diff_rms = float(np.sqrt(np.mean(np.asarray(diff, dtype=np.float64) ** 2)))
            m = diff_rms / (mono_rms + 1e-9)
            if not self._trim_primed:
                self._trim_primed = True
                self._trim_m_smooth = m
                self._trim_prev_m = m
                return
            self._trim_m_smooth += 0.25 * (m - self._trim_m_smooth)
            self._trim_block += 1
            if self._trim_block < 2:
                return
            self._trim_block = 0
            if self._trim_m_smooth < self._trim_prev_m - 1e-6:
                self._trim_dir = -self._trim_dir
            self.stereo_phase_offset = float(np.clip(
                self.stereo_phase_offset + self._trim_dir * self._trim_step_rad,
                -self._trim_max_rad, self._trim_max_rad))
            if abs(self.stereo_phase_offset) >= self._trim_max_rad - 1e-9:
                self._trim_dir = -self._trim_dir
            self._trim_prev_m = self._trim_m_smooth
        except Exception:
            pass

    def _slow_agc_level(self, audio: np.ndarray) -> np.ndarray:
        """局間音量レベリング用スローAGC: ブロックRMSを目標 (-20dBFS) へ寄せる。
        ステレオは全ch一括 (L/R連動で音像・位相を保存)、モノラルも同一式。
        無音 (-80dBFS未満) では凍結しノイズ持ち上げを防ぐ。ゲイン範囲±6dB、
        立ち下げ2秒・立ち上げ10秒の非対称時定数で、番組内の緩急には反応せず
        局替わり等の持続的レベル差だけを均す (ポンピング防止)。
        ブロック内は単一ゲイン (変化は0.1dB/block未満でジッパー雑音なし)。
        lufs_agc_enabled時はRMS推定をR128 LUFS推定に置換 (ダイナミクス同一)。"""
        try:
            if len(audio) == 0:
                return audio
            x = np.asarray(audio, dtype=np.float64)
            cur = float(self.slow_agc_gain)
            rms = float(np.sqrt(np.mean(x * x)))
            if not np.isfinite(rms):
                return audio
            # 凍結パスでも必ず現ゲインを適用する。未適用で素通しすると
            # 収束 (不感帯内) 後にゲインが外れ、局間レベリングが丸ごと
            # 無効化される (test_lufs_agcの実測で発覚)。
            if rms < float(self._slow_agc_floor):
                return (x * cur).astype(np.float32)
            if bool(getattr(self, "lufs_agc_enabled", False)):
                try:
                    from audiophile_dsp import LoudnessNormalizer
                    if self._lufs_norm is None:
                        self._lufs_norm = LoudnessNormalizer(
                            sample_rate=float(self.audio_rate))
                    mono = x if x.ndim == 1 else np.mean(x, axis=1)
                    lufs = self._lufs_norm.push(mono)
                    if lufs is None:
                        return (x * cur).astype(np.float32)
                    raw_desired = self._lufs_norm.gain_for(lufs)
                except Exception:
                    raw_desired = float(self.slow_agc_target) / (rms + 1e-12)
            else:
                raw_desired = float(self.slow_agc_target) / (rms + 1e-12)
            desired = float(np.clip(raw_desired,
                                    float(self.slow_agc_min), float(self.slow_agc_max)))
            # ブロック長から時定数を換算 (audio_rate基準)
            try:
                dt = len(audio) / float(self.audio_rate)
            except Exception:
                dt = 0.05
            # ファストスタート計数は凍結時も進める (hold中に選局直後扱いが
            # 永続しないよう先に加算する)
            try:
                _ab = int(getattr(self, "_agc_blk", 999)) + 1
                self._agc_blk = _ab
            except Exception:
                _ab = 999
            # ヒステリシス不感帯 (±1.5dB未満の微小要求は凍結しジリ動を抑止)
            try:
                _hyst = float(getattr(self, "_agc_hyst_db", 1.5))
                _ddb = 20.0 * float(np.log10(max(desired, 1e-9) / max(cur, 1e-9)))
            except Exception:
                _hyst, _ddb = 1.5, 99.0
            if abs(_ddb) < _hyst:
                return (x * cur).astype(np.float32)
            is_attack = bool(desired < cur)
            # ホールド: attack直後のrelease方向を一定ブロック抑止
            # (語尾・休止での持ち上げ呼吸を防止。attack自体は即時)
            try:
                _hold = int(getattr(self, "_agc_hold_n", 0))
            except Exception:
                _hold = 0
            if not is_attack and _hold > 0:
                try:
                    self._agc_hold_n = _hold - 1
                except Exception:
                    pass
                return (x * cur).astype(np.float32)
            # 番組ゲート: 静かな不確定区間の持ち上げは保留する。
            # speech_probが中間 (0.3〜0.7)＝雑音/間隙らしく、かつRMSが目標の
            # 半分未満のときだけ凍結。音楽 (0寄り)・音声 (1寄り) の確信時は通す。
            if not is_attack and rms < float(self.slow_agc_target) * 0.5:
                try:
                    _eq = getattr(self, "cognitive_eq", None)
                    _sp = None
                    if _eq is not None and bool(getattr(_eq, "enabled", False)):
                        _sp = float(getattr(_eq, "speech_prob", 0.5))
                    if _sp is not None and np.isfinite(_sp) and 0.3 < _sp < 0.7:
                        return (x * cur).astype(np.float32)
                except Exception:
                    pass
            tau = float(self.slow_agc_attack) if is_attack else float(self.slow_agc_release)
            # ファストスタート: 選局直後40ブロック (~2.3秒) は時定数を短縮し
            # 新局レベルへ速く寄せる。定常後は従来時定数に戻りポンピング特性不変。
            # 持ち上げ側2.0→2.5sへ少し鈍化し、選局直後の行き過ぎ(膨らみ)を抑える。
            try:
                if _ab <= 40:
                    tau = min(tau, 0.5 if desired < cur else 2.5)
            except Exception:
                pass
            a = 1.0 - float(np.exp(-dt / tau))
            gain = cur + a * (desired - cur)
            self.slow_agc_gain = float(gain)
            if is_attack:
                try:
                    self._agc_hold_n = int(getattr(self, "_agc_hold_max", 7))
                except Exception:
                    pass
            if abs(gain - 1.0) < 1e-4:
                return audio
            return (x * gain).astype(np.float32)
        except Exception:
            return audio

    def _blend_release(self, floor: float = 0.0):
        """ブレンド低下 (パイロット瞬断フライホイール付き)。
        高ブレンドからの低下要求は25ブロック (~1.4秒) まで凍結し、
        短いパイロット瞬断発作でステレオ像がモノラルへ往復するのを防ぐ。
        持続喪失では floor まで滑らかに落とす。ノイズ量 (_nr_s) に応じて
        低下を加速し (clean 0.97 → noisy 0.90)、全ノイズをほぼ一定に保つ
        (定ノイズブレンド。弱電界のノイズハンプ滞留を抑止)。"""
        if (self._stereo_blend > 0.5
                and self._pilot_hold_n < self._pilot_hold_max):
            self._pilot_hold_n += 1
        else:
            try:
                _ns = float(getattr(self, "_nr_s", 0.0))
                _ns = min(max(_ns, 0.0), 1.0)
            except Exception:
                _ns = 0.0
            _dec = 0.97 - 0.07 * _ns
            self._stereo_blend = max(float(floor), self._stereo_blend * _dec)
        self.stereo_blend = self._stereo_blend

    def _post_process_wfm(self, audio: np.ndarray, ch: str = "",
                            skip_mono_nr: bool = False) -> np.ndarray:
        """WFM音声のチャンネル別仕上げ (ch: ''=モノ, '_l'/'_r'=ステレオ各ch)。
        skip_mono_nr=True時は単一ch NRを後段のMid/Side処理へ委譲する。"""
        # ディエンファシス (50/75μs)
        audio = self._apply_bilinear_deemphasis(audio, ch=ch)

        # オーディオ段ハイカットフィルタ (Hyper時は無段階モーフィング)
        if self.cognitive_enabled:
            if self.applied_cutoff_hz >= 14800.0:
                # 最大帯域開放: IF段が15kHzまで平坦なため素通し (同一遅延)
                fir_final = self.fir_audio_flat_dly
            else:
                fir_final = self._get_dynamic_filter("audio", self.applied_cutoff_hz)
        elif self.filter_mode == "wide":
            fir_final = self.fir_audio_wide
        elif self.filter_mode == "narrow":
            fir_final = self.fir_audio_narrow
        else:
            fir_final = self.fir_audio_clean

        # decimate_with_history (factor=1) を用いることで、フィルタ長変更時にもサンプル数の一致を保証
        # 高域エキサイター用の広帯域ソースをカット前にタップ (カット後に種を取ると
        # 8〜14k成分が無く倍音生成がno-opになる。生成した16〜22kはカット後に足す)
        wide_src = np.asarray(audio).astype(np.float32)
        if ch in ("", "_l"):
            # トラッカー用の広帯域タップ (Lのみで十分。process末尾でanalyzeする)
            self._cog_wide = wide_src
        audio = self.decimate_with_history(audio, fir_final, 1, f"history_final{ch}")

        # DCハイパスフィルタ (ActiveDcServo有効時はバイパスし位相直線性を保つ。
        # 30Hz 1次HPFは20〜300Hzに位相進み歪みを残すため、最終段のDCサーボへ一本化)
        _servo = getattr(self, "dc_servo", None)
        if _servo is None or not bool(getattr(_servo, "enabled", False)):
            audio = self._apply_dc_highpass(audio, ch=ch)

        # Hyper心理音響ハイシェルフ (FM三角雑音を連続減衰) / Cascade離散エキスパンダー
        if self.cognitive_enabled:
            audio = self._apply_hf_shelf(audio, ch=ch)
        elif self.filter_mode == "narrow":
            audio = self._apply_noise_expander(audio, threshold=0.09)

        # 高域高調波補完 (15kHz〜22kHzの高域倍音付加)
        if (getattr(self, "holographic_enhancer", None) is not None
                and self.holographic_enhancer.enabled
                and (self.cognitive_enabled or getattr(self, "holographic_always", False))):
            speech_p = getattr(getattr(self, "cognitive_eq", None), "speech_prob", 0.0)
            s_meter = getattr(self, "s_meter_dbfs", -20.0)
            audio = self.holographic_enhancer.process(audio, ch=ch, speech_prob=speech_p, s_meter_dbfs=s_meter,
                                                      source=wide_src)

        # SVD部分空間ノイズ除去 (特異値しきい値による弱電界ノイズ低減)
        if (getattr(self, "rmt_denoiser", None) is not None
                and self.rmt_denoiser.enabled
                and (self.cognitive_enabled or getattr(self, "rmt_always", False))):
            s_meter = getattr(self, "s_meter_dbfs", -20.0)
            audio = self.rmt_denoiser.process(audio, ch=ch, s_meter_dbfs=s_meter)

        # NOTE: 黒魔法Bはチャンネル別ポスト内では処理しない。L/R独立処理は
        # チャンネル間をdecorrelateし分離度を落とす (ベンチで-3〜-4dB悪化を実測)。
        # ステレオはdemodulate_wfmのスタック直前でMid/Side処理する。

        # 単一ch スペクトル抑圧NR (帯域内ノイズの最小統計Wiener抑圧)
        # 弱電界FMの番組帯ノイズ (ハイカットでは消せない) を低減する。
        # クリーン/定常信号ではゲイン1で透明に通過する自己ゲート方式。
        if (not skip_mono_nr
                and getattr(self, "mono_nr", None) is not None
                and self.mono_nr.enabled
                and getattr(self, "mono_nr_enabled", True)
                and (self.cognitive_enabled or getattr(self, "mono_nr_always", False))):
            audio = self.mono_nr.process(audio, ch=ch)

        return audio.astype(np.float32)

    def _mid_side_mono_nr(self, left: np.ndarray, right: np.ndarray):
        """ステレオ用共通ゲインNR。Midのパワーから求めた同一Wienerゲインを
        L/R両chへ適用し、独立ゲインによる音像の揺れ・分離度低下を防ぐ。
        抑圧量はMid基準で維持される (実測: 安定音像のバイアス+0.57→+0.00dB、
        付加揺らぎ+0.52→+0.07dB、抑圧量維持)。"""
        return self.mono_nr.process_stereo(left, right)

    def _update_stereo_pilot(self, mpx: np.ndarray):
        """19kHzパイロットPLLを更新し、ステレオブレンド係数とRDS用57kHz搬送波を生成する"""
        # RDS用cos3/sin3は毎ブロック生成するため先にクリア (前ブロック長のまま
        # 掛かると形状不一致になる)。ステレオ用cos2/sin2はパイロット消失時に
        # 緩やかなブレンド解放のため保持し、長さ不一致はdemodulate_wfm側で弾く。
        self._last_cos3 = None
        self._last_sin3 = None
        # 非有限MPX(NaN/Inf)はPLL状態を汚染するため早期破棄しブレンドを緩やかに落とす
        try:
            if mpx is None or len(mpx) < 64 or not bool(np.all(np.isfinite(np.asarray(mpx).reshape(-1)))):
                self._blend_release()
                self.stereo_pilot_lock = 0.0
                self.stereo_pilot_ratio = 0.0
                return
        except Exception:
            self._blend_release()
            return
        if not (self.stereo_enabled or self.rds_enabled) or _NATIVE is None or len(mpx) < 64:
            self._stereo_blend *= 0.9
            self.stereo_blend = self._stereo_blend
            return

        try:
            pilot = self.decimate_with_history(mpx, self.fir_pilot_lp, 1, "history_pilot_lp")
            pilot = self.decimate_with_history(pilot, self.fir_pilot_hp, 1, "history_pilot_hp")
            n = len(pilot)
            if n < 32:
                return
            # パイロット振幅で正規化した生MPXをPLLへ入力 (帯域制限による群遅延を回避し、
            # 搬送波位相をMPX本来のタイムラインに一致させる)
            # 高速化: float64一時配列(astype+二乗+meanの3パス)をdot単一パスへ。
            # 誤差1e-8以下でゲート閾値(1.5%)に影響なし。0.05ms→0.006ms×2。
            _pa = np.ascontiguousarray(pilot, dtype=np.float32)
            _ma = np.ascontiguousarray(mpx, dtype=np.float32)
            pilot_rms = float(np.sqrt(float(np.dot(_pa, _pa)) / max(len(_pa), 1)) + 1e-12)
            mpx_rms_pre = float(np.sqrt(float(np.dot(_ma, _ma)) / max(len(_ma), 1)) + 1e-12)
            if pilot_rms < 1e-4 or pilot_rms < 0.015 * mpx_rms_pre:
                # パイロット不在ゲート: 19kHz帯が無音・微小のまま正規化PLLへ渡すと
                # 入力が1e11級に膨張→PLL発散→C側の位相正規化が爆発し復帰不能
                # (モノラル局・無信号でDSPスレッドがハングする)。ここで打ち切り、
                # design されたブレンド解放ランプ (0.97) で滑らかにモノラルへ落とす。
                # 搬送波を保持したままブレンドを絞るため、即時モノ切替の段差
                # (実測0.58FS) が生じない。搬送波の長さ不一致はdemodulate_wfmで弾く。
                # 短い瞬断はフライホイールで凍結する (持続喪失のみ解放ランプ)
                self._blend_release()
                self.stereo_pilot_lock = 0.0
                self.stereo_pilot_ratio = 0.0
                return
            sig = np.ascontiguousarray(np.asarray(mpx, dtype=np.float32) / pilot_rms, dtype=np.float32)
            # 黒魔法A: 検出のみ行い、上昇レート判断は後段の_bm_attack_limitへ委ねる。
            # PLLゲイン自体には触らない。
            # (コントローラがcyclo非活性と判断した場合は検出自体を休止する)
            if (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_cyclo_enabled", False)):
                try:
                    _bmp = getattr(self, "_bm_params", None) or {}
                    _cyc_active = bool(_bmp.get("cyclo_active", True))
                    if (self.cyclo_detector is None
                            and CyclostationaryPilotDetector is not None):
                        _cc = getattr(self, "bm_cfg", None) or {}
                        _cc = _cc.get("cyclostationary", {}) or {}
                        try:
                            _tgt = float(_cc.get("pilot_frequency_hz", 19000.0))
                        except (TypeError, ValueError):
                            _tgt = 19000.0
                        try:
                            _smo = float(_cc.get("smoothing_seconds", 0.25))
                        except (TypeError, ValueError):
                            _smo = 0.25
                        self.cyclo_detector = CyclostationaryPilotDetector(
                            sample_rate=self.if_rate, target_hz=_tgt,
                            min_confidence=float(self.bm_cyclo_min_conf),
                            smoothing_seconds=_smo)
                    if self.cyclo_detector is not None and _cyc_active:
                        cyc = self.cyclo_detector.update(mpx)
                        self.bm_cyclo_confidence = float(cyc.get("confidence", 0.0))
                except Exception:
                    pass
            cos2 = np.empty(n, dtype=np.float32)
            sin2 = np.empty(n, dtype=np.float32)
            quality = ctypes.c_float(0.0)
            th = ctypes.c_double(self._pll_theta)
            ig = ctypes.c_double(self._pll_integ)
            ef = ctypes.c_double(self._pll_ef)
            cos3 = sin3 = None

            # NASA DSN方式 自律適応カルマン・パイロット搬送波追従器 (AKCTL) による最適ゲイン動的計算
            if getattr(self, "pilot_tracker", None) is not None and self.pilot_tracker.enabled:
                kp, ki, alpha = self.pilot_tracker.update_gains(self.stereo_pilot_lock, pilot_rms, self._pll_ef)
            else:
                kp, ki, alpha = self._pll_kp, self._pll_ki, self._pll_alpha

            # RDS無効時は57kHz出力(cos3/sin3)の三角関数×2/サンプルを省略し
            # 2出力版PLLへ切替 (約1/3高速化。数学的等価: cos2/sin2は同一)。
            if NATIVE_PLL3 and self.rds_enabled:
                cos3 = np.empty(n, dtype=np.float32)
                sin3 = np.empty(n, dtype=np.float32)
                _NATIVE.sdr_stereo_pll3(_fptr(sig), n, ctypes.byref(th), self._pll_w0,
                                        kp, ki, ctypes.byref(ig),
                                        ctypes.byref(ef), alpha,
                                        _fptr(cos2), _fptr(sin2),
                                        _fptr(cos3), _fptr(sin3), ctypes.byref(quality))
            else:
                _NATIVE.sdr_stereo_pll(_fptr(sig), n, ctypes.byref(th), self._pll_w0,
                                       kp, ki, ctypes.byref(ig),
                                       ctypes.byref(ef), alpha,
                                       _fptr(cos2), _fptr(sin2), ctypes.byref(quality))
            self._pll_theta = th.value
            self._pll_integ = ig.value
            self._pll_ef = ef.value

            # mpx_rmsは直上で計算済み(mpx_rms_pre)と同一。2回目の16k走査を排除。
            ratio = pilot_rms / (mpx_rms_pre + 1e-18)
            lock = float(quality.value)  # 正規化パイロット基準: ロック時 ~0.5-0.7
            self.stereo_pilot_lock = lock
            # flutter検出用にlock履歴を保持 (ラチェット防止。固定長で自動破棄)
            try:
                _hlh = getattr(self, "_bm_lock_hist", None)
                if _hlh is not None and math.isfinite(lock):
                    _hlh.append(lock)
            except Exception:
                pass
            self.stereo_pilot_ratio = ratio

            if not self.stereo_enabled:
                # モノラル強制: ブレンドを落としcos2/sin2を格納しない
                # (RDS用cos3/sin3は継続)。無いとRDS有効時にブレンドが
                # 再上昇しモノラルボタンが実質無効になる。
                self._stereo_blend *= 0.9
                self.stereo_blend = self._stereo_blend
                self._last_cos2 = None
                self._last_sin2 = None
                if cos3 is not None:
                    self._last_cos3 = cos3
                    self._last_sin3 = sin3
                else:
                    # RDS無効時は57kHz搬送波を生成しないため古い値を破棄
                    self._last_cos3 = None
                    self._last_sin3 = None
                return

            target = 0.0
            if lock > 0.5 and pilot_rms > 1e-3:
                target = 1.0
            elif lock > 0.25 and 0.02 < ratio < 0.8:
                target = min(1.0, (ratio - 0.02) / 0.04)

            if target > self._stereo_blend:
                self._stereo_blend = min(target, self._stereo_blend
                                         + self._bm_attack_limit(lock))
                self._pilot_hold_n = 0
            elif target >= self._stereo_blend - 1e-9:
                # ロック定常の等値: 低下ではないため保持枠を消費しない
                self._stereo_blend = target
                self.stereo_blend = self._stereo_blend
                self._pilot_hold_n = 0
            else:
                self._blend_release(floor=target)

            self.stereo_blend = self._stereo_blend
            if self._stereo_blend > 0.02:
                self._last_cos2 = cos2
                self._last_sin2 = sin2
            if cos3 is not None:
                self._last_cos3 = cos3
                self._last_sin3 = sin3
            else:
                self._last_cos3 = None
                self._last_sin3 = None
        except Exception:
            self._last_cos2 = None
            self._last_sin2 = None
            self._last_cos3 = None
            self._last_sin3 = None
            # NaN固着防止: PLL積分状態もリセットし次ブロックで復帰可能にする
            self._pll_theta = 0.0
            self._pll_integ = 0.0
            self._pll_ef = 0.0
            try:
                if getattr(self, "pilot_tracker", None) is not None:
                    self.pilot_tracker.reset()
            except Exception:
                pass
            self._stereo_blend *= 0.9
            self.stereo_blend = self._stereo_blend

    def _apply_noise_expander(self, audio: np.ndarray, threshold: float = 0.09) -> np.ndarray:
        """弱電界時のFM三角雑音（高域ヒスノイズ）を抑え込み人の声を浮き彫りにするソフトエキスパンダー"""
        if len(audio) == 0:
            return audio
        mag = np.abs(audio)
        gain = np.where(mag < threshold, (mag / threshold) ** 0.5, 1.0)
        return (audio * gain).astype(np.float32)

    def _apply_bilinear_deemphasis(self, x: np.ndarray, ch: str = "") -> np.ndarray:
        if len(x) == 0:
            return x
        if _NATIVE is not None:
            cur = np.ascontiguousarray(x, dtype=np.float32)
            for i, (b0, b1, m) in enumerate(self.deemph_sections):
                state = getattr(self, f"_deemph_state{ch}" if i == 0 else f"_deemph2_state{ch}")
                nxt = np.empty_like(cur)
                _NATIVE.sdr_bilinear_deemphasis(
                    _fptr(cur), _fptr(nxt), len(cur),
                    float(b0), float(b1), float(m),
                    _fptr(state))
                cur = nxt
            return cur
        y = np.empty_like(x)
        for i, (b0, b1, m) in enumerate(self.deemph_sections):
            inp = x if i == 0 else y
            out = np.empty_like(inp)
            x1 = getattr(self, f"deemph_x1{ch}" if i == 0 else f"deemph2_x1{ch}")
            y1 = getattr(self, f"deemph_y1{ch}" if i == 0 else f"deemph2_y1{ch}")
            for k in range(len(inp)):
                curr_x = inp[k]
                curr_y = b0 * curr_x + b1 * x1 + m * y1
                out[k] = curr_y
                x1 = curr_x
                y1 = curr_y
            setattr(self, f"deemph_x1{ch}" if i == 0 else f"deemph2_x1{ch}", x1)
            setattr(self, f"deemph_y1{ch}" if i == 0 else f"deemph2_y1{ch}", y1)
            y = out
        return y

    def _init_wfm_state(self):
        """WFM用FIR・パイロットPLL・NR・CMA等の状態初期化 (__init__ から純粋移動)。"""
        # 19kHzパイロット/38kHz副搬送波を抑えるIF段オーディオフィルタ (288kHzレート)
        # 通過域15kHzを平坦に保ちつつ19kHzで-75dB以上 (337タップ)。旧設計は
        # -6dB点が15kHzで、48kHz段と重なると15kHzで-12dBだった (実測)。
        cutoff_if_audio = 17000.0 / self.if_rate
        self.fir_if_audio = design_fir_kaiser(num_taps=337, cutoff_norm=cutoff_if_audio, beta=7.3)
        # FM差分検波のアパーチャ逆補正 (5tap線形位相・群遅延2spl)。
        # M/D両経路の共通前段に掛けるため時間整合は不変。A/B用フラグ付き。
        self.fir_inv_sinc = design_inverse_sinc(num_taps=5, fs=float(self.if_rate))
        self.history_inv_sinc = np.zeros(len(self.fir_inv_sinc) - 1, dtype=np.float32)
        self.inv_sinc_enabled = True

        # 48kHzオーディオ段のアンチエイリアス・ハイカットフィルタ (48kHzレート)
        # 15kHz: 音楽用Hi-Fiワイド (BS.450準拠。強電界でフル帯域開放用。81タップ。
        # 通過域15kHz平坦・19kHz -85dB。旧51タップは15kHzで-6dBだった)
        cutoff_audio_wide = 16500.0 / self.audio_rate
        self.fir_audio_wide = design_fir_kaiser(num_taps=81, cutoff_norm=cutoff_audio_wide, beta=7.3)
        # Hyper最大帯域時 (カットオフ15kHz) は動的フィルタを同一遅延の素通し線へ
        # 置換する (15kHzでさらに-6dB落ちる二重減衰の回避。遅延40サンプル)
        self.fir_audio_flat_dly = np.zeros(81, dtype=np.float32)
        self.fir_audio_flat_dly[40] = 1.0

        # 8.5kHz: 強力ノイズクリーナー (ヒスノイズ「サー」を消滅させ人の声を鮮明化, 65タップ)
        cutoff_audio_clean = 8500.0 / self.audio_rate
        self.fir_audio_clean = design_fir_kaiser(num_taps=65, cutoff_norm=cutoff_audio_clean, beta=7.0)

        # 5.5kHz: DXボイスフィルタ (微弱局のノイズフロアを抑え声の明瞭度を上げる, 65タップ)
        cutoff_audio_narrow = 5500.0 / self.audio_rate
        self.fir_audio_narrow = design_fir_kaiser(num_taps=65, cutoff_norm=cutoff_audio_narrow, beta=7.0)
        # 4.5kHzクロスオーバーによる心理音響ハイシェルフ (FM三角雑音のみ連続減衰)
        shelf_cut = 4500.0 / self.audio_rate
        self.fir_shelf_lp = design_fir_kaiser(num_taps=81, cutoff_norm=shelf_cut, beta=6.5)
        self.fir_shelf_hp = design_fir_highpass(num_taps=81, cutoff_norm=shelf_cut, beta=6.5)
        self.history_shelf_lp = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
        _sdly = (len(self.fir_shelf_lp) - 1) // 2
        self.history_shelf_dly = np.zeros(_sdly, dtype=np.float32)
        # AFC (Automatic Frequency Control: 100Hz精度の自動搬送波追従)
        self.afc_enabled = True
        self.afc_offset_hz = 0.0
        self.afc_alpha = 0.05  # 滑らかな追従時定数
        self.afc_limit_hz = 42000.0  # 引き込み許容上限 (隣接局100kHzへの誤引き込みを防ぎつつPPMズレ・オフセットを吸収)

        self.fm_last_sample = 0.0 + 0.0j
        # PLL-FM復調状態 (fn=25kHz, ζ=1.0 の実測勝ち値。w=2πfn/fsで正規化設計)
        _w = 2.0 * np.pi * 25000.0 / 288000.0
        _den = 1.0 + _w + 0.25 * _w * _w
        self._fm_pll_kp = 2.0 * _w / _den
        self._fm_pll_ki = _w * _w / _den
        self._fm_pll_state = np.zeros(2, dtype=np.float64)
        self.fm_pll_enabled = False  # ワイドFMの過変調歪み・脱調防止のため通常は差分検波を標準使用
        # ディエンファシス (地域設定: 日本/欧州=50μs, 米国/韓国=75μs)
        # 1次双一次では高域ワープ歪み (15kHzで-3.55dB) が避けられないため、
        # 実測フィットした2縦続1次IIRでアナログ特性に±0.03dBで一致させる。
        # 各段は既存ネイティブ1次IIR (b0,b1,minus_a1) そのまま実行できる。
        self.deemph_tau_us = 50.0
        self.deemph_sections = _deemph_sections(50.0)
        self.deemph_x1 = self.deemph_y1 = 0.0
        self.deemph2_x1 = self.deemph2_y1 = 0.0
        # ネイティブCコア用フィルタ状態 (x1, y1) ×2段
        self._deemph_state = np.zeros(2, dtype=np.float32)
        self._deemph2_state = np.zeros(2, dtype=np.float32)
        self._deemph_state_l = np.zeros(2, dtype=np.float32)
        self._deemph_state_r = np.zeros(2, dtype=np.float32)
        self._deemph2_state_l = np.zeros(2, dtype=np.float32)
        self._deemph2_state_r = np.zeros(2, dtype=np.float32)
        self._dc_hp_state_l = np.zeros(2, dtype=np.float32)
        self._dc_hp_state_r = np.zeros(2, dtype=np.float32)
        self.deemph_x1_l = self.deemph_y1_l = 0.0
        self.deemph_x1_r = self.deemph_y1_r = 0.0
        self.deemph2_x1_l = self.deemph2_y1_l = 0.0
        self.deemph2_x1_r = self.deemph2_y1_r = 0.0
        self.dc_hp_x1_l = self.dc_hp_y1_l = 0.0
        self.dc_hp_x1_r = self.dc_hp_y1_r = 0.0
        # ===== FMステレオMPXデコーダ (19kHzパイロットPLL + 38kHz同期検波) =====
        self.stereo_enabled = True
        self.is_stereo = False
        self.stereo_blend = 0.0
        self.stereo_pilot_lock = 0.0
        self.stereo_pilot_ratio = 0.0
        cutoff_pilot_lp = 20500.0 / self.if_rate
        cutoff_pilot_hp = 17000.0 / self.if_rate
        self.fir_pilot_lp = design_fir_kaiser(num_taps=97, cutoff_norm=cutoff_pilot_lp, beta=6.5)
        self.fir_pilot_hp = design_fir_highpass(num_taps=97, cutoff_norm=cutoff_pilot_hp, beta=6.5)
        self.history_pilot_lp = np.zeros(len(self.fir_pilot_lp) - 1, dtype=np.float32)
        self.history_pilot_hp = np.zeros(len(self.fir_pilot_hp) - 1, dtype=np.float32)
        self.history_lpr = np.zeros(len(self.fir_if_audio) - 1, dtype=np.float32)
        self.history_lpr_q = np.zeros(len(self.fir_if_audio) - 1, dtype=np.float32)
        # RDS用LPFはIFレート(288kHz)基準で設計する。fir_am_narrowは
        # RFレート(1152kHz)基準のため流用すると実効カット875Hzになり
        # 1187bps BPSK帯域を絞ってしまう。タップ数は同一(97)のため
        # history長・群遅延は不変。
        self.fir_rds = design_fir_kaiser(
            num_taps=97, cutoff_norm=2600.0 / self.if_rate, beta=7.0)
        self.history_rds = np.zeros(len(self.fir_rds) - 1, dtype=np.float32)
        self.history_final_l = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)
        self.history_final_r = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)
        self.history_shelf_lp_l = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp_l = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
        self.history_shelf_lp_r = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp_r = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
        self.history_shelf_dly_l = np.zeros(_sdly, dtype=np.float32)
        self.history_shelf_dly_r = np.zeros(_sdly, dtype=np.float32)
        self._pll_theta = 0.0
        self._pll_integ = 0.0
        self._pll_w0 = 2.0 * np.pi * 19000.0 / self.if_rate
        # 低ジッターPLL設計 (fn=16Hz, ζ=0.85, ループフィルタ遮断 20Hz)
        # 従来の過大帯域(205Hz)による低音変調漏れ・位相揺らぎ・定位のあるノイズを抑制
        _fn_pll = 16.0
        _wn_pll = 2.0 * np.pi * _fn_pll
        self._pll_kp = float(2.0 * 0.85 * _wn_pll / self.if_rate)
        self._pll_ki = float((_wn_pll / self.if_rate) ** 2)
        self._pll_alpha = float(2.0 * np.pi * 20.0 / self.if_rate)
        self._pll_ef = 0.0
        self._last_cos2 = None
        self._last_sin2 = None
        self._last_cos3 = None
        self._last_sin3 = None

        # 適応カルマン・パイロット搬送波トラッカー (19kHz追従)
        self.pilot_tracker = KalmanPilotTracker(sample_rate=self.if_rate)
        # 高域高調波補完エキサイター (Harmonic Exciter: 15kHz以上の高域倍音付加)
        # NOTE: 実機実測(ラッキーFM 94.6MHz 強電界)で無音時の12-15kHzを+18.7dB
        # 持ち上げ、静かな場面に合成ヒスが乗ることを確認したため既定OFF。
        # 再有効化は .enabled=True (弱局で空気感を出したい場合のみ推奨)。
        self.holographic_enhancer = HolographicAudioEnhancer(sample_rate=self.audio_rate, air_gain=0.08)
        self.holographic_enhancer.enabled = False

        # 位相スリップ抑制型FM復調器 (特異点クリック防止)
        self.riemann_demodulator = RiemannianTopologicalDemodulator(sample_rate=self.if_rate)
        # 弱電界クロスフェードのC/N窓 (既定=従来の直値28/22/上限0.75と同一)。
        # EKFと同じく推定C/N基準。弱電界了解度チューニング用に属性化。
        self.riemann_cn_hi = 28.0
        self.riemann_cn_lo = 22.0
        self.riemann_w_max = 0.75
        # 位相スリップ (ライスクリック) 補修: 振幅ディップ区間の累積位相残差で
        # 検出し、該当サンプルだけ補間へ差し替える。弱電界 (C/N推定30dB未満)
        # のみ作動し、クリック非検出時は元の復調列とビット等価。
        self.tda_click = TopologicalClickSuppressor(sample_rate=self.if_rate,
                                                    max_dev_hz=75000.0)
        self.tda_click_gate_db = 30.0
        self.tda_clicks = 0

        # 独立成分分析ステレオ復調器 (BSS / FastICA によるヒス低減)
        self.bss_separator = SuperSpatialBssStereoSeparator(sample_rate=self.audio_rate)

        # ハンケル行列SVD部分空間ノイズフィルター (SVD特異値しきい値処理)
        # NOTE: 実機実測で番組の12-15kHzを+7.1dB変形 (入出力相関0.9916=非透明) し、
        # 弱局でのノイズ低減効果も確認できなかったため既定OFF (CPUも節約)。
        self.rmt_denoiser = RmtHankelDenoiser(sample_rate=self.audio_rate, embed_dim=24)
        self.rmt_denoiser.enabled = False

        # 単一ch スペクトル抑圧NR (帯域内ノイズの最小統計Wiener抑圧。
        # 弱電界FMでハイカットでは消せない番組帯ノイズを低減。クリーン時は透明)
        self.mono_nr = MonoNoiseSuppressor(sample_rate=self.audio_rate)
        self.mono_nr_enabled = True
        # ===== RDS (57kHz) =====
        self.rds_enabled = True
        self.rds = None            # 遅延生成 (rds.RdsDecoder)
        self.rds_ps = ""
        self.rds_rt = ""
        self.rds_pi = 0
        self.rds_pty = None
        self.rds_groups = 0
        # BLER早期警報用 (デコーダ未生成時の既定。feed後に実値で上書き)
        self.rds_bler = 0.0
        self.rds_groups_checked = 0
        self.rds_groups_failed = 0
        self.rds_sync_losses = 0
        # BS.450/EN 50067準拠キャリア生成のため追加回転は不要 (0.0)
        self.rds_phase_offset = 0.0
        self._stereo_blend = 0.0
        # パイロット瞬断用フライホイール: ロック喪失直後はブレンドを凍結し、
        # 短い発作 (1.4秒/25ブロックまで) ではステレオ像を維持する
        self._pilot_hold_max = 25
        self._pilot_hold_n = 0
        # CコアのPLL 1サンプル進みが解消されたため、副搬送波オフセットは 0.0
        self.stereo_phase_offset = 0.0
        # 直交復調の残差 (位相誤差の符号付き観測用。MPXキャンセラ経路でのみ有効)
        self._last_diff_q = None
        # 38kHz再生位相オートトリム (ドングル個体差・温度ドリフトによる分離度劣化を
        # L-R電力最大化サーボで吸収。stereo_phase_offsetをアクチュエータに使う)
        self.stereo_trim_enabled = True
        self._trim_dir = 1.0
        self._trim_step_rad = float(np.deg2rad(0.5))
        self._trim_max_rad = float(np.deg2rad(15.0))
        self._trim_block = 0
        self._trim_m_smooth = 0.0
        self._trim_prev_m = 0.0
        self._trim_primed = False
        self._trim_err_ema = 0.0
        # 差信号振幅校正: 狭帯域IF(±50kHz)が75kHz偏移FMの外側ベッセル側波帯を
        # 切り落とすため、38kHz DSB由来の(L-R)が(L+R)より約6%小さく復調される
        # 差信号振幅校正: 旧1.06はアパーチャ損失込みの実測値
        # (|D|/|M|=0.9428@75kHz偏移) だった。アパーチャ分は逆sinc FIRで
        # 周波数特性ごと補正したため、残差 (ベッセル切落とし) のみを補正
        # する 1.03 へ再校正した (L単音75kHz偏移で55.6dB=旧55.9dBと等価、
        # 2トーン75kHzで-40.0dB・30kHzで-30.3dB。内容依存の最適値は
        # 1.02〜1.03に分布し、どちらも可聴限界 (-30dB) を十分上回る)。
        self.stereo_diff_gain = 1.03
        # ===== ステレオノイズリダクション =====
        # 弱電界でステレオ化すると増えるヒスノイズを、(L-R)差信号の高域/中域パワー比から
        # 検出し、ノイズ量に応じて 1) サブバンドWiener抑圧 2) 可変ローパス
        # 3) ブレンドでモノラルへ寄せる。実測はNR適用前の生差信号で行うため発振しない。
        self.stereo_nr_enabled = True
        self.stereo_status = "MONO"       # "STEREO" / "BLEND" / "MONO"
        self.stereo_nr_gain = 1.0         # ノイズ由来ブレンド (1=フルステレオ, 0=モノラル)
        self.stereo_cut_hz = 15000.0      # 差信号ローパス遮断周波数 (平滑)
        self.stereo_hiss_db = -60.0       # (L-R)ヒス指標 (初期値=クリーン, NR不発動)
        self._nr_cut_max_hz = 15000.0
        self._nr_cut_min_hz = 5000.0
        # 高域ブレンド上限の noisy 端アンカー: ヒス大時は従来通り
        # 13kHzで抑える (カーラジオ標準の高域ブレンド)。クリーン時は
        # _update_stereo_nr 側の適応式で15kHzまで開放する。
        self._nr_cut_fixed_hz = 13000.0
        # ブレンド量 (極端に弱い局のみモノラル化。通常はWienerが周波数別に処理)
        self._nr_lo_db = -18.0
        self._nr_hi_db = -4.0
        # 物理SNRゲート: 強電界で音声ドメイン床が番組HF由来になるのを防ぐ
        # (radiko_labオラクル実験: 合成連続音声IF37-47dBで全モノラル化、
        # 実録Lucky IF26dBは崩壊なし。ゲート境界はその間)
        self._nr_snr_gate_hi = 44.0
        self._nr_snr_gate_lo = 32.0
        self._nr_snr_penalty_db = 30.0
        # Wiener適用量 (これより上のノイズで段階的にサブバンド抑圧)
        # 実測: 強局(ラッキーFM 94.6)でも副搬送波ヒスは-36dBあり、旧-40/-18では
        # 適用度0.1しか立たず12-15kHzのヒスが残った。サブバンドWienerは
        # 知覚マスキングゲート内蔵で番組高域を保護するため、適用域を下げて
        # 「聞こえるヒス」を抑える (ブレンド側しきい値は据え置き=高域ブレンド不要)。
        self._nr_wiener_lo_db = -46.0
        self._nr_wiener_hi_db = -26.0
        self._nr_primed = False
        self._nr_s_w = 0.0                # 平滑化されたWiener適用度 (0=off, 1=full)
        self._nr_s = 0.0                  # 平滑化されたノイズ度 (0=クリーン, 1=ノイズ)
        # モノラル番組検出 (M-S相関＋S/M比ゲート)。相関だけではパン振りと
        # 区別できないため、S/M≪1 (漏れ支配) のときのみ全権を与える。
        self._nr_mono_rho = 0.0
        self._nr_mono_w = 0.0
        self._nr_mono_primed = False
        self._nr_sm_db = -30.0            # S/Mエネルギー比の平滑値 (dB)
        self._nr_sw_eff = 0.0             # 有効Wiener適用度 (モノラル判定反映)
        self._nr_sw_eff_prev = 0.0        # 前ブロック値 (スルーレート制限用)
        # マッピング専用の超低速ヒス推定 (τ12s)。ヒス推定は番組の高域/中域比
        # なので音楽の構成でゆっくり動く (実測27秒周期±6dB)。速い値でWiener
        # 適用度を駆動すると「幅の呼吸」になるため、適用度の計算だけ鈍らせる。
        # 超局は初期化時に速い値から開始するため立ち上がりは従来通り。
        self._nr_hiss_slow = None
        self._nr_cut_eff = 15000.0        # 有効S側カットオフ (モノラル判定反映)
        self._nr_hist = deque(maxlen=100)  # 差分HFパワー履歴 (下位10%をノイズフロア推定に使用)
        self._nr_mf_smooth = 0.0  # 番組パワー平滑値 (未初期化=0で初回に即時セット)
        self._nr_cut_levels = np.array([2500.0, 4000.0, 6500.0, 10000.0, 15000.0])
        self._nr_filters = [
            design_fir_kaiser(num_taps=65, cutoff_norm=float(c) / self.audio_rate, beta=6.5)
            for c in self._nr_cut_levels
        ]
        self.history_nr_lp = np.zeros(64, dtype=np.float32)
        self._nr_delay = (len(self._nr_filters[0]) - 1) // 2  # 線形位相FIRの群遅延

        # ===== サブバンドWiener NR (STFT 128pt / hop 64 / 平方根Hann(Sine窓) 50%オーバーラップ) =====
        # 差信号を周波数ごとにWiener抑圧。低域(ノイズが少なく音が濃い)はステレオのまま、
        # ノイズに埋もれた高域のみを選択的に落とすため、単一ローパスより音場が広い。
        # 平方根Hann窓 (Sine窓: sin(pi*(n+0.5)/N)) を分析・合成の両面で適用することで、
        # 50% OLAの二乗和が sin² + cos² ≡ 1.0 となり、再構成時の振幅変調リップル(750Hzとその倍音)が
        # ほぼゼロ(0.000000dB)になる。
        self._wf_n = 128
        self._wf_hop = 64
        self._wf_win = np.sin(np.pi * (np.arange(self._wf_n) + 0.5) / self._wf_n).astype(np.float32)
        self._wf_cola = np.ones(self._wf_n, dtype=np.float32)
        self._wf_in = np.zeros(0, dtype=np.float32)
        self._wf_out = np.zeros(self._wf_hop, dtype=np.float32)  # 初期プリフィル=固定遅延
        # STFTがゼロ埋めで生じた追加遅延。mono側を同量遅らせて時間整合を保つ
        # (ブロック長がホップの倍数でない場合に分離度が崩壊するのを防ぐ)
        self._wf_extra_delay = 0
        self._wf_ola = np.zeros(self._wf_n, dtype=np.float32)
        self._wf_p = None                  # 番組パワーの時間平滑
        self._wf_g = None
        self._wf_xi = None                 # decision-directed事前SNR
        self._wf_gamma_prev = None         # 前フレーム事後SNR
        self._wf_mask_slow = None          # マスキング閾値の低速EMA (幅呼吸防止)
        self._wf_mask_cnt = 0              # 8フレーム間引きカウンタ
        self._wf_f2 = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) ** 2
        # 知覚マスキング行列 (Bark拡散・Schroeder): T = P @ S でビン別マスキング閾値。
        # マスクされるノイズは抑圧不要 (g=1) とし、音楽性ノイズを設計上出さない。
        _bf = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate)
        _bk = 13.0 * np.arctan(0.76 * _bf / 1000.0) + 3.5 * np.arctan((_bf / 7500.0) ** 2)
        _dz = _bk[:, None] - _bk[None, :]
        _sp = 15.81 + 7.5 * (_dz + 0.474) - 17.5 * np.sqrt(1.0 + (_dz + 0.474) ** 2)
        self._wf_spread = (10.0 ** (_sp / 10.0)).astype(np.float32)
        # マスキング閾値オフセット。合成ステレオ (音声L/R・75µsプリエンファシス、
        # 強/弱電界) で0.1→0.05に半減: ヒス10-14kが+5〜6dB深くなり、side番組
        # 誤差は-35dB以下・幅変化0.3dB以下を維持 (0.02相当まで攻めると誤差-21dB
        # と可聴域に入る)。バイパス時 (sw<0.02) は従来通り無処理。
        self._wf_mask_offset = 0.05
        # 周波数依存マスキングオフセット (固定spreadの細分化):
        # 耳が敏感な2〜8kHzは保護寄り (最大2倍→gateが1側へ→番組保全)、
        # 番組疎でヒス支配の12kHz超は抑圧寄り (最小0.4倍→深く落とす)。
        # スカラー時と同一の要素積1回で計算量不変。
        _mid = np.exp(-(((_bf - 4000.0) / 3000.0) ** 2))
        _hi = 1.0 / (1.0 + np.exp(-(_bf - 12000.0) / 2500.0))
        self._wf_mask_offset_vec = (
            0.05 * np.clip(1.0 + 1.0 * _mid - 0.6 * _hi, 0.3, 2.0)).astype(np.float32)
        hf_mask = (np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) >= 6000.0) & \
                  (np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) <= 15000.0)
        self._wf_hf_f2_mean = float(np.mean(self._wf_f2[hf_mask]) + 1e-12)
        # 1024点指標→STFTドメインへのパワースケール補正 (E|X|²=σ²Σw²)
        self._wf_scale = float(np.sum(self._wf_win ** 2) / np.sum(np.hanning(1024) ** 2))
        self._nr_floor_pow = 0.0           # ブロードバンド指標のノイズ床 (HF帯)
        self._nr_floor_bias = 2.3          # 下位タイル→平均ノイズへの補正 (Sine窓特性に最適化)
        self._nr_gmin = 0.05               # 最大抑圧 (-26dB)
        self.stereo_wiener_gain = 1.0
        self._nr_delay += self._wf_hop     # Wiener経路の遅延をmono側で補償
        self.history_mono_delay = np.zeros(self._nr_delay, dtype=np.float32)
        self._nr_window = np.hanning(1024).astype(np.float32)
        # 周波数依存ブレンド用クロスオーバー状態 (1次相補: lo + hi = diff で再構成。
        # blend=1時は_lo/_hiとも1.0で旧スカラー動作とビット一致)
        self.freq_blend_enabled = True
        # 3.5k→4.5kHzへ引き上げ、低域ステレオを広く残す (10-14k抑圧は不変で
        # テストのhi_cut<-1.5dBを維持しつつ、声・楽器の芯を痩せさせない)
        self.freq_blend_xo_hz = 4500.0
        self._blend_xo_y1 = 0.0
        # 38kHz 直交副搬送波マルチパス適応キャンセラ (サ行シピシピ歪み・混濁の逆位相相殺)
        self.mpx_canceller = QuadratureMpxCanceller(sample_rate=self.audio_rate)
        # 拡張カルマンフィルタ (EKF) FM復調エンジン
        self.ekf_demod = DeepSpaceEkfDemodulator(sample_rate=self.if_rate)
        self.ekf_enabled = True
        # EKFクロスフェードのC/N窓 (既定=従来の直値26/20/上限1.0と同一)。
        self.ekf_cn_hi = 26.0
        self.ekf_cn_lo = 20.0
        self.ekf_w_max = 1.0
        self._linear_fir_audio_clean = self.fir_audio_clean.copy()
        self._linear_fir_audio_narrow = self.fir_audio_narrow.copy()
        self._linear_fir_am_audio = self.fir_am_audio.copy()
        # ===== FMマルチパス検出 =====
        # 反射波(マルチパス)はFM波に振幅変動(PM→AM変換)を与える。IF信号の包絡線変動を
        # 検出し、強い時はステレオ/帯域を絞って耳障りな歪みを抑える。
        self.multipath_enabled = True
        self.multipath_amount = 0.0
        self.multipath_gain = 1.0
        # 隣接妨害 (ACI) ガード: 左右いずれかのD/U悪化でL-R側を先に絞る
        # (38kHz副搬送波が先に汚れるため)。クリーン時は1.0でビット等価。
        # depth=0.6 は測定基盤の判定 (側波帯妨害/モノラル番組比が平均-6.3dB・
        # 36/36勝・p<1e-4、mid SI-SDRは完全不変) に基づく。sweep hook: aci.depth。
        self.aci_gain = 1.0
        self.aci_depth = 0.6
        self._mp_var = 0.0
        self.mp_lo = 0.10
        self.mp_hi = 0.35
        # 0.7→0.5へ緩和 (反射波での三重積によるステレオ痩せ・呼吸を軽減。
        # 最小gain 0.3→0.5。歪み抑制はCMA自動等化側に委ねる)
        self.mp_depth = 0.5
        # CMAブラインド等化器 (マルチパス・キャンセル)。手動は実機アンテナの安定性のためデフォルトOFF。
        # cognitive時の強い反射波には、multipath量ヒステリシス＋信号存在ゲートで自動介入する。
        self.multipath_cancel_enabled = False
        self.multipath_auto_cancel = True
        self._cma_auto = False
        self._cma_taps = 33
        # μは0.03→0.02へ (長遅延強エコー d=40/g=1.2で0.03はlock 0.18・
        # 0.02は0.39・0.015は0.56。短エコー・flutterは同等以上。
        # 0.015が最良だが実フラッターの追従速度を残すため0.02を採用)。
        self._cma_mu = 0.02
        self._cma_w = np.zeros(2 * self._cma_taps, dtype=np.float32)
        self._cma_w[2 * (self._cma_taps // 2)] = 1.0  # 中央タップ=デルタ初期化
        self._cma_hist = np.zeros(self._cma_taps - 1, dtype=np.complex64)
        self.cma_active = False
        # CMAクロスフェードのC/N窓 (EKF/Riemannと同型)。
        # 強歪み (低CN) では等化が効くが、比較的クリーンなマルチパスでは
        # 等化人工物が逆に了解度を落とす (ESTOIで+0.09/-0.03を確認)。
        # CNで重み付けし、効く条件でのみ混ぜる。
        self.cma_cn_hi = 28.0
        self.cma_cn_lo = 23.0
        self.cma_w_max = 1.0
