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
    SuperSpatialBssStereoSeparator,
    RmtHankelDenoiser,
    MonoNoiseSuppressor,
)
from dsp_filters import (
    design_fir_kaiser,
    design_fir_highpass,
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
                    blend *= self.multipath_gain
                    diff = self._diff_lowpass(diff_raw, cut)
                    diff = self._wiener_diff(diff)
                    # 周波数依存ブレンド: 弱電界で高域から先にモノラル化
                    # (低域のステレオ感を残す。blend=1時は旧スカラー動作と一致)
                    stereo_diff = self._freq_dependent_blend(diff, blend)
                    # 差信号FIRの群遅延を補償 (mono/diffの位相ズレによる分離度劣化を防止)
                    mono = self._delay_mono(mono)
                else:
                    # NR無効時はLPF/STFT往復をせず生差信号へブレンドのみ
                    # (15kHz LPF＋COLAリップルが可聴域を変える問題とCPU浪費を回避)。
                    # mono遅延履歴だけは更新し、再有効時の継ぎ目を無くす
                    # (出力は遅延させない。遅延させると未遅延diffと2msずれる)。
                    stereo_diff = (diff_raw
                                   * (self._stereo_blend * self.multipath_gain)).astype(np.float32)
                    self._delay_mono(mono)
                    self.stereo_wiener_gain = 1.0

        # 6. チャンネル別ポスト処理 (ディエンファシス・ハイカット・DCカット・シェルフ)
        if stereo_diff is not None:
            # 注: L/Rのスレッド並列化は実測で逆効果 (CPython GIL + C呼び出しが短く
            # オーバーヘッドが上回る)。逐次実行が最速。
            left = self._post_process_wfm(mono + stereo_diff, "_l")
            right = self._post_process_wfm(mono - stereo_diff, "_r")

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
                * self.multipath_gain
            self.is_stereo = blend > 0.5
            if blend > 0.85:
                self.stereo_status = "STEREO"
            elif blend > 0.02:
                self.stereo_status = "BLEND"
            else:
                self.stereo_status = "MONO"
            # WFM経路も±1.0へクリップ (AM/SSBと統一。過偏移・弱電界ノイズで
            # ±1.82超→後段int16変換でのラップ歪みを防止)
            return np.clip(np.stack([left, right], axis=1), -1.0, 1.0).astype(np.float32)
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

        # 1. CMA等化 (マルチパス・キャンセル) + ハードリミッター適用
        use_cma = self._update_cma_auto_gate()
        if (use_cma and _NATIVE is not None and NATIVE_CMA
                and self.multipath_amount > 0.15
                and abs(self.stereo_pilot_lock) > 0.2
                and self.s_meter_dbfs > -60.0):
            # CMA等化 (ハードリミット前。リミット後は包絡線一定で誤差が出ない)。
            # 信号存在ゲート: lock必須＋ノイズ床veto (S-meterだけでの作動は
            # 深フェードでblendを下げるため廃止。ゲート側と条件を一致させる)。
            iq_if = self._apply_cma(iq_if)
            self.cma_active = True
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

        # 弱電界・モノラル時におけるEKFのシームレス・クロスフェード
        # (ステレオ時は38kHz副搬送波の広帯域通過のため広帯域差分法を維持し、
        # 弱電界・モノラル局でEKFを併用してクリックスパイクと三角雑音を抑える)
        # NOTE: 旧条件は fm_pll_enabled(False既定) とのANDで常時OFFになっていた。
        # EKFは ekf_enabled 単独で制御する (PLLとは独立)。
        if (getattr(self, "ekf_enabled", False)
                and getattr(self, "ekf_demod", None) is not None
                and self._stereo_blend <= 0.05):
            w_ekf = float(np.clip((-38.0 - self.s_meter_dbfs) / 10.0, 0.0, 1.0))
            if w_ekf > 0.01:
                demod_ekf = self.ekf_demod.demodulate(limited)
                if len(demod_ekf) == len(demod):
                    demod = ((1.0 - w_ekf) * demod + w_ekf * demod_ekf).astype(np.float32)

        # 位相スリップ防止FM復調 (弱電界フェージング時のクリック雑音抑制)
        if (getattr(self, "riemann_demodulator", None) is not None
                and self.riemann_demodulator.enabled
                and (self.cognitive_enabled or getattr(self, "riemann_always", False))
                and self.s_meter_dbfs < -35.0):
            w_riemann = float(np.clip((-35.0 - self.s_meter_dbfs) / 12.0, 0.0, 0.75))
            demod_riemann = self.riemann_demodulator.demodulate(iq_if)
            if len(demod_riemann) == len(demod):
                demod = ((1.0 - w_riemann) * demod + w_riemann * demod_riemann).astype(np.float32)

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
            if abs(freq_error_hz) < 20000.0:  # ±20kHz以内の偏差に自動追従
                # 20Hz未満の微小ジッターは補正を休止しロックを維持
                if abs(freq_error_hz) > 20.0:
                    self.afc_offset_hz = float(np.clip(
                        self.afc_offset_hz - self.afc_alpha * freq_error_hz,
                        -20000.0,
                        20000.0
                    ))

        # 3. 適切な名目オーディオゲインにスケーリング
        # 日本規格の最大周波数偏移(±75kHz)でも振幅0.95以内に収め、過変調時のソフトリミッターポンピング歪みを抑制
        demod_scaled = demod * 0.58

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
                rds_base = self.decimate_with_history(rds_mix, self.fir_am_narrow, 24, "history_rds")
                self.rds.feed(rds_base)
                self.rds_ps = self.rds.ps_name
                self.rds_rt = self.rds.radio_text
                self.rds_pi = self.rds.pi
                self.rds_pty = self.rds.pty
                self.rds_groups = self.rds.groups
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
        return np.clip(out_mono, -1.0, 1.0)

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
        self._nr_floor_pow = floor
        ratio_db = 10.0 * np.log10((floor + 1e-12) / (mf + 1e-12))

        dt = len(diff) / self.audio_rate
        if not self._nr_primed:
            # 初回は実測値で即座に初期化 (起動直後のランプを排除)
            self._nr_primed = True
            self.stereo_hiss_db = ratio_db
        else:
            a = 1.0 - np.exp(-dt / 0.35)
            self.stereo_hiss_db += a * (ratio_db - self.stereo_hiss_db)

        def amount(value_db: float, lo: float, hi: float) -> float:
            x = float(np.clip((value_db - lo) / (hi - lo), 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)  # smoothstep

        s_b = amount(self.stereo_hiss_db, self._nr_lo_db, self._nr_hi_db)
        s_w = amount(self.stereo_hiss_db, self._nr_wiener_lo_db, self._nr_wiener_hi_db)
        # 非対称スムージング: ノイズ増加時は速く、回復はゆっくり
        tau = 0.3 if s_b > self._nr_s else 1.5
        self._nr_s += (1.0 - np.exp(-dt / tau)) * (s_b - self._nr_s)
        tau_w = 0.25 if s_w > self._nr_s_w else 2.5
        self._nr_s_w += (1.0 - np.exp(-dt / tau_w)) * (s_w - self._nr_s_w)
        self.stereo_nr_gain = 1.0 - self._nr_s
        self.stereo_cut_hz = self._nr_cut_max_hz * (
            (self._nr_cut_min_hz / self._nr_cut_max_hz) ** self._nr_s
        )
        # 固定高域ブレンド: 副搬送波ヒス対策で常時上限を適用 (適応側がそれ以上
        # 絞る場合はそちらを優先)
        self.stereo_cut_hz = min(self.stereo_cut_hz, self._nr_cut_fixed_hz)
        # モノラル番組判定: 実際のステレオミックスでは M=(L+R)/2 と S=(L-R)/2 は
        # 直交するため、M-S相関は「番組でないS成分」(分離漏れクロストーク+ノイズ)
        # の割合を示す。相関が高ければS側を積極抑圧しても番組を損なわない
        # (実測: ラッキーFM 94.6 の番組はモノラルで相関+0.63、S/M -24dB)。
        # 真のステレオ番組 (相関≈0) では従来の控えめ設定を維持する。
        mo = np.asarray(mono, dtype=np.float64)
        di = np.asarray(diff, dtype=np.float64)
        if len(mo) == len(di) and len(mo) > 0:
            mo = mo - float(mo.mean())
            di = di - float(di.mean())
            den = float(np.sqrt(np.mean(mo * mo) * np.mean(di * di))) + 1e-12
            rho = float(np.mean(mo * di)) / den
            if not self._nr_mono_primed:
                self._nr_mono_primed = True
                self._nr_mono_rho = rho
            else:
                a_r = 1.0 - np.exp(-dt / 2.0)
                self._nr_mono_rho += a_r * (rho - self._nr_mono_rho)
        x_m = float(np.clip((self._nr_mono_rho - 0.15) / 0.35, 0.0, 1.0))
        self._nr_mono_w = x_m * x_m * (3.0 - 2.0 * x_m)
        # 有効値: モノラル番組ではWienerを全力(1.0)へ、S高域カットを8kHzへ寄せる
        self._nr_sw_eff = max(self._nr_s_w, self._nr_mono_w)
        self._nr_cut_eff = min(self.stereo_cut_hz, 13000.0 - 5000.0 * self._nr_mono_w)

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
        y = np.convolve(x_ext, self._nr_filters[i0], mode="valid")
        if w > 1e-3:
            y2 = np.convolve(x_ext, self._nr_filters[i0 + 1], mode="valid")
            y = y * (1.0 - w) + y2 * w
        return y.astype(np.float32)

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
            powers = np.abs(specs) ** 2 + 1e-12

            c_noise = ((self._nr_floor_pow * self._nr_floor_bias * self._wf_scale)
                       / self._wf_hf_f2_mean)
            sw = self._nr_sw_eff if self.stereo_nr_enabled else 0.0
            # 平滑再帰のみ逐次 (フレーム間依存のため。ベクトル演算のみでFFTなし)
            gmix = np.empty_like(specs, dtype=np.float32)
            for j in range(nframes):
                power = powers[j]
                self._wf_p = 0.5 * power + 0.5 * self._wf_p
                g_w = np.maximum(1.0 - (c_noise * self._wf_f2) / (self._wf_p + 1e-12),
                                 self._nr_gmin).astype(np.float32)
                g_w[:3] = 1.0  # DC〜低域は保護
                if self._wf_g is None or len(self._wf_g) != len(g_w):
                    self._wf_g = g_w
                else:
                    a = np.where(g_w < self._wf_g, 0.7, 0.1)
                    self._wf_g = self._wf_g + a * (g_w - self._wf_g)
                # 知覚マスキングフロア: 番組にマスクされるノイズは抑圧不要 (g→1)。
                # Wienerの過剰抑圧（音楽性ノイズ・高域の曇り）を可聴性基準で緩和する。
                # マスキング算出はクリーン推定 (P-N) から行う (ノイズ込み電力では
                # ヒス自身がマスクを上げて抑圧不能になるため)。
                noise_bin = c_noise * self._wf_f2
                p_clean = np.maximum(
                    self._wf_p.astype(np.float64) - noise_bin, 0.0)
                mask_thr = (p_clean @ self._wf_spread) * self._wf_mask_offset
                gate = np.minimum(1.0, mask_thr / (noise_bin + 1e-12)).astype(np.float32)
                g_use = np.maximum(self._wf_g, gate)
                g_use[:3] = 1.0  # DC〜低域は保護
                sw = self._nr_sw_eff if self.stereo_nr_enabled else 0.0
                g_mix = 1.0 - sw * (1.0 - g_use)
                gmix[j] = g_mix
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
        a = 1.0 - float(np.exp(-2.0 * np.pi * float(self.freq_blend_xo_hz) / float(self.audio_rate)))
        y1 = float(self._blend_xo_y1)
        # ベクトル化指数平滑の厳密逐次と等価なIIRを、ブロック内は
        # lfilter相当の逐次ループで実行 (N~1kで0.02ms級)
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
            rms = float(np.sqrt(np.mean(x * x)))
            if not np.isfinite(rms) or rms < float(self._slow_agc_floor):
                return audio
            if bool(getattr(self, "lufs_agc_enabled", False)):
                try:
                    from audiophile_dsp import LoudnessNormalizer
                    if self._lufs_norm is None:
                        self._lufs_norm = LoudnessNormalizer(
                            sample_rate=float(self.audio_rate))
                    mono = x if x.ndim == 1 else np.mean(x, axis=1)
                    lufs = self._lufs_norm.push(mono)
                    if lufs is None:
                        return audio
                    raw_desired = self._lufs_norm.gain_for(lufs)
                except Exception:
                    raw_desired = float(self.slow_agc_target) / (rms + 1e-12)
            else:
                raw_desired = float(self.slow_agc_target) / (rms + 1e-12)
            desired = float(np.clip(raw_desired,
                                    float(self.slow_agc_min), float(self.slow_agc_max)))
            cur = float(self.slow_agc_gain)
            # ブロック長から時定数を換算 (audio_rate基準)
            try:
                dt = len(audio) / float(self.audio_rate)
            except Exception:
                dt = 0.05
            tau = float(self.slow_agc_attack) if desired < cur else float(self.slow_agc_release)
            a = 1.0 - float(np.exp(-dt / tau))
            gain = cur + a * (desired - cur)
            self.slow_agc_gain = float(gain)
            if abs(gain - 1.0) < 1e-4:
                return audio
            return (x * gain).astype(np.float32)
        except Exception:
            return audio

    def _blend_release(self, floor: float = 0.0):
        """ブレンド低下 (パイロット瞬断フライホイール付き)。
        高ブレンドからの低下要求は25ブロック (~1.4秒) まで凍結し、
        短いパイロット瞬断発作でステレオ像がモノラルへ往復するのを防ぐ。
        持続喪失では従来通り0.97ランプで floor まで滑らかに落とす。"""
        if (self._stereo_blend > 0.5
                and self._pilot_hold_n < self._pilot_hold_max):
            self._pilot_hold_n += 1
        else:
            self._stereo_blend = max(float(floor), self._stereo_blend * 0.97)
        self.stereo_blend = self._stereo_blend

    def _post_process_wfm(self, audio: np.ndarray, ch: str = "") -> np.ndarray:
        """WFM音声のチャンネル別仕上げ (ch: ''=モノ, '_l'/'_r'=ステレオ各ch)"""
        # ディエンファシス (50/75μs)
        audio = self._apply_bilinear_deemphasis(audio, ch=ch)

        # オーディオ段ハイカットフィルタ (Hyper時は無段階モーフィング)
        if self.cognitive_enabled:
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

        # DCハイパスフィルタ
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
        if (getattr(self, "mono_nr", None) is not None
                and self.mono_nr.enabled
                and getattr(self, "mono_nr_enabled", True)
                and (self.cognitive_enabled or getattr(self, "mono_nr_always", False))):
            audio = self.mono_nr.process(audio, ch=ch)

        return audio.astype(np.float32)

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
            pilot_rms = float(np.sqrt(np.mean(pilot.astype(np.float64) ** 2)) + 1e-12)
            mpx_rms_pre = float(np.sqrt(np.mean(np.asarray(mpx, dtype=np.float64) ** 2)) + 1e-12)
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

            mpx_rms = float(np.sqrt(np.mean(np.asarray(mpx, dtype=np.float64) ** 2)) + 1e-12)
            ratio = pilot_rms / mpx_rms
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
        # 19kHzパイロットトーンを抑えるIF段オーディオフィルタ (288kHzレート)
        # カットオフ 15kHz, 19kHzで -60dB以上の急峻減衰 (257タップ。
        # 97タップでは19kHzで-25.8dBしかなく超音波漏洩していた)
        cutoff_if_audio = 15000.0 / self.if_rate
        self.fir_if_audio = design_fir_kaiser(num_taps=257, cutoff_norm=cutoff_if_audio, beta=7.0)

        # 48kHzオーディオ段のアンチエイリアス・ハイカットフィルタ (48kHzレート)
        # 14kHz: 音楽用Hi-Fiワイド (51タップ)
        cutoff_audio_wide = 14000.0 / self.audio_rate
        self.fir_audio_wide = design_fir_kaiser(num_taps=51, cutoff_norm=cutoff_audio_wide, beta=6.0)

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
        # AFC (Automatic Frequency Control: 100Hz精度の自動搬送波追従)
        self.afc_enabled = True
        self.afc_offset_hz = 0.0
        self.afc_alpha = 0.05  # 滑らかな追従時定数

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
        self.history_rds = np.zeros(len(self.fir_am_narrow) - 1, dtype=np.float32)
        self.history_final_l = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)
        self.history_final_r = np.zeros(len(self.fir_audio_clean) - 1, dtype=np.float32)
        self.history_shelf_lp_l = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp_l = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
        self.history_shelf_lp_r = np.zeros(len(self.fir_shelf_lp) - 1, dtype=np.float32)
        self.history_shelf_hp_r = np.zeros(len(self.fir_shelf_hp) - 1, dtype=np.float32)
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
        # 固定高域ブレンド上限: FMステレオ副搬送波(38kHz DSB)の三角雑音は
        # 高域ほど大きく、強局でも12-15kHzで番組と同程度まで残る (実測: ラッキーFM
        # 94.6MHz 強電界で S高域ノイズが番組-5dB)。ヒス指標に依らず常時S側を
        # 13kHzで緩く減衰させる (カーラジオ標準の高域ブレンド。低域のステレオ感は不変)。
        self._nr_cut_fixed_hz = 13000.0
        # ブレンド量 (極端に弱い局のみモノラル化。通常はWienerが周波数別に処理)
        self._nr_lo_db = -18.0
        self._nr_hi_db = -4.0
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
        # モノラル番組検出 (M-S相関): 真のステレオでは直交するため、相関が高い=
        # S成分が分離漏れ+ノイズ。モノラル番組ではS側を積極抑圧してヒスを消す
        self._nr_mono_rho = 0.0
        self._nr_mono_w = 0.0
        self._nr_mono_primed = False
        self._nr_sw_eff = 0.0             # 有効Wiener適用度 (モノラル判定反映)
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
        self._wf_f2 = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate) ** 2
        # 知覚マスキング行列 (Bark拡散・Schroeder): T = P @ S でビン別マスキング閾値。
        # マスクされるノイズは抑圧不要 (g=1) とし、音楽性ノイズを設計上出さない。
        _bf = np.fft.rfftfreq(self._wf_n, 1.0 / self.audio_rate)
        _bk = 13.0 * np.arctan(0.76 * _bf / 1000.0) + 3.5 * np.arctan((_bf / 7500.0) ** 2)
        _dz = _bk[:, None] - _bk[None, :]
        _sp = 15.81 + 7.5 * (_dz + 0.474) - 17.5 * np.sqrt(1.0 + (_dz + 0.474) ** 2)
        self._wf_spread = (10.0 ** (_sp / 10.0)).astype(np.float32)
        self._wf_mask_offset = 0.1  # マスキング閾値オフセット (同時マスキング-10dB相当)
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
        self.freq_blend_xo_hz = 3500.0
        self._blend_xo_y1 = 0.0
        # 38kHz 直交副搬送波マルチパス適応キャンセラ (サ行シピシピ歪み・混濁の逆位相相殺)
        self.mpx_canceller = QuadratureMpxCanceller(sample_rate=self.audio_rate)
        # 拡張カルマンフィルタ (EKF) FM復調エンジン
        self.ekf_demod = DeepSpaceEkfDemodulator(sample_rate=self.if_rate)
        self.ekf_enabled = True
        self._linear_fir_audio_clean = self.fir_audio_clean.copy()
        self._linear_fir_audio_narrow = self.fir_audio_narrow.copy()
        self._linear_fir_am_audio = self.fir_am_audio.copy()
        # ===== FMマルチパス検出 =====
        # 反射波(マルチパス)はFM波に振幅変動(PM→AM変換)を与える。IF信号の包絡線変動を
        # 検出し、強い時はステレオ/帯域を絞って耳障りな歪みを抑える。
        self.multipath_enabled = True
        self.multipath_amount = 0.0
        self.multipath_gain = 1.0
        self._mp_var = 0.0
        self.mp_lo = 0.10
        self.mp_hi = 0.35
        self.mp_depth = 0.7
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
