"""Black Magic wiring for SdrDspPipeline (extracted from dsp.py).

黒魔法三点＋一点の配線・生成・スケルチ統合。SdrDspPipeline の mixin として
動作する (純粋移動・動作同一)。既定OFF・失敗時フォールバックの方針は維持。
"""

import math
from collections import deque

import numpy as np

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


class DspBlackMagicMixin:
    """黒魔法配線メソッド群 (SdrDspPipeline に mixin される)"""

    def _bm_update_squelch_assist(self) -> float:
        """cyclo検出→スケルチの二基準ヒステリシス＋ソフトフェード (既定OFF)。

        開: conf>open_conf (pilot在り) または S>open_smeter (強信号)。
        閉: conf<close_conf かつ S<close_smeter (弱い局間ノイズのみ)。
        閉閾値は-25dBFS: それより強いノイズ (-16dB級の熱い局間) は
        電力スケルチの仕事とし、ここでは閉じない (弱局-27dBFSとの
        エネルギー重なりを避ける。conf側で分別する)。
        モノラル強局 (conf=0だがS高) は開のまま＝誤ミュートしない。
        開は速く (0.25/block)、閉は遅く (0.05/block) しchatterを防ぐ。
        戻り値は0.0〜1.0のゲイン。例外時は現状維持 (跳ばせない)。
        """
        try:
            if not (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_sq_assist_enabled", False)):
                self._bm_sq_open = True
                self._bm_sq_gain = 1.0
                return 1.0
            conf = float(getattr(self, "bm_cyclo_confidence", 0.0))
            s_db = float(getattr(self, "s_meter_dbfs", -90.0))
            if not (math.isfinite(conf) and math.isfinite(s_db)):
                return float(self._bm_sq_gain)
            open_conf = float(getattr(self, "bm_sq_open_conf", 0.75))
            close_conf = float(getattr(self, "bm_sq_close_conf", 0.55))
            close_s = float(getattr(self, "bm_sq_close_smeter_db", -25.0))
            open_s = float(getattr(self, "bm_sq_open_smeter_db", -40.0))
            if conf > open_conf:
                self._bm_sq_open = True
            elif conf < close_conf and s_db < close_s:
                self._bm_sq_open = False
            elif s_db > open_s:
                self._bm_sq_open = True
            # else: 保持 (ヒステリシス)
            # 最小閉保持: 一度閉じたら一定ブロックは開き直さない。
            # 深フェードで開閉が呼吸 (ポンピング) するのを防ぐ。
            try:
                hold_n = int(getattr(self, "bm_sq_min_close_blocks", 20))
            except Exception:
                hold_n = 20
            if self._bm_sq_open:
                if self._bm_sq_hold > 0:
                    self._bm_sq_hold -= 1
                    if self._bm_sq_hold > 0:
                        self._bm_sq_open = False
            else:
                self._bm_sq_hold = max(0, hold_n)
            target = 1.0 if self._bm_sq_open else 0.0
            g = float(self._bm_sq_gain)
            step = 0.25 if target > g else 0.05
            g += max(-step, min(step, target - g))
            self._bm_sq_gain = float(min(max(g, 0.0), 1.0))
            return self._bm_sq_gain
        except Exception:
            try:
                return float(self._bm_sq_gain)
            except Exception:
                return 1.0

    def _bm_cfg_section(self, name: str) -> dict:
        """bm_cfgの指定節を辞書で返す (main.pyのconfig反映用。破損時は空)。"""
        try:
            cfg = getattr(self, "bm_cfg", None) or {}
            sec = cfg.get(name, None) or {}
            return dict(sec) if isinstance(sec, dict) else {}
        except Exception:
            return {}

    def _bm_make_rmt(self):
        """bm_cfgを反映したSafeRmtDenoiserを生成 (数値破損時は内蔵既定)。"""
        if SafeRmtDenoiser is None:
            raise ImportError("SafeRmtDenoiser unavailable")
        cfg = self._bm_cfg_section("rmt_denoiser")

        def _num(key, default, lo, hi):
            try:
                v = cfg.get(key, default)
                if isinstance(v, bool):
                    return default
                return min(max(float(v), lo), hi)
            except (TypeError, ValueError):
                return default

        return SafeRmtDenoiser(sample_rate=self.audio_rate,
                               max_strength=_num("max_strength", 0.65, 0.0, 1.0),
                               max_rank=int(_num("max_rank", 8, 1, 64)),
                               max_matrix_size=int(_num("max_matrix_size", 256, 16, 4096)),
                               cpu_budget_percent=_num("cpu_budget_percent", 20.0, 0.0, 100.0))

    def _bm_rmt_strength_cap(self) -> float:
        """番組適応したRMT強度上限。トーク (speech_prob→1) で半減する。
        音楽 (prob=0)・適応OFF・分類器不在時は従来上限と一致する。"""
        try:
            cap = min(max(float(getattr(self, "bm_rmt_cap", 0.65)), 0.0), 0.85)
        except (TypeError, ValueError):
            cap = 0.65
        if bool(getattr(self, "bm_rmt_speech_adapt", True)):
            try:
                p = float(getattr(getattr(self, "cognitive_eq", None),
                                  "speech_prob", 0.0))
            except (TypeError, ValueError):
                p = 0.0
            if not (p >= 0.0 and p <= 1.0):
                p = 0.0
            cap *= 1.0 - 0.5 * p
        return float(cap)

    def _bm_make_notch(self):
        """bm_cfgを反映したAdaptiveNotchCancellerを生成。"""
        if AdaptiveNotchCanceller is None:
            raise ImportError("AdaptiveNotchCanceller unavailable")
        cfg = self._bm_cfg_section("adaptive_notch")

        def _num(key, default, lo, hi):
            try:
                v = cfg.get(key, default)
                if isinstance(v, bool):
                    return default
                return min(max(float(v), lo), hi)
            except (TypeError, ValueError):
                return default

        return AdaptiveNotchCanceller(sample_rate=self.audio_rate,
                                      base_hz=_num("base_hz", 0.0, 0.0, 100.0),
                                      max_harmonic=int(_num("max_harmonic", 5, 1, 9)),
                                      line_on_db=_num("line_on_db", 8.0, 0.0, 40.0))

    def _bm_make_sr(self):
        """bm_cfgを反映したStochasticResonanceDetectorを生成。"""
        if StochasticResonanceDetector is None:
            raise ImportError("StochasticResonanceDetector unavailable")
        cfg = self._bm_cfg_section("stochastic_resonance")

        def _num(key, default, lo, hi):
            try:
                v = cfg.get(key, default)
                if isinstance(v, bool):
                    return default
                return min(max(float(v), lo), hi)
            except (TypeError, ValueError):
                return default

        return StochasticResonanceDetector(
            detector_only=True,
            sigma_ratio_min=_num("sigma_ratio_min", 0.01, 0.0, 1.0),
            sigma_ratio_max=_num("sigma_ratio_max", 0.10, 0.0, 1.0),
            trials=int(_num("trials", 4, 2, 16)),
            min_snr_db=_num("min_snr_db", -5.0, -40.0, 40.0),
            max_snr_db=_num("max_snr_db", 12.0, -40.0, 40.0))

    def _bm_attack_limit(self, lock: float) -> float:
        """黒魔法Aのblend上昇レート (既定0.25=従来動作)。

        confidence低＋lock低の安定弱信号のときのみ鈍化させる。
        flutter中 (直近lockの標準偏差が閾値超) は介入すると上昇だけ
        鈍ってblendが下げ方向にラチェットするため、従来レートに戻す
        (77.5MHz実測で崩落を確認した副作用の対策)。
        """
        try:
            if not (getattr(self, "black_magic_enabled", False)
                    and getattr(self, "bm_cyclo_enabled", False)):
                return 0.25
            lock_f = float(lock)
            if not math.isfinite(lock_f):
                return 0.25
            # 既存PLLが正常ロックしている場合は介入しない
            if lock_f > 0.5:
                return 0.25
            hist = getattr(self, "_bm_lock_hist", None)
            if hist is not None and len(hist) >= 8:
                sd = float(np.std(np.asarray(hist, dtype=np.float64)))
                if math.isfinite(sd) and sd > float(getattr(self, "bm_flutter_std",
                                                             0.15)):
                    return 0.25
            conf = float(getattr(self, "bm_cyclo_confidence", 0.0))
            if conf < float(getattr(self, "bm_cyclo_min_conf", 0.55)):
                _bmp = getattr(self, "_bm_params", None) or {}
                return float(_bmp.get("blend_attack_limit", 0.05))
            return 0.25
        except Exception:
            return 0.25

    def _init_bm_state(self):
        """黒魔法フラグ・履歴の初期化 (__init__ から純粋移動)。"""
        # ===== 黒魔法三点セット (弱電界検出補助。既定は全て無効) =====
        # master=self.black_magic_enabled がFalseの間は一切動作せず、
        # 既存経路とビット同一の出力を保つ (tests/test_black_magic.pyで検証)。
        # 各インスタンスは遅延生成 (有効化時のみ) し、失敗時はNoneのまま
        # 既存経路へフォールバックする。
        self.black_magic_enabled = False
        self.bm_cyclo_enabled = False
        self.bm_rmt_enabled = False
        self.bm_sr_enabled = False
        self.bm_notch_enabled = False
        self.bm_sq_assist_enabled = False
        self.bm_sq_open_conf = 0.75
        self.bm_sq_close_conf = 0.55
        self.bm_sq_close_smeter_db = -25.0
        self.bm_sq_open_smeter_db = -40.0
        self.bm_sq_min_close_blocks = 20
        self.bm_seek_enabled = False
        self._bm_sq_open = True  # 起動時は開 (いきなりミュートしない)
        self._bm_sq_gain = 1.0
        self._bm_sq_hold = 0
        self.cyclo_detector = None
        self.bm_rmt = None
        self.bm_sr = None
        self.bm_notch = None
        self.bm_controller = None
        self.bm_cyclo_min_conf = 0.55
        self.bm_cyclo_confidence = 0.0
        self.bm_sr_confidence = 0.0
        self.bm_rmt_cap = 0.65
        # 番組適応RMT (既定ON): トーク時は強度上限を半減し了解度を優先、
        # 音楽時は全開で快適性を優先。speech_prob=0では係数1.0で旧動作と一致。
        self.bm_rmt_speech_adapt = True
        # main.pyから渡される黒魔法パラメータ (configのblack_magic節)。
        # 遅延生成インスタンスのコンストラクタに反映する。既定は空=内蔵既定。
        self.bm_cfg = {}
        # flutter検出用lock履歴 (直近32ブロック) と判定閾値
        self._bm_lock_hist = deque(maxlen=32)
        self.bm_flutter_std = 0.15
        self._bm_params = None
        self._bm_last_rmt_info = None
