"""黒魔法三点セットの統合オートマネージャ。

責務: SNR推定・クリップ状態・pilot confidence・音声ノイズ推定・
stereo blend・squelch confidence・CPU使用率を統合し、各機能の安全な
パラメータ (cyclo助言・RMT強度上限・SR有効/無効) を決める。
音声・IQ自体には触らない (パラメータ計算のみ。副作用なし)。

設計方針:
1. 強信号では処理を減らす (RMT上限→小、SR無効)。
2. 弱信号では検出系を強化する (cyclo助言・SR有効)。
3. 音質劣化兆候があれば感度より音質を優先し安全側へ戻す (ラッチ＋緩やか回復)。
4. CPU逼迫時はRMTから先に弱める。
5. 非有限・例外時は全機能バイパス (フォールバック)。
6. パラメータはEMA平滑し急変させない。
7. 選局時はreset()で全状態クリア。
"""

import math
import time


def _f(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# モード別プリセット (Phase 3-1: 自動プロファイルの表)。
# 音声を壊しうる処理はAM/SSB等の非FMでは使わない。
PROFILES = {
    "WFM": {"cyclo": True, "rmt": True, "sr": True, "notch": True},
    "NFM": {"cyclo": False, "rmt": True, "sr": False, "notch": True},
    "AM": {"cyclo": False, "rmt": True, "sr": False, "notch": True},
    "AM_NARROW": {"cyclo": False, "rmt": True, "sr": False, "notch": True},
    "USB": {"cyclo": False, "rmt": True, "sr": False, "notch": False},
    "LSB": {"cyclo": False, "rmt": True, "sr": False, "notch": False},
    "CW": {"cyclo": False, "rmt": False, "sr": False, "notch": False},
}


def profile_for(mode, snr_db):
    """受信モード＋SNR→推奨フラグ (GUIが表示・適用するための提案。副作用なし)。

    強信号 (SNR>30) では検出補助のみ残し、処理系は落とす。
    未知モードは全OFF (安全側)。
    """
    try:
        base = dict(PROFILES.get(str(mode), {}))
    except Exception:
        base = {}
    if not base:
        return {"cyclo": False, "rmt": False, "sr": False, "notch": False,
                "reason": "unknown-mode"}
    try:
        snr = float(snr_db)
    except (TypeError, ValueError):
        snr = 0.0
    if not math.isfinite(snr):
        snr = 0.0
    if snr > 30.0:
        base.update({"rmt": False, "sr": False, "notch": False,
                     "reason": "strong-signal"})
    else:
        base["reason"] = "weak-signal" if snr <= 12.0 else "normal"
    return base


class BlackMagicController:
    """process_metrics(metrics)→params。metricsキーはすべて任意 (欠損=安全側)。"""

    def __init__(self, enabled=False, smooth_alpha=0.2,
                 strong_snr_db=30.0, weak_snr_db=12.0,
                 cpu_warn_percent=70.0, cpu_max_percent=90.0,
                 rmt_strong_cap=0.05):
        self.enabled = bool(enabled)
        self.alpha = float(min(max(smooth_alpha, 0.01), 1.0))
        self.strong_snr = float(strong_snr_db)
        self.weak_snr = float(weak_snr_db)
        self.cpu_warn = float(cpu_warn_percent)
        self.cpu_max = float(cpu_max_percent)
        self.rmt_strong_cap = float(rmt_strong_cap)
        self.reset()

    def reset(self):
        """選局時に全状態をクリア (平滑値・劣化ラッチを持ち越さない)。"""
        self._snr = None
        self._cpu = None
        self._rmt_cap = None
        self._degraded_latch = False
        self._clear_n = 0
        self.blocks = 0
        self._last_out = None

    def describe(self):
        """現在の状態を人間可読辞書で返す (Phase 3-2: チューニング表示用API)。
        GUIがそのまま表示できる形式。DSP状態には触らない。"""
        o = self._last_out or {}
        try:
            snr = self._snr
        except Exception:
            snr = None
        return {
            "enabled": bool(self.enabled),
            "snr_db": None if snr is None else float(snr),
            "cyclo": "active" if o.get("cyclo_active") else "standby",
            "rmt": ("capped %.2f" % o.get("rmt_cap", 0.0)) if not o.get("bypass_all", True) else "bypass",
            "sr": "active" if o.get("sr_active") else "standby",
            "reason": str(o.get("reason", "disabled")),
            "blocks": int(self.blocks),
        }

    def _ema(self, prev, target):
        # 初回は安全側0から開始 (いきなり目標値へ跳ばせない)
        if prev is None:
            prev = 0.0
        return prev + self.alpha * (target - prev)

    def process_metrics(self, metrics):
        """統合判定 → 安全パラメータ辞書 (例外時は全バイパス)。"""
        t0 = time.perf_counter()
        out = {"bypass_all": True, "reason": "disabled",
               "cyclo_active": False, "blend_attack_limit": 0.25,
               "rmt_cap": 0.0, "sr_active": False,
               "snr_smooth": 0.0, "processing_ms": 0.0}
        try:
            if not self.enabled:
                return out
            if not isinstance(metrics, dict):
                out["reason"] = "bad-metrics"
                return out
            snr = _f(metrics.get("snr_db", 0.0))
            cpu = _f(metrics.get("cpu_percent", 0.0))
            clipped = bool(metrics.get("clipped", False))
            pilot_conf = _f(metrics.get("pilot_confidence", 0.0))
            degraded = bool(metrics.get("audio_degraded", False))
            # 非有限の混入チェック (各値は_fで有限化済みだが、元の欠損扱いは別途)
            for k in ("snr_db", "cpu_percent", "pilot_confidence"):
                v = metrics.get(k, 0.0)
                try:
                    if not math.isfinite(float(v)):
                        out["reason"] = "failsafe-nonfinite"
                        return out
                except (TypeError, ValueError):
                    out["reason"] = "failsafe-badtype"
                    return out

            self._snr = self._ema(self._snr, snr)
            self._cpu = self._ema(self._cpu, cpu)
            self.blocks += 1

            strong = self._snr >= self.strong_snr
            weak = self._snr <= self.weak_snr

            # 音質劣化ラッチ: 一度劣化したら回復は緩やか (ヒステリシス)。
            # 弱信号中は回復させず、非弱信号で8ブロック連続クリアしたら解除。
            if degraded:
                self._degraded_latch = True
                self._clear_n = 0
            elif self._degraded_latch:
                if weak:
                    self._clear_n = 0
                else:
                    self._clear_n += 1
                    if self._clear_n >= 8:
                        self._degraded_latch = False

            # RMT上限: 強信号→小、弱信号→設定上限、劣化ラッチ→0
            if self._degraded_latch:
                rmt = 0.0
                reason = "audio-degraded"
            elif strong:
                rmt = self.rmt_strong_cap
                reason = "strong-signal"
            elif weak:
                rmt = 0.65
                reason = "weak-signal"
            else:
                rmt = 0.2
                reason = "normal"
            # CPU逼迫時はRMTから先に弱める
            if self._cpu >= self.cpu_max:
                rmt = 0.0
                reason = "cpu-max"
            elif self._cpu >= self.cpu_warn:
                rmt *= 0.3
                reason = "cpu-warn"
            if clipped:
                rmt = 0.0
                reason = "clipped"
            self._rmt_cap = self._ema(self._rmt_cap, rmt)

            # SR: 弱信号・中間confidence・非クリップでのみ有効
            sr = (weak and not clipped
                  and 0.3 <= pilot_conf <= 0.7
                  and self._cpu < self.cpu_max)
            # cyclo助言: 弱信号またはconfidence低で有効 (PLL置換ではなく助言)
            cyclo = weak or pilot_conf < 0.55
            out.update({
                "bypass_all": False, "reason": reason,
                "cyclo_active": bool(cyclo),
                "blend_attack_limit": 0.05 if (cyclo and pilot_conf < 0.55) else 0.25,
                "rmt_cap": float(max(0.0, self._rmt_cap)),
                "sr_active": bool(sr),
                "snr_smooth": float(self._snr),
            })
        except Exception:
            out.update({"bypass_all": True, "reason": "failsafe-exception",
                        "cyclo_active": False, "rmt_cap": 0.0,
                        "sr_active": False})
        out["processing_ms"] = (time.perf_counter() - t0) * 1000.0
        self._last_out = dict(out)
        return out
