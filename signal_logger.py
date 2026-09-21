"""
Signal health logger for station-specific fluctuation diagnosis.

特定局だけで音が揺れる場合の切り分け用: パイロットロック・ブレンド・NR・
マルチパス・AFC・S値などを1HzでCSV記録する。後から要約表示し、
最も暴れている指標から犯人候補 (パイロット不安定/反射波/ヒス推定/AFC/電界変動)
を推定する。記録は軽量 (1Hz・上限付きローテーション)。
"""

import csv
import os
import time

from config import config_dir

LOG_NAME = "signal_log.csv"
MAX_BYTES = 2 * 1024 * 1024  # 2MBでローテーション (約10時間分)

# ブランキング: 起動直後と選局直後はAFC・PLL等の過渡が載るため記録しない
WARMUP_SEC = 15.0   # ワーカー開始後の無記録期間
TUNE_BLANK_SEC = 8.0  # 最終選局からの無記録期間


def is_settled(now_m: float, tune_m: float, worker_t0_m: float,
               warmup: float = WARMUP_SEC, blank: float = TUNE_BLANK_SEC) -> bool:
    """記録してよい定常状態か。起動/選局の過渡を除外する純粋関数。"""
    try:
        if now_m - float(worker_t0_m) < float(warmup):
            return False
        if now_m - float(tune_m) < float(blank):
            return False
    except (TypeError, ValueError):
        return False
    return True

COLUMNS = [
    "ts", "freq_hz", "mode", "gain_db",
    "pilot_lock", "blend", "nr_gain", "cut_hz", "wiener_gain",
    "multipath_gain", "afc_hz", "s_units",
    "snr_db", "audio_snr_db",
]


class SignalLogger:
    """受信健康状態のCSVロガー (スレッドセーフ不要: ワーカーからのみ呼ぶ想定)"""

    def __init__(self, path: str = None, max_bytes: int = MAX_BYTES):
        d = config_dir()
        self.path = path or os.path.join(d, LOG_NAME)
        self.max_bytes = int(max_bytes)
        self._need_header = not (os.path.exists(self.path)
                                 and os.path.getsize(self.path) > 0)

    def _rotate(self):
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) >= self.max_bytes:
                bak = self.path + ".1"
                try:
                    if os.path.exists(bak):
                        os.remove(bak)
                except OSError:
                    pass
                os.replace(self.path, bak)
                self._need_header = True
        except OSError:
            pass

    def log(self, row: dict) -> bool:
        """1行記録する。成功でTrue。"""
        try:
            self._rotate()
            write_header = self._need_header
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                    self._need_header = False
                out = {"ts": time.time()}
                out.update(row)
                w.writerow({k: out.get(k, "") for k in COLUMNS})
            return True
        except OSError:
            return False


def load_rows(path: str) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {}
            for k in COLUMNS:
                v = r.get(k, "")
                if k == "mode":
                    row[k] = v
                    continue
                try:
                    row[k] = float(v) if v != "" else float("nan")
                except (ValueError, TypeError):
                    row[k] = float("nan")
            row["ts"] = row.get("ts", float("nan"))
            rows.append(row)
    return rows


def summarize(path: str, freq_hz: int = None) -> dict:
    """ログを要約し、指標ごとの変動と犯人候補ヒントを返す。
    freq_hz指定時はその局 (±50kHz) のみを対象にする。"""
    import numpy as np
    rows = load_rows(path)
    if freq_hz is not None:
        rows = [r for r in rows if abs(r.get("freq_hz", -1) - freq_hz) <= 50000]
    if len(rows) < 10:
        return {"n": len(rows), "hints": ["データ不足 (10行以上記録してから判定)"]}
    keys = ["pilot_lock", "blend", "nr_gain", "cut_hz", "wiener_gain",
            "multipath_gain", "afc_hz", "s_units", "gain_db"]
    stats = {}
    for k in keys:
        v = np.array([r.get(k, float("nan")) for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        stats[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "min": float(np.min(v)), "max": float(np.max(v))}
    hints = []
    b = stats.get("blend", {}).get("std", 0.0)
    l = stats.get("pilot_lock", {}).get("std", 0.0)
    mp = stats.get("multipath_gain", {}).get("mean", 1.0)
    nr = stats.get("nr_gain", {}).get("std", 0.0)
    afc = stats.get("afc_hz", {})
    afc_range = abs(afc.get("max", 0.0) - afc.get("min", 0.0)) if afc else 0.0
    s = stats.get("s_units", {}).get("std", 0.0)
    g = stats.get("gain_db", {}).get("std", 0.0)
    if b > 0.15 or l > 0.15:
        hints.append("パイロット不安定: ステレオ⇔モノラル往復の可能性 (マルチパス/弱電界)")
    if mp < 0.9:
        hints.append("反射波検出中: マルチパス抑圧が介入 (都市部反射・山岳反射)")
    if nr > 0.15:
        hints.append("ヒス推定が変動: NRの効きが番組連動 (弱電界ぎりぎり)")
    if afc_range > 500.0:
        hints.append("AFC大変動: 送信周波数ドリフト or ドングル温度ドリフト")
    if s > 1.5:
        hints.append("電界変動大: フェージング (遠距離局の対流圏変動)")
    if g > 3.0:
        hints.append("ゲイン探索中: 自動ゲインが収束していない (強入力・混変調の疑い)")
    if not hints:
        hints.append("電波側は安定: 揺れは番組内容・音声処理側の可能性")
    # 変動の大きい順に指標を並べる (正規化なしの参考順)
    order = sorted(stats, key=lambda k: stats[k]["std"], reverse=True)
    return {"n": len(rows), "stats": stats, "order": order, "hints": hints}


def diagnose_environment(path: str, freq_hz: int = None, recent_n: int = 120) -> dict:
    """現在の受信環境を多角的に精密診断する。
    台風・強風によるマルチパスフェージング、19kHz/38kHzステレオ副搬送波の乱れ、
    および適応フィルターの過剰反応によるミュージカルノイズ(ピロピロ音)の危険度を算出。
    """
    import numpy as np
    rows = load_rows(path)
    if freq_hz is not None:
        rows = [r for r in rows if abs(r.get("freq_hz", -1) - freq_hz) <= 50000]
    if len(rows) < 10:
        return {"status": "error", "message": "データ不足 (10行以上記録してから実行してください)"}

    # 直近 recent_n 行（直近約2分間）を分析
    recent = rows[-recent_n:] if len(rows) > recent_n else rows
    n = len(recent)

    def get_arr(key: str, default: float = 0.0) -> np.ndarray:
        v = np.array([r.get(key, np.nan) for r in recent], dtype=np.float64)
        v = v[np.isfinite(v)]
        return v if len(v) > 0 else np.array([default], dtype=np.float64)

    s_arr = get_arr("s_units", 9.0)
    mp_arr = get_arr("multipath_gain", 1.0)
    pl_arr = get_arr("pilot_lock", 1.0)
    bl_arr = get_arr("blend", 1.0)
    w_arr = get_arr("wiener_gain", 1.0)
    nr_arr = get_arr("nr_gain", 1.0)
    cut_arr = get_arr("cut_hz", 15000.0)
    gain_arr = get_arr("gain_db", 30.0)
    afc_arr = get_arr("afc_hz", 0.0)
    snr_arr = get_arr("snr_db", 0.0)
    aud_arr = get_arr("audio_snr_db", 0.0)

    # 1. 電界フェージング診断 (RF Fading)
    s_depth = float(np.max(s_arr) - np.min(s_arr))
    s_std = float(np.std(s_arr))
    is_fading = s_depth > 0.8 or s_std > 0.25

    # 2. マルチパス干渉 (Multipath Distortion)
    mp_mean = float(np.mean(mp_arr))
    mp_min = float(np.min(mp_arr))
    mp_active_ratio = float(np.mean(mp_arr < 0.85))
    if mp_mean < 0.60 or mp_min < 0.40:
        mp_severity = "深刻 (SEVERE: 強風・反射物揺れによる激しい多重波)"
    elif mp_mean < 0.85 or mp_active_ratio > 0.3:
        mp_severity = "中等度 (MODERATE: 反射波が継続混入中)"
    else:
        mp_severity = "軽微 (CLEAN: 直接波が支配的)"

    # 3. 19kHz/38kHz ステレオ副搬送波・パイロット同期
    pl_min = float(np.min(pl_arr))
    pl_mean = float(np.mean(pl_arr))
    pl_unlock_ratio = float(np.mean(pl_arr < 0.70))
    # ブレンドのハンティング回数 (0.4以下と0.7以上の横断)
    b_crosses = 0
    for i in range(1, len(bl_arr)):
        if (bl_arr[i - 1] < 0.4 and bl_arr[i] > 0.7) or (bl_arr[i - 1] > 0.7 and bl_arr[i] < 0.4):
            b_crosses += 1

    # 4. ミュージカルノイズ (ピロピロ音) 危険度指数 (0 - 100%)
    wiener_deep_ratio = float(np.mean(w_arr < 0.40))
    cut_diffs = np.abs(np.diff(cut_arr)) if len(cut_arr) > 1 else np.array([0.0])
    cut_chatter_ratio = float(np.mean(cut_diffs > 800.0))
    
    # 複合リスクスコア
    risk_score = int(np.clip(
        (wiener_deep_ratio * 45.0) +
        (cut_chatter_ratio * 35.0) +
        ((1.0 - mp_mean) * 20.0),
        0.0, 100.0
    ))
    if risk_score >= 60:
        risk_level = "CRITICAL (ピロピロ・バーディ音 激甚発生中)"
    elif risk_score >= 35:
        risk_level = "HIGH (ピロピロ音・違和感が顕著に聴こえるレベル)"
    elif risk_score >= 15:
        risk_level = "MODERATE (わずかな残差・変調感あり)"
    else:
        risk_level = "LOW (極めて自然・ピロピロ音なし)"

    # 5. 推奨アクション
    recommendations = []
    if pl_unlock_ratio > 0.2 or mp_mean < 0.75 or b_crosses >= 2:
        recommendations.append("[Stereo] ボタンを押して「MONO」に切り替える (台風フェージングによる38kHz副搬送波の位相乱れを完全遮断)")
    if wiener_deep_ratio > 0.3 or cut_chatter_ratio > 0.2:
        recommendations.append("[NR] ボタンを押して「OFF」にする (適応ウィーナーフィルターの急峻な追従によるミュージカルノイズを停止)")
    if float(np.std(gain_arr)) > 2.0 or is_fading:
        recommendations.append("[Auto Gain] を手動ゲイン固定にする (電波の揺れに追従するAGCハンティングを抑止)")
    if not recommendations:
        recommendations.append("現在の受信状態は極めて安定しています。特に対処は不要です。")

    latest_freq_mhz = float(recent[-1].get("freq_hz", 0.0)) / 1e6

    return {
        "status": "ok",
        "sample_count": n,
        "latest_freq_mhz": latest_freq_mhz,
        "fading": {
            "s_mean": float(np.mean(s_arr)),
            "s_depth": s_depth,
            "s_std": s_std,
            "is_fading": is_fading,
            "status": "台風・強風フェージング検出 (Deep Fast Fading)" if is_fading else "安定電界",
        },
        "multipath": {
            "gain_mean": mp_mean,
            "gain_min": mp_min,
            "active_ratio_pct": mp_active_ratio * 100.0,
            "severity": mp_severity,
        },
        "stereo_stability": {
            "pilot_lock_mean": pl_mean,
            "pilot_lock_min": pl_min,
            "pilot_unlock_ratio_pct": pl_unlock_ratio * 100.0,
            "blend_hunting_count": b_crosses,
        },
        "musical_noise": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "wiener_deep_suppression_pct": wiener_deep_ratio * 100.0,
            "cutoff_chatter_pct": cut_chatter_ratio * 100.0,
        },
        "recommendations": recommendations,
    }


def format_environment_report(diag: dict) -> str:
    """診断結果を人間が読みやすい日本語テキストレポートに整形"""
    if diag.get("status") != "ok":
        return f"[エラー] {diag.get('message', '不明なエラー')}"

    f = diag["fading"]
    m = diag["multipath"]
    s = diag["stereo_stability"]
    mn = diag["musical_noise"]

    lines = [
        "============================================================",
        f"  SDR 受信環境 & 台風・ピロピロノイズ精密診断レポート",
        f"  対象周波数: {diag['latest_freq_mhz']:.2f} MHz (直近 {diag['sample_count']} 秒間の実測データ)",
        "============================================================",
        f"【1. 電波フェージング (台風・風揺れ)】: {f['status']}",
        f"  - S値 平均: S{f['s_mean']:.1f} | 変動幅 (P-P): {f['s_depth']:.1f} S単位 (std: {f['s_std']:.2f})",
        "",
        f"【2. マルチパス反射波干渉】: {m['severity']}",
        f"  - マルチパス係数 平均: {m['gain_mean']:.2f} (最小値: {m['gain_min']:.2f})",
        f"  - 反射波抑圧 発動率: {m['active_ratio_pct']:.1f}%",
        "",
        f"【3. ステレオ副搬送波 (38kHz) 安定度】",
        f"  - 19kHzパイロットロック: 平均 {s['pilot_lock_mean']:.2f} (最低: {s['pilot_lock_min']:.2f})",
        f"  - パイロット脱調・低下率: {s['pilot_unlock_ratio_pct']:.1f}%",
        f"  - ステレオ⇔モノラル急変 (ハンティング) 回数: {s['blend_hunting_count']} 回",
        "",
        f"【4. ピロピロ音 (ミュージカルノイズ) 危険度】: {mn['risk_level']}",
        f"  - リスク総合スコア: {mn['risk_score']} / 100",
        f"  - ウィーナー強抑圧率 (wiener<0.40): {mn['wiener_deep_suppression_pct']:.1f}%",
        f"  - カットオフ周波数の急変頻度: {mn['cutoff_chatter_pct']:.1f}%",
        "",
        "------------------------------------------------------------",
        "【診断結論と推奨対処アクション】",
    ]
    for idx, rec in enumerate(diag["recommendations"], start=1):
        lines.append(f"  {idx}. {rec}")
    lines.append("============================================================")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import json
    target = os.path.join(config_dir(), LOG_NAME)
    freq = None
    show_env = True

    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        target = args[0]
        args = args[1:]
    if args and not args[0].startswith("-"):
        try:
            freq = int(args[0])
            args = args[1:]
        except ValueError:
            pass

    if "--json" in args:
        print(json.dumps(diagnose_environment(target, freq), indent=2, ensure_ascii=False))
    elif "--summary-json" in args:
        print(json.dumps(summarize(target, freq), indent=2, ensure_ascii=False))
    else:
        diag = diagnose_environment(target, freq)
        print(format_environment_report(diag))
