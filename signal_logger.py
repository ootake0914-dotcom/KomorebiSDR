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


if __name__ == "__main__":
    import sys
    import json
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config_dir(), LOG_NAME)
    freq = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if not os.path.exists(target):
        print(f"ログがありません: {target}\n"
              f"先に main.py を起動して受信してください (記録は受信中に1Hzで追記されます)。")
        raise SystemExit(2)
    print(json.dumps(summarize(target, freq), indent=2, ensure_ascii=False))
