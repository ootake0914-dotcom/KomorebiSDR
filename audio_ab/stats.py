"""機械指標の統計ヘルパ (audio_ab)。

CER等は試行間分散が大きく、原稿16.8秒では量子化±0.014だった。
文単位の試行を増やし、平均の信頼区間と対の符号検定で
「1文字差を有意と誤認する」事故を防ぐ。scipy非依存 (numpy + math)。
"""

import math
from statistics import NormalDist

import numpy as np


def bootstrap_ci(x, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """標本平均のブートストラップ信頼区間 (percentile法)。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    if len(x) == 1:
        v = float(x[0])
        return {"mean": v, "lo": v, "hi": v, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0,
                                   100.0 * (1.0 - alpha / 2.0)])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(len(x))}


def sign_test(d, tol: float = 0.0) -> float:
    """対の差dの両側符号検定p値。|d|<=tolの対は無視する。"""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    d = d[np.abs(d) > tol]
    n = len(d)
    if n == 0:
        return 1.0
    k = int(np.sum(d > 0.0))
    k = max(k, n - k)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * p))


def paired(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
           tol: float = 0.0):
    """対応のあるb-aの平均・CI・符号検定。significantはCIが0を外れた時。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    d = b[:n] - a[:n]
    out = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    out["p_sign"] = sign_test(d, tol=tol)
    out["significant"] = bool(out["lo"] > 0.0 or out["hi"] < 0.0)
    return out


def verdict(delta, min_effect: float = 0.01, alpha: float = 0.05) -> str:
    """実用ゲート: 効果量と有意性の両方を満たして初めて改善/悪化と言う。"""
    m = float(delta.get("mean", 0.0))
    if not np.isfinite(m) or abs(m) < min_effect:
        return "no meaningful difference"
    if not bool(delta.get("significant")):
        return "trend only (CI crosses 0)"
    if float(delta.get("p_sign", 1.0)) >= alpha:
        return "trend only (sign test n.s.)"
    return "improvement" if m < 0.0 else "regression"


def holm(pvals, alpha: float = 0.05):
    """Holm法の調整p値 (step-up)。多数バリアントの多重比較補正。"""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    adj = np.empty(m)
    if m == 0:
        return adj
    prev = 0.0
    order = np.argsort(p)
    for rank, idx in enumerate(order):
        prev = max(prev, (m - rank) * float(p[idx]))
        adj[idx] = min(1.0, prev)
    return adj


def holm_reject(pvals, alpha: float = 0.05):
    """Holm補正後に棄却されるか (alpha水準)。"""
    return holm(pvals, alpha) < alpha


def n_for_effect(sigma, effect, alpha: float = 0.05, power: float = 0.8) -> int:
    """対応のある平均差を検出力powerで検出するのに必要な試行数 (正規近似)。
    例: sigma=0.1, effect=0.1 → 8試行。スイープ前に必要数を決める。"""
    s = float(sigma)
    e = float(effect)
    if s <= 0.0 or e <= 0.0:
        return 0
    z_a = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_p = NormalDist().inv_cdf(power)
    return int(np.ceil(((z_a + z_p) ** 2) * (s ** 2) / (e ** 2)))
