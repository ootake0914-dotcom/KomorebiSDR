"""参照あり音声品質指標 (audio_ab)。

CERは「伝わるか」しか見ない。クリーン原稿がある強みを活かし、
レベル・遅延にロバストな参照あり指標を標準装備する:
- SI-SDR: 最適スケール後の信号対歪み比 (AGC/ゲイン差を吸収)
- segSNR: 20msフレームSNRを有声フレームで平均 (±35dBクリップ)
- STOI: tools/score_wav.py の既存実装 (時間整合が前提なので内部で整列)
いずれもDSP出力にAGC遅延・ゲイン差があってそのままでは比較できないため、
相互相関で整列してから評価する。
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))


def xcorr_lag(ref, test, max_lag=4800):
    """testをshiftしてrefに合わせるラグ (正=testが遅れ)。"""
    a = np.asarray(ref, dtype=np.float64).reshape(-1)
    b = np.asarray(test, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    a, b = a[:n] - np.mean(a[:n]), b[:n] - np.mean(b[:n])
    xc = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
    lag = int(np.argmax(xc))
    if lag > n:
        lag -= 2 * n
    return int(np.clip(lag, -max_lag, max_lag))


def shift(x, lag):
    x = np.asarray(x)
    if lag == 0:
        return x
    if lag > 0:
        return np.concatenate((np.zeros(lag, dtype=x.dtype), x))[:len(x)]
    return np.concatenate((x[-lag:], np.zeros(-lag, dtype=x.dtype)))[:len(x)]


def si_sdr(ref, test, align=True, max_lag=4800) -> float:
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    t = np.asarray(test, dtype=np.float64).reshape(-1)
    n = min(len(r), len(t))
    r, t = r[:n], t[:n]
    if align:
        t = shift(t, xcorr_lag(r, t, max_lag))
    r = r - np.mean(r)
    t = t - np.mean(t)
    denom = float(np.dot(r, r)) + 1e-18
    alpha = float(np.dot(t, r)) / denom
    target = alpha * r
    noise = t - target
    en = float(np.sum(noise ** 2))
    if en <= 1e-24:
        return 100.0
    return float(10.0 * np.log10(float(np.sum(target ** 2)) / en))


def seg_snr(ref, test, fs=48000, align=True, frame_ms=20.0, clip_db=35.0,
            floor_db=-40.0) -> float:
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    t = np.asarray(test, dtype=np.float64).reshape(-1)
    n = min(len(r), len(t))
    r, t = r[:n], t[:n]
    if align:
        t = shift(t, xcorr_lag(r, t))
    r = r - np.mean(r)
    t = t - np.mean(t)
    alpha = float(np.dot(t, r)) / (float(np.dot(r, r)) + 1e-18)
    t = t - alpha * r  # 残差=歪み
    fl = max(8, int(fs * frame_ms / 1000.0))
    vals = []
    for k in range(0, n - fl + 1, fl):
        er = float(np.sum(r[k:k + fl] ** 2))
        en = float(np.sum(t[k:k + fl] ** 2))
        if er <= 1e-18:
            continue
        vals.append(10.0 * np.log10(er / (en + 1e-24)))
    if not vals:
        return 0.0
    v = np.clip(np.asarray(vals), -clip_db, clip_db)
    keep = np.asarray(vals) >= (np.max(vals) + floor_db)
    if not np.any(keep):
        keep = np.ones_like(v, dtype=bool)
    return float(np.mean(v[keep]))


def stoi_score(ref, test, sr=48000, align=True) -> float:
    from score_wav import stoi
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    t = np.asarray(test, dtype=np.float64).reshape(-1)
    if align:
        t = shift(t, xcorr_lag(r, t))
    n = min(len(r), len(t))
    return float(stoi(r[:n], t[:n], sr))


def evaluate(ref, test, sr=48000) -> dict:
    return {"si_sdr": round(si_sdr(ref, test), 2),
            "seg_snr": round(seg_snr(ref, test, sr), 2),
            "stoi": round(stoi_score(ref, test, sr), 4)}
