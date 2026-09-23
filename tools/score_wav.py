"""OSS比較用WAV採点器。

同一番組の録音WAV (48kHz推奨・他レート可) を複数受け取り、
以下を採点して比較表を出す:
- 音声SNR proxy (4kHz以上のヒス床 vs 300Hz-3kHz番組帯)
- ステレオ分離度 proxy (L/R相関。1.0=モノラル/同一、低いほど分離)
- ハム量 (50/100Hz線成分 vs 番組帯)
- 音量正規化済みRMS (聴感レベルの参考)

Usage:
  python tools/score_wav.py a.wav b.wav [--out result.json]
"""

import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        ch = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        data = w.readframes(n)
    if sw == 1:
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 127.5) / 127.5
    elif sw == 2:
        x = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
    else:
        raise ValueError(f"unsupported width: {sw}")
    if ch == 2:
        return (x[0::2] + x[1::2]) * 0.5, x[0::2].copy(), x[1::2].copy(), sr
    return x, x.copy(), x.copy(), sr


def band_pow(x, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-18
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (f >= lo) & (f < hi)
    return float(np.sum(spec[m])) if np.any(m) else 1e-18


# --- STOI (簡易標準実装。numpyのみ。10kHz/15帯域/384ms文脈) ---
_STOI_FS = 10000.0
_STOI_NFFT = 256
_STOI_HOP = 128
_STOI_CTX = 30
_STOI_BETA_DB = -15.0
_STOI_CF = [150.0 * (2.0 ** (j / 3.0)) for j in range(15)]


def _to_10k(x, sr):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr != _STOI_FS:
        # FFT brickwall (5kHz) 後に線形補間で10kHzへ
        n = len(x)
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / float(sr))
        spec[f > 5000.0] = 0.0
        x = np.fft.irfft(spec, n)
        t_old = np.arange(n) / float(sr)
        n_new = int(round(n * _STOI_FS / float(sr)))
        x = np.interp(np.arange(n_new) / _STOI_FS, t_old, x)
    return x


def _third_octave_bands(x):
    n = len(x)
    wins = []
    for s in range(0, max(n - _STOI_NFFT + 1, 1), _STOI_HOP):
        wins.append(x[s:s + _STOI_NFFT] * np.hanning(_STOI_NFFT))
    if not wins:
        return np.zeros((15, 1))
    W = np.stack(wins, axis=1)  # (256, T)
    spec = np.abs(np.fft.rfft(W, axis=0)) ** 2  # (129, T)
    freqs = np.fft.rfftfreq(_STOI_NFFT, 1.0 / _STOI_FS)
    bands = np.zeros((15, W.shape[1]))
    for j, cf in enumerate(_STOI_CF):
        lo, hi = cf * (2.0 ** (-1.0 / 6.0)), cf * (2.0 ** (1.0 / 6.0))
        m = (freqs >= lo) & (freqs < hi)
        if np.any(m):
            bands[j] = np.sum(spec[m], axis=0)
    return np.maximum(bands, 1e-18)


def stoi(clean, proc, sr=48000):
    """cleanを参照とするSTOI (0〜1)。procはcleanと時間整合していること。"""
    xc = _to_10k(clean, sr)
    xp = _to_10k(proc, sr)
    n = min(len(xc), len(xp))
    xc, xp = xc[:n], xp[:n]
    X = _third_octave_bands(xc)
    Y = _third_octave_bands(xp)
    T = X.shape[1]
    if T <= _STOI_CTX:
        return 0.0
    # 無音フレーム除外 (全帯域和が最大-40dB未満)
    e = np.sum(X, axis=0)
    keep = e > (np.max(e) * 1e-4)
    c = 10.0 ** (-_STOI_BETA_DB / 20.0)
    ds = []
    for m in range(_STOI_CTX, T):
        seg = slice(m - _STOI_CTX, m)
        if not bool(np.all(keep[seg])):
            continue
        for j in range(15):
            xv = X[j, seg].copy()
            yv = Y[j, seg].copy()
            nx = float(np.sqrt(np.sum(xv ** 2))) + 1e-18
            ny = float(np.sqrt(np.sum(yv ** 2))) + 1e-18
            # 論文順序: 包絡のままスケール→クリップ→平均除去→相関
            yv = yv * (nx / ny)
            if ny < 1e-12:
                # 処理後が無音＝了解度ゼロ (定数相関の不定値を0に確定)
                ds.append(0.0)
                continue
            yv = np.minimum(yv, (1.0 + c) * xv)
            xv -= np.mean(xv)
            yv -= np.mean(yv)
            denom = float(np.sqrt(np.sum(xv ** 2) * np.sum(yv ** 2))) + 1e-18
            ds.append(float(np.sum(xv * yv) / denom))
    if not ds:
        return 0.0
    return float(np.clip(np.mean(ds), -1.0, 1.0))


def score(path):
    mono, l, r, sr = load_wav(path)
    # 無音冒頭・末尾を除外 (全域の中央80%で評価)
    n = len(mono)
    seg = slice(n // 10, n * 9 // 10)
    m, l, r = mono[seg], l[seg], r[seg]
    prog = band_pow(m, sr, 300, 3000)
    hiss = band_pow(m, sr, 6000, min(12000, sr / 2 - 100))
    snr = 10.0 * float(np.log10(prog / hiss))
    # 可聴ヒス (5.5-11kHz。STOIは4.3kHz以上盲目のため別指標が必須。
    # NFMで+22dB可聴ヒスがSTOI=1.000を通過した実例あり)
    ear_hi = min(11000, sr / 2 - 100)
    ear_hiss = 10.0 * float(np.log10(band_pow(m, sr, 5500, ear_hi) / prog))
    corr = float(np.corrcoef(l, r)[0, 1]) if len(l) > 1 else 1.0
    hum = 10.0 * float(np.log10(
        (band_pow(m, sr, 45, 55) + band_pow(m, sr, 95, 105)) / prog))
    rms = float(20.0 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-18))
    return {"file": os.path.basename(path), "snr_db": round(snr, 1),
            "stereo_corr": round(corr, 3), "hum_db": round(hum, 1),
            "ear_hiss_db": round(ear_hiss, 1),
            "rms_dbfs": round(rms, 1)}


def main(argv):
    files = [a for a in argv if not a.startswith("--")]
    out = None
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    if "--stoi" in argv and len(files) >= 2:
        # 先頭がclean参照、残りが評価対象
        mono_c, _, _, sr_c = load_wav(files[0])
        results = []
        for f in files[1:]:
            try:
                mono_p, _, _, sr_p = load_wav(f)
                sr = sr_c if sr_c == sr_p else 48000
                v = stoi(mono_c, mono_p, sr=sr)
                r = {"file": os.path.basename(f), "stoi": round(v, 4)}
            except Exception as e:
                r = {"file": os.path.basename(f), "error": str(e)}
            results.append(r)
            print(r)
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(results, fh, ensure_ascii=False, indent=2)
        return 0
    if not files:
        print(__doc__)
        return 1
    results = []
    for f in files:
        try:
            r = score(f)
        except Exception as e:
            r = {"file": os.path.basename(f), "error": str(e)}
        results.append(r)
        print(r)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
