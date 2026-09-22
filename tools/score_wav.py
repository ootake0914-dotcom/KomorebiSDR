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


def score(path):
    mono, l, r, sr = load_wav(path)
    # 無音冒頭・末尾を除外 (全域の中央80%で評価)
    n = len(mono)
    seg = slice(n // 10, n * 9 // 10)
    m, l, r = mono[seg], l[seg], r[seg]
    prog = band_pow(m, sr, 300, 3000)
    hiss = band_pow(m, sr, 6000, min(12000, sr / 2 - 100))
    snr = 10.0 * float(np.log10(prog / hiss))
    corr = float(np.corrcoef(l, r)[0, 1]) if len(l) > 1 else 1.0
    hum = 10.0 * float(np.log10(
        (band_pow(m, sr, 45, 55) + band_pow(m, sr, 95, 105)) / prog))
    rms = float(20.0 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-18))
    return {"file": os.path.basename(path), "snr_db": round(snr, 1),
            "stereo_corr": round(corr, 3), "hum_db": round(hum, 1),
            "rms_dbfs": round(rms, 1)}


def main(argv):
    files = [a for a in argv if not a.startswith("--")]
    out = None
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
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
