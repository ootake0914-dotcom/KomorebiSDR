"""CER了解度AB (audio_abプロジェクト)。

VOICEVOX日本語TTS (原稿既知) → SSB雑音チェーン → NR off/on →
faster-whisper書き起こし → CER + Scoreq MOS + ブラインド試聴ペア。
母音プロキシより言語内容がある分、本物の了解度に近い。

Usage:
  python audio_ab/cer_ab.py [--snrs 0,5,10] [--out audio_ab/out]
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsp import SdrDspPipeline

FS = 48000
VV = "http://127.0.0.1:50021"
SPEAKER = 1  # ずんだもんノーマル

# 平易な原稿 (カタカナ・英字・数字はWhisper誤読の種なので避ける)
SENTENCES = [
    "おはようございます。きょうも良い天気です。",
    "交通情報をお伝えします。電車は順調です。",
    "気温はぐんぐん上がります。水分をとりましょう。",
    "こちらは放送です。聞こえ方を教えてください。",
]


def tts(text, speaker=SPEAKER):
    q = urllib.parse.quote(text)
    req = urllib.request.Request(
        f"{VV}/audio_query?text={q}&speaker={speaker}", method="POST")
    aq = json.load(urllib.request.urlopen(req, data=b""))
    req2 = urllib.request.Request(
        f"{VV}/synthesis?speaker={speaker}",
        data=json.dumps(aq).encode(), headers={"Content-Type": "application/json"},
        method="POST")
    raw = urllib.request.urlopen(req2).read()
    # wav parse (VOICEVOXは24kHz mono 16bit)
    import io
    with wave.open(io.BytesIO(raw), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        data = w.readframes(w.getnframes())
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    if sr != FS:
        t_old = np.arange(len(x)) / sr
        t_new = np.arange(int(len(x) * FS / sr)) / FS
        x = np.interp(t_new, t_old, x).astype(np.float32)
    return x


def norm_ja(s):
    import re
    s = re.sub(r"[\s、。！？「」『』（）・…ー]+", "", s)
    return s


def cer(truth, hyp):
    a, b = norm_ja(truth), norm_ja(hyp)
    n, m = len(a), len(b)
    if n == 0:
        return 1.0 if m else 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        ndp = [i] + [0] * m
        for j in range(1, m + 1):
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + (a[i - 1] != b[j - 1]))
        dp = ndp
    return dp[m] / n


def main(argv):
    snrs = [0.0, 5.0, 10.0]
    out = os.path.join(ROOT, "audio_ab", "out")
    i = 0
    while i < len(argv):
        if argv[i] == "--snrs" and i + 1 < len(argv):
            snrs = [float(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        else:
            i += 1
    os.makedirs(out, exist_ok=True)
    ttsdir = os.path.join(ROOT, "audio_ab", "tts")
    os.makedirs(ttsdir, exist_ok=True)

    # 1. TTS生成 (キャッシュ)
    refs = []
    for k, s in enumerate(SENTENCES):
        p = os.path.join(ttsdir, f"s{k}.wav")
        if not os.path.exists(p):
            x = tts(s)
            with wave.open(p, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(FS)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
            print(f"tts {k}: {len(x) / FS:.1f}s", flush=True)
        with wave.open(p, "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16).astype(np.float32) / 32768.0
        refs.append((s, x))
    full_text = "".join(s for s, _ in refs)
    prog = np.concatenate([x for _, x in refs])

    # 2. Whisper用意 (48k直入れが壊れているビルドのため16k自前変換)
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "audio_ab"))
    from score_noref import AsrSpotter, MosScorer
    spotter = AsrSpotter("small")

    def asr(x):
        return spotter.transcribe(np.asarray(x, dtype=np.float32), FS)

    # 3. クリーン床の校正
    base_cer = cer(full_text, asr(prog))
    print(f"clean CER floor: {base_cer:.3f}", flush=True)

    # 4. SSB雑音チェーン ±NR
    mos = MosScorer()

    results = {"clean_cer_floor": round(base_cer, 4)}
    import random
    key = {}
    try:
        with open(os.path.join(out, "scores.json"), encoding="utf-8") as f:
            scores_all = json.load(f)
    except Exception:
        scores_all = {}
    try:
        with open(os.path.join(out, "key.json"), encoding="utf-8") as f:
            key = json.load(f)
    except Exception:
        pass

    for snr_db in snrs:
        import sys as _sys
        _sys.path.insert(0, os.path.join(ROOT, "audio_ab"))
        from ssb_synth import noisy_ssb_iq
        n = len(prog)
        iq = noisy_ssb_iq(prog, snr_db, seed=21)
        bl = 2208
        nb = n // bl
        outs = {}
        for tag, en in (("off", False), ("on", True)):
            d = SdrDspPipeline(1152000, FS)
            d.nbm_nr_enabled = en
            ys = [d.demodulate_ssb(iq[k * bl:(k + 1) * bl], "USB") for k in range(nb)]
            outs[tag] = np.concatenate(ys).astype(np.float32)
        item = f"C1_cer_ssb_snr{snr_db:g}"
        row = {}
        wavs = {}
        for tag in ("off", "on"):
            y = outs[tag]
            row[f"cer_{tag}"] = round(cer(full_text, asr(y)), 4)
            try:
                row[f"mos_{tag}"] = round(float(mos.score_array(y, FS)), 4)
            except Exception as e:
                row[f"mos_{tag}"] = f"ERR {e}"
            wavs[tag] = y
        row["cer_clean_ref"] = round(base_cer, 4)
        flip = random.Random(item).random() < 0.5
        A, B = (wavs["off"], wavs["on"]) if flip else (wavs["on"], wavs["off"])
        key[item] = {"A": ("off" if flip else "on"), "B": ("on" if flip else "off")}
        for tag, y in (("A", A), ("B", B)):
            a = np.stack([y, y], axis=1)
            with wave.open(os.path.join(out, f"{item}_{tag}.wav"), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(FS)
                w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        scores_all[item] = row
        print(f"{item}: CER off={row['cer_off']} on={row['cer_on']} "
              f"MOS off={row['mos_off']} on={row['mos_on']}", flush=True)

    with open(os.path.join(out, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores_all, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
