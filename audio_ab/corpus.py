"""TTSコーパス管理 (audio_ab)。

1文=1試行としてCERの標本数を稼ぐため、文を増やし話者も混ぜる。
VOICEVOX (127.0.0.1:50021) で合成し audio_ab/tts/ にキャッシュする
(gitignore済)。参照文は norm_ja で表記ゆれを落として比較する。
"""

import io
import json
import os
import re
import urllib.parse
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 48000
VV = "http://127.0.0.1:50021"

# カタカナ・英字・数字はWhisper誤読の種なので避けた平易な文
SENTENCES = [
    "おはようございます。きょうも良い天気です。",
    "交通情報をお伝えします。電車は順調です。",
    "気温はぐんぐん上がります。水分をとりましょう。",
    "こちらは放送です。聞こえ方を教えてください。",
    "夕方から雨が降るそうです。傘を持って出かけましょう。",
    "新しいお知らせがあります。静かに聞いてください。",
    "山の向こうに大きな虹が見えます。",
    "今日は早めに帰って休みましょう。",
    "この道をまっすぐ行くと駅に着きます。",
    "みなさん、こんにちは。よい一日を。",
]
# ずんだもんノーマル / 四国めたんノーマル / 春日部つむぎ / 玄野武宏
SPEAKERS = (3, 2, 8, 11)


def norm_ja(s: str) -> str:
    return re.sub(r"[\s、。！？「」『』（）・…ー]+", "", s)


def tts(text: str, speaker: int = 3) -> np.ndarray:
    """VOICEVOX合成→48kHz mono float32。"""
    q = urllib.parse.quote(text)
    req = urllib.request.Request(
        f"{VV}/audio_query?text={q}&speaker={speaker}", method="POST")
    aq = json.load(urllib.request.urlopen(req, data=b""))
    req2 = urllib.request.Request(
        f"{VV}/synthesis?speaker={speaker}",
        data=json.dumps(aq).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    raw = urllib.request.urlopen(req2).read()
    with wave.open(io.BytesIO(raw), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = w.readframes(w.getnframes())
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    if sr != FS:
        t_old = np.arange(len(x)) / sr
        t_new = np.arange(int(len(x) * FS / sr)) / FS
        x = np.interp(t_new, t_old, x).astype(np.float32)
    return x


def load(utts: int = None, speakers=SPEAKERS, refresh: bool = False):
    """文×話者を割り当てて読み込む。1文=1 dict {text, x, speaker, index}。"""
    ttsdir = os.path.join(ROOT, "audio_ab", "tts")
    os.makedirs(ttsdir, exist_ok=True)
    n = len(SENTENCES) if utts is None else min(int(utts), len(SENTENCES))
    out = []
    for i in range(n):
        spk = int(speakers[i % len(speakers)])
        p = os.path.join(ttsdir, f"u{spk:02d}_{i:02d}.wav")
        if refresh or not os.path.exists(p):
            x = tts(SENTENCES[i], spk)
            with wave.open(p, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(FS)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        with wave.open(p, "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16).astype(np.float32) / 32768.0
        out.append({"text": SENTENCES[i], "x": x, "speaker": spk, "index": i})
    return out


def program(items):
    """旧cer_ab互換: コーパスを連結して (full_text, x) を返す。"""
    full = "".join(it["text"] for it in items)
    return full, np.concatenate([it["x"] for it in items])
