"""計測の高速化 (audio_ab)。

Whisperはこの4コア機で1試行~6.6秒かかり、スイープの律速だった。
(スレッド並列はCT2が1推論で全コアを使うため効果なしを実測済み。
プロセス並列も同じ理由で見込みなし。)
そこでASR結果を「デコード後音声のハッシュ」でディスクキャッシュする。
DSPコード・パラメータを変えれば音声が変わり自動で無効化されるため、
キーにコード版を混ぜる必要がない。

Usage:
  from fast import CachedAsr
  asr = CachedAsr("small", beam=5)
  text = asr.transcribe(y, 48000)
"""

import hashlib
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CachedAsr:
    """AsrSpotter互換 + ディスクキャッシュ + beam選択。"""

    def __init__(self, model: str = "small", beam: int = 5,
                 cache: bool = True, cache_dir: str = None):
        from score_noref import AsrSpotter
        self._sp = AsrSpotter(model)
        self.model = model
        self.beam = int(beam)
        self.cache = bool(cache)
        self.dir = cache_dir or os.path.join(ROOT, "audio_ab", "out", "asr_cache")
        if self.cache:
            os.makedirs(self.dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, x, sr) -> str:
        a = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32).reshape(-1))
        h = hashlib.sha1()
        h.update(a.tobytes())
        h.update(f"|{int(sr)}|{self.model}|{self.beam}|ja".encode())
        return h.hexdigest()

    def transcribe(self, x, sr) -> str:
        if not self.cache:
            return self._sp.transcribe(x, sr, beam_size=self.beam)
        key = self._key(x, sr)
        p = os.path.join(self.dir, key + ".txt")
        if os.path.exists(p):
            self.hits += 1
            with open(p, encoding="utf-8") as f:
                return f.read()
        text = self._sp.transcribe(x, sr, beam_size=self.beam)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        self.misses += 1
        return text

    def stats(self) -> str:
        n = self.hits + self.misses
        return (f"asr cache: {self.hits}/{n} hits"
                + ("" if n == 0 else f" ({100.0 * self.hits / n:.0f}%)"))
