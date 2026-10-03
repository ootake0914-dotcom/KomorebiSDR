"""耳代替の機械評価ヘルパ (audio_abプロジェクト)。

- Scoreq NR-MOS: torchaudio/torchcodecを経由せずONNXセッション直叩き
  (この環境のtorchaudioはFFmpeg共有DLL不足でwavを読めないため)。
- faster-whisper: 48k直入れが壊れているビルドのため、必ず自前16k化して渡す。

どちらも「ネットで取ってきた耳以上の解析」をローカルで回すための接着層。
"""

import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fft_resample(x, sr_in, sr_out=16000):
    """FFT帯域制限リサンプル (numpyのみ)。音声帯域では透過的。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_in = len(x)
    n_out = int(round(n_in * sr_out / sr_in))
    if n_out < 16:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    X = np.fft.rfft(x)
    nbin_out = n_out // 2 + 1
    if len(X) >= nbin_out:
        Y = X[:nbin_out]
    else:
        Y = np.zeros(nbin_out, dtype=np.complex128)
        Y[:len(X)] = X
    y = np.fft.irfft(Y, n=n_out) * (n_out / n_in)
    return y.astype(np.float32)


def load_wav_mono(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
    else:
        raise ValueError(f"sampwidth {sw}")
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x.astype(np.float32), sr


class MosScorer:
    """Scoreq natural/NRラッパ (ONNX直実行)。"""

    def __init__(self):
        from scoreq import Scoreq
        self._m = Scoreq(data_domain="natural", mode="nr")
        self._inp = self._m.session.get_inputs()[0].name

    def score_array(self, x, sr):
        import torch
        x16 = fft_resample(x, sr, 16000)
        if len(x16) < 16000:
            pad = np.zeros(16000 - len(x16), dtype=np.float32)
            x16 = np.concatenate((x16, pad))
        t = torch.from_numpy(np.asarray([x16], dtype=np.float32))
        with torch.no_grad():
            out = self._m.session.run(None, {self._inp: t.numpy()})[0]
        try:
            return float(out.item() if hasattr(out, "item") else out.flat[0])
        except Exception:
            return float("nan")

    def score_wav(self, path):
        x, sr = load_wav_mono(path)
        return self.score_array(x, sr)


class AesScorer:
    """Meta Audiobox-Aesthetics (PQ/CE/CU/PC)。音声・音楽・効果音の統一品質。

    参照なしの制作品質 (PQ) と楽しさ (CE) が音楽/一般音の自然さを測る。
    既存のScoreq/DNSMOS (音声中心) と相補的。CPUで数秒かかるため
    SHA1キーのディスクキャッシュ (fast.CachedAsrと同型)。
    モデルはHF `facebook/audiobox-aesthetics` (初回DL、gitignore外のHF cache)。
    """

    AXES = ("PQ", "CE", "CU", "PC")

    def __init__(self, cache=True, cache_dir=None, max_files=20000):
        self._p = None
        self.cache = bool(cache)
        self.dir = cache_dir or os.path.join(ROOT, "audio_ab", "out",
                                             "aes_cache")
        if self.cache:
            os.makedirs(self.dir, exist_ok=True)
            self._evict(int(max_files))

    def _evict(self, max_files):
        import glob as _g
        import time as _t
        names = sorted(_g.glob(os.path.join(self.dir, "*.json")),
                       key=os.path.getmtime)
        if len(names) <= max_files:
            return
        keep = max(1, int(max_files * 0.9))
        for p in names[:len(names) - keep]:
            try:
                if _t.time() - os.path.getmtime(p) > 60:
                    os.unlink(p)
            except Exception:
                pass

    def _key(self, x, sr):
        import hashlib
        h = hashlib.sha1()
        h.update(np.asarray(x, dtype=np.float32).tobytes())
        h.update(str(int(sr)).encode())
        return h.hexdigest() + ".json"

    def _predict(self, x, sr):
        if self._p is None:
            from audiobox_aesthetics.infer import initialize_predictor
            self._p = initialize_predictor(None)
        import torch
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.ndim > 1:
            x = x.mean(axis=1)
        out = self._p.forward([{
            "path": torch.from_numpy(x).unsqueeze(0),
            "sample_rate": int(sr),
        }])[0]
        return {k: float(out[k]) for k in self.AXES}

    def score_array(self, x, sr):
        if not self.cache:
            return self._predict(x, sr)
        import json
        p = os.path.join(self.dir, self._key(x, sr))
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if all(k in d for k in self.AXES):
                return {k: float(d[k]) for k in self.AXES}
        except Exception:
            pass
        d = self._predict(x, sr)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f)
        except Exception:
            pass
        return d

    def score_wav(self, path):
        x, sr = load_wav_mono(path)
        return self.score_array(x, sr)


class AsrSpotter:
    """faster-whisperラッパ (16k自前変換・small/int8 CPU)。"""

    def __init__(self, model="small"):
        from faster_whisper import WhisperModel
        self._m = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, x, sr, language="ja", beam_size=5):
        x16 = fft_resample(x, sr, 16000).astype(np.float32)
        segs, _ = self._m.transcribe(x16, language=language, beam_size=beam_size)
        return "".join(s.text for s in segs)


class DnsmosScorer:
    """DNSMOS P.835ラッパ (SIG/BAK/OVRL＋P808)。
    Microsoft DNS-ChallengeのComputeScoreを16k wavで呼ぶ
    (48kを渡すとlibrosa新APIと衝突するため自前16k化)。
    モデルはaudio_ab/models/*.onnx (HF mirror取得、gitignore)。
    """

    def __init__(self, models_dir=None):
        import importlib.util
        self._models = models_dir or os.path.join(ROOT, "audio_ab", "models")
        spec = importlib.util.spec_from_file_location(
            "dnsmos_local",
            "C:/Users/ootak/AppData/Local/Temp/opencode/dnssrc/DNSMOS/dnsmos_local.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._cs = mod.ComputeScore(
            os.path.join(self._models, "sig_bak_ovr.onnx"),
            os.path.join(self._models, "model_v8.onnx"))

    def score_wav16k(self, path16k):
        return self._cs(path16k, 16000, False)

    def score_array(self, x, sr):
        import tempfile
        x16 = fft_resample(x, sr, 16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tp = tf.name
        import wave as _w
        with _w.open(tp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(x16, -1, 1) * 32767).astype(np.int16).tobytes())
        try:
            return self.score_wav16k(tp)
        finally:
            try:
                os.unlink(tp)
            except Exception:
                pass
