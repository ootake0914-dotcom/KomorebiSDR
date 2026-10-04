"""埋め込み分布距離 (audio_ab)。

クリップ集合間の分布距離 (FAD) を EnCodec 埋め込みで測る。
radiko_lab のオラクル比較や NR on/off 集合比較など、
「数クリップしかない条件」での相対 A/B 用。

注意:
- この fadtk 版に KAD は無いため FAD で代替。小標本バイアスがあるので
  絶対値ではなく同一条件の相対比較に使う (集合サイズを固定すること)。
- 埋め込みは SHA1 キャッシュ (out/emb_cache)。モデルは encodec 24k
  (初回DL、HF cache)。
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audio_ab"))

_MODEL = None
CACHE_DIR = os.path.join(ROOT, "audio_ab", "out", "emb_cache")


def _get_model():
    global _MODEL
    if _MODEL is None:
        from fadtk import EncodecEmbModel
        _MODEL = EncodecEmbModel()
        _MODEL.load_model()
    return _MODEL


def _key(x, sr):
    import hashlib
    h = hashlib.sha1()
    h.update(np.asarray(x, dtype=np.float32).tobytes())
    h.update(str(int(sr)).encode())
    return h.hexdigest() + ".npy"


def embed(x, sr, cache=True):
    """1クリップの埋め込み (frames, 128) を返す。"""
    import torch
    from score_noref import fft_resample
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        p = os.path.join(CACHE_DIR, _key(x, sr))
        if os.path.exists(p):
            try:
                return np.load(p)
            except Exception:
                pass
    ml = _get_model()
    x24 = fft_resample(x, sr, ml.sr)
    t = torch.from_numpy(np.asarray([x24], dtype=np.float32)).unsqueeze(0)
    e = np.asarray(ml.get_embedding(t), dtype=np.float32)
    if cache:
        try:
            np.save(p, e)
        except Exception:
            pass
    return e


def fad_sets(ref_clips, test_clips, sr) -> float:
    """2つのクリップ集合間のFAD。同一集合なら≈0。"""
    from fadtk.fad import calc_embd_statistics, calc_frechet_distance
    er = np.concatenate([np.asarray(embed(x, sr), dtype=np.float64)
                         for x in ref_clips], axis=0)
    et = np.concatenate([np.asarray(embed(x, sr), dtype=np.float64)
                         for x in test_clips], axis=0)
    mu1, cov1 = calc_embd_statistics(er)
    mu2, cov2 = calc_embd_statistics(et)
    return float(calc_frechet_distance(mu1, cov1, mu2, cov2))
