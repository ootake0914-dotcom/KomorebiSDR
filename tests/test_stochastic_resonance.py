"""C: 確率共鳴検出補助のテスト (合成信号のみ・実機不要)。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic_resonance import StochasticResonanceDetector


def _tone_present(x, thresh=0.02):
    # 簡易基礎検出器: 19kHzビン振幅の閾値判定
    n = len(x)
    k = int(round(19000.0 * n / 288000.0))
    tw = np.exp(-2j * np.pi * k * np.arange(n) / n)
    return abs(np.dot(np.asarray(x, dtype=np.float64), tw)) * (2.0 / n) > thresh


def _block(freq=19000.0, amp=0.03, noise=0.01, seed=0, n=8192):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 288000.0
    return amp * np.sin(2.0 * np.pi * freq * t) + noise * rng.standard_normal(n)


def test_main_path_untouched():
    sr = StochasticResonanceDetector()
    x = _block()
    before = x.copy()
    sr.assess(x, _tone_present, noise_floor=0.01, snr_db=5.0,
              base_confidence=0.5)
    assert np.array_equal(x, before), "入力配列が書き換えられた"


def test_detector_only_enforced():
    try:
        StochasticResonanceDetector(detector_only=False)
    except ValueError:
        pass
    else:
        raise AssertionError("detector_only=Falseが拒否されない")
    sr = StochasticResonanceDetector()
    assert sr.detector_only is True


def test_strong_signal_disabled():
    sr = StochasticResonanceDetector()
    x = _block(amp=0.5, noise=0.001)
    out = sr.assess(x, _tone_present, noise_floor=0.001, snr_db=30.0,
                    base_confidence=0.5)
    assert out["enabled"] is False
    assert out["reason"] == "snr-out-of-range"


def test_confident_cases_passthrough():
    sr = StochasticResonanceDetector()
    x = _block()
    o1 = sr.assess(x, _tone_present, 0.01, 5.0, base_confidence=0.9)
    assert o1["enabled"] is False and o1["reason"] == "confident-present"
    o2 = sr.assess(x, _tone_present, 0.01, 5.0, base_confidence=0.1)
    assert o2["enabled"] is False and o2["reason"] == "confident-absent"


def test_mid_zone_assists():
    sr = StochasticResonanceDetector(trials=4)
    x = _block(amp=0.03, noise=0.01)
    out = sr.assess(x, _tone_present, noise_floor=0.01, snr_db=5.0,
                    base_confidence=0.5, has_signal_truth=True)
    assert out["enabled"] is True
    assert out["adopt"] is True
    assert 0.0 <= out["sr_confidence"] <= 1.0
    st = sr.stats()
    assert st["assessed"] == 1 and st["adopted"] == 1


def test_seed_reproducible():
    kw = dict(noise_floor=0.01, snr_db=5.0, base_confidence=0.5)
    a = StochasticResonanceDetector(seed=42)
    b = StochasticResonanceDetector(seed=42)
    x = _block(seed=3)
    oa = a.assess(x, _tone_present, **kw)
    ob = b.assess(x, _tone_present, **kw)
    assert oa["agreement"] == ob["agreement"]
    assert oa["sr_confidence"] == ob["sr_confidence"]


def test_fa_worse_rejects():
    # 付加ノイズで誤検出が増える基礎検出器 (分散閾値・閾値をクリーン分散の
    # すぐ上に置く): クリーンでは稀にしかTrueにならないが、ノイズ付加試行の
    # 分散は systematic に上がるため共鳴の誤検出が通常を上回りrejectされる
    sr = StochasticResonanceDetector(trials=4)
    rng = np.random.default_rng(0)
    out = None
    for i in range(6):
        x = 0.01 * rng.standard_normal(8192)  # 信号なし
        out = sr.assess(x, lambda v: float(np.var(v)) > 1.0005e-4,
                        noise_floor=0.02, snr_db=-5.0,
                        base_confidence=0.5, has_signal_truth=False)
    st = sr.stats()
    print(f"[*] FA通常={st['normal_fa']} 共鳴={st['sr_fa']}")
    assert out["adopt"] is False
    assert out["reason"] == "fa-worse-reject"
    assert st["rejected"] >= 1


def test_clip_and_nonfinite_disabled():
    sr = StochasticResonanceDetector()
    x = _block()
    o = sr.assess(x, _tone_present, 0.01, 5.0, 0.5, clip=True)
    assert o["enabled"] is False and o["reason"] == "clipped"
    o2 = sr.assess(np.full(8192, np.nan), _tone_present, 0.01, 5.0, 0.5)
    assert o2["enabled"] is False and o2["reason"] == "non-finite"


def main() -> int:
    try:
        test_main_path_untouched()
        print("[*] メイン経路無汚染 OK")
        test_detector_only_enforced()
        print("[*] detector_only強制 OK")
        test_strong_signal_disabled()
        print("[*] 強信号無効 OK")
        test_confident_cases_passthrough()
        print("[*] 確定例素通し OK")
        test_mid_zone_assists()
        print("[*] 中間帯補助 OK")
        test_seed_reproducible()
        print("[*] seed再現性 OK")
        test_fa_worse_rejects()
        print("[*] 誤検出悪化時reject OK")
        test_clip_and_nonfinite_disabled()
        print("[*] クリップ/非有限無効 OK")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return 1
    print("\nALL STOCHASTIC RESONANCE TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
