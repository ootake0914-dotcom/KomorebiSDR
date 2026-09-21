"""Robustness regression: controllers must survive non-finite measurements.

発端: NaN混じりスペクトル → _measure_channel_snrがNaN返却 →
カルマン状態が永久汚染 → int(round(nan)) でワーカーがクラッシュ。
測定→融合の各段で非有限を遮断することを回帰保証する。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline
from hyper_controller import HyperController

RF = 1152000
BLOCK = 132096


class StubDriver:
    def __init__(self):
        self.gain = 33.8

    def get_gains(self):
        return [0.0, 8.7, 12.5, 20.7, 28.0, 33.8, 42.1, 49.6]

    def set_gain_mode(self, m):
        pass

    def set_gain(self, g):
        self.gain = float(g)


def test_nan_spectrum_survival():
    print("\n===== test_nan_spectrum_survival =====")
    dsp = SdrDspPipeline(RF, 48000)
    dsp.set_offset_freq(0.0)
    dsp.cognitive_enabled = False
    drv = StubDriver()
    c = HyperController(drv, dsp, None)
    if not getattr(c, "available_gains", None):
        c.available_gains = sorted(drv.get_gains())

    rng = np.random.default_rng(0)
    n = BLOCK // 2
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.3
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)

    # 正常フレームで状態を温める
    spec = np.full(1024, -70.0)
    for _ in range(3):
        c.process_frame(raw, spectrum_db=spec, audio=np.zeros(4096, dtype=np.float32), mode="WFM")

    # NaN / inf 混じりスペクトル＋NaN音声でも死なない
    for bad_spec in (np.full(1024, np.nan),
                     np.full(1024, np.inf),
                     np.concatenate([np.full(512, -70.0), np.full(512, np.nan)])):
        st = c.process_frame(raw, spectrum_db=bad_spec,
                             audio=np.full(4096, np.nan, dtype=np.float32), mode="WFM")
        assert isinstance(st, dict), "stats must stay dict"
    # 状態が有限のまま
    for k in ("state_channel_snr", "state_audio_snr", "state_quality",
              "target_cutoff_hz", "target_if_bw_hz", "target_hf_gain"):
        v = float(getattr(c, k))
        assert np.isfinite(v), f"{k} poisoned: {v}"
    # ゲインは有効集合内
    assert float(drv.gain) in c.available_gains, f"gain escaped: {drv.gain}"
    # 正常フレームで復帰できる
    st = c.process_frame(raw, spectrum_db=spec, audio=np.zeros(4096, dtype=np.float32), mode="WFM")
    assert isinstance(st, dict)
    print(f"[*] survived NaN/inf frames; states finite; gain={drv.gain}")
    print("[OK] nan spectrum survival")


if __name__ == "__main__":
    test_nan_spectrum_survival()
    print("\nALL CONTROLLER ROBUSTNESS TESTS PASSED!")
