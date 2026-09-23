"""ADC clip flag test - synthetic, no hardware.

process()にクリーン/飽和した生IQを流し、adc_clip_pct/adc_clippedが
正しく立ち・選局でリセットされることを検証する。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline

BLOCK = 132096


def main() -> int:
    ok = True
    rng = np.random.default_rng(3)

    dsp = SdrDspPipeline(1152000, 48000)
    assert dsp.adc_clipped is False and dsp.adc_clip_pct == 0.0

    # クリーン信号: 飽和なし
    clean = (127.5 + 20.0 * rng.standard_normal(BLOCK)).clip(2, 253).astype(np.uint8)
    for _ in range(4):
        dsp.process(clean, mode="WFM")
    print(f"[{'OK' if not dsp.adc_clipped else 'FAIL'}] clean stays clear "
          f"(pct={dsp.adc_clip_pct:.3f})")
    ok &= not dsp.adc_clipped

    # 飽和信号: 30%を0/255へ
    sat = clean.copy()
    m = rng.random(BLOCK) < 0.30
    sat[m] = np.where(rng.random(m.sum()) < 0.5, 0, 255).astype(np.uint8)
    for _ in range(8):
        dsp.process(sat, mode="WFM")
    print(f"[{'OK' if dsp.adc_clipped else 'FAIL'}] saturated raises flag "
          f"(pct={dsp.adc_clip_pct:.2f})")
    ok &= dsp.adc_clipped

    # 選局でリセット
    dsp.set_offset_freq(150000.0)
    reset_ok = (dsp.adc_clipped is False and dsp.adc_clip_pct == 0.0)
    print(f"[{'OK' if reset_ok else 'FAIL'}] tune resets flag")
    ok &= reset_ok

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
