"""Resampler integer-bypass regression tests (d==1 shape bug)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import AdaptiveDriftResampler


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)

    for channels in (1, 2):
        r = AdaptiveDriftResampler()
        r.phase = 0.6  # 端数>=0.5 → d==1 分岐を強制
        x = rng.standard_normal(2752).astype(np.float32)
        if channels == 2:
            x = np.stack([x, x], axis=1)
        y = r.process(x)
        expected = (2751,) if channels == 1 else (2751, 2)
        good = y.shape == expected
        print(f"[{'OK' if good else 'FAIL'}] d==1 bypass len {channels}ch: "
              f"{tuple(x.shape)}->{tuple(y.shape)} (want {expected})")
        ok &= good
        # 連続性: 先頭は前ブロック最終サンプル (last_sample 初期値 0)
        if channels == 2:
            ok &= np.allclose(y[0], 0.0, atol=1e-6)
        else:
            ok &= abs(float(y[0])) < 1e-6
        # 途中でクラッシュしない (ValueError再発防止)。位相は毎回リセットして再現
        for _ in range(5):
            r.phase = 0.6
            z = r.process(x)
            ok &= z.shape == expected

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())