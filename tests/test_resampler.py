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
        # 連続性: d==1 は先頭1サンプルを捨てるだけで重複・欠落が無いこと
        # (旧実装は last_sample を重複挿入し末尾を落としていた)
        r2 = AdaptiveDriftResampler()
        r2.phase = 0.6
        a = np.arange(64, dtype=np.float32)
        b = np.arange(64, 128, dtype=np.float32)
        r2.process(a)
        r2.phase = 0.6
        o = r2.process(b)
        good = (len(o) == 63 and not np.any(o == 63.0)
                and abs(float(o[0]) - 65.0) < 1e-6 and abs(float(o[-1]) - 127.0) < 1e-6)
        print(f"[{'OK' if good else 'FAIL'}] d==1 drops one sample without "
              f"duplication (head={o[0]:.0f} tail={o[-1]:.0f} len={len(o)})")
        ok &= good
        # 形状 (両ch) の回帰。位相は毎回リセットして再現
        for _ in range(5):
            r.phase = 0.6
            z = r.process(x)
            ok &= z.shape == expected
        # d==1 は先頭1サンプルを捨てる (last_sample重複挿入はしない)
        if channels == 2:
            ok &= bool(np.allclose(y[0], x[1]))
        else:
            ok &= abs(float(y[0]) - float(x[1])) < 1e-6

    # ドリフト補正作動時のHF平坦性 (回帰: Catmull-Romは15kで-1.44dBだった。
    # ポリフェーズ窓sinc 8tap/128phaseで±50ppmでも-0.11dB以内)。
    fs = 48000.0

    def tone_level(freq, ratio):
        rr = AdaptiveDriftResampler()
        rr.current_ratio = ratio
        n = 48000
        t = np.arange(n) / fs
        xx = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        oo = [rr.process(xx[k:k + 2752]) for k in range(0, n, 2752)]
        yy = np.concatenate(oo)
        yy = yy[len(yy) // 4:]
        W = np.abs(np.fft.rfft(yy * np.hanning(len(yy)))) ** 2
        i = int(round(freq * len(yy) / fs))
        return 10 * np.log10(float(W[max(0, i - 2):i + 3].sum()) + 1e-24)

    for freq, tol in ((14000.0, 0.15), (15000.0, 0.2)):
        base = tone_level(freq, 1.0)
        for ratio in (1.00005, 0.99995):
            dv = tone_level(freq, ratio) - base
            good = abs(dv) < tol
            print(f"[{'OK' if good else 'FAIL'}] drift {ratio - 1:+.0e} "
                  f"@ {freq / 1000:.0f}k: {dv:+.2f} dB (tol {tol})")
            ok &= good

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())