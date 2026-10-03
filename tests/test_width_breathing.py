"""幅呼吸の回帰テスト (純ロジック、HW不要)。

発端: ステレオ差信号NRのヒス推定が番組内容でゆっくり動き、Wiener適用度
(s_w) が「幅の呼吸」として聞こえた。対策として
- マッピング用ヒス推定を超低速化 (τ12s)
- 適用度に非対称スルーレート (閉じる+0.06/ブロック、開く-0.01)
- マスキング閾値の平滑化 (τ8s)
を入れた。ここでは「ヒスが急に増えても2〜3秒では適用度が立ち上がらない
(番組起因の揺れに追従しない) が、持続変化には数十秒で適応する」契約を検証する。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline

N = 2048


def _hiss(amp, rng):
    return (amp * rng.standard_normal(N)).astype(np.float32)


def main() -> int:
    dsp = SdrDspPipeline(1152000, 48000)
    rng = np.random.default_rng(0)
    t = np.arange(N) / 48000.0
    mono = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    for _ in range(200):
        dsp._update_stereo_nr(_hiss(0.005, rng), mono)
    sw_base = float(dsp._nr_s_w)

    for _ in range(60):
        dsp._update_stereo_nr(_hiss(0.05, rng), mono)
    sw_fast = float(dsp._nr_s_w)

    for _ in range(540):
        dsp._update_stereo_nr(_hiss(0.05, rng), mono)
    sw_slow = float(dsp._nr_s_w)

    ok1 = sw_fast < 0.2 and abs(sw_fast - sw_base) < 0.2
    print(f"[{'OK' if ok1 else 'FAIL'}] no quick engage on hiss step "
          f"(base={sw_base:.3f} after3.4s={sw_fast:.3f}, need <0.2)")
    ok2 = sw_slow > 0.6
    print(f"[{'OK' if ok2 else 'FAIL'}] sustained change adapts "
          f"(after34s={sw_slow:.3f}, need >0.6)")
    ok3 = abs(float(dsp.stereo_hiss_db) - (-26.9)) < 6.0
    print(f"[{'OK' if ok3 else 'FAIL'}] fast hiss telemetry tracks "
          f"(hiss={dsp.stereo_hiss_db:.1f}dB)")
    ok = bool(ok1 and ok2 and ok3)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
