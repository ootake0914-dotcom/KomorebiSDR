"""Hyper帯域マッピングの呼吸防止テスト (純ロジック、HW不要)。

発端: 83.2MHz WFM実録で target_cutoff が10.4k〜15kを0.1〜0.4Hz往復し
「音量の呼吸」として聞こえた。原因は帯域マッピングが直近21ms窓の
聴感SNR (番組内容で4〜41dB振れる) を直接使っていたこと。

検証:
1) 強信号で聴感SNRが激しく振れても target_cutoff が安定 (遅いEMA＋
   不感帯＋スルーレート)
2) 強信号ではカットオフ下限13kHzが効く (C/N十分で絞る理由がない)
3) 弱信号へは数秒で狭窄側へ適応する (鈍らせても追従は死なない)
4) 手動overrideは即時反映 (遅延を入れない)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hyper_controller as hc_mod
from dsp import SdrDspPipeline
from hyper_controller import HyperController


class _FakeClock:
    """time.timeをブロック周期で進める (実sleepでテストを遅くしない)。"""

    def __init__(self, dt=0.0573):
        self.t = 1000.0
        self.dt = dt

    def time(self):
        self.t += self.dt
        return self.t


class FakeDriver:
    def __init__(self):
        self.gains = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7,
                      16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8,
                      36.4, 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]

    def get_gains(self):
        return list(self.gains)

    def set_gain(self, g):
        pass

    def set_gain_mode(self, auto):
        pass


def _make():
    dsp = SdrDspPipeline(1152000, 48000)
    hc = HyperController(FakeDriver(), dsp, audio=None)
    hc.init_gains()
    hc.set_hard_lock(True)
    return dsp, hc


def test_strong_stable() -> bool:
    real_time = hc_mod.time.time
    hc_mod.time.time = _FakeClock(0.0573).time
    try:
        dsp, hc = _make()
        hc.state_channel_snr = 36.0
        vals = []
        for i in range(400):
            # 番組内容の激しい揺れを模擬 (10⇔30dBを毎ブロック)
            hc.state_audio_snr = 10.0 if (i % 2 == 0) else 30.0
            hc._map_continuous_parameters("WFM")
            vals.append(hc.target_cutoff_hz)
    finally:
        hc_mod.time.time = real_time
    tail = np.asarray(vals[300:])
    std = float(np.std(tail))
    p2p = float(np.max(tail) - np.min(tail))
    floor_ok = float(np.min(tail)) >= 13000.0
    ok = std < 150.0 and p2p < 600.0
    print(f"[{'OK' if ok else 'FAIL'}] strong stable: std={std:.1f}Hz "
          f"p2p={p2p:.0f}Hz floor13k={floor_ok} "
          f"(need std<150, p2p<600)")
    return bool(ok and floor_ok)


def test_weak_adapts() -> bool:
    real_time = hc_mod.time.time
    hc_mod.time.time = _FakeClock(0.0573).time
    try:
        dsp, hc = _make()
        hc.state_channel_snr = 5.0
        hc.state_audio_snr = 5.0
        for _ in range(400):
            hc._map_continuous_parameters("WFM")
        final = float(hc.target_cutoff_hz)
    finally:
        hc_mod.time.time = real_time
    ok = final < 7000.0
    print(f"[{'OK' if ok else 'FAIL'}] weak adapts: final={final:.0f}Hz "
          f"(need <7000)")
    return ok


def test_override_immediate() -> bool:
    dsp, hc = _make()
    hc.state_channel_snr = 36.0
    hc.state_audio_snr = 26.0
    hc._map_continuous_parameters("WFM")
    hc.set_filter_override("wide")
    hc._map_continuous_parameters("WFM")
    ok = abs(float(hc.target_cutoff_hz) - 15000.0) < 1e-6
    print(f"[{'OK' if ok else 'FAIL'}] override immediate: "
          f"target={hc.target_cutoff_hz:.0f}Hz (need 15000)")
    return ok


def test_mode_switch_settles() -> bool:
    # WFM→SSBで数秒以内に通信帯域へ落ちること (鈍らせても追従は死なない)。
    # 不感帯±300Hzのため最終値は目標近傍 (±600Hz) で静止すれば合格。
    real_time = hc_mod.time.time
    hc_mod.time.time = _FakeClock(0.0573).time
    try:
        dsp, hc = _make()
        hc.state_channel_snr = 36.0
        hc.state_audio_snr = 26.0
        for _ in range(50):
            hc._map_continuous_parameters("WFM")
        n = 0
        for _ in range(600):
            hc._map_continuous_parameters("USB")
            n += 1
            if abs(hc.target_cutoff_hz - 3500.0) < 600.0:
                break
        final = float(hc.target_cutoff_hz)
    finally:
        hc_mod.time.time = real_time
    ok = abs(final - 3500.0) < 600.0
    print(f"[{'OK' if ok else 'FAIL'}] WFM->USB settles: "
          f"target={final:.0f}Hz in {n} blocks (need |err|<600)")
    return ok


def main() -> int:
    ok = test_strong_stable()
    ok &= test_weak_adapts()
    ok &= test_override_immediate()
    ok &= test_mode_switch_settles()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
