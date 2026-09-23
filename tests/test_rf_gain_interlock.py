"""RF Health Governor と HyperController のハードウェアゲイン連動テスト。

過大入力・ADCクリッピング発生時に、自律保護FSMからの指示 (gain_step_db) に基づき
チューナーハードウェアゲインが安全段へ即座に引き下げられることを検証する。
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hyper_controller import HyperController
from rf_health import RfHealthState


class MockDriver:
    def __init__(self, gains=None):
        if gains is None:
            self.gains = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
                          19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
                          38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
        else:
            self.gains = sorted(gains)
        self.gain = 43.4
        self.gain_mode = False

    def get_gains(self):
        return list(self.gains)

    def set_gain_mode(self, manual: bool):
        self.gain_mode = manual

    def set_gain(self, gain: float):
        self.gain = float(gain)


class MockDsp:
    def __init__(self):
        self.audio_rate = 48000
        self.rf_rate = 1152000
        self.rf_health_state = RfHealthState.HEALTHY
        self.rf_health_info = {"state": "HEALTHY", "clip_rate": 0.0, "gain_step_db": 0.0}
        self.applied_cutoff_hz = 14000.0
        self.hf_gain_applied = 1.0

    def set_cognitive_parameters(self, **kwargs):
        pass


class TestRfGainInterlock(unittest.TestCase):

    def setUp(self):
        self.driver = MockDriver()
        self.dsp = MockDsp()
        self.controller = HyperController(self.driver, self.dsp)
        self.controller.init_gains()

    def test_healthy_steady_state(self):
        """1. HEALTHY 状態では正常なゲインが維持されること"""
        self.dsp.rf_health_state = RfHealthState.HEALTHY
        self.dsp.rf_health_info = {"state": "HEALTHY", "clip_rate": 0.0, "gain_step_db": 0.0}

        raw = np.full(2048, 128, dtype=np.uint8)
        initial_gain = self.driver.gain
        self.controller.process_frame(raw, mode="WFM")

        # 正常時、急激なゲイン落ちは発生しない
        self.assertGreaterEqual(self.driver.gain, initial_gain - 5.0)

    def test_overload_warning_step(self):
        """2. OVERLOAD_WARNING (-2dB要求) でチューナーゲインが引き下げられること"""
        self.dsp.rf_health_state = RfHealthState.OVERLOAD_WARNING
        self.dsp.rf_health_info = {"state": "OVERLOAD_WARNING", "clip_rate": 0.08, "gain_step_db": -2.0}

        # 43.4dB 動作中
        self.driver.gain = 43.4
        self.controller.current_gain_idx = self.controller.available_gains.index(43.4)

        raw = np.full(2048, 128, dtype=np.uint8)
        self.controller.process_frame(raw, mode="WFM")

        # 43.4dB から -2dB 要求 -> 41.4dB 以下 (例: 40.2dB や 38.6dB) へ減衰
        self.assertLess(self.driver.gain, 43.4)
        self.assertLessEqual(self.driver.gain, 43.4 - 2.0 + 1e-4)

    def test_overload_hard_emergency_step_with_hard_lock(self):
        """3. ハードロック中であっても OVERLOAD_HARD (-12dB要求) で即座に緊急回避減衰すること"""
        self.dsp.rf_health_state = RfHealthState.OVERLOAD_HARD
        self.dsp.rf_health_info = {"state": "OVERLOAD_HARD", "clip_rate": 0.40, "gain_step_db": -12.0}

        # 43.4dB でハードロック (自動再探索停止状態)
        self.driver.gain = 43.4
        self.controller.current_gain_idx = self.controller.available_gains.index(43.4)
        self.controller.hard_lock = True
        self.controller.locked = True

        raw = np.full(2048, 128, dtype=np.uint8)
        self.controller.process_frame(raw, mode="WFM")

        # ハードロックを突破して -12dB 要求 (43.4 - 12.0 = 31.4dB 以下) まで退避
        self.assertLessEqual(self.driver.gain, 31.4 + 1e-4)
        # クリップ領域として上位段が記録されていること
        self.assertIsNotNone(self.controller.clip_upper_idx)
        # ハードロック自体は維持され、勝手な乱探索に落ちないこと
        self.assertTrue(self.controller.hard_lock)

    def test_floor_clamp(self):
        """4. 連続過大入力でも安全下限を下回らないこと"""
        self.dsp.rf_health_state = RfHealthState.OVERLOAD_HARD
        self.dsp.rf_health_info = {"state": "OVERLOAD_HARD", "clip_rate": 0.50, "gain_step_db": -12.0}

        # 22.9dB 動作中に -12dB 要求 (10.9dB) -> 安全下限 (19.7dB) にクランプされる
        self.driver.gain = 22.9
        self.controller.current_gain_idx = self.controller.available_gains.index(22.9)

        raw = np.full(2048, 128, dtype=np.uint8)
        self.controller.process_frame(raw, mode="WFM")

        min_safe = self.controller.available_gains[self.controller._min_safe_idx()]
        self.assertEqual(self.driver.gain, min_safe)


if __name__ == "__main__":
    unittest.main()
