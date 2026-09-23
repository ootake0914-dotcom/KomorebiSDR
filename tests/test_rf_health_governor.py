"""RF Health Governor 受入テスト (Receiver Resilience & Overload Protection)。

テスト項目:
1. test_fsm_state_transitions:
   - 正常信号: HEALTHY
   - 中度過入力: OVERLOAD_WARNING
   - 重度過入力: OVERLOAD_HARD
   - 復帰時のヒステリシス動作 (チャタリング防止)
2. test_overload_gain_action:
   - OVERLOAD_HARD 時にハードウェアゲイン引下げ要求 (例: -6dB〜-12dB) が発行されること
3. test_mode_switch_smoothness:
   - モード切り替え時の過渡段差が抑えられ、先頭が滑らかに立ち上がること
4. test_healthy_transparency:
   - HEALTHY 状態では音声出力が完全透明 (同一) であること
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rf_health import RfHealthGovernor, RfHealthState


class TestRfHealthGovernor(unittest.TestCase):

    def setUp(self):
        self.fs = 48000.0
        self.rf = 1152000.0
        self.block_size = 132096  # 66048 IQ samples

    def _make_raw_iq(self, amp: float = 0.5, n_samples: int = 66048, offset_hz: float = 0.0) -> np.ndarray:
        t = np.arange(n_samples) / self.rf
        iq = amp * np.exp(2j * np.pi * (1000.0 * t + offset_hz * t))
        raw = np.empty(2 * n_samples, dtype=np.uint8)
        raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
        raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
        return raw

    def test_fsm_state_transitions(self):
        """1. 状態遷移とヒステリシスの検証"""
        gov = RfHealthGovernor(sample_rate=self.fs)

        # (a) 正常入力 -> HEALTHY
        normal_raw = self._make_raw_iq(amp=0.4)
        state, info = gov.update(normal_raw, s_meter_dbfs=-20.0, pilot_lock=0.8)
        self.assertEqual(state, RfHealthState.HEALTHY)
        self.assertEqual(info["gain_step_db"], 0.0)

        # (b) 中度過入力 (クリップ率 ~8%) -> OVERLOAD_WARNING
        warn_raw = self._make_raw_iq(amp=1.02)
        state, info = gov.update(warn_raw, s_meter_dbfs=-2.0, pilot_lock=0.5)
        self.assertEqual(state, RfHealthState.OVERLOAD_WARNING)

        # (c) 重度過入力 (クリップ率 >20%) -> OVERLOAD_HARD
        hard_raw = self._make_raw_iq(amp=4.0)
        state, info = gov.update(hard_raw, s_meter_dbfs=+1.0, pilot_lock=0.2)
        self.assertEqual(state, RfHealthState.OVERLOAD_HARD)
        self.assertLess(info["gain_step_db"], 0.0)

        # (d) 正常入力への復帰ヒステリシス (即時復帰せず数ブロックホールド)
        state_hold, _ = gov.update(normal_raw, s_meter_dbfs=-20.0, pilot_lock=0.8)
        self.assertIn(state_hold, [RfHealthState.OVERLOAD_HARD, RfHealthState.OVERLOAD_WARNING])

        # ホールド時間経過後に HEALTHY へ復帰
        for _ in range(10):
            state, _ = gov.update(normal_raw, s_meter_dbfs=-20.0, pilot_lock=0.8)
        self.assertEqual(state, RfHealthState.HEALTHY)

    def test_overload_gain_action(self):
        """2. 過大入力時の自律ゲイン要求値の検証"""
        gov = RfHealthGovernor(sample_rate=self.fs)
        hard_raw = self._make_raw_iq(amp=5.0)

        state, info = gov.update(hard_raw, s_meter_dbfs=+2.0, pilot_lock=0.1)
        self.assertEqual(state, RfHealthState.OVERLOAD_HARD)
        # 最低 -6dB 以上の引き下げ要求
        self.assertLessEqual(info["gain_step_db"], -6.0)

    def test_mode_switch_smoothness(self):
        """3. モード切り替えクロスフェードの段差検証"""
        gov = RfHealthGovernor(sample_rate=self.fs)

        audio_prev = np.ones(2752, dtype=np.float32) * 0.8
        audio_next = np.ones(2752, dtype=np.float32) * -0.8  # 完全逆相の大振幅

        # 直前ブロックを通過させて末尾サンプル (0.8) を記憶
        _ = gov.apply_audio_guard(audio_prev)

        # モード遷移イベント通知
        gov.notify_mode_switch("WFM", "AM")
        protected_next = gov.apply_audio_guard(audio_next)

        # 直前末尾サンプル (0.8) と新ブロック先頭サンプルの境界段差がゼロであること
        boundary_step = abs(float(protected_next[0]) - float(audio_prev[-1]))
        self.assertLess(boundary_step, 1e-4, f"境界段差がゼロではありません: {boundary_step}")

        # 終端サンプルは新モードの元の振幅 (-0.8) に回復していること
        self.assertAlmostEqual(float(protected_next[-1]), -0.8, places=3)

    def test_healthy_transparency(self):
        """4. HEALTHY 状態でのビット完全透明性"""
        gov = RfHealthGovernor(sample_rate=self.fs)
        normal_raw = self._make_raw_iq(amp=0.4)
        gov.update(normal_raw, s_meter_dbfs=-20.0, pilot_lock=0.8)

        audio_in = (np.random.randn(2752) * 0.2).astype(np.float32)
        audio_out = gov.apply_audio_guard(audio_in)
        self.assertTrue(np.array_equal(audio_in, audio_out), "HEALTHY 時に原音が完全素通しされていません")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRfHealthGovernor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
