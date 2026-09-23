"""RF Health Governor (受信機自律保護・ヘルスガバナー)。

RTL-SDR特有の狭ダイナミックレンジ (8-bit ADC ~48dB) や
過大入力・近接強信号・モード切替時の破綻を防ぐ自律防衛機構。

状態 (3段階):
- HEALTHY: 正常運用。原音完全素通し (ゼロオーバーヘッド)。
- OVERLOAD_WARNING: 軽度過大入力 (クリップ率 3-15% または高入力)。
  ゲイン微小引下げ助言 (-2dB) を発行。
- OVERLOAD_HARD: 重度過大入力 (クリップ率 > 20% または極大飽和)。
  ゲイン大幅引下げ助言 (-6dB〜-12dB) を発行。

過渡保護:
- モード切り替え直後 (WFM ↔ AM ↔ NFM ↔ USB) の破裂音 (クリックスパイク) を
  20msのコサイン窓で滑らかにフェードインし、完全に根絶。
"""

from enum import Enum
import numpy as np


class RfHealthState(str, Enum):
    HEALTHY = "HEALTHY"
    OVERLOAD_WARNING = "OVERLOAD_WARNING"
    OVERLOAD_HARD = "OVERLOAD_HARD"


class RfHealthGovernor:
    """RF過大入力・クリップ・モード過渡を自律統制するガバナー"""

    def __init__(self, sample_rate: float = 48000.0):
        self.sample_rate = float(sample_rate)
        self.state = RfHealthState.HEALTHY
        
        # 統計とヒステリシス
        self._clip_rate = 0.0
        self._hold_counter = 0
        self._hold_blocks = 8  # 悪化状態からの復帰に必要な連続正常ブロック数 (~450ms)
        self._gain_step_db = 0.0
        
        # モード切替過渡保護フェードフラグと直前サンプル保持
        self._mode_switch_fade = False
        self._fade_len = int(self.sample_rate * 0.02)  # 20ms フェード窓
        self._last_audio_sample = None

    def reset(self):
        """状態リセット"""
        self.state = RfHealthState.HEALTHY
        self._clip_rate = 0.0
        self._hold_counter = 0
        self._gain_step_db = 0.0
        self._mode_switch_fade = False
        self._last_audio_sample = None

    def update(self, raw_bytes: np.ndarray, s_meter_dbfs: float = -20.0, pilot_lock: float = 1.0) -> tuple[RfHealthState, dict]:
        """ブロック毎のRF入力品質を観測し、FSM状態とアクションを決定する"""
        if len(raw_bytes) == 0:
            return self.state, {"gain_step_db": 0.0}

        # 1. ADCクリッピング率の測定 (uint8 の 0 または 255 の比率)
        n_raw = len(raw_bytes)
        n_clipped = np.count_nonzero((raw_bytes == 0) | (raw_bytes == 255))
        clip_rate = float(n_clipped / max(n_raw, 1))
        self._clip_rate = clip_rate

        # 2. 状態判定 (悪化は即時、回復はヒステリシス)
        target_state = RfHealthState.HEALTHY
        gain_step = 0.0

        if clip_rate > 0.20 or s_meter_dbfs > 0.0:
            target_state = RfHealthState.OVERLOAD_HARD
            gain_step = -12.0 if clip_rate > 0.35 else -6.0
        elif clip_rate > 0.03 or s_meter_dbfs > -3.0:
            target_state = RfHealthState.OVERLOAD_WARNING
            gain_step = -2.0

        # FSM遷移ロジック
        if target_state != RfHealthState.HEALTHY:
            # 異常状態への移行は即座
            self.state = target_state
            self._hold_counter = self._hold_blocks
            self._gain_step_db = gain_step
        else:
            # 正常状態への復帰はホールドカウンタを減算
            if self._hold_counter > 0:
                self._hold_counter -= 1
            else:
                self.state = RfHealthState.HEALTHY
                self._gain_step_db = 0.0

        info = {
            "state": self.state.value,
            "clip_rate": self._clip_rate,
            "gain_step_db": self._gain_step_db,
        }
        return self.state, info

    def notify_mode_switch(self, old_mode: str, new_mode: str):
        """復調モード切り替えイベント通知 (クリック保護フェードインを起動)"""
        self._mode_switch_fade = True

    def apply_audio_guard(self, audio: np.ndarray) -> np.ndarray:
        """オーディオ出力に対するセーフティガード (モード切替クリック抑圧・境界平滑化)"""
        if len(audio) == 0:
            return audio

        is_stereo = (audio.ndim == 2)
        n = len(audio)

        if self._mode_switch_fade:
            out = np.array(audio, dtype=np.float32, copy=True)
            flen = min(self._fade_len, n)
            idx = np.arange(flen, dtype=np.float32) / float(flen)
            # 0.0 -> 1.0 の滑らかなコサイン窓
            w = 0.5 * (1.0 - np.cos(np.pi * idx))

            if self._last_audio_sample is not None:
                # 直前の末尾サンプルから新モード信号へ滑らかにクロスフェード (境界段差ゼロ化)
                last = np.asarray(self._last_audio_sample, dtype=np.float32)
                if is_stereo:
                    if last.ndim == 0 or len(last) != 2:
                        last = np.array([float(last), float(last)], dtype=np.float32)
                    fade_from = (1.0 - w[:, None]) * last[None, :]
                    out[:flen, :] = fade_from + w[:, None] * out[:flen, :]
                else:
                    if last.ndim > 0:
                        last = float(last[0])
                    fade_from = (1.0 - w) * last
                    out[:flen] = fade_from + w * out[:flen]
            else:
                # 直前サンプルがない場合はゼロからのソフト立ち上げ
                if is_stereo:
                    out[:flen, :] *= w[:, None]
                else:
                    out[:flen] *= w
            self._mode_switch_fade = False
        else:
            out = audio

        # 次回モード切替時の境界連続性のために末尾サンプルを記憶
        if is_stereo:
            self._last_audio_sample = np.array(out[-1, :], dtype=np.float32, copy=True)
        else:
            self._last_audio_sample = float(out[-1])

        return out
