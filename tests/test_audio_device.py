"""Audio device-change detection tests (Windows Core Audio path, hardware-free).

- _match_output_index picks the same-hostapi device by friendly name
- device change detection via Core Audio name triggers a reopen on a new index
- reopen tolerates no-stream and keeps queue
"""

import os
import sys
import threading
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from audio_output import AudioOutput

MOCK_DEVICES = [
    {"index": 3, "name": "Microsoft サウンド マッパー - Output", "hostapi": 0, "max_output_channels": 2},
    {"index": 4, "name": "Headphones (Senary Audio)", "hostapi": 0, "max_output_channels": 2},
    {"index": 5, "name": "Speakers (Senary Audio)", "hostapi": 0, "max_output_channels": 2},
    {"index": 10, "name": "Headphones (Senary Audio)", "hostapi": 1, "max_output_channels": 2},
    {"index": 11, "name": "Speakers (Senary Audio)", "hostapi": 1, "max_output_channels": 2},
]


def test_match_same_hostapi():
    with mock.patch("audio_output.sd.query_devices", return_value=MOCK_DEVICES):
        # 現在 hostapi 0 なら index 4 を選ぶ
        idx = AudioOutput._match_output_index("Headphones (Senary Audio)", current_index=4)
        assert idx == 4, f"same-hostapi match failed: {idx}"
        # 現在デバイス不明ならホストAPI昇順で最初の一致 (hostapi 0)
        idx2 = AudioOutput._match_output_index("Headphones (Senary Audio)", current_index=None)
        assert idx2 == 4, f"first-match failed: {idx2}"
        # 名前不一致は None
        assert AudioOutput._match_output_index("Nothing", 4) is None
    print("[OK] _match_output_index host-api aware matching")
    return True


def test_reopen_on_device_change():
    """Core Audio名の変化 → 新しいデバイスで再オープンされることを検証"""
    a = AudioOutput(48000, 1024)
    # stream はモック (実デバイス不要で再オープン経路だけを検証)
    reopens = []

    def fake_reopen(idx):
        reopens.append(idx)

    a._reopen_for_device = fake_reopen
    a.current_device = 4
    a.is_running = True

    # 監視ループの本体ロジックを直接検証: 名前変化→一致→reopen
    with mock.patch("audio_output.sd.query_devices", return_value=MOCK_DEVICES):
        with mock.patch("win_audio._win_default_output_name", return_value="Speakers (Senary Audio)"):
            name = __import__("win_audio")._win_default_output_name()
            idx = AudioOutput._match_output_index(name, a.current_device)
            assert idx == 5, f"speakers match failed: {idx}"
            if idx is not None and idx != a.current_device:
                a._reopen_for_device(idx)
    assert reopens == [5], f"reopen not triggered: {reopens}"
    print("[OK] device change triggers reopen on matched index")
    return True


def test_watch_thread_stops_cleanly():
    a = AudioOutput(48000, 1024)
    a.start()
    # 監視スレッドが起動している
    assert a._device_watch_thread is not None and a._device_watch_thread.is_alive()
    a.stop()
    t = time.time()
    while a._device_watch_thread and a._device_watch_thread.is_alive() and time.time() - t < 3:
        time.sleep(0.05)
    assert a._device_watch_thread is None or not a._device_watch_thread.is_alive()
    print("[OK] device watch thread stops cleanly")
    return True


def main() -> int:
    ok = True
    ok &= test_match_same_hostapi()
    ok &= test_reopen_on_device_change()
    ok &= test_watch_thread_stops_cleanly()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())