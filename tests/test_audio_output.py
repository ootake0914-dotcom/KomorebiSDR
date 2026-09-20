"""AudioOutput callback tests (no hardware, no sounddevice stream).

Verifies the preallocated-scratch callback path:
- sample-accurate reassembly across chunk boundaries
- preroll gating
- underrun fade + re-preroll
- volume/limiter correctness
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from audio_output import AudioOutput

BLOCK = 1024


def put(ao, arr):
    """stream無しでもキューへ直接投入 (テスト用)"""
    ao.audio_queue.put_nowait(np.asarray(arr, dtype=np.float32))


def main() -> int:
    ok = True
    ao = AudioOutput(48000, blocksize=BLOCK)
    ao.volume = 1.0
    ao._vol_current = 1.0  # 音量ランプを整定済みにする (直接代入はテスト用)

    # 1) プレロール前は無音
    out = np.ones((BLOCK, 2), dtype=np.float32)
    ao._audio_callback(out, BLOCK, None, None)
    ok &= bool(np.all(out == 0.0))
    print(f"[{'OK' if ok else 'FAIL'}] preroll outputs silence")

    # 2) チャンク境界をまたぐ再構成 (2752サンプル × 8)
    ao.preroll_threshold = 2
    chunks = []
    for i in range(8):
        n = 2752
        t = np.arange(i * n, (i + 1) * n) / 48000.0
        # 振幅0.5 (ソフトリミッターしきい値0.85未満で素通しを検証)
        chunks.append((0.5 * np.stack([np.sin(2 * np.pi * 440 * t),
                                       np.sin(2 * np.pi * 660 * t)], axis=1)).astype(np.float32))
    for c in chunks:
        put(ao, c)
    ref = np.concatenate(chunks, axis=0)
    nblocks = len(ref) // BLOCK  # 端数ブロックは意図的に除外
    got = []
    for _ in range(nblocks):
        out = np.zeros((BLOCK, 2), dtype=np.float32)
        ao._audio_callback(out, BLOCK, None, None)
        got.append(out.copy())
    y = np.concatenate(got, axis=0)
    err = float(np.max(np.abs(y - ref[:len(y)])))
    good = err < 1e-6 and ao.underrun_count == 0
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] reassembly across chunk boundaries "
          f"(max error {err:.2e}, underruns={ao.underrun_count})")

    # 3) アンダーラン時: 減衰して0へ (クリックなし)
    ao2 = AudioOutput(48000, blocksize=BLOCK)
    ao2.is_prerolled = True
    put(ao2, np.full((4096, 2), 0.5, dtype=np.float32))
    tail = None
    for _ in range(8):
        out = np.zeros((BLOCK, 2), dtype=np.float32)
        ao2._audio_callback(out, BLOCK, None, None)
        tail = out.copy()
    good = ao2.underrun_count > 0 and float(np.max(np.abs(tail))) < 0.02
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] underrun fades to silence "
          f"(underruns={ao2.underrun_count}, tail={float(np.max(np.abs(tail))):.4f})")

    # 4) 音量とソフトリミッター
    ao3 = AudioOutput(48000, blocksize=BLOCK)
    ao3.is_prerolled = True
    ao3.volume = 0.5
    ao3._vol_current = 0.5  # 音量ランプ整定済み扱い
    put(ao3, np.full((2048, 2), 0.4, dtype=np.float32))
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    ao3._audio_callback(out, BLOCK, None, None)
    good = abs(float(out[0, 0]) - 0.2) < 1e-6
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] volume scaling (got {out[0, 0]:.3f}, expect 0.200)")

    ao3.volume = 2.0
    ao3._vol_current = 2.0
    put(ao3, np.full((2048, 2), 0.9, dtype=np.float32))
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    ao3._audio_callback(out, BLOCK, None, None)
    peak = float(np.max(np.abs(out)))
    good = threshold_ok = peak <= 1.0 and peak > 0.8
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] soft limiter keeps peak <= 1.0 (peak {peak:.3f})")

    # 5) アンダーラン復帰時のフェードイン (無音→任意振幅の段差クリック防止)
    ao4 = AudioOutput(48000, blocksize=BLOCK)
    ao4.is_prerolled = True
    ao4.volume = 1.0
    ao4._vol_current = 1.0
    put(ao4, np.full((BLOCK, 2), 0.5, dtype=np.float32))
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    ao4._audio_callback(out, BLOCK, None, None)
    # キューを空にしてアンダーラン → 再プレロール → 復帰
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    ao4._audio_callback(out, BLOCK, None, None)
    # 再プレロール閾値(4)を満たすよう個別チャンクで4つ投入
    for _ in range(4):
        put(ao4, np.full((BLOCK, 2), 0.5, dtype=np.float32))
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    ao4._audio_callback(out, BLOCK, None, None)
    head, tail = float(abs(out[0, 0])), float(out[-1, 0])
    good = head < 0.05 and abs(tail - 0.5) < 1e-6
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] underrun resume fades in (head={head:.3f}, tail={tail:.3f})")

    # 6) last_out_samplesは音量適用前 (次回アンダーランでの二重適用-6dB段差防止)
    ao5 = AudioOutput(48000, blocksize=BLOCK)
    ao5.is_prerolled = True
    ao5.volume = 0.5
    ao5._vol_current = 0.5
    put(ao5, np.full((BLOCK, 2), 0.4, dtype=np.float32))
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    ao5._audio_callback(out, BLOCK, None, None)
    saved = float(ao5.last_out_samples[0])
    good = abs(saved - 0.4) < 1e-6
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] last_out_samples is pre-volume (got {saved:.3f}, expect 0.400)")

    print(f"     callback max time: {ao.callback_us_max:.0f} us")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
