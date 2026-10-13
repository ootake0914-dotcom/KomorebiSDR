"""
透明性チェック: 音声出力段 (AudioOutput) が信号を変えていないことを検証する。

検証内容:
  - 合成正弦波 (振幅0.5 = リミッタ閾値0.98未満) を put_audio -> コールバックへ通す
  - 出力は入力を「ルックアヘッドリミッタの遅延分 (72サンプル)」だけ遅らせたものと
    ビット単位で一致するかを確認する
  - ボリューム1.0・アンダーランなし・フェードなしの条件で実行する

注意: DSPパイプライン (SdrDspPipeline) の透明性は検証しない。出力段のみ。
使い方: python tools/transparency_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from audio_output import AudioOutput  # noqa: E402


def run_output_stage_check(seconds: float = 2.0, amplitude: float = 0.5) -> bool:
    ao = AudioOutput(sample_rate=48000, blocksize=1024)
    ao.stream = object()  # put_audio は stream が無いと破棄するためダミーを置く
    ao.volume = 1.0
    ao._vol_current = 1.0
    ao._played_once = False  # 初回再生としてフェードを掛けない

    frames = 1024  # コールバック単位
    t = np.arange(int(48000 * seconds)) / 48000.0
    mono = (amplitude * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    src = np.stack([mono, mono * 0.7], axis=1).astype(np.float32)
    # 末尾端数は最終コールバックのアンダーラン扱いになり、同コールバック内の
    # fade-inで有効サンプルまで変わる。framesの整数倍に無音パディングする。
    pad = (-len(src)) % frames
    if pad:
        src = np.concatenate([src, np.zeros((pad, 2), dtype=np.float32)])

    chunk = 2752
    for i in range(0, len(src), chunk):
        ao.put_audio(src[i:i + chunk])

    out = []
    buf = np.zeros((frames, 2), dtype=np.float32)
    while not ao.audio_queue.empty() or len(ao.remainder) > 0:
        ao._audio_callback(buf, frames, None, None)
        out.append(buf.copy())
    if not out:
        print("NG: 出力が得られなかった")
        return False
    out = np.concatenate(out, axis=0)

    delay = ao._lim_delay_n  # ルックアヘッドリミッタの遅延 (72サンプル)
    n = min(len(src) - delay, len(out) - delay)
    expected = src[:n]
    actual = out[delay:delay + n]

    diff = np.abs(actual - expected)
    max_err = float(diff.max())
    underruns = ao.underrun_count

    print(f"比較サンプル数: {n} / 遅延: {delay} サンプル / 末尾パディング: {pad}")
    print(f"最大誤差: {max_err:.3e} / アンダーラン回数: {underruns}")
    ok = max_err == 0.0 and underruns == 0
    print("OK: 出力段は透明 (ビット一致)" if ok else "NG: 出力段で信号が変化、またはアンダーラン発生")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_output_stage_check() else 1)
