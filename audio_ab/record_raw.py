"""生IQ録音＋アクティビティ判定 (audio_abプロジェクト用)。

復調音ではなく生IQ (.npy uint8) を保存する。耳ABは同一録音の往復比較の
ため、録りっぱなしの生IQが要る (tools/record_station.pyは復調WAVのみ)。

Usage:
  python audio_ab/record_raw.py 7.100 LSB 15 --tag day40m
  -> testdata/day40m_7.100_LSB_15s.npy (+ activity判定を表示)

activity: 音声帯域エネルギーの窓別ダイナミクス (無音/無変調はflat)。
  speechy スコアが高いほど音声らしい。目安: >6dBで要確認、>10dBで有望。
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rtlsdr_driver import RtlSdrDriver

BLOCK = 132096
FS = 48000


def activity_score(audio, fs=FS):
    """音声らしさ (dB)。0.5秒窓の300-3000Hzエネルギーのmax/min比。
    連続トーン・無変調・定常ノイズは低く、音声は高い。"""
    try:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if len(x) < fs:
            return -99.0
        w = fs // 2
        n = len(x) // w
        e = []
        for k in range(n):
            seg = x[k * w:(k + 1) * w]
            if len(seg) < w:
                break
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
            fr = np.fft.rfftfreq(len(seg), 1.0 / fs)
            m = (fr >= 300.0) & (fr <= 3000.0)
            e.append(float(np.sum(spec[m])) + 1e-18)
        e = np.asarray(e)
        return float(10 * np.log10(np.max(e) / (np.median(e) + 1e-18)))
    except Exception:
        return -99.0


def main(argv):
    if len(argv) < 3 or "--help" in argv:
        print(__doc__)
        return 1
    freq_mhz = float(argv[0])
    mode = argv[1].upper()
    secs = float(argv[2])
    tag = ""
    for i, a in enumerate(argv):
        if a == "--tag" and i + 1 < len(argv):
            tag = argv[i + 1]
    freq_hz = int(freq_mhz * 1e6)

    drv = RtlSdrDriver()
    if drv.get_device_count() == 0:
        print("no device", file=sys.stderr)
        return 1
    drv.open(0)
    try:
        drv.set_sample_rate(1152000)
        drv.set_direct_sampling(2 if freq_hz < 24000000 else 0)
        offset = 150000
        drv.set_center_freq(freq_hz + offset)
        drv.set_gain_mode(True)
        drv.set_gain(33.8)
        try:
            drv.reset_buffer()
        except Exception:
            pass
        n_blocks = max(1, int(secs * 1152000 * 2 / BLOCK))
        bufs = []
        t0 = time.monotonic()
        for _ in range(n_blocks):
            raw = drv.read_sync(BLOCK)
            if len(raw) < BLOCK:
                continue
            bufs.append(np.asarray(raw, dtype=np.uint8))
    finally:
        try:
            drv.close()
        except Exception:
            pass
    if not bufs:
        print("no data captured", file=sys.stderr)
        return 1
    raw = np.concatenate(bufs)
    dt = time.monotonic() - t0

    name = f"{tag + '_' if tag else ''}{freq_mhz:.3f}_{mode}_{secs:g}s.npy"
    out = os.path.join(ROOT, "testdata", name)
    np.save(out, raw)
    print(f"saved {out} ({len(raw) / 1e6:.1f}MB, {dt:.1f}s wall)")

    # アクティビティ判定 (簡易復調→音声帯ダイナミクス)
    try:
        from dsp import SdrDspPipeline
        dsp = SdrDspPipeline(1152000, FS)
        dsp.set_offset_freq(offset)
        outs = []
        for k in range(0, len(raw) - BLOCK + 1, BLOCK):
            audio, _ = dsp.process(np.asarray(raw[k:k + BLOCK], dtype=np.uint8),
                                   mode=mode)
            a = np.asarray(audio, dtype=np.float32)
            outs.append(a.reshape(-1))
        y = np.concatenate(outs)
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        print(f"mode={mode} rms={rms:.4f} speechy={activity_score(y):.1f}dB")
    except Exception as e:
        print(f"activity check skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
