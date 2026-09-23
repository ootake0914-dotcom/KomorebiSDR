"""指定局を録音する ( KomorebiSDR 受信品質確認用CLI )。

使い方:
    python tools/record_station.py 439.56 NFM 15
    -> recordings/rec_439.560_NFM_15s.wav (+ .npy 生IQは --raw で保存)

選局は main.py と同一手順 (DC回避 +150kHzオフセット)。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rtlsdr_driver import RtlSdrDriver
from dsp import SdrDspPipeline

BLOCK = 132096  # main.py と同一 (57.3ms空中時間)


def main() -> int:
    freq_mhz = float(sys.argv[1]) if len(sys.argv) > 1 else 439.56
    mode = (sys.argv[2] if len(sys.argv) > 2 else "NFM").upper()
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    freq_hz = int(freq_mhz * 1e6)

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "recordings")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"rec_{freq_mhz:.3f}_{mode}_{secs:g}s")

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

        dsp = SdrDspPipeline(1152000, 48000)
        dsp.set_offset_freq(offset)

        n_blocks = max(1, int(secs * 1152000 * 2 / BLOCK))
        audios = []
        t0 = time.monotonic()
        for _ in range(n_blocks):
            raw = drv.read_sync(BLOCK)
            if len(raw) < BLOCK:
                continue
            audio, _spec = dsp.process(np.asarray(raw, dtype=np.uint8), mode=mode)
            audios.append(np.asarray(audio).reshape(-1))
        dt = time.monotonic() - t0
    finally:
        try:
            drv.close()
        except Exception:
            pass

    if not audios:
        print("no audio captured", file=sys.stderr)
        return 1
    y = np.concatenate(audios).astype(np.float32)
    print(f"captured {len(y) / 48000.0:.1f}s audio in {dt:.1f}s wall "
          f"(rms={float(np.sqrt(np.mean(y ** 2))):.4f})")

    import wave
    pcm = np.clip(y, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(base + ".wav", "wb") as wf:
        wf.setnchannels(1 if pcm.ndim == 1 else 2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(pcm.tobytes())
    print("wrote", base + ".wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
