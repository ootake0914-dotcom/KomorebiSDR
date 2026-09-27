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
    # --cog: live Hyper相当 (認知制御+mono抑圧+ヒスゲート有効) で録る。
    # 既定はwide固定の素性評価用。live挙動の検証は --cog を付ける。
    live = any(a == "--cog" for a in sys.argv[1:])
    freq_hz = int(freq_mhz * 1e6)

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "recordings")
    os.makedirs(outdir, exist_ok=True)
    tag = "_live" if live else ""
    base = os.path.join(outdir, f"rec_{freq_mhz:.3f}_{mode}_{secs:g}s{tag}")

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
        # 本体Hyper経路の強電界相当で録る (既定clean=8.5kHzでは高域評価できない)。
        # WFMのみwide (15kHz) を使用。AM系は既定のまま narrow/clean を維持する。
        if live:
            try:
                # 強電界相当の認知パラメータでlive挙動を再現する
                dsp.set_cognitive_parameters(cutoff_hz=15000.0, hf_gain=1.0,
                                             if_bw_hz=190000.0, enabled=True)
            except Exception:
                pass
        elif mode == "WFM":
            try:
                dsp.filter_mode = "wide"
            except Exception:
                pass

        n_blocks = max(1, int(secs * 1152000 * 2 / BLOCK))
        audios = []
        t0 = time.monotonic()
        for _ in range(n_blocks):
            raw = drv.read_sync(BLOCK)
            if len(raw) < BLOCK:
                continue
            audio, _spec = dsp.process(np.asarray(raw, dtype=np.uint8), mode=mode)
            a = np.asarray(audio, dtype=np.float32)
            # ステレオ (N,2) はそのまま保持する。reshape(-1) で潰すと
            # L/R交互サンプルが1chに化けて解析・再生が壊れるため。
            # 弱電界でモノラル (N,) /(N,1) に落ちたブロックは2chへ複製し、
            # 混在時のconcat不一致 (ValueError) を防ぐ。本体put_audioと同一扱い。
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.shape[1] == 1:
                a = np.concatenate((a, a), axis=1)
            audios.append(a.reshape(a.shape[0], 2))
        dt = time.monotonic() - t0
    finally:
        try:
            drv.close()
        except Exception:
            pass

    if not audios:
        print("no audio captured", file=sys.stderr)
        return 1
    y = np.concatenate(audios, axis=0).astype(np.float32)
    n_ch = y.shape[1] if y.ndim == 2 else 1
    print(f"captured {y.shape[0] / 48000.0:.1f}s audio in {dt:.1f}s wall "
          f"(ch={n_ch}, rms={float(np.sqrt(np.mean(y ** 2))):.4f})")

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
