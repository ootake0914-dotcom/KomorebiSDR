"""AM sideband-diversity synthesis tests (synthetic, no hardware).

2 (AM側波帯合成) の回帰テスト:
1) クリーン信号では enabled 出力が disabled とビット等価 (透過性)。
   加算パスが現状と一致することの証拠。通らなければ設計ミス。
2) 片側 (USB) のみに隣接妨害を加えた素材で、妨害残留が減り、
   番組トーンが保たれること (SINR改善)。
3) 切替時に非有限・段差クリックが出ないこと。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsp import SdrDspPipeline

RATE = 288000
BLK = 16512
NBLK = 12


def _run(dsp, iq):
    out = []
    for k in range(NBLK):
        out.append(dsp.demodulate_am(iq[k * BLK:(k + 1) * BLK]))
    return np.concatenate(out)


def _spectrum(audio, rate=48000, tail=32768):
    seg = np.asarray(audio[-tail:], dtype=np.float64)
    seg = seg - float(np.mean(seg))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / rate)
    return spec, freqs


def _peak(spec, freqs, f, tol=30.0):
    m = (freqs > f - tol) & (freqs < f + tol)
    return float(np.max(spec[m]))


def test_clean_transparent() -> bool:
    rng = np.random.default_rng(7)
    n = BLK * NBLK
    t = np.arange(n) / RATE
    audio = np.cos(2 * np.pi * 1000.0 * t)
    env = 1e-4 * (1.0 + 0.5 * audio)
    iq = (env * np.exp(1j * 2 * np.pi * 15.0 * t)).astype(np.complex64)
    iq = iq + (5e-6 * (rng.standard_normal(n)
                       + 1j * rng.standard_normal(n))).astype(np.complex64)

    d_off = SdrDspPipeline(1152000, 48000)
    y_off = _run(d_off, iq)
    d_on = SdrDspPipeline(1152000, 48000)
    d_on.am_sideband_enabled = True
    y_on = _run(d_on, iq)
    diff = float(np.max(np.abs(y_on - y_off)))
    ok = diff == 0.0
    print(f"[{'OK' if ok else 'FAIL'}] clean transparent: max|on-off|={diff:.3e} "
          f"(need exactly 0)")
    return ok


def test_onesided_interference() -> bool:
    # 現実的な到来: クリーンでロック→途中 (6ブロック目) からUSB側妨害が始まる。
    # baselineはクリーン時に学習される。評価は後半のみ。
    rng = np.random.default_rng(11)
    n = BLK * NBLK
    t = np.arange(n) / RATE
    audio = np.cos(2 * np.pi * 1000.0 * t)
    env = 1e-4 * (1.0 + 0.5 * audio)
    iq = (env * np.exp(1j * 2 * np.pi * 15.0 * t)).astype(np.complex64)
    # USB側のみの隣接妨害: 搬送波+2.5kHzの無変調トーン (LSBには存在しない)
    gate = np.ones(n)
    gate[:BLK * 6] = 0.0
    iq = iq + (gate * 3e-5 * np.exp(1j * 2 * np.pi * 2500.0 * t)).astype(np.complex64)
    iq = iq + (5e-6 * (rng.standard_normal(n)
                       + 1j * rng.standard_normal(n))).astype(np.complex64)

    d_off = SdrDspPipeline(1152000, 48000)
    y_off = _run(d_off, iq)
    d_on = SdrDspPipeline(1152000, 48000)
    d_on.am_sideband_enabled = True
    y_on = _run(d_on, iq)

    # 評価は妨害到来後の後半のみ (最後6ブロック≒16512サンプル)
    s_off, f = _spectrum(y_off, tail=16384)
    s_on, _ = _spectrum(y_on, tail=16384)
    prog_off, prog_on = _peak(s_off, f, 1000.0), _peak(s_on, f, 1000.0)
    intf_off, intf_on = _peak(s_off, f, 2500.0), _peak(s_on, f, 2500.0)
    sinr_off = 20 * np.log10(prog_off / (intf_off + 1e-18))
    sinr_on = 20 * np.log10(prog_on / (intf_on + 1e-18))
    # 番組は保たれる (±3dB以内) かつ妨害が減ること
    prog_kept = abs(20 * np.log10(prog_on / (prog_off + 1e-18))) < 3.0
    improved = sinr_on > sinr_off + 1.0
    ok = bool(prog_kept and improved)
    print(f"[{'OK' if ok else 'FAIL'}] one-sided: SINR off={sinr_off:.1f}dB "
          f"on={sinr_on:.1f}dB prog_kept={prog_kept}")
    # クリック: 非有限なし＋ブロック境界の段差が番組振幅内に収まること
    finite = bool(np.all(np.isfinite(y_on)))
    step = float(np.max(np.abs(np.diff(y_on))))
    rms = float(np.sqrt(np.mean(y_on ** 2))) + 1e-18
    click_ok = bool(finite and step < 20.0 * rms)
    print(f"[{'OK' if click_ok else 'FAIL'}] no-click: finite={finite} "
          f"maxstep/rms={step / rms:.2f}")
    return bool(ok and click_ok)


def test_unlocked_fallback() -> bool:
    # 無搬送波ノイズでは側波帯パスが発動せず包絡線相当 (発散・爆音なし)
    rng = np.random.default_rng(3)
    n = BLK * 4
    noise = (2e-5 * (rng.standard_normal(n)
                     + 1j * rng.standard_normal(n))).astype(np.complex64)
    d = SdrDspPipeline(1152000, 48000)
    d.am_sideband_enabled = True
    out = []
    for k in range(4):
        out.append(d.demodulate_am(noise[k * BLK:(k + 1) * BLK]))
    y = np.concatenate(out)
    ok = bool(np.all(np.isfinite(y))) and float(np.max(np.abs(y))) <= 1.0
    print(f"[{'OK' if ok else 'FAIL'}] unlocked fallback: finite, "
          f"bounded (max={float(np.max(np.abs(y))):.3f})")
    return ok


def main() -> int:
    ok = test_clean_transparent()
    ok &= test_onesided_interference()
    ok &= test_unlocked_fallback()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
