"""SSB合成ヘルパ (audio_abプロジェクト)。

重要: dspのdemodulate_ssbは周波数変換をしない (側波帯選択のみ)。
複素IF上の周波数がそのまま音声周波数になる。したがってSSB合成は
「解析信号のベースバンド直入れ」が正解である。実信号に+1500Hz掛けると
DSB (両側波が重なる) になり、了解度が壊れる (実測で確認済み)。

Usage:
  iq = ssb_usb_iq(voice_48k)                    # クリーンUSB
  iq = noisy_ssb_iq(voice_48k, snr_db, seed)    # 複素白色雑音つき
"""

import numpy as np


def analytic(x):
    """実信号→解析信号 (負周波数ゼロ。irfftではなくifftで作ること)。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    Xf = np.fft.fft(x)
    H = np.zeros(n)
    H[0] = 1.0
    if n % 2 == 0:
        H[1:n // 2] = 2.0
        H[n // 2] = 1.0
    else:
        H[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(Xf * H)


def ssb_usb_iq(voice):
    """実音声 (48k) → USB複素IF (音声帯域そのまま、複素化のみ)。"""
    return np.asarray(analytic(voice), dtype=np.complex64)


def noisy_ssb_iq(voice, snr_db, seed=0):
    """音声＋複素白色雑音のUSB複素IF。雑音は複素 (両側) でよい
    (USBフィルタが+150〜+2850Hzだけ通すため、帯域内雑音のみ効く)。"""
    v = np.asarray(voice, dtype=np.float32).reshape(-1)
    n = len(v)
    rng = np.random.default_rng(seed)
    nz = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    nz = nz / (float(np.sqrt(np.mean(np.abs(nz) ** 2))) + 1e-12)
    sig_pow = float(np.mean(v.astype(np.float64) ** 2))
    nz = (nz * np.sqrt(sig_pow / (10.0 ** (snr_db / 10.0)))).astype(np.complex64)
    return (np.asarray(analytic(v), dtype=np.complex64) + nz).astype(np.complex64)
