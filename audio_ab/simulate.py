"""標準チャネルシミュレータ (audio_ab)。

「適切なSNR・妨害・クリックを再現性高く作る」ための共通資産。
これまで各スイープがtempスクリプトに書き捨てていた素材・妨害を
名前付きシナリオに集約する。測定側 (cer_ab等) はSCENARIOSを介して
モード非依存に扱う。

build(utts, snr_db, seed, **kw):
  utts は corpus.load() のdictリスト (wfmはL/Rの2本、他は1本使用)。
  戻りはcomplex64 (SSB/NFM/AM) またはuint8 raw (WFM)。
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audio_ab"))

from ssb_synth import noisy_ssb_iq  # noqa: E402

FS = 48000
IF = 288000
RF = 1152000


def add_clicks(x, fs, seed=0, rate_per_s=15.0, amp=12.0, width=(1, 6)):
    """ランダム位相の振幅置換でインパルスクリック列を注入する。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) + 1e-18
    out = x.copy()
    nclk = max(0, int(n / fs * rate_per_s))
    for p in rng.integers(1, max(2, n - 8), max(1, nclk)):
        w = int(rng.integers(width[0], width[1] + 1))
        out[p:p + w] = (amp * rms * np.exp(
            1j * rng.uniform(-np.pi, np.pi, w))).astype(out.dtype)
    return out


def _tone_iq(n, fs, f_hz, amp, phase=0.0):
    t = np.arange(n) / fs
    return (amp * np.exp(1j * (2 * np.pi * f_hz * t + phase))).astype(np.complex64)


def build_ssb(utts, snr_db, seed, clicks=0.0, intf_db=None, intf_hz=2500.0,
              click_amp=12.0, width=(1, 6)):
    x = np.asarray(utts[0]["x"], dtype=np.float32)
    iq = noisy_ssb_iq(x, float(snr_db), seed=int(seed))
    if clicks:
        iq = add_clicks(iq, FS, seed=seed + 100, rate_per_s=float(clicks),
                        amp=float(click_amp), width=width)
    if intf_db is not None:
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) + 1e-18
        iq = iq + _tone_iq(len(iq), FS, intf_hz, rms * 10 ** (intf_db / 20.0))
    return iq.astype(np.complex64)


def build_nfm(utts, snr_db, seed, dev_hz=3000.0, clicks=0.0,
              click_amp=10.0, width=(2, 10)):
    x = np.asarray(utts[0]["x"], dtype=np.float64).reshape(-1)
    n = int(IF * len(x) / FS)
    t = np.arange(n) / IF
    v = np.interp(t, np.arange(len(x)) / FS, x)
    ph = 2 * np.pi * float(dev_hz) * np.cumsum(v) / IF
    car = 0.5 * np.exp(1j * ph)
    rng = np.random.default_rng(seed)
    nz = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    nz = nz / (float(np.sqrt(np.mean(np.abs(nz) ** 2))) + 1e-12)
    sig_pow = float(np.mean(np.abs(car) ** 2))
    iq = (car + nz * np.sqrt(sig_pow / 10 ** (float(snr_db) / 10.0))).astype(np.complex64)
    if clicks:
        iq = add_clicks(iq, IF, seed=seed + 100, rate_per_s=float(clicks),
                        amp=float(click_amp), width=width)
    return iq


def build_am(utts, snr_db, seed, intf_db=None, intf_hz=2500.0, sides="one",
             gate_frac=0.0, carrier_hz=15.0):
    x = np.asarray(utts[0]["x"], dtype=np.float64).reshape(-1)
    n = int(IF * len(x) / FS)
    t = np.arange(n) / IF
    v = np.interp(t, np.arange(len(x)) / FS, x)
    env = 1e-4 * (1.0 + 0.5 * v)
    iq = (env * np.exp(1j * 2 * np.pi * carrier_hz * t)).astype(np.complex64)
    rng = np.random.default_rng(seed)
    nz = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.float64)
    nz = nz / (float(np.sqrt(np.mean(np.abs(nz) ** 2))) + 1e-12)
    sig_pow = float(np.mean(env ** 2))
    iq = iq + (nz * np.sqrt(sig_pow / 10 ** (float(snr_db) / 10.0))).astype(np.complex64)
    if intf_db is not None:
        gate = np.ones(n)
        gate[:int(n * float(gate_frac))] = 0.0
        amp = 1e-4 * 10 ** (float(intf_db) / 20.0)
        iq = iq + (gate * _tone_iq(n, IF, intf_hz, amp)).astype(np.complex64)
        if sides == "both":
            iq = iq + (gate * _tone_iq(n, IF, -intf_hz, amp)).astype(np.complex64)
    return iq


def preemph(x, tau_us=75.0, fs=FS):
    """放送プリエンファシス H=1+j2πfτ (音声帯域のFFTフィルタ)。"""
    X = np.fft.rfft(np.asarray(x, dtype=np.float64))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return np.fft.irfft(X * (1 + 1j * 2 * np.pi * f * tau_us * 1e-6), n=len(x))


def build_wfm(utts, snr_db, seed, dev_hz=30000.0, preemph_us=75.0):
    """BS.450ステレオMPX→FM→雑音→ADC生uint8 (dsp.process入力形式)。"""
    dur = max(len(np.asarray(u["x"])) for u in utts) / FS
    N = int(dur * RF)
    t = np.arange(N) / RF
    L = np.interp(t, np.arange(len(utts[0]["x"])) / FS, preemph(utts[0]["x"], preemph_us))
    R = np.interp(t, np.arange(len(utts[1 % len(utts)]["x"])) / FS,
                  preemph(utts[1 % len(utts)]["x"], preemph_us))
    mpx = (0.45 * (L + R) + 0.45 * (L - R) * np.sin(2 * np.pi * 38000.0 * t)
           + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
    mpx = mpx / (float(np.max(np.abs(mpx))) + 1e-9)
    ph = 2 * np.pi * float(dev_hz) * np.cumsum(mpx) / RF
    iq = 0.6 * np.exp(1j * ph)
    rng = np.random.default_rng(seed)
    p_noise = 0.36 / 10 ** (float(snr_db) / 10.0)
    iq = iq + np.sqrt(p_noise / 2.0) * (rng.standard_normal(N)
                                        + 1j * rng.standard_normal(N))
    raw = np.empty(2 * N, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
    return raw


# シナリオ定義: mode/block/io/demod/enable/utts/build/kw
SCENARIOS = {
    "ssb": {
        "mode": "SSB", "block": 2208, "io": "iq", "utts": 1,
        "build": build_ssb, "kw": {},
        "demod": ("demodulate_ssb", ("USB",)),
        "enable": ("attr", "nbm_nr_enabled"),
    },
    "ssb-clicks": {
        "mode": "SSB", "block": 2208, "io": "iq", "utts": 1,
        "build": build_ssb, "kw": {"clicks": 15.0},
        "demod": ("demodulate_ssb", ("USB",)),
        "enable": ("attr", "nbm_nr_enabled"),
    },
    "ssb-onesided": {
        "mode": "SSB", "block": 2208, "io": "iq", "utts": 1,
        "build": build_ssb, "kw": {"intf_db": -10.0},
        "demod": ("demodulate_ssb", ("USB",)),
        "enable": ("attr", "nbm_nr_enabled"),
    },
    "am": {
        "mode": "AM", "block": 16512, "io": "iq", "utts": 1,
        "build": build_am, "kw": {},
        "demod": ("demodulate_am", ()),
        "enable": ("attr", "am_sideband_enabled"),
    },
    "am-onesided": {
        "mode": "AM", "block": 16512, "io": "iq", "utts": 1,
        "build": build_am, "kw": {"intf_db": -10.0, "gate_frac": 0.35},
        "demod": ("demodulate_am", ()),
        "enable": ("attr", "am_sideband_enabled"),
    },
    "nfm": {
        "mode": "NFM", "block": 16512, "io": "iq", "utts": 1,
        "build": build_nfm, "kw": {},
        "demod": ("demodulate_nfm", ()),
        "enable": ("attr", "nbm_nr_enabled"),
    },
    "nfm-clicks": {
        "mode": "NFM", "block": 16512, "io": "iq", "utts": 1,
        "build": build_nfm, "kw": {"clicks": 15.0},
        "demod": ("demodulate_nfm", ()),
        "enable": ("attr", "nbm_nr_enabled"),
    },
    "wfm": {
        "mode": "WFM", "block": 132096, "io": "raw", "utts": 2,
        "build": build_wfm, "kw": {},
        "demod": ("process", ("WFM",)),
        "enable": ("method", "set_stereo_nr"),
    },
}


def build(name, utts, snr_db, seed, **over):
    sc = SCENARIOS[name]
    kw = dict(sc["kw"])
    kw.update(over)
    return sc["build"](utts, snr_db, seed, **kw)


if __name__ == "__main__":
    for name, sc in SCENARIOS.items():
        print(f"{name:14s} mode={sc['mode']:3s} block={sc['block']:6d} "
              f"utt={sc['utts']} kw={sc['kw']}")
