"""自動化の完成: 選局直後 (未設定状態) から主要状態が3秒以内に収束すること。

合成IQ (WFM/AM/NFM/USB) を冷えたdspに流し、ブロック毎の主要状態
(Sメーター・同期ロック・ステレオブレンド・NRヒス推定・AFC・SSB AGC) が
最終値の許容内に入り、以後安定するまでの時間を測る。

注: スローAGC (attack 2s/release 10s) は局間レベリング用で意図的に遅い
(ポンピング防止) ため合否対象外とし、収束時間を参考表示する。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dsp import SdrDspPipeline

RF = 1152000
FS = 48000
BLOCK = 132096          # raw bytes (66048 IQ samples)
IQ_PER_BLOCK = BLOCK // 2
BLOCK_S = IQ_PER_BLOCK / RF
NBLK = 80               # ≈4.59s
DEADLINE_S = 3.0


def _to_raw(iq):
    iq = np.asarray(iq, dtype=np.complex128)
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def _add_noise(iq, snr_db, seed):
    rng = np.random.default_rng(seed)
    p = float(np.mean(np.abs(iq) ** 2)) / (10.0 ** (snr_db / 10.0))
    return iq + np.sqrt(p / 2.0) * (rng.standard_normal(len(iq))
                                    + 1j * rng.standard_normal(len(iq)))


def _fm_raw(mpx_fn, dev_hz, snr_db, seed=7):
    n = NBLK * IQ_PER_BLOCK
    iq = np.empty(n, dtype=np.complex128)
    ph = 0.0
    pos = 0
    while pos < n:
        m = min(IQ_PER_BLOCK, n - pos)
        t = np.arange(pos, pos + m) / RF
        mpx = mpx_fn(t)
        ph_arr = ph + 2.0 * np.pi * dev_hz * np.cumsum(mpx) / RF
        iq[pos:pos + m] = 0.6 * np.exp(1j * ph_arr)
        ph = float(ph_arr[-1])
        pos += m
    return _to_raw(_add_noise(iq, snr_db, seed))


def raw_wfm(snr_db=15.0):
    def mpx(t):
        left = np.sin(2 * np.pi * 1000.0 * t)
        right = np.sin(2 * np.pi * 5000.0 * t)
        m = (0.45 * (left + right)
             + 0.45 * (left - right) * np.sin(2 * np.pi * 38000.0 * t)
             + 0.09 * np.sin(2 * np.pi * 19000.0 * t))
        return m / (np.max(np.abs(m)) + 1e-9)
    return _fm_raw(mpx, 30000.0, snr_db)


def raw_nfm(snr_db=15.0):
    # +300Hzキャリアオフセット→AFCが-300Hzへ収束する (追従の実証)
    def mpx(t):
        return np.sin(2 * np.pi * 1000.0 * t)

    def gen():
        n = NBLK * IQ_PER_BLOCK
        iq = np.empty(n, dtype=np.complex128)
        ph = 0.0
        pos = 0
        while pos < n:
            m = min(IQ_PER_BLOCK, n - pos)
            t = np.arange(pos, pos + m) / RF
            msg = mpx(t)
            ph_arr = ph + 2.0 * np.pi * (300.0 * np.arange(m) / RF
                                         + 3000.0 * np.cumsum(msg) / RF)
            iq[pos:pos + m] = 0.6 * np.exp(1j * ph_arr)
            ph = float(ph_arr[-1])
            pos += m
        return iq
    return _to_raw(_add_noise(gen(), snr_db, seed=8))


def raw_am(snr_db=15.0):
    n = NBLK * IQ_PER_BLOCK
    t = np.arange(n) / RF
    iq = (0.5 + 0.4 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.complex128)
    return _to_raw(_add_noise(iq, snr_db, seed=9))


def raw_usb(snr_db=15.0):
    n = NBLK * IQ_PER_BLOCK
    t = np.arange(n) / RF
    iq = 0.3 * np.exp(2j * np.pi * 2500.0 * t)
    return _to_raw(_add_noise(iq, snr_db, seed=10))


def _states(dsp, mode):
    s = {
        "s_meter": float(getattr(dsp, "s_meter_dbfs", -90.0)),
        "speech": float(getattr(getattr(dsp, "cognitive_eq", None),
                                "speech_prob", 0.0)),
        "slow_agc": float(getattr(dsp, "slow_agc_gain", 1.0)),
    }
    if mode == "WFM":
        s["pilot"] = float(getattr(dsp, "stereo_pilot_lock", 0.0))
        s["blend"] = float(getattr(dsp, "stereo_blend", 0.0))
        s["hiss"] = float(getattr(dsp, "stereo_hiss_db", 0.0))
        s["afc"] = float(getattr(dsp, "afc_offset_hz", 0.0))
    elif mode == "AM":
        s["am_lock"] = float(getattr(dsp, "am_sync_lock", 0.0))
    elif mode == "NFM":
        s["afc"] = float(getattr(dsp, "nfm_afc_offset_hz", 0.0))
    elif mode == "USB":
        lv = float(getattr(dsp, "ssb_agc_level", 1e-4))
        s["ssb_agc_db"] = 20.0 * float(np.log10(lv + 1e-18))
    return s


TOLS = {"s_meter": 1.5, "speech": 0.20, "pilot": 0.10, "blend": 0.10,
        "hiss": 2.0, "afc": 60.0, "am_lock": 0.10, "ssb_agc_db": 1.5,
        "slow_agc": 0.10}


def _settle_time(series, ref, tol):
    ok = np.abs(np.asarray(series) - ref) <= tol
    stable = np.logical_and.accumulate(ok[::-1])[::-1]
    if not bool(stable.any()):
        return None
    return float(np.argmax(stable)) * BLOCK_S


def _run(mode, raw):
    dsp = SdrDspPipeline(RF, FS)
    dsp.set_offset_freq(0.0)
    dsp.cognitive_enabled = True
    dsp.set_cognitive_parameters(cutoff_hz=8500.0, hf_gain=1.0,
                                 if_bw_hz=150000.0)
    hist = []
    for k in range(NBLK):
        dsp.process(raw[k * BLOCK:(k + 1) * BLOCK], mode=mode)
        hist.append(_states(dsp, mode))
    ref = {key: float(np.mean([h[key] for h in hist[-8:]]))
           for key in hist[-1]}
    times = {}
    for key in hist[-1]:
        series = [h[key] for h in hist]
        t = _settle_time(series, ref[key], TOLS[key])
        times[key] = t
    return ref, times


def test_convergence():
    all_ok = True
    for mode, raw in (("WFM", raw_wfm()), ("AM", raw_am()),
                      ("NFM", raw_nfm()), ("USB", raw_usb())):
        ref, times = _run(mode, raw)
        parts = []
        worst = 0.0
        for key, t in times.items():
            if key == "slow_agc":
                continue  # 設計上遅い (レベリング用)。参考表示のみ
            parts.append(f"{key}={t if t is None else round(t, 2)}s")
            if t is None or t > DEADLINE_S:
                all_ok = False
        print(f"{mode}: " + " ".join(parts)
              + f" | slow_agc={times['slow_agc'] if times['slow_agc'] is None else round(times['slow_agc'], 2)}s(参考)")
        print(f"    ref: " + " ".join(
            f"{k}={v:.3f}" for k, v in ref.items() if k != "slow_agc"))
    assert all_ok, "主要状態が3秒以内に収束しなかった"
    print(f"convergence OK (<{DEADLINE_S}s)")


def test_mode_final_values():
    # 収束値の意味的妥当性 (誤判定ゼロの裏付け)
    ref_wfm, _ = _run("WFM", raw_wfm())
    assert ref_wfm["pilot"] > 0.5, ref_wfm
    assert ref_wfm["blend"] > 0.8, ref_wfm
    ref_nfm, _ = _run("NFM", raw_nfm())
    assert abs(ref_nfm["afc"] + 300.0) < 40.0, ref_nfm  # +300Hzを補正
    ref_am, _ = _run("AM", raw_am())
    assert ref_am["am_lock"] > 0.5, ref_am
    print("final values OK")


if __name__ == "__main__":
    test_convergence()
    test_mode_final_values()
    print("ALL CONVERGENCE TESTS PASSED!")
