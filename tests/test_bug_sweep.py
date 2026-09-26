"""Bug-sweep regression tests (no hardware).

Covers fixes from the repo-wide bug hunt:
- auto_tuner: per-band scan TTL (FM/SW no longer share freshness)
- main.ham_seek_candidate: wrap-around + self-only + empty
- main._update_presets_from_scan: out-of-band presets (ISS) survive FM scan
- main._apply_frequency_and_mode: SSB BFO kept, CW auto-pitch reset kept
- rtlsdr_driver.set_center_freq: out-of-range rejected (no c_uint32 wrap)
- audio_output.put_audio: NaN/Inf sanitized; get_queue_size fractional
- gui: spectrum x<->freq mapping roundtrip, list auto-scroll, modal wheel
  swallow, waterfall empty guard
- dsp: AM/SSB AGC NaN guard (no stuck state), NFM mute phase continuity
"""

import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame  # noqa: F401  (gui needs it; dummy driver above)


class _StubDriver:
    def __init__(self):
        self.is_open = True
        self.center_freq = 0
        self.direct_sampling = 0
        self.sample_rate = 1152000

    def set_sample_rate(self, r):
        self.sample_rate = r

    def set_direct_sampling(self, m):
        self.direct_sampling = m

    def set_center_freq(self, f):
        self.center_freq = int(f)


def test_autotuner_band_ttl():
    from auto_tuner import AutoTuner
    calls = []

    t = AutoTuner(driver=_StubDriver())
    t.discovered_stations = [{"freq_hz": 80000000, "snr_db": 10.0}]
    t.discovered_sw = [{"freq_hz": 6000000, "snr_db": 10.0}]
    t.scan_band = lambda *a, **k: calls.append("fm") or []
    t.scan_band_hf = lambda *a, **k: calls.append("sw") or []

    # SW stale + FM fresh -> SW seek must rescan HF (not fooled by FM time)
    t.last_scan_time_sw = time.time() - 3600.0
    t.last_scan_time_fm = time.time()
    t.last_scan_time = time.time()
    try:
        t.seek_next(6000000, direction=1, use_sw=True)
    except Exception:
        pass
    assert "sw" in calls, f"stale SW list was reused: {calls}"

    # FM stale + SW fresh -> FM seek must rescan FM
    calls.clear()
    t.last_scan_time_fm = time.time() - 3600.0
    t.last_scan_time_sw = time.time()
    try:
        t.seek_next(80000000, direction=1, use_sw=False)
    except Exception:
        pass
    assert "fm" in calls, f"stale FM list was reused: {calls}"

    # both fresh -> no rescan
    calls.clear()
    t.last_scan_time_fm = time.time()
    t.last_scan_time_sw = time.time()
    t.seek_next(80000000, direction=1, use_sw=False)
    assert calls == [], f"fresh list rescanned: {calls}"
    print("[OK] autotuner per-band TTL")


def test_ham_seek_candidate():
    import main as main_mod
    H = main_mod.SdrApp.ham_seek_candidate
    ham = [{"freq_hz": f} for f in (7000000, 7100000, 7200000)]
    assert H(ham, 7050000, +1)["freq_hz"] == 7100000
    assert H(ham, 7050000, -1)["freq_hz"] == 7000000
    # wrap at edges
    assert H(ham, 7300000, +1)["freq_hz"] == 7000000
    assert H(ham, 6900000, -1)["freq_hz"] == 7200000
    # self-only list does not loop onto itself
    assert H([{"freq_hz": 7100000}], 7100000, +1) is None
    assert H([{"freq_hz": 7100000}], 7100000, -1) is None
    assert H([], 7100000, +1) is None
    print("[OK] ham seek candidate wrap")


def _make_app():
    import main as main_mod
    app = main_mod.SdrApp(initial_freq=83200000, initial_mode="WFM",
                          controller_type="hyper")
    app.driver = _StubDriver()
    return app


def test_presets_keep_iss():
    import main as main_mod
    app = _make_app()
    app.profile = main_mod.region_profile("JP")
    app.config["presets_fm"] = [
        {"name": "ISS 145.8", "freq_hz": 145800000, "mode": "NFM"},
    ]
    app.config["presets_region"] = "JP"
    main_mod.save_config = lambda cfg: None  # no disk side effects
    scan = [{"freq_hz": 80000000 + i * 100000, "freq_mhz": 80.0 + i * 0.1,
             "snr_db": 20.0 - i, "mode": "WFM"} for i in range(10)]
    app._update_presets_from_scan(scan)
    kept = [p["freq_hz"] for p in app.config["presets_fm"]]
    assert 145800000 in kept, f"ISS preset lost: {kept}"
    assert len([f for f in kept if 76000000 <= f <= 95000000]) == 10
    print("[OK] FM scan keeps out-of-band presets (ISS)")


def test_bfo_kept_for_ssb():
    app = _make_app()
    app.dsp.bfo_offset_hz = 300.0
    app._apply_frequency_and_mode(7074000, "USB")
    assert float(app.dsp.bfo_offset_hz) == 300.0, app.dsp.bfo_offset_hz
    app.dsp.bfo_offset_hz = 300.0
    app._apply_frequency_and_mode(7074000, "CW")
    assert float(app.dsp.bfo_offset_hz) == 0.0, app.dsp.bfo_offset_hz
    print("[OK] SSB BFO kept, CW auto-pitch reset kept")


def test_driver_freq_validation():
    from rtlsdr_driver import RtlSdrDriver
    drv = RtlSdrDriver.__new__(RtlSdrDriver)
    drv.is_open = True
    drv.ppm = 0
    drv._ppm_hw = True
    for bad in (-1, 0, 9999, 2000000000):
        try:
            drv.set_center_freq(bad)
        except ValueError:
            continue
        raise AssertionError(f"no validation for {bad}")
    print("[OK] driver center-freq validation")


def test_audio_sanitize_and_fraction():
    from audio_output import AudioOutput
    ao = AudioOutput()
    ao.stream = object()  # pretend open (queue path only, no HW)
    bad = np.array([[np.nan, np.inf], [1.0, -np.inf]], dtype=np.float32)
    ao.put_audio(bad)
    got = ao.audio_queue.get_nowait()
    assert bool(np.all(np.isfinite(got))), f"NaN leaked: {got}"
    assert float(np.max(np.abs(got))) <= 1.0
    ao.remainder = np.zeros((100, 2), dtype=np.float32)
    q = ao.get_queue_size()
    assert 0.0 < q < 1.0, f"remainder over-counted: {q}"
    print("[OK] audio NaN sanitize + fractional remainder")


def test_gui_mapping_and_list():
    from gui import SdrGui
    gui = SdrGui()
    gui.center_freq = 80000000
    gui.sample_rate = 1152000
    r = gui.spec_rect
    for mx in (r.x + 4, r.centerx, r.right - 4):
        f = gui._spec_x_to_freq(mx)
        back = gui._spec_freq_to_x(f)
        assert abs(back - mx) <= 1, f"mapping roundtrip {mx}->{f}->{back}"
    # center maps to center pixel (symmetric inset)
    assert gui._spec_freq_to_x(80000000) == r.centerx
    # auto-scroll: selection at end becomes visible on open
    gui.detected_stations = [
        {"freq_hz": 76000000 + i * 700000,
         "freq_mhz": 76.0 + i * 0.7, "name": f"ST{i:02d}",
         "snr_db": 10.0, "quality": "STRONG"} for i in range(25)]
    gui.center_freq = 76000000 + 24 * 700000
    gui._toggle_station_list()
    assert gui.station_list_open
    _, rows, total = gui._station_list_layout()
    start = gui.station_list_scroll
    sel = gui._selected_station_index()
    assert sel == 24, sel
    assert start <= sel < start + len(rows), f"sel {sel} outside view {start}"
    # modal wheel outside panel: no background tuning, list stays open
    from unittest import mock
    before = gui.center_freq
    ev = pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": 1})
    pygame.event.post(ev)
    with mock.patch("pygame.mouse.get_pos", return_value=(5, 5)):
        gui.handle_events()
    assert gui.center_freq == before, "wheel leaked through modal"
    assert gui.station_list_open, "wheel closed modal unexpectedly"
    # empty waterfall must not raise
    gui.update_waterfall(np.zeros(0, dtype=np.float32))
    gui.close()
    print("[OK] gui mapping/scroll/modal/waterfall")


def test_dsp_nan_guards():
    from dsp import SdrDspPipeline
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(0.0)
    bad = (np.nan * np.ones(4096, dtype=np.complex64))
    for _ in range(2):
        out = dsp._am_agc_normalize(np.abs(bad).astype(np.float32))
        assert bool(np.all(np.isfinite(out))), "AM NaN leaked"
    assert np.isfinite(dsp.am_agc_level), "AM AGC stuck at NaN"
    bad48 = (np.nan * np.ones(2752, dtype=np.complex64))
    out = dsp.demodulate_ssb(bad48, "USB")
    assert bool(np.all(np.isfinite(out))), "SSB NaN leaked"
    assert np.isfinite(dsp.ssb_agc_level), "SSB AGC stuck at NaN"
    print("[OK] dsp AM/SSB NaN guards")


def test_nfm_mute_continuity():
    from dsp import SdrDspPipeline
    dsp = SdrDspPipeline(1152000, 48000)
    dsp.set_offset_freq(0.0)
    rng = np.random.default_rng(1)
    quiet = (rng.standard_normal(16512) * 1e-4
             + 1j * rng.standard_normal(16512) * 1e-4).astype(np.complex64)
    out0 = dsp.demodulate_nfm(quiet)
    assert len(out0) > 0 and bool(np.all(np.isfinite(out0)))
    assert bool(np.all(np.isfinite([dsp.nfm_last_sample.real,
                                    dsp.nfm_last_sample.imag]))), \
        "mute left stale phase reference"
    print("[OK] NFM mute phase continuity")


def main() -> int:
    test_autotuner_band_ttl()
    test_ham_seek_candidate()
    test_presets_keep_iss()
    test_bfo_kept_for_ssb()
    test_driver_freq_validation()
    test_audio_sanitize_and_fraction()
    test_gui_mapping_and_list()
    test_dsp_nan_guards()
    test_nfm_mute_continuity()
    print("ALL BUG-SWEEP TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
