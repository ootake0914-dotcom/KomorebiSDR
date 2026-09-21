"""Hardware-free worker/scan lifecycle test using a fake SDR driver.

Verifies that band scanning safely stops the async USB stream, that the stream
always resumes (including after a simulated scan failure), and that a failing
command does not kill the receiver worker.
"""

import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
import main as main_mod


class FakeDriver:
    def __init__(self):
        self.is_open = False
        self._cancel = threading.Event()
        self._active = threading.Event()
        self.sample_rate = 1152000
        self.center_freq = 83000000
        self.async_calls = 0
        self.sync_while_active = 0
        self.sync_calls = 0
        self.fail_sync = False
        self.gains = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
                      19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
                      38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]

    def get_device_count(self): return 1
    def get_device_name(self, i=0): return "FAKE RTL-SDR"
    def open(self, i=0): self.is_open = True
    def close(self): self.is_open = False
    def set_sample_rate(self, r): self.sample_rate = r
    def get_sample_rate(self): return self.sample_rate
    def set_center_freq(self, f): self.center_freq = f
    def get_center_freq(self): return self.center_freq
    def set_gain_mode(self, m): pass
    def get_gains(self): return list(self.gains)
    def set_gain(self, g): pass
    def set_direct_sampling(self, m): pass
    def reset_buffer(self): pass

    def read_async(self, cb, num_buffers=16, buffer_len=132096):
        if self._cancel.is_set():
            return
        self._active.set()
        self.async_calls += 1
        try:
            while not self._cancel.is_set():
                time.sleep(0.057)
                cb(np.random.randint(0, 256, size=buffer_len, dtype=np.uint8))
        finally:
            self._active.clear()

    def cancel_async(self, timeout=2.0):
        self._cancel.set()
        self._active.wait(timeout)

    def resume_async(self):
        self._cancel.clear()

    def is_async_active(self):
        return self._active.is_set()

    def read_sync(self, n):
        self.sync_calls += 1
        if self._active.is_set():
            self.sync_while_active += 1
        if self.fail_sync and self.sync_calls > 1:
            raise RuntimeError("simulated scan read failure")
        time.sleep(0.005)
        return np.random.randint(0, 256, size=n, dtype=np.uint8)


def main() -> int:
    app = main_mod.SdrApp(initial_freq=83200000, initial_mode="WFM", controller_type="hyper")
    fake = FakeDriver()
    app.driver = fake
    app.controller.driver = fake
    app.tuner.driver = fake

    frames = [0]
    orig_process = app.dsp.process

    def counting_process(raw, mode="WFM"):
        frames[0] += 1
        return orig_process(raw, mode=mode)

    app.dsp.process = counting_process

    scans = []

    def fake_scan(start=76000000, end=95000000, step_hz=1800000, snr_threshold=4.2):
        scans.append(fake.is_async_active())
        time.sleep(0.1)
        app.tuner.discovered_stations = [
            {"freq_hz": 83200000, "freq_mhz": 83.2, "name": "83.2", "snr_db": 30.0,
             "peak_power_db": -30.0, "afc_offset_hz": 0.0, "quality": "STRONG"},
        ]
        app.tuner.last_scan_time = time.time()
        return list(app.tuner.discovered_stations)

    app.tuner.scan_band = fake_scan

    errors = []
    result = {"ok": False}

    def click(x, y):
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))

    def click_scan():
        # ハードコード座標ではなく実ボタン矩形中心を押す (レイアウト変更に頑健)
        r = app.gui.btn_scan_band.rect
        click(r.centerx, r.centery)

    def script():
        try:
            time.sleep(2.0)
            assert frames[0] > 5, f"no frames processed before scan: {frames[0]}"

            click_scan()  # scan button (glass layout)
            time.sleep(1.5)
            assert len(scans) == 1 and scans[0] is False, f"scan raced with async: {scans}"
            f1 = frames[0]
            time.sleep(1.0)
            assert frames[0] > f1, "streaming did not resume after scan"

            fake.fail_sync = True
            click_scan()
            time.sleep(2.0)
            assert len(scans) == 2 and scans[1] is False, f"failed scan raced: {scans}"
            f2 = frames[0]
            time.sleep(1.0)
            assert frames[0] > f2, "streaming did not resume after failed scan"

            app.tuner.discovered_stations = []
            click(1031, 314)  # Auto Seek >>
            time.sleep(1.5)
            assert len(scans) == 3 and scans[2] is False, f"seek scan raced: {scans}"
            assert fake.sync_while_active == 0, "read_sync ran while async stream active"
            assert fake.async_calls >= 4, f"async stream not restarted: {fake.async_calls}"

            result["ok"] = True
        except AssertionError as e:
            errors.append(str(e))
            traceback.print_exc()
        finally:
            app.gui.running = False

    threading.Thread(target=script, daemon=True).start()
    threading.Thread(target=lambda: (time.sleep(25), setattr(app.gui, "running", False)),
                     daemon=True).start()

    app.run()

    if errors:
        print("FAILED:", errors)
        return 1
    print(f"async_calls={fake.async_calls} sync_while_active={fake.sync_while_active} "
          f"scans={scans} frames={frames[0]}")
    print("OK" if result["ok"] else "FAILED (script did not complete)")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
