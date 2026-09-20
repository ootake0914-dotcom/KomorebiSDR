"""Run all hardware-free tests.

Usage: python tests/run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TESTS = [
    "test_regions.py",
    "test_shortwave.py",
    "test_am_sync.py",
    "test_ssb.py",
    "test_sw_schedule.py",
    "test_rds.py",
    "test_audio_output.py",
    "test_rt_profile.py",
    "test_native_equiv.py",
    "test_stereo.py",
    "test_adaptive_dsp.py",
    "test_gui.py",
    "test_live_peaks.py",
    "test_station_list.py",
    "test_mock_worker.py",
]


def main() -> int:
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    failed = []
    for name in TESTS:
        path = os.path.join(HERE, name)
        print(f"\n===== {name} =====", flush=True)
        result = subprocess.run([sys.executable, path], env=env)
        if result.returncode != 0:
            failed.append(name)

    print("\n====================")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
