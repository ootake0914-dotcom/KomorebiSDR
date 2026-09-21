"""Run all hardware-free tests.

Usage: python tests/run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# TESTSハードコードでは新規テストが無視されるため、test_*.pyを自動検出する
import glob as _glob

TESTS = sorted(
    os.path.basename(p) for p in _glob.glob(os.path.join(HERE, "test_*.py"))
)


def main() -> int:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    failed = []
    for name in TESTS:
        path = os.path.join(HERE, name)
        print(f"\n===== {name} =====", flush=True)
        try:
            result = subprocess.run([sys.executable, path], env=env, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {name} exceeded 300s")
            failed.append(name)
            continue
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
