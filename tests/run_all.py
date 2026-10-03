"""Run all hardware-free tests.

Usage:
  python tests/run_all.py                     # 全テスト
  python tests/run_all.py --only audio_ab,lufs
  python tests/run_all.py --skip sic
"""

import glob as _glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# TESTSハードコードでは新規テストが無視されるため、test_*.pyを自動検出する
TESTS = sorted(
    os.path.basename(p) for p in _glob.glob(os.path.join(HERE, "test_*.py"))
)


def _filters(argv):
    only, skip = [], []
    i = 0
    while i < len(argv):
        if argv[i] == "--only" and i + 1 < len(argv):
            only = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            i += 2
        elif argv[i] == "--skip" and i + 1 < len(argv):
            skip = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            i += 2
        else:
            i += 1
    return only, skip


def main() -> int:
    only, skip = _filters(sys.argv[1:])
    names = TESTS
    if only:
        names = [n for n in names if any(s in n for s in only)]
    if skip:
        names = [n for n in names if not any(s in n for s in skip)]
    if not names:
        print("no tests selected")
        return 1

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    failed = []
    elapsed = {}
    for name in names:
        path = os.path.join(HERE, name)
        print(f"\n===== {name} =====", flush=True)
        t0 = time.perf_counter()
        try:
            result = subprocess.run([sys.executable, path], env=env, timeout=300)
        except subprocess.TimeoutExpired:
            elapsed[name] = time.perf_counter() - t0
            print(f"[TIMEOUT] {name} exceeded 300s")
            failed.append(name)
            continue
        elapsed[name] = time.perf_counter() - t0
        print(f"--- {name}: {elapsed[name]:.1f}s ---", flush=True)
        if result.returncode != 0:
            failed.append(name)

    print("\n====================")
    slow = sorted(elapsed.items(), key=lambda kv: -kv[1])[:5]
    print("slowest: " + ", ".join(f"{n}={t:.1f}s" for n, t in slow))
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"ALL TESTS PASSED ({len(names)} tests, "
          f"{sum(elapsed.values()):.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
