"""Run every test file, each in its own process; exit 1 if any test fails.

    python tests/run_all.py
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    failed = []
    t0 = time.time()
    for f in files:
        r = subprocess.run([sys.executable, f], capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        print(f"{os.path.basename(f):24s} {last}")
        if r.returncode:
            failed.append(f)
            print(r.stdout[-4000:], r.stderr[-4000:], sep="\n")
    print(f"\n{len(files) - len(failed)} of {len(files)} test files passed "
          f"({time.time() - t0:.0f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
