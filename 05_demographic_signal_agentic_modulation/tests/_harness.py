"""Shared test helpers. The tests run with or without pytest:

    python tests/run_all.py          # every file, each in its own process
    python tests/test_filters.py     # one file
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def run_module_tests(namespace):
    names = sorted(n for n, f in namespace.items() if n.startswith("test_") and callable(f))
    failed = []
    for n in names:
        try:
            namespace[n]()
            print(f"  ok   {n}")
        except Exception:                                    # noqa: BLE001
            failed.append(n)
            print(f"  FAIL {n}")
            traceback.print_exc()
    print(f"{len(names) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


def raises(exc, fn, *a, **kw):
    """True if fn(*a, **kw) raises exc."""
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False
