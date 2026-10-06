"""Test-only capture support. Product arithmetic remains in compiled Mojo.

Fixtures may use NumPy for input construction and independent quality oracles;
none of these functions is imported by mojolearn's public runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))


def apple_fast(load_product=True):
    if not __debug__:raise RuntimeError('quality qualification requires Python assertions enabled')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'fast' or os.environ.get('MOJOLEARN_VENDOR') != 'apple':
        raise RuntimeError('requires explicit Apple FAST mode')
    chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if 'Apple M3 Ultra' not in chip:
        raise RuntimeError('qualification is restricted to the existing M3 Ultra queue')
    if load_product:
        import mojolearn
        return mojolearn
    return None


def binding_check(binding, family):
    mode = getattr(binding, family + '_numeric_mode')()
    vendor = getattr(binding, family + '_vendor')()
    if int(mode) != 0 or str(vendor) != 'metal':
        raise RuntimeError('hidden non-FAST/non-Apple binding: ' + family)
    path = Path(binding.__file__).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mode='fast', vendor='metal')


def consumed(call):
    """Time actual public work through first read of every requested output."""
    import numpy as np
    start = time.perf_counter_ns()
    result = call()
    values = result if isinstance(result, tuple) else (result,)
    for value in values:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise AssertionError('nonfinite public output')
        array.tobytes()  # includes mandatory first read, never just launch timing
    return result, (time.perf_counter_ns() - start) / 1e6


def capture_main(exercise):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--variant', default='default')
    args = parser.parse_args()
    apple_fast()
    packet = exercise(args)
    packet.update(schema=1, arm=args.arm, variant=args.variant,
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  mode='fast', vendor='apple', timing_contract='public caller through first read',
                  digest_equality_required=False, promotion_authorized=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(packet, stream, indent=2, allow_nan=False)
    print('APPLE_FAST_CAPTURE status=PASS cases=' + str(len(packet['cases'])) + ' output=' + str(args.output))
