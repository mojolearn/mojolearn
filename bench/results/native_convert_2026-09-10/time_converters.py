"""Time the host converters, DEVIATION 2470/2471, against NumPy INSIDE ONE
THERMAL WINDOW, per docs/lanes/BRIEF_native_convert_2026-09-10.md.

Three arms per shape, interleaved, at least PAIRS rounds:
    numpy    -> np.ascontiguousarray / np.asfortranarray (x, dtype=float32)
    native   -> _buffer.as_f32_c / as_f32_colmajor with the binding present
    python   -> the same, with the binding forced absent (_NATIVE[key]=None)

Reports the MINIMUM per arm (the least thermally damaged sample), the pair
count, and the spread between the first and last NumPy sample. A spread
above 20 percent voids the window; the script says so and exits 3.

Run from the repo root with the package on the path and the base binding
built:

    PYTHONPATH=python nice -n 19 python bench/results/native_convert_2026-09-10/time_converters.py

NumPy is the opponent here, so it is imported; the module under test does
not import it. The float32 F-order case is included to confirm the
zero-copy borrow still costs nothing on every arm.
"""
import sys
import time

import numpy as np

from mojolearn import _buffer

PAIRS = 7
SHAPES = [(1_000_000, 10), (2_000_000, 20)]
KEYS = ("cast_f64_to_f32", "cast_colmajor_f64_to_f32")


def _ms(fn, *args):
    t0 = time.perf_counter()
    r = fn(*args)
    return (time.perf_counter() - t0) * 1e3, r


def _force_absent():
    for k in KEYS:
        _buffer._NATIVE[k] = None


def _force_present():
    for k in KEYS:
        _buffer._NATIVE.pop(k, None)
    if _buffer._native(KEYS[0]) is None:
        sys.exit("base binding not built; nothing to time")


def run_case(label, x, numpy_fn, ours_fn, ref_bytes):
    """Interleave the three arms PAIRS times; return per-arm minimums and
    the NumPy first/last spread. Every sample is checked against the
    NumPy bytes so a fast wrong answer cannot post a time."""
    samples = {"numpy": [], "native": [], "python": []}
    for _ in range(PAIRS):
        t, r = _ms(numpy_fn, x)
        assert r.tobytes(order="F" if r.flags.f_contiguous and not r.flags.c_contiguous else "C") == ref_bytes
        samples["numpy"].append(t)
        _force_present()
        t, (a, _) = _ms(ours_fn, x)
        assert a.tobytes() == ref_bytes, "native arm bytes differ from NumPy"
        samples["native"].append(t)
        _force_absent()
        t, (a, _) = _ms(ours_fn, x)
        assert a.tobytes() == ref_bytes, "python arm bytes differ from NumPy"
        samples["python"].append(t)
    _force_present()
    first, last = samples["numpy"][0], samples["numpy"][-1]
    spread = abs(last - first) / min(first, last)
    mins = {k: min(v) for k, v in samples.items()}
    return label, mins, spread, samples


def main():
    _force_present()
    rng = np.random.default_rng(2470)
    rows = []
    void = False
    for shape in SHAPES:
        x64 = np.ascontiguousarray(rng.standard_normal(shape))
        # flat cast: float64 C -> float32 C
        ref_c = np.ascontiguousarray(x64, dtype=np.float32).tobytes()
        rows.append(run_case(
            f"{shape[0]:,} x {shape[1]} f64 C -> f32 C (as_f32_c)",
            x64,
            lambda x: np.ascontiguousarray(x, dtype=np.float32),
            lambda x: _buffer.as_f32_c(x, name="X"),
            ref_c,
        ))
        # fused cast+transpose: float64 C -> float32 F
        ref_f = np.asfortranarray(x64, dtype=np.float32).tobytes(order="F")
        rows.append(run_case(
            f"{shape[0]:,} x {shape[1]} f64 C -> f32 F (as_f32_colmajor)",
            x64,
            lambda x: np.asfortranarray(x, dtype=np.float32),
            lambda x: _buffer.as_f32_colmajor(x, name="X"),
            ref_f,
        ))
        # the zero-copy borrow: float32 F -> float32 F
        x32f = np.asfortranarray(x64, dtype=np.float32)
        rows.append(run_case(
            f"{shape[0]:,} x {shape[1]} f32 F -> f32 F (borrow)",
            x32f,
            lambda x: np.asfortranarray(x, dtype=np.float32),
            lambda x: _buffer.as_f32_colmajor(x, name="X"),
            ref_f,
        ))

    print(f"pairs per case: {PAIRS}; minimum of each arm reported (ms)")
    print(f"{'case':<52} {'numpy':>9} {'native':>9} {'python':>10} {'np spread':>10}")
    for label, mins, spread, _ in rows:
        flag = "  VOID" if spread > 0.20 else ""
        void |= spread > 0.20
        print(f"{label:<52} {mins['numpy']:>9.3f} {mins['native']:>9.3f} "
              f"{mins['python']:>10.3f} {spread*100:>9.1f}%{flag}")
    if void:
        print("\nAT LEAST ONE WINDOW VOID: NumPy drifted more than 20% first to "
              "last; the box was heating. Take it again.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
