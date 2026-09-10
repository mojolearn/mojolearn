"""Time the host converters, DEVIATION 2470/2471/2472, against NumPy INSIDE ONE
THERMAL WINDOW, per docs/lanes/BRIEF_native_convert_2026-09-10.md.

Two arms per shape, interleaved, at least PAIRS rounds:
    numpy    -> np.ascontiguousarray / np.asfortranarray (x, dtype=float32)
    native   -> _buffer.as_f32_c / as_f32_colmajor

run1..run5 in this directory were taken with a third arm, `python`, the
pure-Python converter the binding replaced, forced by `_NATIVE[key]=None`.
That arm and the forcing mechanism were removed from the package on
2026-09-10 (a missing symbol is a stale build, not a fallback), so this
script now times two arms; the recorded runs keep their third column.

Reports the MINIMUM per arm (the least thermally damaged sample), the pair
count, and the spread between the first and last NumPy sample. A spread
above 20 percent voids the window; the script says so and exits 3. Cases
under a millisecond (the zero-copy borrow) are exempt: their spread is
timer noise, not heat.

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

PAIRS = 9
SHAPES = [(1_000_000, 10), (2_000_000, 20)]
KEYS = ("cast_f64_to_f32", "cast_colmajor_f64_to_f32", "transpose_f32")


def _ms(fn, *args):
    t0 = time.perf_counter()
    r = fn(*args)
    return (time.perf_counter() - t0) * 1e3, r


def _force_present():
    for k in KEYS:
        _buffer._native(k)  # raises by name if the binding predates it


def run_case(label, x, numpy_fn, ours_fn, ref_bytes):
    """Interleave the two arms PAIRS times; return per-arm minimums and
    the NumPy first/last spread. Every sample is checked against the
    NumPy bytes so a fast wrong answer cannot post a time."""
    samples = {"numpy": [], "native": []}
    # One untimed round first. Runs 1-4 voided windows on a COLD first NumPy
    # sample (3.84 ms then a flat 3.1; 82.8 then a flat 54) followed by no
    # upward trend: page-cache and allocator warm-up, not heat. The drift
    # rule is for heat, so the cold sample is spent here and not compared.
    numpy_fn(x)
    _force_present()
    ours_fn(x)
    for _ in range(PAIRS):
        t, r = _ms(numpy_fn, x)
        assert r.tobytes(order="F" if r.flags.f_contiguous and not r.flags.c_contiguous else "C") == ref_bytes
        samples["numpy"].append(t)
        t, (a, _) = _ms(ours_fn, x)
        assert a.tobytes() == ref_bytes, "native arm bytes differ from NumPy"
        samples["native"].append(t)
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
        # float32 transpose only, both directions (DEVIATION 2472)
        x32c = np.ascontiguousarray(x64, dtype=np.float32)
        rows.append(run_case(
            f"{shape[0]:,} x {shape[1]} f32 C -> f32 F (as_f32_colmajor)",
            x32c,
            lambda x: np.asfortranarray(x),
            lambda x: _buffer.as_f32_colmajor(x, name="X"),
            ref_f,
        ))
        x32f = np.asfortranarray(x64, dtype=np.float32)
        rows.append(run_case(
            f"{shape[0]:,} x {shape[1]} f32 F -> f32 C (as_f32_c)",
            x32f,
            lambda x: np.ascontiguousarray(x),
            lambda x: _buffer.as_f32_c(x, name="X"),
            ref_c,
        ))
        # the zero-copy borrow: float32 F -> float32 F
        rows.append(run_case(
            f"{shape[0]:,} x {shape[1]} f32 F -> f32 F (borrow)",
            x32f,
            lambda x: np.asfortranarray(x, dtype=np.float32),
            lambda x: _buffer.as_f32_colmajor(x, name="X"),
            ref_f,
        ))

    print(f"pairs per case: {PAIRS}; minimum of each arm reported (ms)")
    print(f"{'case':<52} {'numpy':>9} {'native':>9} {'np spread':>10}")
    for label, mins, spread, _ in rows:
        # the drift rule is about heat; a case that finishes in under a
        # millisecond (the zero-copy borrow) reports timer noise, not heat
        drifted = spread > 0.20 and mins["numpy"] >= 1.0
        flag = "  VOID" if drifted else ("  (sub-ms, drift rule n/a)" if spread > 0.20 else "")
        void |= drifted
        print(f"{label:<52} {mins['numpy']:>9.3f} {mins['native']:>9.3f} "
              f"{spread*100:>9.1f}%{flag}")
    print("\nall samples (ms), in run order, for every case:")
    for label, _, _, samples in rows:
        print(f"  {label}")
        for arm in ("numpy", "native"):
            print(f"    {arm:<7} " + " ".join(f"{t:8.3f}" for t in samples[arm]))
    if void:
        print("\nAT LEAST ONE WINDOW VOID: NumPy drifted more than 20% first to "
              "last; the box was heating. Take it again.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
