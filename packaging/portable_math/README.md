# Owned host math and wheel guard

Product Python modules use `mojolearn._portable_math`. The bundled runtime's
standard math imports are renamed to `mojolearn_*` and resolved by a small
repository-owned shared library. Neither helper imports any external symbols.
The final wheel audit rejects bundled libm/libmvec files, direct libm dependencies,
standard C math imports, and Python `math`/`cmath` imports. It also checks that
the owned helper is present and exports its required ABI.

This describes the wheel payload, not the Python process or operating system.
Apple's required `libSystem` ABI dependency remains. Python, GPU drivers, and
optional third-party packages may themselves depend on system math. Independent
numerical test oracles outside the wheel retain their platform math calls.
Mojo device intrinsics and FAST device arithmetic are unchanged; all native
wheel files are checked for external host math imports.

## Arithmetic

`portable_math.c` translates the existing `checks/numerics.mojo` FP64 log,
log2, and exp polynomials, with explicit FMA and contraction disabled elsewhere.
CPU sqrt is hardware rounded. log10 scales the pinned log and preserves exact
normal decimal-power results for runtime formatting; log2f rounds the
pinned double log2 to float. Binary decomposition, scaling, splitting, and integer
rounding use owned bit operations. The Python wrapper implements classification,
sign, integer rounding, products, exact summation, and binary scaling without math.
The schedule's variable floating power of two is replaced by exact binary scaling.

These are portable approximations, not a correctly-rounded transcendental API.
The existing exp primitive flushes outputs below the smallest normal to zero.
`fsum` rounds an exact finite sum once; unlike CPython's partial-sum algorithm,
it permits cancellation of a temporarily overflowing intermediate sum. Product
callers use bounded finite terms. The internal log wrapper accepts float-sized
inputs, not arbitrary-sized Python integers. A few former platform log/exp results
can move in their last bit; prior evidence is not relabeled as proof of these new
host paths.

## Build and check

LIEF 1.0.0 is a **build tool only**, pinned in the pkg environment and PEP 517
build requirements. It is not a runtime dependency and is not bundled. Linux
build_sets stages the helper on its native host before calculating the manifest.
The Linux packer can therefore run on macOS without a cross compiler. For older
cached sets, supply `--portable-math-helper /path/to/Linux-built/libMojolearnMath.so`.
Linux compilation uses the existing x86-64-v3 baseline; macOS uses arm64/M1 and
macOS 11. No GPU compilation is needed for this helper.

Both direct setuptools wheel builds and the Linux packer finalize and audit the
artifact. Build from the repository checkout so `python/setup.py` can access the
build-only `packaging/portable_math` tools. Runtime files are rewritten in a
temporary copy, wheel RECORD hashes are regenerated, and failed build artifacts
are removed. Publishing workflows audit the final immutable bytes again before
upload. This is a short static check, not a new full numerical release gate.

```sh
pixi run -e pkg python packaging/portable_math/wheel.py --audit-only path/to/*.whl
# With the private candidate installed in a test environment:
python -m pytest -q packaging/portable_math/test_host_math.py
# Build-only guard controls, including a real foreign sin import:
python -m pytest -q packaging/portable_math/test_wheel.py
```

Validation for this change is retained under
`bench/results/portable_math/2026-09-19/`: installed macOS/ARM64 and Linux/x86-64
(initial emulation, then physical AMD host) tests, identical arithmetic digests, exact comparison to the
existing compiled Mojo GP primitives, full wheel import/dependency inventories,
a direct PEP 517 wheel build, and small real Apple/AMD GPU GP/forest smokes with matching output hashes.
No full matrix recertification or new NVIDIA GPU smoke is claimed for the patched runtimes.
Published 0.8.8 remains unchanged; these are private candidates for a later release.
