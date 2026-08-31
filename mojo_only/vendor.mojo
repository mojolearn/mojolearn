# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which accelerator API this binary was COMPILED for, as a constant.

THE QUESTION THIS ANSWERS, AND WHO ASKS IT. As of the Linux wheel
(docs/LINUX_WHEEL.md) one `pip install mojolearn` on Linux carries SIX
binary sets, three numeric tiers for CUDA and three for HIP, and
`python/mojolearn/_backend.py` picks a vendor directory at import time. A
directory name is a label. The failure the tier read-back
(`gbdt_numeric_mode`, `linalg_numeric_mode`, `svm_numeric_mode`) exists to
catch, a binary sitting under the wrong label, has an exact vendor-shaped
twin: a CUDA `.so` filed under `hip/` imports cleanly on an NVIDIA box and a
HIP `.so` filed under `cuda/` imports cleanly on an AMD box, because a
Python extension does not touch the device until the first fit. So the
selector reads the vendor BACK OUT OF EVERY BINARY it loads and refuses at
import when the answer disagrees with the directory, exactly as it refuses a
tier mismatch. Every binding exports `<prefix>_vendor()` returning
`COMPILED_VENDOR`.

WHY COMPILE TIME AND NOT A RUNTIME QUERY. A `DeviceContext` at import would
(a) open the device inside `PyInit_*`, which the ten bindings deliberately
never do, and (b) report the device the PROCESS can see, which is the
environment's answer, not the binary's. The claim being checked is about
the binary. `has_nvidia_gpu_accelerator()`, `has_amd_gpu_accelerator()` and
`has_apple_gpu_accelerator()` in `std.sys.info` are resolved by the
compiler against the accelerator TARGET of the build, the same predicates
`mojo_only/kernel_matrix.mojo` uses to pick `DETECTED_COLUMN`, and the
value is folded into the binary as a constant. The Python side can then
compare that constant against the directory it loaded from and against what
the box appears to have, in that order of trust.

THE NAMES ARE THE API, NOT THE MARKETING. `metal`, `cuda`, `hip`. Not
`apple`, `nvidia`, `amd`, which are vendors and which the kernel matrix
already uses for its COLUMNS. A column is a scheduling decision (CDNA vs
RDNA is a column split under one API); the wheel layout and the selector
are keyed by the API a set was compiled against, and there is exactly one
directory per API. `none` is what a build with no accelerator target
reports, and the selector refuses it by name rather than guessing.

UNVERIFIED AT THE TIME OF WRITING (2026-08-29, written under the no-run
order): that these three predicates fold to the expected value inside a
`--emit shared-lib` build on a Linux box. They are the predicates
`kernel_matrix.mojo` has used in every E1/E2 leg to pick the column, and
the column has come out right on H100 and MI325X, so the expectation is
not a guess, but this file has not been compiled. The first Linux build leg
that runs `packaging/linux/build_sets.sh` is the measurement, and its
manifest records what every binary answered.
"""

from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

#: `metal`, `cuda`, `hip` or `none`. Resolved by the compiler, folded into
#: the binary, read back by `python/mojolearn/_backend.py`.
comptime COMPILED_VENDOR: StaticString = (
    "hip" if has_amd_gpu_accelerator() else
    "cuda" if has_nvidia_gpu_accelerator() else
    "metal" if has_apple_gpu_accelerator() else
    "none"
)
