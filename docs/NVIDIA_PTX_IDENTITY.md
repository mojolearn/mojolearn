# NVIDIA PTX: a normal target

Andrew 2026-10-10: PTX is a normal target; no flag.

## Distribution layout

The release packages native `sm_89` in `mojolearn-nvidia` and PTX `sm_80`
in `mojolearn-nvidia-ptx`. NVIDIA requires the PTX distribution at exactly
the same version, so `pip install mojolearn` installs both automatically.
Their files occupy separate directories at the existing loader paths.
Both package markers bind the PTX manifest's SHA256.

Releases through 0.8.52 shipped native `sm_89` only. The native and PTX
sets together packed at 134.1 MiB, over PyPI's 100 MiB file limit. Splitting
the files lets each distribution fit under that limit.

## Available build slots

The NVIDIA install supports two kinds of set, registered in
`python/mojolearn/gpu_plugins.py`:

| Slot | Directory | Code | Runs on |
| --- | --- | --- | --- |
| native `sm_89` | `cuda_native/sm_89` | cubins (IDENTICAL compiled with a pinned ptxas, contraction off) | Ada (L40S, L40, RTX 4090, RTX 6000 Ada) |
| PTX `sm_80` | `cuda_ptx/sm_80` | rounding-pinned PTX, compiled by the driver on the user's machine | every NVIDIA GPU of compute capability 8.0 or newer |

Other native architectures stay registered (`sm_90`/`sm_90a`, Blackwell
`sm_100`/`sm_100a`, `sm_103`/`sm_103a`, `sm_120`/`sm_120a`, `sm_121`/`sm_121a`):
a wheel that carries one of those sets loads it. The release ships sm_89
because of PyPI's 100 MiB file limit (`pack_wheel.py` checks every wheel at
pack time and fails with the per-set sizes).

## What the loader does

`_backend._vendor_base`: native first, then PTX.

1. `MOJOLEARN_GPU_ARCH` picks a directory by name, as for any architecture;
   `MOJOLEARN_GPU_ARCH=sm_80` selects the PTX set.
2. A native set the device can run (exact match, the `a` suffix, or the
   same-family rule) is loaded.
3. Otherwise, **if the PTX set is installed**, on any NVIDIA device of compute capability 8.0 or newer (A100
   sm_80, A10/A40 sm_86, H100/H200 sm_90, Blackwell sm_100/sm_120, ...), the
   PTX set is loaded, in every numeric mode, IDENTICAL included. A device older
   than 8.0 refuses. Without a compatible native set or an installed PTX set,
   the GPU is unsupported by that install. The CPU is never substituted.

There is no flag, no admission record and no local qualification step. The
PTX set is checked like a native set before any binding loads: the vendor
marker (`gpu_plugin.json`) binds the PTX manifest (`PTX_BASELINE.json`,
schema `mojolearn.ptx-set.v2`) by SHA256, every binding must match the
manifest, and the manifest's source commit must be the clean installed
source. A missing or altered PTX set refuses at import.
`_backend.baseline_selection_receipt()` records the selection and every PTX
binding the process loaded.

## Identity: the PTX column

The PTX set's identity is decided the way every native set's is, by the
reference table (`python/mojolearn/verify_reference/table.json`):

- **Recorded once** on any NVIDIA box, `REPEATS=1`, through the PTX set:

  ```sh
  # in a built checkout of the frozen commit, with the cuda-sm_80 leg's
  # sets/cuda/sm_80 copied to python/mojolearn/cuda/sm_80
  tools/record_identity_column.sh nvidia-ptx-<gpu>-sm80 /abs/outdir
  ```

  The label `nvidia-ptx-*` makes the record the `ptx` device class; the
  script sets `MOJOLEARN_GPU_ARCH=sm_80` and its preflight refuses unless the
  PTX set is what loaded. `tools/identity_break.py` refuses a PTX run without
  a PTX label and a PTX label without the PTX set.
- **Admitted** with the other columns by `tools/admit_identity_columns.sh`.
  The PTX column never decides a reference: the reference is NVIDIA == AMD.
  Every PTX digest must equal it. A differing part is listed in the table's
  `ptx_divergent` and fails the admission (`PTX rc=1`).
- **IDENTICAL on PTX is allowed** when the PTX digests equal the NVIDIA ==
  AMD digests. A mismatch is a bug to fix in the PTX codegen, never a reason
  to rerun and never a reason for a separate qualification step.

## Where PTX codegen can differ from the native cubins

The native IDENTICAL sets are compiled here with a pinned ptxas (12.5) and
`--fmad=false`; the PTX set is compiled by the user's driver. These are the
places the two can produce different bits, and what holds each one:

1. **Contraction (FMA).** The driver JIT compiles with fmad on, so a plain
   `mul.f32` feeding a plain `add.f32` may become one `fma.rn.f32`.
   `packaging/linux/ptx_contract.py` gives every plain float `mul`/`add`/`sub`
   in the IDENTICAL set its `.rn` spelling (ptxas never contracts `.rn`
   instructions), and `ptx_baseline.py` refuses an IDENTICAL module that still
   carries one. A divergence here means an op the pass missed (an `f64` form,
   a vector form, a new mnemonic) or an explicit `fma` in the source that the
   native path reads differently.
2. **Flush to zero (ftz).** `.ftz` variants flush subnormal inputs and results
   to zero. The two paths must agree on every `.ftz` spelling; a driver default
   or a `-ftz` JIT option that differs from the cubin build flushes subnormals
   on one side only. Look for `.ftz` in the PTX and the denormal mode of the
   cubin build.
3. **Approximate math.** `rcp.approx`, `sqrt.approx`, `rsqrt.approx`,
   `ex2.approx`, `lg2.approx`, `sin.approx`, `cos.approx`, `tanh.approx` and
   `div.approx`/`div.full` are implementation-defined: their results can change
   between the ptxas that built the cubins and the user's driver, and between
   driver versions. `ptx_baseline.py` inventories every approximate instruction
   per module (`approx` in the manifest). IDENTICAL kernels must use the
   IEEE-rounded forms (`div.rn`, `sqrt.rn`, `rcp.rn`) or the owned math
   library; an approximate instruction in an IDENTICAL module is the first
   suspect for a divergent part.
4. **Target differences.** The PTX targets sm_80; the cubins target sm_89.
   Code paths keyed on the architecture at compile time (warp-level or
   tensor-core instructions, `__nanosleep`, async copies) can take a different
   branch, and a different fold order changes bits. IDENTICAL kernels must not
   key their arithmetic on the architecture.

## Build, pack, release

- The release builds the PTX set as the leg `cuda-sm_80` on every build
  backend, through the same route as any arch: `tools/release061_remote_build.sh`
  gives cuda/sm_80 the `ptx` code format, so `packaging/linux/build_sets.sh`
  keeps the rounding-pinned PTX and writes `PTX_BASELINE.json` instead of
  converting the IDENTICAL modules to cubins. It needs no GPU.
- `cross-compile-check` compiles sm_80 like any arch.
- `pack_wheel.py --profile release-split` packs the PTX set into
  `mojolearn-nvidia-ptx` with its own build proof. The split audit checks
  disjoint file ownership and matching PTX manifest digests across packages.
  Publication uploads the Linux core first, PTX before NVIDIA, and the
  aggregate Linux `mojolearn` package after the vendor packages.

## Installed artifact qualification

The installed artifact path is
`tools/release_installed_checks.sh qualify-identity`, which dispatches to
`tools/release_identity_installed.sh`. It accepts `cuda sm_80` for PTX.
Use a held NVIDIA GPU with no compatible native set in the wheel, such as
an A40 (`sm_86`) when the wheel ships native `sm_89` only. Leave
`MOJOLEARN_GPU_ARCH` unset: this checks the user's automatic selection.
The qualifier removes inherited routing overrides itself.

Run on the Linux GPU box, using the exact retained wheels and an explicit
selection JSON containing a nonempty `lanes` list:

```sh
bash tools/release_installed_checks.sh qualify-identity \
  "$CORE_WHEEL" "$CORE_SHA256" cuda /root/results-ptx-qualification \
  "$WHEEL_SOURCE_COMMIT" sm_80 "$SELECTION_JSON" \
  "$NVIDIA_WHEEL" "$PTX_WHEEL"
```

The source commit is the one inside the wheel's `identity_columns/COMMIT`,
not the later commit of the qualification tools. No native rebuild is
needed to exercise retained wheel bytes.

The gate requires both the existing expanded CLI smoke and the selected
base-fixture identity checks. For PTX it additionally requires:

- A CUDA `sm_80` selection receipt identifying the PTX format, the manifest
  hash, the installed source commit and an actually loaded identical-mode
  binding.
- Numerical worker binding hashes belonging to that identical PTX payload.
  A native CUDA run, an import-only receipt, or a worker loading fast-mode
  bytes cannot establish PTX identity.
- The existing complete-cell and reference comparisons. Missing, refused,
  divergent or timed-out work remains a failure.

The selected set (`sm_80`) and physical device architecture (`sm_86` on an
A40) are recorded separately. `identity-receipt.json`, `identity.json`,
the smoke receipt, wheel hashes, selection hash, tool hashes and complete
logs stay together. The receipt states the exact selection, base fixture
and one repetition; it does not claim the full surface suite or a
performance measurement. Use a fresh output directory for each changed
artifact or selection and preserve previous failures.

The lower-level `tools/qualify_verifier_wheel.py --require-ptx` option
requires this route for its expanded smoke. It does not force PTX or admit
an identity column by itself.

## Paper evidence

A passing installed PTX result can support an additional tested hardware
and driver configuration in the MLSys paper. Retain the GPU model, driver,
exact wheel bytes and their source commit, PTX selection and loaded-binding
hashes, reference-table hash, selected methods, fixtures, parts and verdicts.
Keep these as a dated evidence record; later code changes do not update it.
Archive rented-box outputs on the Mac and in R2 before teardown.

The paper's evidence is pinned separately in
`mlsys/results/measurement-sources.json`; PDF builds do not import new
results. Import and review a completed PTX record explicitly, add its
scoped identity result to the verification discussion and supporting
artifact, then rebuild the named and anonymous PDFs. Do not alter the
fixed historical verification totals or infer all-device coverage from one
new card. A qualification run supplies no new timing comparison for the
measurement boards or cost tables.
