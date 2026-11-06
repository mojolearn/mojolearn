# NVIDIA PTX: a normal target

Andrew 2026-10-10: PTX is a normal target; no flag.

## Distribution layout

The release packages native `sm_89` and PTX `sm_80` in one wheel,
`mojolearn-nvidia` (Andrew 2026-11-05), at the existing loader paths
(`cuda_native/sm_89`, `cuda_portable/sm_80`). Its marker binds the PTX manifest's
SHA256.

Releases through 0.8.52 shipped native `sm_89` only. The native and PTX
sets together packed at 134.1 MiB, over PyPI's 100 MiB file limit, so
releases 0.8.53 to 0.8.65 shipped the PTX set in its own exactly pinned
`mojolearn-nvidia-ptx`. PyPI granted `mojolearn-nvidia` 250 MiB per file on
2026-11-04 (`packaging/pypi_limits.json`) and the two sets pack to about
63 MiB, so the split is undone.

Upgrading from a release with `mojolearn-nvidia-ptx`: the old package
installed the PTX set at `mojolearn/cuda_ptx/sm_80`; the combined wheel
installs it at `mojolearn/cuda_portable/sm_80`, a path no published
distribution ever owned (`gpu_plugins.PTX_DIRECTORY`, with the old one in
`LEGACY_PTX_DIRECTORIES`). pip leaves the orphaned `mojolearn-nvidia-ptx`
installed; the loader never reads its directory or its .dist-info, and
`pip uninstall mojolearn-nvidia-ptx` removes only its own old files.

## Available build slots

The NVIDIA install supports two kinds of set, registered in
`python/mojolearn/gpu_plugins.py`:

| Slot | Directory | Code | Runs on |
| --- | --- | --- | --- |
| native `sm_89` | `cuda_native/sm_89` | cubins (IDENTICAL compiled with a pinned ptxas, contraction off) | Ada (L40S, L40, RTX 4090, RTX 6000 Ada) |
| PTX `sm_80` | `cuda_portable/sm_80` | rounding-pinned PTX, compiled by the driver on the user's machine | every NVIDIA GPU of compute capability 8.0 or newer |

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
  `mojolearn-nvidia` beside the native set, with its own build proof. The
  split audit checks the set's bytes against the manifest the NVIDIA marker
  binds, and the pack's size check prints both sets against the 250 MiB limit.
  Publication uploads the Linux core first, then the vendor packages, and the
  aggregate Linux `mojolearn` package last.

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
  "$NVIDIA_WHEEL"
```

The source commit is the one inside the wheel's `identity_columns/COMMIT`,
not the later commit of the qualification tools. No native rebuild is
needed to exercise retained wheel bytes.

The shell qualifier uses the CLI's routine verification profile. Do not put
neural-training lanes into that selection: the CLI requires the explicit
`--neural-training` profile for them. Keep a separate profile receipt when
running those checks, as in the retained RTX 3090 record below.

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

## Retained RTX 3090 check, 2026-10-20

[The dated result](../bench/results/ptx_qualification/2026-10-20/summary.json)
checks the published 0.8.55 artifacts on an RTX 3090 (`sm_86`), driver
580.126.09. Automatic selection loaded the `sm_80` PTX payload. The expanded
installed smoke passed; the broader identity qualification did **not** pass.

The 393-lane routine profile reported 1,247 identical parts and 718 N/A
parts, including four undeclared batch probes. The 64-lane neural-training
profile reported 187 identical, 129 N/A and four divergent parts:
`sequence-adamax/train`, `sequence-nadam/train`, and
`sequence-lr-schedulers/{train,batch}`. The already completed `knn` smoke
supplied the remaining lane without repeating it. Both numerical profiles
loaded the pinned PTX binding bytes. The original profile-selection refusal
and all later failures remain in the retained evidence.

The missing batch declarations now have real row-wise prediction probes and
batch-only revisions in `tools/identity_break.py`. Their new cross-vendor
reference evidence is pending. The three sequence lanes need review against
the pending autograd changes; this older wheel does not test those changes.

The same machine passed the three small GEMM measurement-smoke recipes
(`gemm`, `gemm-bf16`, `gemm-int8`) with Torch opponents and worker provenance.
These check the timing pipeline, not full-dataset performance. No board
performance result or broad paper identity claim is admitted from this run.

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
