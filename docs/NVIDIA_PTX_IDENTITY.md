# NVIDIA PTX fallback and IDENTICAL mode

The PTX path must preserve the same bitwise results as the supported native
NVIDIA, AMD, and Apple paths in IDENTICAL mode. Driver compilation is not an
exception to that contract. FAST mode has no cross-vendor bitwise requirement.

Native kernels remain the first choice. PTX is the fallback when no compatible
native kernel exists. FAST and DETERMINISTIC take it automatically. IDENTICAL
takes it only after qualification of the exact configuration, by the release
or by the user. A numerical mismatch is a defect to fix, not a reason to relax
the comparison or silently change the numeric mode.

## Current implementation

The experimental sm_80 build retains PTX rather than embedding native CUDA
machine code. It uses the existing IDENTICAL arithmetic and the existing pass
that pins rounding on floating-point instructions. The build audits rounding,
target, embedded code format, and payload hashes before packaging. Approximate
instructions are inventoried; passing this static audit alone does not establish
cross-vendor identity.

The forced investigation route requires both
`MOJOLEARN_CUDA_PATH=ptx-baseline` and `MOJOLEARN_EXPERIMENTAL_PTX=1`. Its build
manifest and experimental selection receipt retain `identical_qualified=false`.
A successful build or a local repeatability check must not change that status.

## What the loader does

Native kernels are the first choice on every NVIDIA device. They ship for
sm_89 and sm_90a. The PTX fallback is reached in exactly one case: the device
was detected, and no native set the install carries can run on it. A missing
or invalid payload, a native load failure, or a provenance failure is not that
case and still raises. The CPU is never substituted.

The payload is `cuda_ptx/sm_80` inside `mojolearn-nvidia`. Before any tier
uses it, the loader requires that the manifest hash equals the hash in the
NVIDIA vendor marker, that every binding file matches the manifest, and that
the manifest's source commit is the clean installed source. A release
admission named by the marker is part of that provenance: if it is missing or
altered, every tier refuses.

| Mode | PTX fallback |
| --- | --- |
| `fast` | Automatic. No admission is needed, because FAST makes no bitwise claim. |
| `deterministic` | Automatic. The tier promises the same bits on the same box and build and says nothing about a second box, so it makes no cross-device claim. |
| `identical` | Only on a release-admitted or locally qualified configuration. Otherwise it refuses and names the command to run. |

The selection receipt (`_backend.baseline_selection_receipt()`) records
`fallback="ptx"`, the numeric mode, `identical_qualified`, and `admission`,
which is `"bundled"`, `"local"` or `None`.

A per-call `numeric_mode="identical"` in a FAST or DETERMINISTIC process that
runs the PTX fallback follows the same rule. It is served when an admission
covers the configuration and refused, with the same command, when none does.
The identical-only bindings (the neural surface) are therefore unavailable on
an unqualified PTX configuration.

## Release admission

`PTX_IDENTITY_ADMISSION.json` is bound by hashes in the NVIDIA vendor marker
to the exact PTX manifest and installed source. It lists the device, compute
capability, driver text and CUDA API version of each measured configuration.
A release can only list the configurations that were measured before
publication. No production admission record is provided here.

The loader also accepts a vendor marker that binds the PTX manifest alone,
with no release admission. The packer does not write that form yet: today it
bundles PTX only together with an admission (`--bundle-ptx-admission`).

## Local qualification

Every other configuration can qualify itself:

```sh
python -m mojolearn verify --qualify-gpu
```

The command needs one visible CUDA device. It runs the identity suite the
wheel ships, at full depth, through the PTX payload on that device:
`verify --all`, then `verify --all --neural-training`. Together they select
every lane the harness defines, on every fixture, with every part. Each part
is compared with the reference table in the wheel
(`mojolearn/verify_reference/table.json`), which holds the results recorded on
NVIDIA, AMD and Apple hardware and the host column.

It writes a local admission only when all of the following hold in both runs:

- every lane in scope reads VERIFIED or NOT APPLICABLE (the two-device `par-*`
  drivers, which a one-device run cannot state; they stay named in the record);
- no part reads DIVERGENT, REFUSED or OWED;
- every fixture in the reference table ran, and no cell was interrupted;
- the comparator self-test passed, and the GPU/CPU inference comparison passed;
- every GPU binding the run loaded has a hash in the PTX manifest's identical
  tier.

This is stricter than `verify` itself. A lane that is OWED, HELD, NOT RUN or
UNDECLARED does not cost `verify` its pass, but it does deny qualification,
because it is a lane the wheel ships no usable reference for. The command
lists such lanes and parts as missing reference data. It does not admit on
partial coverage.

Current reference coverage, from a static read of the table in this source
tree (no device run): lane `gemm-int15` has no committed record, and nine
other lanes lack a usable reference for some fixture parts
(`gbdt-class-weights` and `gbdt-multiclass-offgrid` batch; `umap` and
`x-decomp-umap-options` on `ties`; single train parts of
`x-cluster-optics-metrics`, `x-neighbors-nearest-centroid`,
`x-prep-inverse-transforms`, `x-prep-score-edges` and `x-prep-select-kbest`).
A qualification run reports whichever of these the device produces a value
for as missing reference data and does not admit until a release record
carries them.

A failed qualification writes no admission. It reports the differing lanes,
the parts with no shipped reference and anything that did not run, and it
keeps the two verifier reports and logs as evidence. IDENTICAL mode stays
refused.

The local admission is a separate file with its own schema
(`mojolearn.ptx-local-identity-admission.v1`). It lives in
`$MOJOLEARN_PTX_ADMISSION_DIR`, or else in
`$XDG_STATE_HOME/mojolearn/ptx-admissions` (default
`~/.local/state/mojolearn/ptx-admissions`). It is bound to the installed
source commit, the PTX manifest hash, the device name, compute capability,
driver text and CUDA API version, and the hashes of the reference table and
identity harness, and it carries the comparison summary and the report
hashes. The loader accepts it for exactly that configuration and reports
`admission="local"`. A driver update, another device or a different wheel
does not match it, so IDENTICAL refuses again and names the command.

A local admission is the user's own evidence about one machine. It is not a
release claim, and it does not add a configuration to any release admission.

While the command runs, its verifier processes load the IDENTICAL PTX set
with `identical_qualified=False` and `qualifying=True` in the receipt. That
state exists so the comparison can run, and it is not an admission.

## Evidence required before admission

- Compare actual installed PTX execution with native NVIDIA execution on the
  supported GPU configurations, retaining loaded binary hashes, source commit,
  compiler, driver, and exact test inputs and outputs.
- Compare with supported AMD and Apple execution under the same identity
  protocols. NVIDIA-only agreement does not by itself establish vendor coverage.
- Account for every applicable lane, fixture, and output part. A comparison over
  three fixtures cannot be described as a nine-fixture comparison; structural
  exclusions and missing evidence must remain explicit.
- Investigate every mismatch, refusal, and incomplete run. Do not update golden
  values solely to make PTX pass, change tolerances, or count absent hashes as
  equal outputs.
- Bound admission by the configurations and arithmetic contract actually
  supported. PTX target compatibility alone does not qualify all NVIDIA GPUs
  or future driver versions.

`tools/admit_nvidia_ptx.py` generates the separate admission record only after
rechecking the actual payload and full NVIDIA nine-fixture comparisons, direct
canonical three-fixture Apple/AMD comparisons over every applicable lane, and
supplementary measured CUDA configuration witnesses. These two coverage scopes
remain distinct in the record. The tool retains comparison reports and their
hashes and writes the admission decision last. It does not rewrite input
artifacts, their source commits, or experimental manifests.

Two lanes (`gbdt-class-weights`, `gbdt-multiclass-offgrid`) have no batch
declaration in the harness, so their batch part reads `n/a:UNDECLARED` in every
column. The checker pins that (lane, part) list: each must read exactly that
value in every compared column, is reported as excluded in the comparison and
in the admission's NVIDIA coverage, and is never counted as an equal hash. Any
other undeclared part fails.

The fallback only triggers on a device with no native payload (an A100,
capability 8.0), where no native column can exist. `--gpu ampere` in
`tools/nvidia_baseline_gpu_batch.py` collects the forced-PTX nine-fixture
column, the extra capture and the configuration witness there. Such a receipt
is admitted only when its column equals the native reference columns of the
natively supported devices from the same source, cell for cell, and its three
shared fixtures equal the pinned Apple and AMD columns. It never counts toward
the two natively supported capabilities. The record keeps these as a separate
`native_absent` scope with its own comparison hashes.

`--fallback-stage` then tests the automatic route in the same rental
(`tools/nvidia_ptx_fallback_stage.py`): this machine generates the admission
and packs the vendor wheel, and the pod installs the core plus that wheel with
no forcing variable. The selection receipt must name the admitted fallback, the
canonical three-fixture column must equal the Apple reference, and an admission
without this device configuration and a vendor wheel without bundled PTX must
both refuse with `GpuPluginError`. It needs wheels whose core has the
admitted-fallback loader.

The packer's `--bundle-ptx-admission` option places the separately admitted PTX
inside `mojolearn-nvidia`; it requires matching source and manifest bytes and
cannot also emit a separate experimental owner for those paths. The requested
250 MiB project allowance must still be confirmed before publication.

Until a matching release admission is included or a local qualification
passes, IDENTICAL mode does not select PTX. An unsupported configuration
reports the limitation and the command, and keeps the requested numeric mode.
Forced-path numerical comparisons do not replace an end-to-end test of
automatic fallback on a device without compatible native code.
