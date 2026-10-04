# NVIDIA PTX fallback and IDENTICAL mode

The PTX path must preserve the same bitwise results as the supported native
NVIDIA, AMD, and Apple paths in IDENTICAL mode. Driver compilation is not an
exception to that contract. FAST mode has no cross-vendor bitwise requirement.

Native kernels remain the first choice. PTX is intended as a fallback when no
compatible native kernel exists, after qualification for the supported execution
configurations. A numerical mismatch is a defect to fix, not a reason to relax
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

The production loader supports a separate native-first route. Only a detected
NVIDIA device without compatible native code may reach it. A missing or invalid
payload, a native load failure, or a provenance failure is not that condition.
This route requires `PTX_IDENTITY_ADMISSION.json`, bound by hashes in the NVIDIA
vendor marker to the exact PTX manifest, installed source, numeric tier, and
measured device/driver configuration. Unknown configurations refuse instead of
substituting CPU execution. No production admission record is provided here.

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

Until the required evidence passes and a matching record is included, IDENTICAL
mode must not automatically select PTX. An unsupported configuration reports the
limitation while preserving the requested numeric mode. Forced-path numerical
comparisons do not replace an end-to-end test of automatic fallback on a device
without compatible native code.
