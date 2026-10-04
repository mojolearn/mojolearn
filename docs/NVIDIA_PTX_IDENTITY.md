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

The loader currently requires both `MOJOLEARN_CUDA_PATH=ptx-baseline` and
`MOJOLEARN_EXPERIMENTAL_PTX=1`. This is an investigation route, not an admitted
IDENTICAL fallback. Its manifest and selection receipt explicitly retain
`identical_qualified=false`. A successful build or a local repeatability check
must not change that status.

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

Until runtime admission and its evidence are implemented, IDENTICAL mode must
not automatically select this experimental payload. An unsupported configuration
must report the limitation while preserving the requested numeric mode.
