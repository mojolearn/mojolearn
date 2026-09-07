# Mamba public alpha surface and evidence boundaries

Source audit only; no tests, builds, model execution, measurements or publishing were performed by the audit agent. Root alone validates/publishes artifacts with at most three CPU cores under the current session limit.

All implemented **public** block entrypoints are already exported. `mojolearn` and `mojolearn.mamba` expose `Mamba1Block`, `Mamba2Block`, `Mamba3Block` and their corresponding State classes. Native `_mojolearn_mamba` registers all three forward, decode-step and backward functions. No new alias is required to make those methods visible in an alpha wheel.

| Operation | Available public behavior | Modes |
|---|---|---|
| `forward(x)` | Zero-state prefill, FP32 `[B,L,d_model]` | FAST, DETERMINISTIC, IDENTICAL |
| `forward(x, state)` | Prefill/continuation with explicit caller-owned state | Same forward modes |
| `step(x, state)` | One-token decode, state updated in place | Same forward modes |
| `backward(x, dy)` | Recomputed zero-state prefill VJP; `dy` matches `x`; named FP32 gradients | IDENTICAL only |

Mamba1 returns input plus ten weight gradients; Mamba2/3 each return input plus nine weight gradients. Backward requires positive B/L and exact FP32 inputs; unsupported dtype/mode/state arguments are refused. Mamba2/3 model width must be a positive multiple of 32; Mamba1 admits positive widths subject to its fixed profile. Accepted dimensions are not a claim that every dimension has been qualified. Mamba1 fixes state width16/conv4/expand2; Mamba2 fixes state128/head64/groups1/chunk256/expand2; Mamba3 fixes state128/head64/groups1/chunk64/expand2 with32 rotary angles. Constructor docstrings list each exact parameter shape and Mamba2's runtime dt limits.

Not exposed: FAST or DETERMINISTIC backward; gradients through a carried cache or decode sequence; gradients with respect to final recurrent-state outputs; generic PyTorch autograd integration; reduced-precision training; or a full Mamba language-model training API. Historical native diagnostic cases include initial-state gradient work, but the new public synchronous backward adapters are explicitly zero-state-only. Exposing those additional native state-gradient paths needs a separate state/cotangent ABI and qualification, not a Python export alias.

## Packaging audit

`python/mojolearn/_backend.py` registers `_mojolearn_mamba` and `build_mamba.sh`. Linux `packaging/linux/build_sets.sh` and macOS `packaging/macos/build_release_wheel.sh` already include that build script and extension. `python/pyproject.toml` includes Python package modules and native library globs for root, IDENTICAL/DETERMINISTIC, CUDA/sm targets, and HIP/gfx targets. Those declarations package the binaries actually built; they do not manufacture missing vendor/mode binaries or add backward symbols to older extensions. This audit did not build, inspect a new wheel, query PyPI, upload, or claim installation success.

Mamba1's missing-backward-symbol failure now matches the actionable Mamba2/3 behavior: it resolves the entrypoint before allocating output arrays and asks for a current matching alpha wheel or IDENTICAL rebuild. Native arithmetic and method signatures are unchanged.

Root release checks must establish that the **exact candidate wheel**, installed outside the source checkout, contains these six imports and all native entrypoints for each advertised vendor/mode. Forward and backward gates must execute against that installed binary; source tests or old native certificates cannot substitute for wheel evidence. Python `torch` is used by external qualification tooling, not required by the public native methods themselves.

## Retained evidence actually inspected

- Historical IDENTICAL native backward: `bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json` records the three-vendor five-case/54-gradient comparison. The later NVIDIA/AMD same-source comparison is retained in `bench/results/e1/2026-09-05_111524-mojolearn-e2-amd/comparisons.json`. These are native diagnostic certificates, not blanket Python/wheel certificates.
- September 6 NVIDIA source-run forward/API: `bench/results/resume/2026-09-06-root-feature-nvidia/run3/remote/feature-finish/mamba-api-identical.log` and `mamba-api-fast.log` each report **102 checks, zero failures**, with successful retained guard endings. FAST tolerance checks are distinct from IDENTICAL bit assertions. These runs supersede older documentation saying durable NVIDIA validation of the allocation fix is still pending.
- Same retained run's `mamba23-backward.log` reports **five tests passing**, covering host ABI/refusals plus the opted-in Mamba2/3 public FP64 VJP and requested native-dump comparisons at the existing tiny fixture. It does not cover all shapes, all modes, or current installed AMD/Metal wheels.
- New shared B2/L8/D32 harness is `tools/mamba23_shared_shape_backward.py`. Its authored source alone does not prove it ran. Root must use actual retained logs/artifacts for any newer result; this audit does not promote an active or uncollected job into a pass.

The alpha release may expose an implemented experimental method while accurately listing unqualified columns. It must not advertise universal three-vendor bitwise correctness, stateful training, FAST backward, or a newly validated PyPI installation based solely on these declarations.
