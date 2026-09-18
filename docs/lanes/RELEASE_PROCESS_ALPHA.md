# Alpha release process: 0.8.7 review

The alpha publication gate should answer whether the exact wheel installs, exposes
its advertised API, and passes a small representative CPU/GPU smoke. Numerical
certification is a separate deliverable with its own supported-input claims.
Pending certification should remain visible without holding every alpha release.

## What consumed the time

- The final Apple wheel's 13 smoke stages took 31.37 seconds. Fresh native build,
  artifact checks and fits across five Python interpreters took 1,217 seconds.
- Fresh Linux builds took about 36 minutes on H100, 22 minutes on AMD and
  30 minutes on the successful L40S. A slower host hit the original 40-minute
  compile limit and its partial output was discarded. The
  serial compiler and variable rental CPU speed made the sm_89 leg the long pole.
- Each Linux architecture rebuilt the same 32 CPU bindings. Their completed
  NVIDIA/AMD byte hashes matched. Build a compatible shared CPU set once.
- A single five-second NVIDIA telemetry timeout aborted one compiler run.
  Bounded telemetry retry preserves resource checks without wasting a whole build.
- A 40-minute compiler deadline was too tight for heterogeneous rental hosts.
  Compilation and a light test need separate budgets.
- The Linux wrapper accepts a requested job count, then overwrites it with one.
  Remove that inconsistency with a small resource-cap test before the next build.

## Keep for every published alpha wheel

1. Pin source and dependencies; build only changed native inputs.
2. Check wheel metadata, native inventory, package integrity and dependency closure.
3. Install the exact final wheel and run one bounded smoke per wheel platform
   (Metal for macOS, one CUDA target for Linux), with API imports, a small CPU/GPU comparison and the negative control.
4. Publish the admitted wheel digest and verify PyPI serves the same bytes.

Publish ready platforms independently. This is already implemented for 0.8.7.
Use one representative interpreter for numerical smoke and a cheap import/ABI
matrix for other supported Python versions. Run full numerical campaigns on a
schedule or when the affected arithmetic changes; don't repeat them for a
publisher/documentation change. A changed wheel must still get a new smoke.

## Next implementation priorities

1. Make light the explicit default for alpha API releases; preserve an explicit
   full-certification profile. Keep unresolved numerical claims unresolved.
2. Fix compiler-job propagation and retry transient telemetry queries within
   the existing resource bounds. Fail persistent telemetry errors.
3. Cache native artifacts by their actual compile inputs, toolchain, flags and
   target, then relink/package and smoke the exact wheel. Python/verifier-only
   changes should not rebuild unchanged native libraries.
4. Build shared CPU bindings once, then assemble the architecture sets around
   those exact bytes. Preserve compatibility and provenance checks.
5. Use one command to freeze, build, pack, smoke, publish and write evidence;
   resume completed steps by digest rather than repeating them manually.

Tiny smoke fixtures are generated locally. Putting them in R2 would add little
value here. R2 is more useful for immutable native build outputs and wheel transfer
once caching is implemented and measured.

## Completed release

[0.8.7](https://github.com/mojolearn/mojolearn/releases/tag/v0.8.7) is published
for macOS and Linux. Both exact wheels passed all 13 light smoke stages; the
published PyPI hashes match those receipts. Independent platform publication
and the explicit light admission profile are implemented. Caching, shared CPU
builds, default selection and permanent scheduler/telemetry fixes above remain
follow-up work; this review does not claim they are already implemented.
