# Retained Linux group admission

`tools/release_linux_group_composition.py` admits the existing `hip/gfx942`
merged GPU layout together with individually compiled development host groups.
It does not invoke a compiler, fabricate a successful whole-set build, or
qualify numerical results. Original execution receipts and partial-set results
remain unchanged.

The producer inputs JSON names the frozen `source_commit`, `compiler_sha256`,
`leg` (`hip/gfx942`), and the accepted flat package `site`. Its `hosts` list
contains one `{binding, result, receipt}` record per registered host binding;
`devices` contains one `{stem, members, archive, report}` record per registered
merged GPU group. Paths point to retained original files, including their
adjacent execution receipts, compiler logs and raw binaries.

Additional inputs are `adapter_directory`, `dependency_manifest`, `stage_exit`,
`stage_execution`, `stage_log`, and `stage_controller_log`. Adapter inputs must
include the original successful controller result, consumed source closure,
actual commands, pinned Torch readback, raw outputs and staged manifest.
Runtime admission retains the actual portable-math source/constants, compiler
flags and output hash; it does not substitute another compiler's helper bytes.

`readback_directory` is required for complete release admission. It contains
the actual standalone package readbacks (`mojolearn.amd.package-readback.v1`),
each child execution receipt, original JSON output and log, and the standard
`readback.txt`/`arch_readback.txt` files. Binary hashes, source, coverage, native
vendor/host mode and ISA must agree with the composed set. Missing readbacks
produce a useful compilation projection with `complete: false`; they cannot
pass `complete_native_proof` or the release packer.

```sh
python tools/release_linux_group_composition.py \
  --inputs retained-inputs.json --source-root /path/to/frozen-checkout \
  --patchelf /path/to/patchelf --output group-composition.json
```

Validation checks the original development-builder execution and canonical
source entries, locked compiler dependencies, numerical flags, GHA archive keys
and actual compilation logs, complete current per-binding identities, adapter
source/ABI receipts, and exact asset coverage. It replays the existing RPATH
staging operation on copies and requires byte equality with the accepted stage.
The proof records all consulted file hashes and sizes. Retain those actual
files with the proof on the Mac and R2; hashes alone are insufficient. A proof
cannot choose an executable for validation: the validator uses the installed
`patchelf` from its trusted execution environment.

Pass the resulting complete document through the existing release packer's
`--build-proof` argument, with the standard architecture set layout and actual
readback files. The inventory labels its origins `compiled-group-composition`,
retaining individual group keys, raw hashes and original producer references.
It explicitly records `host_binding_layout: per-binding`: the common host can
use its registered individual groups while the GPU uses its registered merged
groups. This shape is enabled only for explicit group composition; ordinary
mixed-layout and incomplete-host refusals remain in place.
The ordinary source, loader, dependency, adapter and release qualification gates
continue to apply. Qualification wheels remain non-publishable until those
independent gates pass.

This route currently covers the retained AMD native producer described above.
It does not treat native NVIDIA execution as PTX execution, infer physical
multi-GPU coverage, or waive cross-vendor identity requirements.

CUDA development groups can use the same composition admission through
`release_linux_cuda_groups.py`: 33 individual images, without a fabricated
merged index. The original compiler command, source closure, flags and every
raw image remain required. PTX staging replays the standard rounding contract.
Native CUDA staging additionally requires the pinned CUDA 12.5.82 tools archive
and successful per-module assembler receipts; raw development PTX is not a
release-native image.

`cubin_contract.py patch --retention-dir DIR` retains exact PTX, cubin, fatbin,
full assembler output, tool identities and commands. An unchanged entry can be
reused; changed inputs, tools, flags or retained bytes are refused. This leaves
the existing assembler flags and placement algorithm unchanged. Admission can
replay placement from those retained outputs and independently disassemble all
moved references, avoiding another assembler run. Actual installed loading,
runtime and identity remain separate requirements.

## Current AMD member overrides

`device_overrides` may list selected gfx942 development groups using the same
`binding`, `result`, and `receipt` fields as individual CUDA groups. Every listed
group must have a successful retained optimized execution, current source
closure, exact locked compiler/dependencies, and canonical raw-to-stage ELF
replay. A name alone never exempts a changed source closure.

Keep the original four merged images and their original compile evidence.
Validate every member not replaced against the current source. Record replaced
exports as inactive in those origins, and record the independently compiled
current images as their active owners. The effective merged index removes each
replaced member from its old group and adds `<member>.so: [member]`; the existing
loader already prefers that own image. This also prevents falling back to the
older merged export if the current image is missing. Reject unknown or duplicate
members, stale untouched closures, missing current images, and fully unused old
merged groups.

The strict split packer parses this effective plan only when an explicit AMD
group-composition proof names the overrides. Full producer, source, binary and
readback rederivation still occurs before packing. Generic packing and arbitrary
own files beside an unmodified merged index remain refused. Actual metadata
readbacks must name the current image for each replaced binding; installed
runtime and cross-vendor identity remain separate requirements.
