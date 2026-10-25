# Retained device artifacts and an independently built host

`tools/retained_release_composition.py` assembles an unchanged Linux merged
device set with a separately built common CPU host. A host-only source change
does not invalidate device compilation. The earlier whole-leg receipt could
not express this because its complete source inventory included that host.

The new `mojolearn.linux.retained-composition.v1` receipt is an assembly receipt,
with action `assemble-retained-native`; it has no `build_exit`. Each device and
host artifact retains its original source revision, original proof hash, and
byte hash. Original build proofs, before-build captures, source inventories,
cache keys, manifests, merged indexes, and readbacks are embedded unchanged.
The package source revision is separate. Neither old proof bytes nor old
artifact bytes are relabeled as a new compilation.

Admission requires exact agreement for the relevant binding source closures,
builders, compiler, target, flags, runtime, and adapter inputs. It reconstructs
both original committed source contexts and the clean target source context.
The original builds must already satisfy the existing complete-build checks.
Device, selected host, index, runtime, and adapter files must match their
original producer's hashes. Runtime libraries and compiler/package environment
must agree between the two producers. Unknown source closures are refused.

For the PTX leg, the original arithmetic manifest is retained as a raw witness.
The current core-source association is explicit; its target, arithmetic
metadata, device bytes, and merged-index bytes remain unchanged and pass the
existing PTX manifest validator.

## Invocation after the host build

Run this for each retained device leg, using the same independently built host:

```sh
python3 tools/retained_release_composition.py \
  --device-build /retained/device-leg/release-build/build \
  --host-build /new/host-containing-leg/release-build/build \
  --source-root /clean/frozen/release-checkout \
  --compiler-sha256 ORIGINAL_AND_CURRENT_COMPILER_SHA256 \
  --out /new/assembled-leg
```

The destination must not exist. It receives `set/` and `assembly-proof.json`.
Pass that set and its proof to the existing Linux packer in the same places as
an original staged set and build proof. The packer reports
`retained-composition` origins and `retained_composed` counts; assembling these
files does not increment its new-compilation count. The stage manifest remains
the original witness, while the assembly proof owns the replacement host.

This tool does not use an arbitrary standalone `.so` as a host witness. The
host must come from an independently proven complete build with retained
before-build inputs and actual host compiler/cache provenance. It does not
qualify runtime behavior, cross-vendor identity, or installed-wheel adapters.
Those checks remain separately required where their inputs changed or valid
evidence is missing.

## Retained assessment and validation

The audit from `c00c4d800` to `b8c91475c` found 76 unchanged and 3 changed
logical identities in each of the three Linux release legs. All 33 device
identities per leg were unchanged. The changed identities were tokenizer,
classical runtime, and model-selection runtime host bindings, contained in one
physical 46-member merged host artifact. Adapter transitive inputs and runtime
inputs also matched. Pending host changes require a fresh comparison at the
final source freeze; a new commit alone does not invalidate evidence.

Source tests cover changed inputs, compiler, flags and target, mutated bytes,
original-source reconciliation, PTX metadata preservation, and packer origins.
Read-only checks against all three retained original legs validated their
actual artifact bytes and reproduced their original source captures. These
checks are not new runtime results. Assembly using the final new host and the
final source is performed by the release owner after that build exists.

Evidence is retained under
`~/mojolearn-evidence/release-artifact-reuse-20261025/`.
