# `python -m mojolearn conformance`

`docs/VERIFY.md` describes `verify`, which checks **this build** against a
reference card. This document describes the next step out. A **conformance
bundle** is a self-describing directory that freezes the pinned fixture,
meaning its input bytes, its expected stage bytes, its identity-trace card,
and a manifest that hashes all of it, so that an **external implementation**
can check itself against mojolearn's IDENTICAL-mode computation **without
running Mojo or Python at all**. An external implementation here means
anything that consumes the bundle without running our code, such as a
simulator, an accelerator bring-up, another library, or a from-scratch
reimplementation.

```
python -m mojolearn conformance export  BUNDLE_DIR      produce a bundle
python -m mojolearn conformance validate BUNDLE_DIR     structural check
python -m mojolearn conformance validate BUNDLE_DIR --report REPORT.json
                                                        grade an external result
python -m mojolearn conformance diff BUNDLE_DIR --report REPORT.json
                                                        name the FIRST diverging stage
python -m mojolearn conformance diff BUNDLE_DIR --self  the corruption witness
```

Exit codes, for continuous integration, aligned with
`tools/identity_trace_diff.py`.

| exit | meaning |
|---|---|
| 0 | PASS. Bundle structurally valid, report conformant, or traces identical. |
| 1 | FAIL, always **named**. The file with the wrong hash, the missing stage, the first diverging stage. |
| 2 | COULD NOT JUDGE. Usage, unparseable input, a refused report, no comparator. Never a verdict. |

---

## The two hashes, and why there are two (DEVIATION 928)

| hash | role | where |
|---|---|---|
| **SHA-256** | the **audit-integrity layer**. "Is this the artifact that was exported, and did anything get corrupted or swapped." | computed host-side (`hashlib`) over every file in the bundle and recorded in `manifest.json`; computed by the external side over its raw stage bytes |
| **FNV-1a64** | the **in-kernel localization checksum**. It is what `core/identity_trace.mojo` can afford to compute on-device at every checkpoint, it is what the cards carry, and it is what the differ aligns and walks to name the FIRST diverging stage. | inside `.card` files, per stage, and per stage in the implementation report |

FNV-1a64 is **not collision-resistant and is never the audit layer**. It
stays because localization is its job and it is the same function the traced
kernels already emit; a bundle that replaced it would orphan every card in
the repository. SHA-256 is never asked to localize; a manifest can tell you a
file is wrong but not which stage inside a computation moved first. One hash
per purpose. A sweep of `docs/` on 2026-09-01 found no sentence presenting
FNV as audit-grade integrity, so this section is the first statement of the
split rather than a correction, and the differ's "DUMP INTEGRITY" step is a
writer/reader **format-agreement** check that remains FNV, correctly.

An external implementation reports **both** per stage. SHA-256 lets
`validate` grade against the audit layer. FNV-1a64 lets `diff` hand the
comparison to `tools/identity_trace_diff.py`, the repository's one comparator
(DEVIATION 924), with the same verdict vocabulary, `RESULT: IDENTICAL` or
`RESULT: DIVERGENT`, and `FIRST DIVERGENCE:` naming a stage.

---

## Bundle format v1

A bundle is a directory.

```
manifest.json                      format version, profile, fixture list,
                                   stage schema, provenance, SHA-256 of every
                                   other file in the bundle
inputs/<fixture>/<buffer>.bin      frozen little-endian input bytes
expected/<fixture>/<stage>.bin     raw expected bytes (stages of 1 MiB or less)
expected/<fixture>/<stage>.sha256  digest only (larger stages): 64 lowercase
                                   hex characters and a newline
traces/<fixture>.card              the identity-trace card as produced
implementation-report.json         written by the EXTERNAL side, section 5
reports/                           optional, external files, never hashed by
                                   the manifest
```

`<stage>` is `<seq>.<sanitized-tag>`, the record's 0-based sequence number,
a dot, and its tag with every character outside `[A-Za-z0-9._-]` replaced by
`_` (the same rule the trace writer and the differ use). The raw/digest
threshold is **1,048,576 bytes** of stage buffer; changing it is a format
change and bumps the version. All bundle-internal paths use forward slashes
on every host.

### manifest.json

```json
{
  "format": "mojolearn-conformance-bundle",
  "format_version": "1",
  "profile": "mojolearn.identical.verify.kmeans.fp32.v1",
  "numeric_profile": {"mode": "identical", "dtype": "fp32", "version": "v1"},
  "raw_threshold_bytes": 1048576,
  "fixtures": [
    {
      "id": "kmeans",
      "profile": "mojolearn.identical.verify.kmeans.fp32.v1",
      "params": {"algorithm": "kmeans", "n": 4096, "d": 8, "k": 8,
                 "init": "array", "n_init": 1, "max_iter": 10,
                 "tol": 0.0001, "seed": 7, "sample_weight": null},
      "inputs": [
        {"name": "x", "file": "inputs/kmeans/x.bin", "dtype": "f32",
         "shape": [4096, 8], "fnv1a64": "<16 hex>", "sha256": "<64 hex>"},
        {"name": "centroids", "file": "inputs/kmeans/centroids.bin",
         "dtype": "f32", "shape": [8, 8], "fnv1a64": "<16 hex>",
         "sha256": "<64 hex>"}
      ],
      "trace": "traces/kmeans.card",
      "stage_schema": [
        {"seq": 0, "tag": "fit.x_norm", "dtype": "f32", "count": 4096,
         "fnv1a64": "<16 hex>", "sha256": "<64 hex>",
         "expected": "expected/kmeans/0.fit.x_norm.bin", "kind": "raw"}
      ]
    }
  ],
  "provenance": { "...": "see 'Provenance and artifact provenance' below" },
  "files": {"inputs/kmeans/x.bin": "<64 hex>", "...": "every file except manifest.json"}
}
```

`stage_schema` is the **ordered** stage list. It repeats the card's records
(seq, tag, dtype, element count, FNV-1a64) and adds each stage's SHA-256 and
its `expected/` path. The card stays in the bundle verbatim because it is the
artifact the whole trace toolchain speaks; the schema exists so a consumer
can implement from JSON without writing a card parser.

The dtype table is identical to the card format's.

| dtype | bytes per element |
|---|---|
| f32, u32, i32 | 4 |
| f64, i64 | 8 |
| u8 | 1 |

All multi-byte values are **little-endian**, elements in index order (row
major for 2-D shapes). All hashes are lowercase hex, SHA-256 as 64
characters, FNV-1a64 as 16. FNV-1a64 uses offset basis
`0xCBF29CE484222325` and prime `0x100000001B3`, one byte at a time, with
`h = (h XOR byte) * prime mod 2^64` starting from the offset basis. Known
answers, so a fresh implementation can check itself before touching a
bundle: the empty string hashes to `cbf29ce484222325`, `"a"` to
`af63dc4c8601ec8c`, `"foobar"` to `85944171f73967e8`.

### Provenance and artifact provenance (DEVIATION 929)

The manifest's `provenance` object records the run that produced the bundle.
Date, mojolearn version, commit, numeric mode, vendor read-back, device,
host, python, the comparator's path and SHA-256, and a per-binding
`artifacts` list carrying, for **every compiled extension (`.so`) the
exporting process had loaded**, its file SHA-256, size, mtime, and
`artifact_source_commit: "unverified"`.

That last field is a stated gap, not a formality. **A bundle certifies the
artifacts it hashes, not the source tree.** Today nothing ties a `.so` to the
commit that built it. The working tree's binaries carry no freshness signal,
and a stale binary announces itself only when a signature happens to have
drifted (three of four bindings exercised on 2026-09-01 were stale against
their own callers; the one caught in a single run was caught because the RF
binding happens to carry a vendor read-back). The manifest's `commit` field
is therefore a claim about the **checkout**, and the `artifacts` list is the
honest record of the **binaries**. The two are connected only by operator
discipline, meaning a rebuild before export as the run list orders, until
the closing mechanism below lands.

**The closing mechanism, specced here and not implemented here** (`.mojo`
and `bindings/` belong to the bindings lane, which owns this row): each
binding embeds, at compile time, the build commit and its numeric mode, the
same pattern as the existing per-binding vendor read-back and
`gbdt_numeric_mode`, and the Python surface reads both back at import,
refusing a binding whose stamp disagrees with the package version the same
way a tier or vendor mismatch is already refused. The precedent for why
read-back works and its absence does not: the RF vendor read-back caught
staleness in one run, while the unstamped bindings needed a signature change
to raise at all. When that lands, `artifact_source_commit` becomes the
read-back value and this section shrinks to a table row.

Bundles assembled with `export --from-card` set `artifacts_recorded: false`
and carry the producing card's own provenance comment lines instead. The
assembly host did not run the fit and does not pretend to know which
binaries did.

---

## What `validate` checks, and how it fails

The structural pass (`conformance validate BUNDLE`) checks that

- every file in `files` exists and re-hashes (SHA-256) to its manifest
  entry, and a wrong hash **names the file**, with both digests;
- files on disk the manifest does not vouch for are **named** as warnings
  (`manifest.json`, `implementation-report.json` and `reports/` excepted);
- the card parses through the one comparator's parser and matches
  `stage_schema` record for record, and a drift **names the stage and
  field**;
- every schema stage has its `expected/` file, and a missing one **names the
  stage**; raw stages match their dtype-times-count size, their card FNV and
  their manifest SHA-256; digest files are well-formed and agree with the
  manifest;
- inputs match their declared shape and dtype size and their FNV, and the
  kmeans fixture's inputs are additionally held to the pinned three-vendor
  E1U values (DEVIATION 922), so a manifest and its inputs cannot drift
  together unnoticed.

Report grading (`--report`) refuses **before comparing any bytes** when

- the report's `profile` differs from the manifest's. A different profile is
  a different computation, and grading across it would manufacture a verdict
  about the wrong thing.
- `bundle_manifest_sha256` does not match this bundle's manifest. The
  consumer validated a different bundle.
- required provenance (`implementation`, `device`, `os`, `date`) is missing.
  A result with no provenance is a number, not evidence, and that rule does
  not soften for other people's runs.
- a tag repeats. Tags are unique within a trace by the writer's own
  invariant, and a report that repeats one cannot be aligned.

After those gates, every schema stage is graded by SHA-256. A mismatch names
the stage and both digests and points at `conformance diff` for
localization. A stage the report does not answer for, a dtype or count
mismatch (structural, meaning the two implementations built different
amounts of work), and any extra stage in the report are each named failures.

## `diff`, localization through the one comparator

`diff` never compares hashes itself. It synthesizes a card, either from the
report's per-stage FNV-1a64 column or, with `--self`, from re-hashing the
bundle's own `expected/` bytes, and hands it with the bundle's card to
`tools/identity_trace_diff.py` (DEVIATION 924, one comparator, never two).
Its output is that tool's report verbatim. Tag-sequence alignment first,
structural findings before numeric ones, then the hash walk and
`FIRST DIVERGENCE:` with the last agreeing stages as context.

`--self` is the **corruption witness**. Flip one byte under `expected/` and
`diff --self` names the stage that byte lives in, while `validate` names the
file. Digest-only stages carry no bytes to re-hash; `--self` copies their
card hash through and **names them as uncovered** rather than silently
skipping them (DEVIATION 930), because a corruption in a digest-only stage
is the SHA-256 check in `validate` that catches it. Fixture #1 ships every
stage raw, so it can witness a corruption at any stage.

---

## implementation-report.json, what an external side writes back

This section is the contract. A team should be able to implement from it
without running mojolearn. One JSON file, UTF-8.

```json
{
  "format": "mojolearn-implementation-report",
  "format_version": "1",
  "profile": "mojolearn.identical.verify.kmeans.fp32.v1",
  "fixture": "kmeans",
  "bundle_manifest_sha256": "<the 64-hex SHA-256 of the manifest.json you ran against>",
  "provenance": {
    "implementation": "<name of your implementation>",
    "version": "<its version or commit>",
    "device": "<hardware that computed the stages>",
    "os": "<operating system / environment>",
    "toolchain": "<compiler / runtime, optional>",
    "date": "<ISO 8601 UTC>",
    "notes": "<optional>"
  },
  "stages": [
    {
      "seq": 0,
      "tag": "fit.x_norm",
      "dtype": "f32",
      "count": 4096,
      "sha256": "<64 hex over your raw little-endian bytes for this stage>",
      "fnv1a64": "<16 hex, FNV-1a64 over the same bytes>"
    }
  ]
}
```

The rules.

1. Run the fixture named in `manifest.json`. `params` gives every parameter
   and `inputs/` gives the exact input bytes. Read them; do not regenerate
   them.
2. At each point in your computation corresponding to a `stage_schema`
   entry, capture the buffer as raw little-endian bytes in element index
   order and record **both** hashes over those bytes. The provenance fields
   `implementation`, `device`, `os` and `date` are required; a report
   without them is refused, not graded.
3. Report **every** stage in the schema, in order, with the schema's `seq`,
   `tag`, `dtype` and `count`. If your implementation has no analogue of a
   stage, omit it and expect a named MISSING STAGE failure. That is the
   honest outcome, not a formatting problem to paper over. Do not invent
   stages.
4. The numeric semantics required to match at all are IEEE-754 binary32
   arithmetic as the profile's kernels perform it. This is deliberately
   hard. The stage card exists so that when you diverge, `conformance diff`
   tells you the first place where, instead of "the answer differs".
5. Write the file as `implementation-report.json` in the bundle directory,
   or anywhere, since `--report` takes a path. Then

```
python -m mojolearn conformance validate BUNDLE --report implementation-report.json
python -m mojolearn conformance diff     BUNDLE --report implementation-report.json
```

---

## What a PASS means, and what it does NOT mean

Blunt, mirroring `docs/VERIFY.md`.

- **A validated bundle round-trip on one machine claims nothing across
  vendors.** Export and validate on the same box prove the format
  round-trips and the artifacts are internally consistent. That is all.
- **The cross-vendor claim lives in the confirmed reference card**, two
  vendors at one commit with cards compared by the one differ
  (`docs/VERIFY.md`, "Where the reference card came from"), and in
  `E3_RESULTS.md` for the whole library. A bundle inherits exactly as much
  cross-vendor weight as the run that produced its card, which is why the
  provenance block is not optional.
- **A conformant external implementation claims agreement on THESE fixtures
  under THIS profile.** Not "bit-identical AI", not agreement on other
  inputs, sizes or algorithms, not anything about stages the trace does not
  checkpoint, and not that the computation was identical. Matching hashes
  mean the buffers agreed at the checkpoints (`core/identity_trace.mojo`,
  "What a matching hash does NOT prove").
- **A bundle certifies the artifacts it hashes, not the source tree.** The
  artifact-provenance section above, until the bindings carry a build stamp.
- One fixture, one algorithm, today. The coverage claim is one k-means fit
  wide and the tooling says so in its own output.

## Fixtures

**Fixture #1, shipped: `kmeans`.** The E1U k-means fixture `verify` already
runs, reused for the same three reasons `docs/VERIFY.md` gives. It is pinned
with three vendors behind its input hashes, every coordinate is exact in
Float32, and it reaches the REPLACE mechanism (fixed-point Int32
accumulation) rather than a convenience. Every stage is small enough to ship
raw, so the bundle can witness a corruption at any stage.

The recommended next fixtures are named here rather than invented, because
each needs its producing card wired on the Mojo side first (adding them is
sanctioned capability growth, but not from this lane).

| future fixture | what it would witness | producing card |
|---|---|---|
| elementwise | the PIN moves (ftz, contraction) in isolation, the cheapest possible cross-check for a bring-up | needs a small traced elementwise driver |
| reduction / GEMM stage | ordered-reduction agreement, the mechanism most accelerators get wrong first | `gemm/` trace territory (`tools/gemm_card.sh`) |
| tree-training card | a real multi-thousand-stage workload, graded by digest and diffed by stage | `gbdt/train.mojo`'s existing `IdentityTrace` |

## Runs owed (nothing on this page has run yet)

This machinery was built write-only on 2026-09-01. **Every command below is
UNVERIFIED, RUN OWED**, in this order. The rebuild comes first because the
on-disk `.so` artifacts must be presumed stale (artifact provenance, above).

```
# 0. rebuild the bindings the fixture path loads (identical arm), then all
#    twelve for a bundle export, since the manifest hashes every loaded one
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh
for s in bindings/build_*.sh; do MOJOLEARN_NUMERIC_MODE=identical bash "$s"; done

# 1. the host-only gate, then the Apple card
PYTHONPATH=python pixi run python3 -m mojolearn check-fixture
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  pixi run python3 -m mojolearn verify --emit-reference /tmp/kmeans.apple.card

# 2. second vendor, same commit (rented leg, the Apple card copied over),
#    one command
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  pixi run python3 -m mojolearn verify \
  --confirm-reference kmeans.apple.card --emit-reference kmeans.nvidia.card

# 3. install the agreed pair, then close the loop on the producing machine
PYTHONPATH=python pixi run python3 -m mojolearn install-reference \
  /tmp/kmeans.apple.card /tmp/kmeans.nvidia.card
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  pixi run python3 -m mojolearn verify

# 4. bundle round trip, including the required-red arm
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  pixi run python3 -m mojolearn conformance export /tmp/kmeans.bundle
PYTHONPATH=python pixi run python3 -m mojolearn conformance validate /tmp/kmeans.bundle
PYTHONPATH=python pixi run python3 -m mojolearn conformance diff /tmp/kmeans.bundle --self   # must print IDENTICAL
python3 - <<'EOF'   # flip one byte in one expected stage
import glob
p = sorted(glob.glob('/tmp/kmeans.bundle/expected/kmeans/*.bin'))[3]
b = bytearray(open(p, 'rb').read()); b[0] ^= 1; open(p, 'wb').write(bytes(b))
print('corrupted', p)
EOF
PYTHONPATH=python pixi run python3 -m mojolearn conformance validate /tmp/kmeans.bundle      # must FAIL naming the file
PYTHONPATH=python pixi run python3 -m mojolearn conformance diff /tmp/kmeans.bundle --self   # must name that stage
```

Until step 3 lands, `verify` keeps exiting 5 (NO REFERENCE), exactly as
`docs/VERIFY.md` documents.
