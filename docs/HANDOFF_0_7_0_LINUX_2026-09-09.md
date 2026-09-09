# 0.7.0 Linux release: handoff, 2026-09-09

Wind-down state. Everything below is committed. Nothing is running and
nothing is billing from this lane.

## Where it stopped

**One step from publishing.** The wheel is built, packed, repaired and
twine-clean. All three architectures have build proofs from one commit and
all three have installed qualification evidence. What remains is a decision
about one failing job, then the manifest, the GitHub pre-release and the
PyPI dispatch.

## The artifact

    mojolearn-0.7.0-py3-none-manylinux_2_35_x86_64.whl
    60,009,457 bytes   sha256 afb5dd8a06ba62ff222168fad7440e6558fb7e1fd43018ac7c648321e4fb9d75
    twine check PASSED

It lives in this session's scratchpad, which is NOT durable:

    /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/
      2bf083ed-3130-44f4-a3e7-e3f9a4424d1b/scratchpad/rel070/
        dist070/audit/repaired/final/   the wheel to publish + dir-entry-strip.json
        qualification/                  the staged admission root
        proofs/                         cuda-sm_89.json cuda-sm_90a.json hip-gfx942.json

**COPY THAT OUT BEFORE IT IS REAPED.** Rebuilding it costs three rentals.

Frozen build commit: `340de7a1`. The three proofs and the wheel all name it.
Later commits changed only files outside the native inventory, so the
proofs still verify against the working tree; `check_release061` confirms
this and it was checked after every commit.

## Qualification

    hip/gfx942   25/25 PASSED       DigitalOcean MI325X
    cuda/sm_89   24/25              RunPod L40S
    cuda/sm_90a  24/25              DigitalOcean H100

The one failure is the same on both NVIDIA columns and is bit-identical
across two chip generations:

    mamba/deterministic
    k_last_ matches ref64 ssd.k_last
    worst excess 3.980e-07 over |dump-ref| <= 1e-06 + 1e-05*|ref|
    flat index 151 (batch 1 / head 0 / component 23)
    got -0.0214189123   ref -0.0214173001

This is a KNOWN OPEN ITEM, not a regression:
`CROSS_VENDOR_FEATURE_IDENTITY_AUDIT.md` records Mamba3 forward/continuation
as "FAST repair FNV validated; **DET pending**", and
`MAMBA_0_6_1_RELEASE_CHECK.md` names this same flat index 151 and says of
the tolerance "do not change it". **It was not changed.**

## The open decision, and Andrew's stated preference

Two mechanisms exist. Andrew's last instruction was to prefer the second.

1. **Declare it** (built, tested, DEVIATION 2299). A full column may carry a
   known failure declared by name and cited to the document recording it
   open. Undeclared failures still refuse; a declaration for a job that
   passed refuses as stale; every declaration is republished. This makes the
   RELEASE RECORD honest.

2. **Disable it** (NOT built). Refuse `deterministic` for the Mamba lane on
   the CUDA column by name, so a user gets an error instead of silently
   wrong numbers. This protects the USER, which the declaration does not.
   Andrew: "rather than shipping and documenting a failed feature flag why
   not just disable it?" -- and he is right. AMD passes this job, so the
   refusal should be CUDA-only rather than deleting a working feature.

   Cost: `bindings/_mojolearn_mamba.mojo` is in the native inventory, so it
   is a full rebuild of all three architectures plus requalification of all
   three. About an hour and twenty dollars. The binding already imports
   `GLOBAL_NUMERIC_MODE`, `NUMERIC_DETERMINISTIC` and `COMPILED_VENDOR`, and
   `mamba3.mojo` has the `raise Error(...)` refusal idiom at :271.
   `expected_jobs()` in `tools/verify_linux_surface_qualification.py` (NOT in
   the inventory) must stop expecting `('mamba','deterministic')` to match.

## What is owed on the mamba defect itself

Two hypotheses were tested and both were wrong; the useful narrowing:

* IDENTICAL passes because `identical_mul_add`
  (`checks/numerics.mojo:84`, IDENTITY_PATHS row 9) emits an explicit FMA
  with one rounding. FAST and DETERMINISTIC both take `a*b + c`.
* `ftz` (`:73`) is a no-op in BOTH fast and deterministic, and the Mamba
  forward has no deterministic-gated row at all. So by inspection fast and
  deterministic should compile the same for this kernel -- **yet fast passes
  and deterministic fails, reproducibly.** That asymmetry is the bug and it
  is not explained by the source.

**Next experiment, one rental, five minutes, about a dollar:** run
`tools/mamba3_mode_state_probe.py` in fast and in deterministic on one
NVIDIA box and diff the retained `k_last_[1,0,22:24]` bits -- the window
containing the failing component 23. Equal bits means the fault is in the
harness; different bits means two modes sharing every documented code path
produce different results, which is the defect.

## Publishing, when the decision is made

1. `check_release061` over the staged qualification root (local, free).
2. Alpha manifest: `{schema: mojolearn.alpha-release.v1, version: "0.7.0",
   release_profile: "alpha-api", files: {<wheel>: <sha256>}}`. One to
   thirty-two wheels are allowed, so Linux-only is fine and no macOS wheel
   is needed.
3. GitHub pre-release `alpha-api-0.7.0-<stamp>` carrying wheel + manifest.
4. `gh workflow run release-provenance.yml --ref <ref> -f publish=pypi
   -f alpha_candidate_tag=<tag> -f alpha_manifest_sha256=<sha>`.

`packaging/verify_alpha_artifacts.py` requires the installed qualification
before upload, so the workflow will refuse a wheel without it. There is no
PyPI API token on this machine; publication goes through the workflow's
Trusted Publisher only.

## Also landed tonight, and owed afterwards

Landed: DEVIATION 2292 (uplink guard in three leg tools), 2293 (the Hopper
slot's two spellings, eight gates, one definition), 2294 (qualify mode on
the DigitalOcean leg), 2295 (RECORD does not list directories), 2296 (the
optimizer flags do not flip), 2297 (two qualification tiers), 2298 (qualify
mode on the RunPod leg), 2299 (declared known failures).

Owed, all blocked on the inventory because fixing them invalidates the three
build proofs:

* The same RECORD-vs-directories bug still lives in
  `tools/linux_surface_qualification.sh`, worked around by
  `tools/strip_wheel_dir_entries.py`. Fix it the next time that file moves.
* Stale comments from the 2026-09-03 fast-path flips:
  `greedy_search_helper.mojo:376-380` still claims AMD routes to the lookback
  partition, and cites `kernel_matrix.mojo:1789` in a 799-line file;
  `reorder_single_pass.mojo:78-79`; `reorder_one_bit.mojo:74-75`;
  `tools/gbdt_accuracy_ab.sh:53-57`. Two defines are now inert:
  `MOJOLEARN_2043_FAST_FUSED_ONE_BYTE` and, on AMD,
  `MOJOLEARN_2042_FAST_NO_LOOKBACK`.

Not blocked: the NVIDIA lanes price sweep with the partition A/B
(`MOJOLEARN_2042_FAST_NO_LOOKBACK` moves NVIDIA only now), and Andrew's
question of whether `deterministic` earns its keep at all on lanes where it
is no faster than `identical` -- decided by measurement, implemented as a
named refusal, never a silent alias.

## The AMD re-measure, which is finished

The reason this round began. At the release commit, on a fresh MI325X:

    gbdt identical/fast    S (1.0M rows)  0.782 -> 1.000
                           M (1.5M rows)  0.731 -> 0.997
                           L (2.0M rows)  never measured -> 1.012

Fast's own time at M fell from 0.727 s to 0.541 s. The paper's worst row is
parity, and the 2M-row point is new evidence. Every other AMD lane held.
