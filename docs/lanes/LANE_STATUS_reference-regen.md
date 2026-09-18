# lane/reference-regen — the shipped reference table, regenerated

**Written 2026-09-17 mid-flight, for a session with no context.** Legs are
still out. Everything below is measured unless it says otherwise.

## What this lane was asked to fix

`python -m mojolearn verify --all` could not reach VERIFIED and
`release_qualified` was false, for five stated reasons:

1. five lanes withheld for `stale reference` (`mamba3`, `transformer`,
   `transformer-window`, `mamba2-dtlimit`, `samba-untied-dropout-accum`);
2. `verify --coverage --json` reported `reference_admission_policy =
   {status: legacy}`;
3. thirty of the 246 appendix entries shipped no reference hash of any kind;
4. the four optional properties (`batchgrad`, `batchscale`, `ragged`,
   `rlpair`) shipped ZERO hashes;
5. `stepfull` was declared by sixteen lanes and carried by nine.

## What the causes turned out to be

**Two of the five were a regeneration that had never been run, not missing
data.** `_verify_reference.build_table` writes the `admission_policy` key and,
under `--batch-checks`, the four `OPTIONAL_PARTS`. The shipped table was built
before either existed, so `_verification_coverage` fell back to the `legacy`
stub and the optional parts were simply never emitted although 19 to 39
committed columns already carried them.

**Item 1 was the opposite: a regeneration alone makes it WORSE.** No committed
column carries those five lanes at their current `LANE_REVISIONS` entry, so
rebuilding drops their cells rather than refreshing them: `stepfull` goes from
9 cells to 4 and the appendix's zero-hash count from 28 to 37. They needed a
record, which is what was rented.

## Measured, before the rented columns (regeneration of committed records only)

    cells 1669 -> 1669, parts 6680 -> 7228 (with --batch-checks)
    admission_policy: legacy -> min_repeats 2, input_witness_required,
                                property_protocol_required
    batchgrad 0 -> 6 lanes at 9/9,  batchscale 0 -> 13,
    ragged    0 -> 4,               rlpair     0 -> 4 plus 8 partial
    stepfull  9 cells -> 4          (the stale lanes' cells drop out)
    appendix entries with no hash: 28 mapped -> 37 mapped

## Measured, with three of the rented columns in place

    cells 1669 -> 1885,  parts -> 11733,  records 87 -> 80
    stepfull  4 -> 1317 cell parts with a reference
    batchgrad 162 -> 927, batchscale 162 -> 927, ragged 163 -> 928
    rlpair    61 -> 94
    stale reference lanes: 5 -> 0
    appendix entries with no hash: 28 mapped -> 17 mapped (plus 2 unmapped)

## The columns

All CPU columns are `MOJOLEARN_NUMERIC_MODE=identical`, `--repeats 2`, nine
fixtures, all 32 host families built from source on the box through the R2
binding cache (`tools/runpod_cpu_leg.sh`). RunPod CPU pods, $0.24/hr at
8 vCPU. Staged under `bench/results/identity_break/2026-09-17_reference-regen/`.

| column | box | lanes | flags | state |
|---|---|---|---|---|
| `cpu-x86-shard2.json` | EPYC 9965 | 55 | all four opt-in parts | complete, 495 cells, 414 STABLE |
| `cpu-x86-shard3.json` | EPYC 9655P | 55 | all four opt-in parts | complete, 495 cells, 423 STABLE |
| `cpu-x86-light1.json` | EPYC 9655 | 115 | `--step-full --no-rlpair` | complete, 1035 cells, 882 STABLE |
| `cpu-x86-shard1.json` | — | 56 | all four opt-in parts | RUNNING, pod `b97wwyyfre2q1m` |
| `cpu-x86-shard4.json` | — | 55 | all four opt-in parts | RUNNING, pod `m98de6kclo0x4l` |
| `cpu-x86-light2.json` | — | 114 | `--step-full --no-rlpair` | RUNNING, pod `2x1owmqj5e6zge` |
| `cpu-x86.json` (neural probe) | — | 8 | all four opt-in parts | RUNNING, pod `4ckx8iprhc6k54` |
| NVIDIA `par-*` | RTX 4090 | 12 | default | RUNNING, pod `ny4j0vvsrkigke` |

**Partials are safe to take at any time** with `scp` from the pod's
`/root/leg_out/`: `tools/identity_break.py` rewrites its JSON after EVERY
cell. The leg also fetches whatever exists when the lease is nearly spent.
Do NOT wait for a leg to finish to get its data.

## Why two kinds of CPU leg

`batchscale` on the Mamba-2-family lanes costs 200 to 370 SECONDS per cell on
a RunPod CPU box (measured: `mamba2/base` 366 s, `mamba2-int8w/base` 277 s,
`mamba2-bf16w/base` 209 s), against 8 to 11 s for the same lanes without it.
A single all-lanes all-parts sweep does not fit in any lease. The `light`
legs therefore run every lane with `--step-full --no-rlpair` and no other
opt-in part (1035 cells in about 17 minutes) and carry items 1, 3 and 5; the
`shard` legs carry the opt-in tail for item 4.

## The check that ran before the table was rebuilt

`~/mojolearn-evidence/reference-regen/refcheck.py` compares every new column
against the hashes the SHIPPED table already carries, because a newest-wins
rebuild would otherwise replace a reference four device classes agree on with
this record's single value and say nothing at all.

    shard3   1176 compared, 1176 agree, 0 differ, 453 new
    shard2   1187 compared, 1187 agree, 0 differ, 442 new
    light1   2555 compared, 2499 agree, 56 differ, 613 new
    light2 (partial, 424 cells)
             980 compared,  943 agree, 37 differ, 563 new

**Every one of the 93 differing cell parts is in one of the five stale
lanes** (`transformer` 28, `transformer-window` 28, `mamba3` 28,
`samba-untied-dropout-accum` 5, `mamba2-dtlimit` 4) and nowhere else. That is
the divergence appearing exactly where `LANE_REVISIONS` says the input moved,
which is the reason those lanes were withheld. No unexplained divergence.

The checker was watched to FAIL first: one STABLE cell of one column was
rewritten to `deadbeefdeadbeef` and it reported that cell by name together
with the four device classes standing behind the shipped value
(`~/mojolearn-evidence/reference-regen/partials/NEGATIVE_CONTROL.txt`).

## Two findings that are NOT fixable by regenerating the table

**`release_qualified` is hard-coded.** `python/mojolearn/_verification_coverage.py`
sets `lane['release_qualified'] = False` and `evidence_summary
release_qualified=False`; `tools/verification_evidence.py` writes
`release_qualified=False` into the shipped evidence data; and
`python/mojolearn/tests/test_verification_coverage.py` asserts all three are
`False`. It is not a property of the table. Flipping it is a claim about a
release record on all four columns at one commit, which this repository takes
once per PyPI release and which needs the Apple column that this lane was told
not to run.

**A full CPU `verify --all` cannot reach VERIFIED while
`PUBLIC_PENDING_LANES` is non-empty.** `_verify_all` line ~1537 makes every
WITHHELD lane a `scope_gap` when the run is a full one with no `--lanes`, and
`verdict()` reads any gap as INCOMPLETE. Of the 51 withheld lanes, 17 are held
for reasons only a three-vendor release record can clear: 12
`PUBLIC_REFERENCE_CANDIDATES` ("reference qualification pending", every
IDENTICAL cell they have rests on the Apple column alone) and 5
`TRAINING_FIX_LANES` ("own record", they are not in `record_covered_lanes()`
until `TRAINING_GPU_COLUMNS` names a record carrying them).

## What this lane changed in code

`PUBLIC_PENDING_LANES` gained the reason **`one column`**: the table carries
cells for the lane at the current revision, so `no reference` and `stale
reference` are both false, but every cell rests on a SINGLE device class, so
the reference has one witness and nothing has ever reproduced it. `unwatched`
said the only thing missing was a run, which would have been a run against a
number no second machine has produced. `samba` was in exactly that position
and was labelled `unwatched`; every cell the shipped table carries for it
comes from the CPU column alone.

The reason is checked against the table, not trusted as prose:
`test_host_surface._reference_classes` counts the device classes in each cell
part's `cols` and the test fails BOTH ways. Watched to fail first: labelling
`gbdt-yeti-rank` (apple and cpu) as `one column` fails the test by name.

`docs/VERIFY_EXTERNALLY.md` gained Recipe 4, the `verify --compare`
two-strangers protocol, which the external-verification document did not
mention at all, with the verdict ladder in the order the code tries it. Its
"Two smaller checks" section, which predated the whole `verify --all` surface,
now describes what the command does. `docs/VERIFY.md` names
`bench/results/verify_reports/` and says plainly that comparing against our
own published document is WEAKER than two strangers comparing.

## What is still owed when the legs land

1. Stage the remaining columns into
   `bench/results/identity_break/2026-09-17_reference-regen/`, run
   `refcheck.py` on each, commit them.
2. `cd python && MOJOLEARN_NUMERIC_MODE=fast python3 -m mojolearn verify --all
   --batch-checks --emit-reference python/mojolearn/verify_reference/table.json`
   — this is the shipped command and it needs NO binding; `fast` mode only
   gets the package imported, and it was verified to produce byte-identical
   output to calling `build_table` directly.
3. Re-derive every `PUBLIC_PENDING_LANES` reason from the new table with
   `~/mojolearn-evidence/reference-regen/reasons.py`, which prints the derived
   reason beside the current one and reproduces the hand-maintained list
   exactly on the shipped table.
4. One more CPU leg with `~/mojolearn-evidence/reference-regen/cmd_reports.sh`:
   `verify --coverage --json-out`, `verify --self-test`, `verify --all
   --json-out`. The last is BOTH the watched promotion run AND the published
   evidence document item 7 asks for. Commit the documents under
   `bench/results/verify_reports/`.
5. `python3 tools/verification_matrix.py --write` then `--check`.
6. `bench/results/verify_reports/` is cited by `docs/VERIFY.md` and
   `docs/VERIFY_EXTERNALLY.md` and does not exist yet.

## Operational notes for whoever picks this up

- **The session scratchpad is SHARED with the other concurrent lanes.**
  Another lane's `gemm_remote_leg` truncated this lane's leg log at
  `/private/tmp/claude-501/.../scratchpad/leg1.log`, and another lane's task
  output appears in the same `tasks/` directory. All of this lane's logs,
  scripts and partials are under `~/mojolearn-evidence/reference-regen/`.
- The first NVIDIA attempt
  (`bench/results/e1g/2026-09-17_201824-nvidia-refregen-par-lanes`) built all
  ten GPU bindings in 8 minutes and then lost the run to
  `REFUSING: no commit witness`. The RunPod runner passes no environment to
  the extra body and ships a git archive with no `.git`. The body now bakes
  the sha in, substituted by `git rev-parse` when the file is generated,
  never typed.
- The `par-*` lanes and `gbdt-tensor-ctr-tables` are the ONLY lanes that
  refuse on a CPU-only box. `par-*` are multi-GPU drivers;
  `gbdt-tensor-ctr-tables` wants
  `verify_reference/ctr_models/gbdt-tensor-ctr-tables.base.npz`, which is not
  in the tree.
- RunPod CPU hosts were running at load 60 to 160 throughout. Cell rates in
  this document are from those boxes and are not a property of the code.


---

# PART 2, 2026-09-18: publishing our own evidence documents

## Why

`docs/VERIFY_EXTERNALLY.md` Recipe 4 is `verify --compare`, the only protocol
here that does not rest on our records. It was documented and INERT: a
stranger who runs `verify --all --json-out mine.json` had nothing to diff
against, because `python/mojolearn/verify_reference/` holds only `models` and
`table.json`.

## State right now

`bench/results/verify_reports/` exists and carries:

| file | what |
|---|---|
| `README.md` | which classes, what the pair proves, the three ways a comparison fails without anything being wrong |
| `verify-all.cpu-a.json.gz` | a real `verify --all` run, AMD EPYC 4564P, at `b9c338384` |
| `verify-all.cpu-a.json.commitment` | its commitment |
| `verify-all.cpu-a.txt` `self-test.cpu-a.txt` | the human output, and the self-test that watched the verifier detect a one-ULP perturbation |
| `compare-same-device-agree.txt` | `AGREE`, 5174 parts, both commitments verified |
| `compare-uncomputed-negative-control.txt` | `INCOMPLETE. Absence is not agreement.` |
| `nvidia-hang/` | the deadlock below |

**DEVICE CLASSES: CPU ONLY.** No Apple, NVIDIA or AMD document exists. Do not
read the published pair as cross-vendor evidence.

**STILL OWED:** `cpu-b` at `b9c338384` (its leg was running when this was
written; pod `lycvc7rkggap9r`, output lands under
`bench/results/runpod_cpu/*refregen-rep6/remote/leg_out/verify-all.cpu6.json`)
and the comparison of cpu-a against it. The two comparison transcripts above
were taken from the EARLIER pair at `1863520e9` and prove the mechanism, not
the published files.

## THE HARNESS DIGEST MOVED TODAY, AND IT INVALIDATES OLDER DOCUMENTS

`lane/kmeans-cosine-capability` deleted the `kmeans-cosine` lane. The harness
went from 229 lanes to 228 and `tools/identity_break.py` went from sha256
`5160aff733ed3c36` to `90bdd7557b3c123b` at commit `b9c338384`.

`compare_documents` refuses to compare across a harness change:
`comparison_context_problems` reads `INCOMPARABLE`, never a silent wrong
answer. So EVERY evidence document emitted before that deletion, including
the four this lane made earlier today at `1863520e9`, is INCOMPARABLE against
anything run at today's main. That is the protection working, and it is why
`cpu-a` was re-emitted. It is also why these documents are RELEASE artifacts:
they need re-emitting whenever the harness moves.

The rebuilt table needed no work for the same event: an independent
`--emit-reference --batch-checks` over the merged records came out BYTE
IDENTICAL to main's at 2052 cells, 228 lanes, no cosine cell, every optional
part present. Two independent regenerations agreeing byte for byte is the
strongest statement available that no hash in that file is typed.

## A GPU INSTALL'S `verify --all` DEADLOCKS AT `transformer-bf16w`

Measured 2026-09-18 on a rented RTX 4090 (sm_89), all ten GPU bindings built
clean, evidence in `bench/results/verify_reports/nvidia-hang/`:

    47 lanes in about four minutes, including mamba1/2/3 and transformer,
    then nothing for over ten minutes at `transformer-bf16w`
    nvidia-smi   utilization 0 %, 978 MiB
    /proc/<pid>  State: S (sleeping), Threads: 195
    wchan        futex_wait_queue

GPU idle, 195 threads, blocked on a futex: a host-side synchronisation
deadlock, not slow arithmetic. THE USER-FACING CONSEQUENCE: the fourteen
low-bit weight lanes are PUBLIC as of 2026-09-17, they were promoted on CPU
evidence, and a GPU install's `verify --all` runs every lane. So `verify
--all` on an NVIDIA box hangs rather than finishing. This is not this lane's
to fix and no lane has been opened for it.

## What a plain `verify --all` on a CPU box reads now, and why not VERIFIED

    RESULT: INCOMPLETE (verified 5165 of 7253 cell parts
                        (0 divergent, 0 owed, 0 refused, 2088 n/a)). exit 5

Zero divergent, zero owed, ZERO REFUSED. The only thing between this and
VERIFIED is `scope_gaps`: `_verify_all` makes every WITHHELD lane a scope gap
on a full run, and `verdict()` reads any gap as INCOMPLETE. Twelve
`PUBLIC_REFERENCE_CANDIDATES` and five `one column` lanes remain, and both
need columns from a device class that is not CPU.

An earlier pair of documents read 90 REFUSED. That was NOT a code defect:
`tools/runpod_cpu_leg.sh` leaves `bench/results` at home, so the two CTR
lanes' digest-checked `.npz` models never reached the box. `resolve_model`
falls back to the checkout copy correctly. The fix is
`--include bench/results/identity_break/2026-09-15_gbdt-ctr-tables/models`.

## THE EXACT NEXT COMMAND

When `rep6` lands, from `~/mojolearn-wt/reference-regen`:

    V=bench/results/verify_reports
    D=$(ls -d bench/results/runpod_cpu/*refregen-rep6/remote/leg_out)
    gzip -9 -c $D/verify-all.cpu6.json > $V/verify-all.cpu-b.json.gz
    cp $D/verify-all.cpu6.json.commitment $V/verify-all.cpu-b.json.commitment
    cd python && MOJOLEARN_NUMERIC_MODE=fast python3 -m mojolearn verify --compare \
        ../bench/results/runpod_cpu/2026-09-18_084015-refregen-rep5/remote/leg_out/verify-all.cpu5.json \
        $D/verify-all.cpu6.json \
        --commitment-a <that file>.commitment --commitment-b <that file>.commitment \
        > ../$V/compare-cpu-a-vs-cpu-b.txt

Then update `$V/README.md` with the verdict and whether the two boxes were
the same CPU model (`rep5` was an EPYC 4564P; if `rep6` differs, the pair is
cross-hardware rather than repeatability, and the comparer prints a WARNING
when they match).
