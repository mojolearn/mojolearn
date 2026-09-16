# Verify the identity claims on your own machine

mojolearn claims three things under the IDENTICAL tier: the same fit gives
the same bits on Apple, NVIDIA, AMD and CPU; a row's answer does not depend
on which other rows were in the call (batch invariance); and a model trained
on a GPU predicts the same bits wherever it is loaded. One command checks
all three on the machine you installed on.

```sh
pip install mojolearn
python -m mojolearn verify --all
```

No repository, no dataset, no network and no other machine-learning library is needed. The
command selects the IDENTICAL tier itself when `MOJOLEARN_NUMERIC_MODE` is
unset; a tier you set explicitly is respected, and a FAST process is refused
(exit 3).

## What `verify --all` runs

The lanes of `tools/identity_break.py`, the harness every committed record
was written with. The wheel carries a byte copy of it
(`mojolearn/_identity_break.py`), and the command imports it and calls its
own fixtures, lanes and hash functions, so a hash printed here is the hash
the harness would record on the same machine.
`python/mojolearn/tests/test_verify_all.py` runs both on one machine and
requires equal values.

- **Fixtures** are generated from fixed seeds inside the package: nine kinds
  (base, ties, hashed, wide, denormal, denormal_ftz, dupes, odd, negative),
  20,000 rows by 16 columns (odd: 12,345 by 17), plus held-out rows from a
  second seed. Their hashes are checked against the record before any fit;
  a machine that draws different bytes stops with exit 4 rather than
  reporting divergences that are not arithmetic.
- **On a GPU install** every lane runs (one per public estimator, plus one
  per constructor value that selects a different numeric path, the linalg
  and metrics functions, and the multi-GPU drivers on one device).
- **On a CPU-only install** the public CPU reference lanes run
  (`host_surface.public_reference_lanes()`, 122 of them), fitted inside the
  verifier's reference scope. The list is DERIVED, not hand-written: every
  lane with a CPU training path that a release record covers and the shipped
  table carries a reference for, less the `par-*` multi-GPU drivers and the
  lanes held in `PUBLIC_PENDING_LANES` with a stated reason. They cover the
  gradient boosting and random forest and Extra Trees fits, k-means and its
  starts, the k-NN, radius and kernel-density variants, DBSCAN, the linear,
  ridge, logistic and decomposition lanes, SVC, SVR and the isolation forest,
  the Gaussian processes and mixtures, the scalers, ARIMA and Holt-Winters,
  the MLP, Mamba and Transformer blocks and the optimizers, the saved UMAP
  embedding's transform, the pinned GEMM and the Cholesky solve, the KPSS
  test, the bootstrap, the permutation test and Monte Carlo integration, and
  tokenizer, which loads the synthetic vocabulary mojolearn trains itself.
  It was 9 lanes until 2026-09-16, then 39 when thirty more were measured
  IDENTICAL against the Apple, NVIDIA and AMD columns, and then 122 when
  every host family began shipping (lane/ship-cpu-host-families) and 83 lanes
  stopped being unreachable on the machine you installed on. Each of the 83
  was watched reading IDENTICAL on a CPU-only install before it was added.
- **Portable models** run on every install: small models trained on a GPU
  and saved, shipped in `mojolearn/verify_reference/models/` (a random
  forest, a symmetric boosting model, a linear regression and a PCA, 179 KB
  together). A GPU install loads them with the class's `load`; a CPU-only
  install with `mojolearn.host_model(path)`. Their file
  bytes must equal the recorded model hash, and the loaded model's answers
  on the held-out rows, whole, row by row and split, must equal the recorded
  GPU answers.

Every cell (a lane on a fixture) has four parts:

| part | question |
|---|---|
| train | does the fit read back the recorded bits? |
| infer | does the fitted model answer held-out rows with the recorded bits? |
| model | are the saved file's bytes the recorded bytes, and does the reloaded file answer like the model in memory? |
| batch | are the held-out answers the same whole, one row at a time, in an uneven split and by prefix, and equal to the recorded bits? |

Each part reads one state.

| state | meaning |
|---|---|
| IDENTICAL | equal to the reference hash |
| DIVERGENT | different from it, or this machine disagreed with itself between repeats, or batch invariance failed here |
| OWED | no committed record carries this part yet, the record's own columns disagree at one commit, or the part changed after the record (a hash here against an `n/a` there, as when a lane gained a batch declaration); not a pass |
| REFUSED | the lane or probe raised; the sentence is printed (for example a function with no CPU implementation, or a lane whose host binding this install lacks, refused by name). **A refused part did not run, so it is not a pass and it costs the whole run its pass**: any refusal makes the verdict INCOMPLETE and the exit non-zero |
| N/A | the estimator has no such output (a transductive clusterer has no held-out answer) |

The command prints a table per family and a verdict. The verdict line leads
with how much of the run was actually checked, as in `verified 44 of 332 cell
parts (0 divergent, 0 owed, 288 refused, 0 n/a)`, so a run that mostly refused
cannot be misread as a run that passed.

## What the CPU training bindings are for

Every wheel since 2026-09-16 carries all thirty-two host (CPU) bindings, the
sixteen training ones included, so that the lanes above can be re-run on the
machine you installed on. **They exist to check the claim, not to train your
models.**

- **Verification.** `verify --all` fits each lane on your CPU and compares the
  bits against what Apple, NVIDIA and AMD recorded. That is the whole reason a
  fit is in the wheel at all.
- **Small data, and reproducibility.** The fixtures are 20,000 rows by 16
  columns. A fit at that size is a check you can read the result of in seconds,
  and it gives the same bits on every machine, this year and next.
- **Air-gapped checking.** No network, no dataset, no second machine and no
  GPU. A box that can never reach one of our GPUs can still hold us to the
  claim.

**They are not a CPU training engine and they are not tuned for speed.** A host
binding is a device kernel restated as a serial host loop so that it produces
the device's bits exactly; it is single-threaded by construction, because a
reduction whose order depends on a thread count does not give one answer.
Timing one against a GPU fit, or against another library's threaded CPU fit,
measures that choice and nothing else, so please do not file it as a
performance bug. To train a model, train it on a supported GPU and load it
anywhere; `verify --all` checks that path too, with the portable models.

This is also why ordinary CPU `fit` still refuses. The bindings ship, but
`mojolearn.LinearRegression().fit(...)` on a CPU-only install raises by name
and tells you to train on a GPU and load the saved model. The verifier reaches
them through a private scope of its own
(`python/mojolearn/_cpu_reference.py`), so shipping the binaries widened what
you can CHECK and changed nothing about what the library will train for you.

## Flags

| flag | effect |
|---|---|
| `--all` | every lane this install runs, on every fixture |
| `--quick` | one lane per family on the base fixture |
| `--full` | the same as `--all`, spelled out |
| `--lanes a,b` | only these lanes |
| `--fixtures a,b` | only these fixtures |
| `--repeats N` | fits per cell (default 1); two or more also catch a cell that moves on this machine |
| `--no-models` | skip the portable models |
| `--json` | one JSON report on stdout, progress on stderr |
| `--reference-table PATH` | compare against another table |
| `--self-test` | show that this verifier can fail (below) |
| `--cross-check [quick\|default\|all]` | compare your GPU against your CPU (below) |
| `--json-out PATH` | with `--all`: also write the evidence document to PATH |

The three checks answer different questions, and are worth more together than
separately:

1. **`--cross-check`**, your GPU against your CPU. Trusts nobody: you generated
   both sides on your own machine.
2. **`--all`**, your machine against our recorded columns. Trusts our table,
   which is auditable because the raw columns are committed under
   `bench/results/identity_break/` and the document names the exact file and
   commit each reference came from.
3. **`--self-test`**, which shows the comparison can fail at all. Without it the
   first two are checks nobody has watched fail.

## Can you watch it fail?

A verifier that only ever passes proves nothing. `verify --self-test` lets you
see the comparison catch a wrong answer, on your machine, in your installation:

    python -m mojolearn verify --self-test

It runs one lane twice through the ordinary comparison, the same code path that
judges every other lane. Once on the untouched fixture, which must read
IDENTICAL, and once with every value of the input's first column moved up by
one ULP, which must read DIVERGENT. The perturbation is real arithmetic at run
time, not a printed verdict, so it needs no special build; the lane computes
correctly over an input whose last bits differ, and the comparison is what
notices. Exit 0 only if **both** arms behave, so a comparison stuck on either
answer fails it.

Two commands, then, and the passing one means something because the other one
can fail.

## Your GPU against your CPU

The strongest of the three checks, because it requires trusting **nobody**:

    python -m mojolearn verify --cross-check

Comparing your machine against our recorded columns asks you to believe we
recorded honestly. This asks you to believe nothing. Each lane is fitted once
on your GPU, then the same fitted model is asked for the same held-out answer
twice: from the GPU estimator, and from the saved model reloaded through the
CPU host binding. You generated both sides, on two genuinely different pieces
of hardware in your own box, and what it demonstrates is exactly the claim the
library makes. It also works on any GPU mojolearn supports, not only the three
vendors we happened to record.

There is **one digest implementation**, the harness's own, used for both sides,
so there is no second comparison path that could drift from the first. The only
difference between the two arms is which binding answers.

It compares two different things, where the lane has both:

- **`infer`** is cross-vendor identity: same input, different hardware, same
  bits.
- **`batch`** is batch invariance: same row, different batch neighbours, same
  bits. That is a different axis, and the one that bites a serving system
  batching dynamically: a prediction that changes with traffic is a real
  problem. A lane that declares `n/a` for batch keeps its `n/a`.

**Scope.** The intersection is every lane with both a shipped GPU path and a
shipped host family, and that is **all 79 declared inference lanes: every one
is reachable from a binding the wheel already carries.** `quick` is one lane
per family on the base fixture, seconds. The default is up to 24 lanes,
minutes, capped because one Apple Metal process may not run a full column
outside a release, which `tools/identity_break.py` enforces rather than merely
advising. `all` runs the whole intersection, and on Apple is refused by that
same rule. `--lanes` and `--fixtures` narrow or widen any of them.

**On a CPU-only install it says so and exits 4.** There is no second piece of
hardware to compare against, so the cross-check did not run: that is neither a
pass nor a failure, and it is never silently skipped.

## Two strangers, with us out of the loop

The honest gap in everything above is that **we** published the reference
table. Nobody outside has rerun these lanes on their own hardware, and we
cannot manufacture that.

But once the evidence document exists, you can do it without us:

    # on a 4090
    python -m mojolearn verify --all --json-out mine.json
    # on an M2, someone else
    python -m mojolearn verify --all --json-out theirs.json
    # either of you, anywhere
    python -m mojolearn verify --compare mine.json theirs.json

It compares every cell hash in the two documents, prints the two provenance
blocks side by side, and says whether the machines were genuinely different.
If the hashes match, two people have demonstrated the claim **to each other**,
with us entirely absent. That is stronger than anything we can publish about
ourselves, and it needs no GPU, no bindings and no network to run.

### A comparer's one failure mode is agreeing too easily

This command is meant to be pointed at us, so the interesting question is not
whether it says `AGREE` when two honest documents match. It is every way of
getting `AGREE` **without** two machines having computed the same bits. Each
one below is its own outcome with its own exit code, and each was built and
watched before the code that catches it existed:

| result | exit | meaning |
|---|---|---|
| `AGREE` | 0 | every shared cell part carries the same hash, both sides computed it, and none is present in only one document |
| `MISMATCH` | 1 | at least one cell part differs; every differing cell is named with **both** values |
| `SELF-CONTRADICTED` | 1 | a cell part reads `MOVED`, `BATCH_MOVED` or `RELOAD-MOVED`, meaning that box gave two different answers for one fit. Two documents carrying the same such string hold the same text and agree only that the claim is false |
| `INCOMPLETE` | 4 | nothing differs, but the runs did not cover the same ground: a cell in only one document, a cell **neither** side computed (`value` null, where the probe raised), or two different `n/a` reasons |
| `SAME DOCUMENT` | 4 | the two files are byte-identical. That is one document handed over twice, and it can only agree with itself |
| `MALFORMED` | 2 | a file is not an evidence document, or names one cell part twice. A duplicated row would otherwise let a party paste the other's answer over their own and hide the loss |
| `SAME FILE` / `CANNOT READ` | 2 | both arguments are one path, or a file is missing or not JSON |

A cell recorded with the **same** `n/a` reason by both sides is an absence they
agreed on, counted separately from agreement. If both documents describe the
same device the output says so: that shows repeatability, not cross-hardware
identity. Nothing is ever summarized as a bare count; every differing,
self-contradicted, uncomputed and one-sided cell is printed by name, and a
truncated listing says how many it hid.

Every path out of the command prints exactly one `RESULT:` line and returns a
code from that table, including a crash, so an empty output or a failed
invocation can never be read as a pass. It dispatches **before** the import and
numeric-tier checks: under `MOJOLEARN_NUMERIC_MODE=fast` every other check
refuses with exit 3 and `--compare` still runs, because a third party has none
of our bindings.

## The evidence document

`--json` (and `--json-out PATH`) emits the run as data rather than a verdict,
for a reader who did not run it and should not have to take a word on trust:

- **every cell**, with the hash computed here, the hash expected, the verdict
  and the wall time, so a run can be audited or two runs diffed;
- **what produced the numbers**: the version and commit, and the sha256 and
  size of every binding actually loaded, host bindings included;
- **where each reference came from**: the committed column, its vendor and the
  commit it was recorded at, as a path under `bench/results/identity_break/`
  that you can open in this repository and re-run;
- **the self-test result**, in the same document, so one artifact shows both
  that the comparison caught a deliberately wrong answer and that the real
  lanes matched;
- **counts that cannot be misread**: lanes checked and skipped with the reason
  for each, portable models counted separately, and cell parts compared and
  refused.

The human report and the JSON are rendered from the same object, so they cannot
drift into describing different runs.

## How long it takes

The fixtures are the harness's own sizes (20,000 rows), which the records
were written at; they are not reduced for this command, because a smaller
fixture would hash different bytes.

| install | `--quick` | `--full` |
|---|---|---|
| CPU-only, Apple M4, one core | 2 s | about 25 min, measured (122 lanes, 9 fixtures, 4 models; 1,488 s summed over 17 chunked processes at one core) |
| CPU-only, x86 Linux (AMD EPYC, 8 vCPU) | 2 s | 6 to 12 s |
| Metal, Apple M4 | 12 s (26 lanes, base fixture) | more than an hour (about 180 lanes x 9 fixtures) |

On a GPU install `--full` holds the GPU for the whole run: spectral
clustering, HDBSCAN, the byte-level language model and the multi-class
boosting lanes take 40 to 140 s each on the M4. On a shared GPU run it in
pieces, `--lanes a,b,...` a group at a time, or check one area with
`--lanes` and `--fixtures base`. `--quick` is the few-seconds check.

**Maintainers: do not run a full Apple column routinely.** There is one Mac
with one GPU, only one Metal job runs at a time, and it cannot be rented or
parallelized, so a full Apple pass (over seven hours, measured) serializes
every other GPU need on that machine behind it. The Apple column is recorded
ONCE PER PyPI RELEASE (`docs/RELEASE_CHECKLIST.md` section 5b,
ENGINEERING_RULES.md section 12), and `tools/identity_break.py` refuses a
full-column Apple run that does not name the release in
`MOJOLEARN_APPLE_RELEASE_RECORD`.

For routine and occasional verification use the **rented CPU column**, which
is bitwise equal to Metal, runs in parallel and costs about $0.24/hour
(`tools/runpod_cpu_leg.sh`, `docs/RUNPOD_CPU_LEG.md`). The only question the
Apple column uniquely answers is whether the Metal backend agrees, and that
is a per-release question.

## Exit codes

| exit | meaning |
|---|---|
| 0 | `VERIFIED`: at least one part IDENTICAL, no part DIVERGENT, and **nothing REFUSED**. OWED and N/A parts are counted, not passed |
| 1 | `MISMATCH`: at least one part DIVERGENT. A wrong answer outranks an absent one, so this is read before INCOMPLETE |
| 2 | invalid usage |
| 3 | refused: the process loaded a tier other than IDENTICAL |
| 4 | `INCOMPLETE`, or cannot run: **any** judged part REFUSED, or the import raised, or the fixtures differ. A part that did not run is not a part that passed, and there is no number of parts that did run which makes up for one that did not |
| 5 | no reference: no table in this install, or every part OWED |

Before 2026-09-16 one IDENTICAL part outranked any number of REFUSED ones, so
an install missing most of its host bindings printed `VERIFIED, exit 0` having
checked 44 of 332 parts. That is fixed; a run in that state now reads
`RESULT: INCOMPLETE (verified 44 of 332 cell parts ...)` and exits 4.

## What a local run proves, and what it does not

A VERIFIED result says that this build, on this device, reproduced the
committed records' bits on every IDENTICAL part it ran. Because each
reference hash was recorded on other vendors' hardware (the report names
which), an IDENTICAL part on your machine joins those columns: your machine
and the recorded Apple, NVIDIA, AMD and CPU columns give the same bits there.

It does not re-measure the other vendors, cover inputs other than the
fixtures, cover lanes added after the table was generated (they read OWED),
or say anything about speed. OWED parts are claims no record backs yet. A
CPU-only install checks the CPU reference lanes and the portable models,
not GPU training.

## Sharing a report

```sh
python -m mojolearn verify --all --json > mojolearn-verify.json
```

The report carries the mojolearn version, the commit witness of the build,
the numeric mode, the vendor and device, the CPU model, the platform, Python
and numpy versions, the sha256 of every loaded binding, the table's sha256,
and for every cell part its value, its state and the reference columns it
was compared with (record directory, vendor label and commit). Attach it to
an issue as is.

## Regenerating the reference table (maintainers)

The table is `python/mojolearn/verify_reference/table.json`, generated from
the committed columns under `bench/results/identity_break/`; no hash in it is
typed. After a release record lands:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify --all \
  --emit-reference python/mojolearn/verify_reference/table.json
```

It admits a column only when it is identical mode, names a commit, ran the
default fixture size on one device, is not a sabotage, partial, probe or
smoke run, and its fixture hashes equal the current harness's. Per cell part
and device class the newest commit wins; a class that differs at an older
commit is kept as superseded, and classes that differ at the same commit
leave the part without a reference. `--records DIR` names other record
directories.

The portable models are saved on a GPU install and kept only when their file
bytes equal the table's model reference, so a shipped file is byte for byte
the file every recorded vendor wrote:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify \
  --emit-models python/mojolearn/verify_reference/models
```

## The pinned k-means card

`verify` without `--all` runs one pinned k-means fixture, captures its
ordered identity-trace card, and compares it with a reference card shipped
in the installation.

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify
```

Use `--json` for automation, `--all-stages` to list every divergent stage,
and `--keep` to retain a matching generated card. `python -m mojolearn env`
shows which binaries and mode the process selected without running the GPU.

| Exit | Meaning |
|---|---|
| 0 | `VERIFIED`: this build reproduced the selected reference card. |
| 1 | `MISMATCH`: the run completed and at least one recorded stage differs. |
| 2 | Invalid command usage. |
| 3 | Refused because the loaded FAST build makes no identity claim. |
| 4 | Could not run or judge: for example no GPU, missing binary, failed fit, or missing comparator. |
| 5 | No usable reference card is installed. |

A green card result proves only that this build and device reproduced this
reference fixture at its recorded checkpoints. Cross-vendor certification
additionally requires completed hardware legs and provenance; see
[the support matrix](../SUPPORT_MATRIX.md).

The fixture uses exactly representable FP32 inputs and exercises the pinned
k-means accumulation path. Check its input hashes without a GPU:

```sh
python -m mojolearn check-fixture
```

The repository uses `tools/identity_trace_diff.py` as its one card comparator.
It aligns stage tags before comparing dtype, element count, and raw-bit hash,
so structural and numerical divergences remain distinguishable.

### Maintainer card workflow

Never replace a reference card from one machine alone.

On the producing device:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify \
  --emit-reference /tmp/producer.card
```

On a second vendor, from the same source state:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify \
  --emit-reference /tmp/confirmer.card \
  --confirm-reference /tmp/producer.card
```

After transferring both cards to one checkout:

```sh
python -m mojolearn install-reference producer.card confirmer.card
```

The installer refuses missing provenance, source/profile disagreement,
placeholder fields, and divergent cards. Reference changes require the same
sabotage-sensitive review as any other numerical contract change.

For portable external-implementation artifacts, use
[conformance bundles](CONFORMANCE.md).
