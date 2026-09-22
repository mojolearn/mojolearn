# Verify the identity claims on your own machine

This page describes the current verifier API. Each published wheel has its own
release evidence; source coverage alone does not qualify a distributed wheel.

Under the IDENTICAL tier, verification compares fixed fixtures with recorded
hashes and checks applicable local contracts, including batch invariance:
the same row must retain its output bits when its batch neighbors change.
A pass covers the selected fixtures, routes and properties; it does not prove
every input, parameter combination or GPU generation.

Inspect the installed package's scope before running the checks:

```sh
python -m mojolearn verify          # the quick sample (--quick) when no card ships
python -m mojolearn verify --coverage
python -m mojolearn verify --coverage --json-out coverage.json
python -m mojolearn verify --all --json-out verification.json
```

The inventory maps **all 246 algorithms and variants of the algorithm inventory**
to lanes or named alternative gates. It also lists every additional registered
lane, so newer features cannot disappear behind the historical count. Entries
are not distinct Python classes and mappings are not execution certificates.
For every lane the inventory shows CPU availability, withheld-reference reasons,
batch applicability and reference-fixture counts. It executes no algorithms.
The JSON also includes `lanes.<name>.reference_support`: for each property,
the number of fixtures with a numerical reference and matching CPU, Apple,
NVIDIA and AMD records. Missing/conflicted, stale and inapplicable fixtures
are counted separately. These counts describe the bundled reference table;
they do not certify the installed wheel or promote a pending lane.

The appendix is a historical list, not the total number of today's public
algorithms. `additional_lanes` names registered checks outside that list.
For the source/API/wheel comparison and the remaining qualification work, see
[the next-wheel coverage audit](NEXT_WHEEL_COVERAGE.md).

The candidate's default CPU selection now includes spectral clustering,
Fowlkes–Mallows, and both ARIMA-with-regressors variants. To replay just those
lanes, including applicable batch and saved-model checks:

```sh
python -m mojolearn verify --lanes spectral,metrics-fowlkes-mallows,arima-exog,arima-exog-seasonal --repeats 2
```

`--all` runs standard whole/individual/split/prefix batch checks where declared,
and the applicable step-versus-full sequence checks. For the additional gradient,
batch-size, ragged-batch and sampler/replay properties, opt in explicitly (these cost more work):

```sh
python -m mojolearn verify --batch-checks --lanes ols --fixtures base --json-out batches.json
```

The additional probes reuse each fitted model. A cell's `local_check: passed`
means its local equality assertions passed, not that it matched a trusted hash.
Without a usable reference it remains `OWED`, and the overall result is nonzero.
The newly exposed optional probes do not yet have bundled reference hashes;
their local checks can pass while reference comparison remains OWED.
Per-property results, including batch results, appear in both the human report
and JSON. A global statistic, corpus-wide vocabulary trainer, or fixed fitted
membership table has a stated reason where per-row batch invariance is inapplicable.

Missing reference parts, failed/refused parts, and withheld lanes in a full
request prevent an overall `VERIFIED` result. Explicit `--lanes` requests can
exercise pending declared CPU routes; they are not silently promoted into the
default qualified set. A successful explicit subset is only that subset.
BPE training and fold construction also run on CPU; they currently owe reference
hashes. Saved-model host APIs still have dedicated gates and portable-model
checks; the inventory does not claim full per-API wheel qualification for them.

The appendix's “seasonal-difference selection” entry maps to `select_d`, which
chooses ordinary order `d` with seasonal order `D` supplied by the caller.
Automatic selection of seasonal `D` is not implemented.

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
  (`host_surface.public_reference_lanes()`; use `--coverage` for the current count), fitted inside the
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
- **Portable models** run on every install: small models trained on a GPU
  and saved, shipped in `mojolearn/verify_reference/models/` (a random
  forest, a symmetric boosting model, a linear regression and a PCA, 179 KB
  together). A GPU install loads them with the class's `load`; a CPU-only
  install with `mojolearn.host_model(path)`. Their file
  bytes must equal the recorded model hash, and the loaded model's answers
  on the held-out rows, whole, row by row and split, must equal the recorded
  GPU answers.

### Every lane is public, and every lane says what this box did with it

The harness defines 262 lanes. Until 2026-09-20 the shipped verifier exposed
186 of them and the other 76 were not reported as anything at all, so
`186 lanes` was indistinguishable from all of them. Fifty-nine left by a
prefix rule and the rest sat in a pending list. **Nothing is hidden now.**
`PUBLIC_PENDING_LANES` and the inapplicable-prefix rule no longer remove a
lane from anything; they annotate. A reason decides what a lane REPORTS, and
whether this box can compare it, never whether you can see it.

The reasoning is short: an absence is indistinguishable from a feature we do
not have, and we do have these. A lane with no reference reads OWED, which
names what is missing. That is more than silence gives anyone.

Every run prints a LANE ACCOUNTING block whose denominator is the whole
harness, and gives each lane a state and a sentence:

| state | what it means | gates? |
|---|---|---|
| VERIFIED | it ran and every part read IDENTICAL or N/A | — |
| SMOKE | its full claim is not expressible on this box, but `--smoke` fitted it **twice** and it ran, kept its shape, stayed finite and gave the same bits both times. Never a verification | — |
| SMOKE FAILED | that smoke test found a raise, a NaN or inf, a changed shape, or different bits from two fits on one box. A real defect | **yes** |
| DIVERGENT | its bits differ from the reference | **yes** |
| REFUSED | a part raised, so it did not run | **yes** |
| OWED | no committed record carries a hash for a part of it | no |
| NOT APPLICABLE | no run on this box could ever state its claim. The 59 `par-*` drivers claim a two-device column hashes equal to the one-device column cell for cell, and `verify` is a one-device run; and a handful of lanes refuse by name on a CPU-only install | no |
| HELD | its comparison is blocked by a named condition | no |
| NOT RUN | this run did not select it, or its fixture moved past the shipped reference | no |
| UNDECLARED | nothing in the manifest says anything about it. This is a bug, and `tools/lane_accounting.py` fails on it | no |

**Only DIVERGENT, REFUSED and SMOKE FAILED cost the run its pass.** Those are the two that
mean something went WRONG rather than something is missing. A lane whose bits
differ from its reference is the single thing this library exists to detect,
and a lane that raised did not run at all; passing over either would make the
word mean nothing. Everything else is reported and counted, not held against
you. That is the same rule the part level has always followed: a part declared
`n/a:no-backward` has never made a run INCOMPLETE.

**`--smoke` is a looser tier, not a weaker one.** For a lane whose real claim
needs hardware you do not have — the 59 `par-*` drivers claim two devices hash
equal to one — it fits the lane **twice, in full**, and reports whether it
ran, kept its shape, stayed finite and produced the same bits both times. Two
full fits, never two probes of one fit: a second probe re-reads the same
arrays and cannot fail, and a cheap check that cannot fail is worse than none.
It is off by default because two full fits is real work, and it skips lanes
the run already executed, so on a GPU install it costs nothing.

**A run that checked nothing is still not a pass.** If a run's whole scope is
NOT APPLICABLE it exits 4 with CANNOT RUN, even though every cell part it
produced read IDENTICAL, because a one-device `par-*` column is compared
against itself and passes whatever the code does.

**The verdict carries its own scope**, which is what keeps a looser gate
honest. Nobody should have to open the JSON to learn that a quarter of the run
was inapplicable:

```
RESULT: VERIFIED (verified 16 of 20 cell parts (0 divergent, 0 owed, 0 refused,
4 n/a); 2 verified, 2 not applicable, 0 owed, 0 held, of 4 lanes). exit 0
```

Every cell (a lane on a fixture) has five standard parts; `--batch-checks` adds the four optional probes:

| part | question |
|---|---|
| train | does the fit read back the recorded bits? |
| infer | does the fitted model answer held-out rows with the recorded bits? |
| model | are the saved file's bytes the recorded bytes, and does the reloaded file answer like the model in memory? |
| batch | are the held-out answers the same whole, one row at a time, in an uneven split and by prefix, and equal to the recorded bits? |
| stepfull | for a model that decodes, is a sequence decoded ONE TOKEN AT A TIME with a carried state, at every position, the bits the same model answers when the whole sequence runs as one fresh-state forward pass? |

`stepfull` is the decode property, exposed on 2026-09-16. It is the claim an
inference server actually depends on, and the place bitwise determinism
usually breaks, so it is asked of the public classes on the machine you
installed on rather than only recorded by us. Eight lanes have it
(`transformer`, `transformer-window`, `mamba1`, `mamba2`, `mamba2-dtlimit`,
`mamba3`, `samba`, `samba-untied-dropout-accum`); every other lane reads
`n/a:no-decode-state`, which is a declaration rather than a pass. A failure
names the FIRST differing position and prints both values.

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

## Choose the verification cost and scope

Release certification builds native bindings and runs clean and injected-fault
sweeps on several architectures. It is a maintainer workflow, not a requirement
for each user installation. Choose a narrower installed-package command:

```sh
python -m mojolearn verify --coverage                    # inspect scope; no fits
python -m mojolearn verify --inference                   # bundled saved models; no fits
python -m mojolearn verify --training --lanes ols --fixtures base --repeats 2
python -m mojolearn verify --training                    # all available fit-based routes
python -m mojolearn verify --all --batch-checks --repeats 2 # comprehensive suite
```

`--inference` is an alias for `--models-only`. It covers the bundled models,
not every possible inference API or input. `--training` excludes that separate
bundled-model suite, but still checks applicable inference/reload/batch properties
of the models it fits. These scopes are mutually exclusive. `--quick` selects
one fit-based route per family; it is smaller, but it still performs training
and is not guaranteed to finish in seconds on every CPU.

Evidence applies to the tested artifact, implementation, inputs, properties and
hardware. Unchanged compatible results can be retained; changing native code,
bindings, numerical behavior or verification protocol requires the affected
checks again. Small targeted checks during development catch bugs before an
expensive final frozen-artifact certification.

The installed CLI defaults to a one-thread setting for supported CPU host and
math libraries. To request up to five threads for an inference check:

```sh
python -m mojolearn verify --inference --cpu-threads 5
```

The request is clamped to at most the detected available logical CPU count
minus one (minimum one). Process affinity is respected where available. Thread
settings are applied in a fresh interpreter before native imports; numerical
evidence records the selected verifier setting. This setting is not five
parallel algorithm fits and is not an OS CPU/RAM limit. Internally parallel
algorithms can have their own workers; a thread setting cannot guarantee that
a large model fits memory. Start with inference or a named training lane on
machines with limited resources. Read-only coverage/comparison commands do not
launch another numerical process for these settings.

There is no corresponding "five GPU cores" setting. GPU kernels are scheduled
across the selected device by its runtime. Workload size, memory, selected GPU
devices and concurrent jobs are the useful controls. Ordinary users need not
run multi-GPU or large-capacity stress tests to check their installation.

Small boundary fixtures remain valuable: duplicate/tied values, odd row counts,
values near numerical limits, and sizes around kernel or batch boundaries.
These can expose incorrect reductions or indexing that a large easy input
misses. The planned small training profile needs independently recorded hashes;
it is not implemented merely by lowering the old fixture size.

## Whole loaded-language-model inference

The expanded 0.8.7 source includes a small, separate composition check for
loaded Llama, Mistral, Qwen2, Qwen3, Phi3, Mamba and Mamba2 models. The v2 profile
has 24 cases, including tied/untied Llama heads, windowed attention, nonzero
QKV biases, QK normalization, fused checkpoint mapping, and FP32/BF16/int8 weights:

```sh
python -m mojolearn verify-causal-lm --device cpu --output cpu-models.json
python -m mojolearn verify-causal-lm --device gpu --output gpu-models.json
python -m mojolearn verify-causal-lm --compare cpu-models.json gpu-models.json
```

This checks complete logits, prefill versus carried-state decoding, batch
invariance, cache reset, reload and greedy outputs. Output paths must be new.
CPU threads default to one. The comparison requires matching fixture/profile
bytes; a successful capture reports `CAPTURED_UNQUALIFIED`, and matching
captures report `NUMERICAL_MATCH_UNQUALIFIED`. These are numerical evidence,
not automatic reference admission or a release certificate. Composition faults
are distinct from faults injected into native arithmetic. Refused architectures
(such as Gemma) are not covered by this profile. Physical multi-GPU execution
requires its own records; `--layer-devices 0 1` selects the experimental explicit
two-layer GPU mapping in this tiny profile.

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
| `--par [quick\|default\|all]` | run your two-device column once and compare it with the recorded one-device values (below) |
| `--par-devices 0,1` | with `--par`: which devices the second column runs on |
| `--par-self-test` | show that `--par` can fail |
| `--json-out PATH` | with `--all`: also write the evidence document to PATH |

The four checks answer different questions, and are worth more together than
separately:

1. **`--cross-check`**, your GPU against your CPU. Trusts nobody: you generated
   both sides on your own machine.
2. **`--all`**, your machine against our recorded columns. Trusts our table,
   which is auditable because the raw columns are committed under
   `bench/results/identity_break/` and the document names the exact file and
   commit each reference came from.
3. **`--par`**, your two-device column against your one-device column, for the
   59 `par-*` multi-device driver lanes. Trusts nobody for the same reason
   `--cross-check` does not, and it is the only command that states the
   drivers' claim at all: every `par-*` cell in the recorded table is a
   ONE-device run, so `--all` cannot ask this question.
4. **`--self-test`** and **`--par-self-test`**, which show the comparisons can
   fail at all. Without them the first three are checks nobody has watched
   fail.

## Your two devices against your one device

A `par-*` lane is a multi-device DRIVER: it cuts one fit or one query into
shards, sends each shard to its own device, and merges the results. Its whole
claim, in `identity_break._par_devices`, is that the two-device column hashes
equal to the one-device column cell for cell. Sharding must not move a bit.

That claim is not in any release record. `identity_break.RECORD_EXCLUDED_PREFIXES`
keeps `par-*` out of a full column, and every `par-*` reference in the shipped
table is a one-device run, so `verify --all` compares one device against one
device and passes whatever the sharding does.

```sh
python -m mojolearn verify --par                  # all 59 par-* lanes, base fixture
python -m mojolearn verify --par quick            # one lane per family
python -m mojolearn verify --par --lanes par-queries-nn --fixtures base,ties
python -m mojolearn verify --par-devices 2,3      # another pair
```

**Which scope to run.** `--par quick` is the two-physical-GPU check: one lane
per family on one device and on two, with the placement witness, in under a
minute of lane time. Run it, with `--par-self-test`, on every routine run and
every rented box. **Never run `--par all` without the maintainer's express
permission.** It is every lane on all nine fixtures: measured on two RTX 4090s
on 2026-09-20, when each lane still ran twice, at about 16 minutes per fixture
and 2.4 hours in all, with `par-resample` alone taking about 300 s of each
fixture. The default scope (every lane, base fixture) also needs a reason.

It runs each lane ONCE, with `MOJOLEARN_PAR_DEVICES=0,1`, and compares the
result with the one-device values the shipped reference table already records
for that lane on your vendor class. Fitting the one-device column again on
your box bought nothing and doubled the run, so it is gone.

**A column that never sharded is refused, not passed.** If a driver silently
falls back to one device, both columns agree instantly and a naive command
would print a confident pass over nothing. So every device pool the two-device
column starts is inventoried through the driver, each worker is required to
resolve to its own process and its own physical GPU by UUID and PCI id, and a
run in which no pool started at all is reported as CANNOT RUN with that
sentence.

The byte-level language-model drivers start no pool: they hand their device
list to the native binding, which opens one device context per device inside
the calling process. Those sessions are inventoried as native sessions. One
counts as a witness only when the binding's own ownership rows put resident
bytes on every requested device and the process's driver inventory shows those
ordinals as distinct physical GPUs. `par-byte-lm-offload` is different again:
its driver admits exactly one device, so both of its columns run on the first
device. Its parts read `N/A` with that reason and are never counted as a match;
a part that differs still gates.

`DIVERGENT` means sharding changed the bits. `ONE-COLUMN` means the two-device
run refused a part the record has a value for (or the reverse), which is a
defect only a two-device run can see. Both exit non-zero.

**A run that ran nothing is not a pass.** The result is a ladder, worst first,
and it is the one `verify --all` already uses:

| result | when | exit |
|---|---|---|
| `MISMATCH` | a `DIVERGENT`, `MOVED` or `ONE-COLUMN` part. A wrong answer outranks an absent one, so this is read first | non-zero |
| `INCOMPLETE` | nothing disagreed, but a part `REFUSED` on BOTH columns. Absence is not agreement, and the parts that did run do not make up for the ones that did not | non-zero |
| `NOTHING COMPARED` | no part produced two hashes. A column of `N/A` and `CPU-ROUTE-LIMIT` establishes nothing | non-zero |
| `VERIFIED` | at least one part was compared and every compared part agreed | 0 |

Until 2026-09-20 the verdict was "no gating verdict was counted", and `REFUSED`
does not gate, so a run in which every part refused on both columns printed
`THE TWO COLUMNS AGREE on 0 compared cell parts` and exited 0. Measured on a
CPU-only install outside the reference-training scope: 65 refused parts, exit
0, `VERIFIED`.

**Apple is excluded structurally, not by policy.** `DevicePool._start` admits
the metal vendor only at a single device and raises for any other group, so no
Mac can produce a two-device `par-*` column with this code. The command says so
by name rather than leaving you to look for hardware that would never have
worked. On a CPU-only install it runs the drivers' own partition and merge
across two worker processes and states, in the report, that this is not a
device claim.

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

### Owed parts the host sabotage cannot move (maintainers)

The CPU identity gate (`.github/workflows/cpu-identity-gate.yml`) has a
negative control of its own. It builds every host binding again with
`-D MOJOLEARN_HOST_SABOTAGE=1`, reruns the covered lanes on that set, and
`tools/cpu_identity_gate_check.py owed` requires every OWED cell part (one no
committed GPU record hashes yet) to move between the production and the
sabotage CPU columns. A part that does not move rests on nothing.

A few parts cannot move, because the sabotage faults never reach the bytes
they hash. They are listed, one entry per lane and part with its fixtures and
its reason, in `SABOTAGE_EXEMPT_ENTRIES` in `tools/cpu_identity_gate_check.py`.
Measured on gate run 35728134044, the same 92 parts, and only those, stayed put
on the x86, ARM64 and macOS runners.

| Lane | Part | Fixtures | Why the sabotage does not reach it |
|---|---|---|---|
| `radius`, `radius-manhattan`, `radius-chebyshev`, `radius-minkowski-p3` | model | all 9 | the saved file is the training rows and the knobs; fit computes nothing and the ball cover is built per query |
| `iforest` | model | all 9 | the saved file is the training matrix, the knobs and a constant `offset_`; the forest is rebuilt on every scoring call and never saved |
| `rbf-sampler` | model | all 9 | the saved Philox weights and offsets depend only on `n_features` and the seed; the sabotaged GEMM is in `transform`, which is not saved |
| `kernel-ridge-poly` | model | 8 (all but `negative`) | reached but inert, since the one ulp flip in the kernel matrix does not survive the `alpha=64` solve into the float32 dual |
| `standard-scaler-no-std` | model | `ties` | reached but inert, since the saved mean is a sum of small integers, exact in float32 in any order |
| `optim-sgd` | batch | all 9 | the batch arms step SGD without a clip, which is elementwise; only the clip norm is a reduction the sabotage moves |
| `select-d` | train, batch | all 9 | the result is an integer differencing order, and a one ulp change in the KPSS statistic crosses no threshold |
| `agglomerative` | batch | `denormal`, `denormal_ftz` | reached but inert, since the 64 held out rows get the same labels with and without the fault |

**These parts are not covered by the sabotage control.** The owed check admits
them on the production CPU hash and the agreement of the other columns alone.
Every other part of these lanes still moves, with one exception. `select-d`
moves in no part at all, so that lane has no negative control in the CPU gate.

The check prints each exempted part as `EXEMPT` and counts them in its verdict
line, for example `owed verdict OK (4111 of 4203 owed cell part(s) moved, 92
exempt part(s) did not move as listed, 0 failure(s))`. An exempted part that
does move fails the check. That is deliberate, not strict. The fixtures are
seeded and the three runners agree, so a move means a code change now lets the
fault reach the part, and the entry is stale.

To remove an entry, add a fault the sabotage build reaches that part with (for
example a value flip in the code that writes the saved file, or a
`MOJOLEARN_HOST_SABOTAGE` arm on the KPSS decision), rerun the gate, and delete
the entry once the part moves. A failing `EXEMPT but MOVED` line asks for the
same deletion.

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

### If you have nobody to swap with

`bench/results/verify_reports/` carries OUR OWN evidence documents, one per
device class, each taken at the commit named inside it, so
`verify --compare mine.json ours.json` works on the day you install the wheel.

**It is weaker than two strangers comparing and it does not replace it.**
Comparing against our document puts us back in the loop: the answer then
depends on our having run what we say we ran, on the hardware we say we ran
it on. The paragraph above is the protocol that removes us; this is the
fallback for someone who has not yet found a second party, and a way to check
that your install produces a comparable document at all. The comparison is
also `INCOMPARABLE` rather than an agreement whenever the harness digest, the
fixture fingerprints or a property protocol differ, so a document of ours from
another commit will refuse to agree with yours rather than appear to.

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
| `AGREED ON A DIVERGENT ANSWER` | 1 | both sides hold the same hash for a cell that one of them judged DIVERGENT against its own reference table. Two machines reaching the same wrong answer is a finding, not a pass |
| `INCOMPLETE` | 4 | nothing differs, but the runs did not cover the same ground: a cell in only one document, a cell **neither** side computed (`value` null, where the probe raised), or two different `n/a` reasons |
| `SAME DOCUMENT` | 4 | the two files are byte-identical. That is one document handed over twice, and it can only agree with itself |
| `MALFORMED` | 2 | a file is not an evidence document, or names one cell part twice. A duplicated row would otherwise let a party paste the other's answer over their own and hide the loss |
| `COMMITMENT BROKEN` | 1 | a document is not the one its party committed to before the exchange. Only `MALFORMED` outranks it: until you know a document is the one that was committed to, no headline about its cells is honest |
| `CHALLENGE BROKEN` | 1 | a challenge is answered and the answer does not hold up: only one document carries one, the two answer different challenges, the challenge is not what its own pair of commitments hashes to, that pair is not the one in hand or the one published, a response does not match the challenge commitment published for it, the two carry the same challenge nonce, or the two answered over different input. It sits directly under `COMMITMENT BROKEN`, and above every cell outcome, because until it is settled the reader does not know these cells were computed rather than written out of the shipped table. Absence of a challenge never lands here |
| `SAME FILE` / `CANNOT READ` | 2 | both arguments are one path, or a file is missing or not JSON |

A cell recorded with the **same** `n/a` reason by both sides is an absence they
agreed on, counted separately from agreement. If both documents describe the
same device the output says so: that shows repeatability, not cross-hardware
identity. Nothing is ever summarized as a bare count; every differing,
self-contradicted, uncomputed and one-sided cell is printed by name, and a
truncated listing says how many it hid.

`AGREE` is the **last** outcome tried, and the exit-1 outcomes are read before
the exit-4 ones. That ordering is the same one `verify --all`'s own verdict
carries: a wrong answer outranks an absent one, and no number of parts that did
run makes up for one that did not. Each document's own verdict and detail line
are printed beside the comparison, because two parties can agree while one of
them checked a fraction of what a reader assumes.

Every path out of the command prints exactly one `RESULT:` line and returns a
code from that table, including a crash, so an empty output or a failed
invocation can never be read as a pass. It dispatches **before** the import and
numeric-tier checks: under `MOJOLEARN_NUMERIC_MODE=fast` every other check
refuses with exit 3 and `--compare` still runs, because a third party has none
of our bindings.

### Commit first, then exchange

Every outcome in that table is about a document that is malformed or that
contradicts itself. None of them is about a **well formed** document whose
numbers were never computed by the machine it names, because the party wrote
them down after reading the other party's file. Whichever side receives the
other's document first can paste its cell values under their own provenance,
and `--compare` reads a clean `AGREE` across two "independent" vendors.

Commit-reveal closes that, and it is the cheap standard fix: before either
party can see the other's file, each publishes a hash of their own document
under a random nonce they hold back.

    # each party, on their own machine, before anything is exchanged
    python -m mojolearn verify --all --json-out mine.json
    python -m mojolearn verify --commitment mine.json     # prints one line

Publish that 64-character line however you like -- a message, a mailing list
post, a gist, a git tag. There is no transport here and there must not be.
Then exchange the documents, which carry the nonces, and either party runs:

    python -m mojolearn verify --compare mine.json theirs.json \
        --commitment-a <the line you published> \
        --commitment-b <the line they published>

Each argument takes the line itself or a path to the `.commitment` file
written beside the document. A party who published a commitment cannot copy,
because what they are bound to was fixed before there was anything to copy.

**What the commitment covers** is the whole design, and the reason lives in
`commitment_preimage` in `python/mojolearn/_verify_all.py`, next to the code,
not here. In short: exactly the fields `compare_documents` reads -- the
format, every cell row, the whole device block, the verification contract,
the binding digests and the document's own verdict and detail line -- and
nothing else. Cells alone would leave a forger free to swap the provenance,
which is the half they never needed to change. The file's bytes would break
on re-indenting, and a check that fires on honest handling is a check people
learn to click past, so it hashes parsed values re-serialized canonically and
excludes the wall-clock fields by name.

**A comparison without commitments is not an error.** Two strangers with two
files and no prior arrangement still get the full comparison and exit 0. What
changes is that the output says what it does not show, in the same place and
the same voice `same_device` is called weaker than `independent`:

    COMMITMENTS: none were exchanged. Nothing here shows that either document's
    numbers were COMPUTED by the machine it names. Whichever party received the
    other's file first could have pasted its cell values into a document
    carrying their own provenance, and this command cannot tell that apart from
    two honest runs. This result is WEAKER for the same reason two documents
    from one device are weaker than two vendors.

and the `RESULT: AGREE` line itself carries `WEAKER THAN IT LOOKS`, so a
reader who greps for `RESULT:` cannot get the strong sentence without the
reason. A commitment carried **inside** a document is reported as
self-declared and never as a check that passed: the nonce ships with it, so
anyone holding the document can recompute it. One commitment presented for
both documents, and two documents carrying the same nonce, are refused.

### The challenge: commit, exchange the lines, then run

**What commit-reveal does not close.** It stops a party copying after seeing.
It does not stop a party synthesizing a document from the reference table
shipped in this wheel, which pins the expected hash of every cell on every
pinned fixture, and committing to that without running anything. A commitment
is evidence about ORDER, never about execution.

The challenge closes that, and the protocol grows one round for it:

    challenge = sha256("mojolearn.verify-challenge.v1\n" || min(c_a, c_b) || max(c_a, c_b))

Both parties publish their commitments, exchange those two lines, and only
then run

    python -m mojolearn verify --challenge mine.json \
        --challenge-from <your line> <their line>

which reseeds `identity_break.fixture('hashed', seed=...)` from the challenge,
reruns the document's own lane set on it, appends a `challenge` block and
prints a SECOND line to publish before the documents are exchanged. Those
hashes are in no table we ship, so they cannot be written in advance; and
because `--compare` reads no table -- it compares two documents to each other
-- unreferenced cells are exactly what it can check. The block is kept out of
`cells` so `verify --all`'s own verdict, which does judge against the table,
never sees a cell it has no reference for.

The second line is not ceremony: two honest responses are IDENTICAL, so a
response nobody was bound to can be copied out of the other party's file after
it arrives. Its digest has its own domain separator and covers the document's
round-one commitment, so it cannot be presented for a different document. The
challenge block is deliberately EXCLUDED from the round-one commitment: it is
derived from that commitment and so arrives after it, and a check that fired
on the protocol working would be worse than no check.

**What it does not prove**, in the same breath as what it does: it does not
prove two PEOPLE (one party with one machine can answer once and write both
documents around it, and the device blocks are self-reported); whoever
publishes their commitment LAST can grind nonces to steer the challenge, so
the fixture is not an unbiased draw, though every candidate still has to be
run to be answered; and it proves execution of what it covers, on one fixture
of one kind, so a party who answers honestly and writes the rest of the
document out of the table still passes.
`docs/VERIFY_EXTERNALLY.md` carries the adversary model as a table.

The comparer takes its lanes from the two documents and never enumerates,
greps or imports a lane list of its own, so it cannot grow a second idea of
what the lane set is. Where a lane list **is** needed, as in `--cross-check`,
it is read from the registry by import, the same way `tools/lane_select.py`
and `tools/verification_matrix.py` read it.

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

**Maintainers: do not run a full Apple column routinely or for every alpha
release.** Use the exact-wheel light smoke and target numerical checks at changed
arithmetic, missing references, or a specific observed failure. A historical full
Apple pass exceeded seven hours; it is not a publication requirement. The bounded
alpha policy is documented in `docs/lanes/RELEASE_PROCESS_ALPHA.md` and
`ENGINEERING_RULES.md` section 12. Full-column diagnostic guards remain in place.

Routine checks can use an implemented CPU route. A CPU result does not establish
Metal correctness, and a GPU-only route has no CPU evidence merely because the
corresponding serial algorithm exists. Preserve successful compatible records;
repeat only the affected work and the final new wheel's small installed smoke.

## Exit codes

| exit | meaning |
|---|---|
| 0 | `VERIFIED`: at least one part IDENTICAL; no DIVERGENT, REFUSED or OWED parts, and no withheld lanes in the requested full scope. N/A is inapplicable, not a passed test |
| 1 | `MISMATCH`: at least one part DIVERGENT. A wrong answer outranks an absent one, so this is read before INCOMPLETE |
| 2 | invalid usage |
| 3 | refused: the process loaded a tier other than IDENTICAL |
| 4 | `INCOMPLETE`, or cannot run: **any** judged part REFUSED, or the import raised, or the fixtures differ. A part that did not run is not a part that passed, and there is no number of parts that did run which makes up for one that did not |
| 5 | missing reference or incomplete scope: no table, any OWED part, or withheld/stale lanes in the requested full scope |

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

New tables require at least two identical, valid digest samples per part,
matching input witnesses on each cell, held-out witnesses on non-training
parts, and matching protocols for batch and other numerical properties.
A probe error refuses the part even if it also returned a matching hash.
Historical tables remain readable; `--coverage` explicitly labels a table
without the new admission policy as legacy. Regenerate and replay references
from the release artifacts before treating them as release qualification.

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

When the installation ships only the card placeholder, bare `verify` runs
`verify --quick` against the reference table instead, says so in one line,
and points to `verify --all` for the full check. A request that names the
card path (`--all-stages`, `--keep`, `--reference-name`, `--emit-reference`,
`--confirm-reference`) still exits 5 when no card is installed.

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

### Recording references for the optional batch checks

Maintainers can include the extra recorded parts with
`verify --all --batch-checks --emit-reference candidate-table.json`, using
the existing record-selection arguments. The source columns must actually
carry `batchgrad`, `batchscale`, `ragged` and `rlpair` results. The builder
uses the same admission, fixture, revision and conflict rules as the five
standard parts; missing observations remain missing. Generating a table is
not final-wheel qualification or permission to publish it.

### Historical evidence and verification inputs

The 0.8.7 candidate's `verify --coverage --json` includes source paths and
SHA-256 digests for historical backend records, build versus harness negative
controls, and matched one-device/multiple-device records. It retains all 246
appendix entries. These are recorded observations on named fixtures and parts,
not qualification of the installed wheel. Missing evidence stays visible.
A parallel lane is not a multi-GPU pass merely because it exists or ran on
one device. The report names requested device indices; it does not independently
attest physical device use.

New execution reports include `verification_contract`: input and held-out
fingerprints, the harness digest, and the actual batch/decode protocols.
`verify --compare A.json B.json` refuses missing or differing contracts with
`INCOMPARABLE` (exit 4), even when output hashes happen to match. Older reports
without these fields must be regenerated before this comparison can pass.

The CPU verifier now bundles the 18 GPU-trained CTR fixtures used by
`gbdt-categorical-ctr-tables` and `gbdt-tensor-ctr-tables`. Both builders validate
their SHA-256 digests. CPU replay finds and validates them automatically; an
external model directory is no longer needed. These lanes test CPU inference
of GPU-trained models, not CPU training of CTR tables. Both lanes passed all
nine fixtures twice from an installed development wheel and join the default
CPU set. This does not replace final release-wheel qualification.

See [the evidence audit and remaining work](lanes/LANE_STATUS_verification_evidence_audit.md).

### Admitting one verified group without rewriting other references

To generate a scoped candidate while preserving every other lane, combine
`--lanes`, `--reference-table` (the base) and `--emit-reference` (the output):

```sh
python -m mojolearn verify --all --lanes lane-a,lane-b \
  --reference-table python/mojolearn/verify_reference/table.json \
  --emit-reference /tmp/candidate-table.json
```

This requires strict generated evidence for every fixture and core part, refuses
conflicts, and refuses to drop an existing part. Unselected cells and record
indices remain unchanged. Coverage reports the admission policy per updated
lane; the whole table keeps its prior policy because adding a few strict lanes
does not qualify legacy references. Replay the candidate from installed wheels
and review native controls before promoting a lane's default availability.


## Broader execution before qualification

`python -m mojolearn verify --include-pending --batch-checks --json` also
executes declared CPU routes whose references are not yet qualified, including
CPU-supported parallel drivers. Pending lanes remain INCOMPLETE; stale
reference bytes are never compared to changed fixtures. Actual hashes and
local invariance outcomes are retained as OWED when no usable reference
exists. A local mismatch still reads DIVERGENT and a missing operation still
reads REFUSED. Use `--lanes`, `--fixtures base` and `--repeats 2` to bound a
particular run. This option does not promote references or certify a release.

CPU parallel execution exercises logical shard splitting, original native
operations and ordered result assembly. It does not simulate or qualify GPU
communication. Coverage names CPU-supported drivers and drivers that still
require GPUs; a broad full request retains the missing drivers as scope gaps.

`python -m mojolearn verify --models-only --json` executes the four bundled
GPU-trained saved models without training: HostForest, HostGBDT, OLS and PCA.
It checks serialized model bytes and batch invariance of loaded-model outputs
against independent reference-table entries. The ordinary `--all` run also
includes these probes unless `--no-models` is specified. The two HostForest /
HostGBDT appendix entries now name this installed command and asset availability
in `--coverage`. These representative models do not replace all configurations
in the separate host gates. `--models-only` does not run the training-based
self-test; run `verify --self-test` separately.

Normal lane verification checks repeated training outputs, held-out inference,
save/reload, row-batch invariance and step-versus-full decoding where declared.
`--batch-checks` adds gradient accumulation, batch-size, ragged sequence and
sampler/replay properties. Non-applicable properties retain their named reason;
a missing reference is OWED, never a successful check. `--cross-check` compares
GPU inference with CPU saved-model inference on the same machine. `--compare`
compares independently produced evidence documents. Native fault controls and
physical multi-GPU pairs remain historical evidence until rerun on the actual
release artifacts; the ordinary verifier self-test is not a substitute.

### Two-GPU distributed checks in the installed package

The expanded 0.8.7 candidate includes a separate, opt-in command. Its fixture
generator requires NumPy, available through the optional `mojolearn[test]` extra:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify-distributed --devices 0,1 --out distributed.json --require-installed
python -m mojolearn verify-distributed --compare first.json second.json
```

This small suite covers ARIMA prediction, Holt-Winters forecasting, Gaussian
process classification and disjoint IVF search. It checkpoints each completed
case, compares one-device, two-device and reversed-device results, and exercises
actual dropped/reordered worker-result controls. Output paths must be new;
retain partial reports if execution stops. `--require-installed` checks loaded
Python and native bytes against the installed wheel's RECORD.

The default is one CPU math thread per worker; GPU indices name whole devices,
not GPU cores. Driver UUID/PCI identities and numerical worker activity establish
placement. They do not replace independent GPU kernel traces, native arithmetic
sabotage, or release qualification. Loaded-language-model checks use the separate
`verify-causal-lm` command above.

GPU cross-validation has its own installed check:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify-cross-validation --devices 0,1 --out cv-evidence --require-installed
```

It checks fold models, predictions and scores across device orderings and
repetitions, retains intermediate evidence, and validates installed file bytes.
Like the distributed suite, it requires the optional NumPy test dependency.
Physical GPU traces and native sabotage remain separate qualification gates.
