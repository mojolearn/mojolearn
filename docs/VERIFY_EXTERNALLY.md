# Verify mojolearn's identity claims yourself

Four recipes at four costs. Each ends in a pass or fail you can read without
trusting a sentence in this repository. What each one proves, and what it does
not, is stated beside it.

Recipes 1 to 3 all end in a comparison against something WE recorded, so they
rest on our records being honest. Recipe 4 does not, and it is the strongest
evidence here: two people who have never met run the shipped verifier on their
own machines and compare with us absent. Read it even if you only ever run
Recipe 1.

The claim under test: for every public estimator at one certified configuration
(a lane), on nine hostile fixtures, training state, held-out predictions and saved
model bytes hash to the same value on an Apple M4, an NVIDIA H100 and an AMD
MI300X, and, for the lanes that have a CPU path, on a plain CPU too. The record is
`bench/results/identity_break/<date>_<n>-lanes/`: one JSON per box, a README
naming the boxes and commits, and `diff.*.txt`, the cell-by-cell comparison.
The lanes and their pinned configurations are the `@lane` functions in
`tools/identity_break.py`; a value that is not pinned there is not inside the
claim (`docs/lanes/BRIEF_claim_surface_census_2026-09-14.md` lists what is not).

## Recipe 1, no GPU, free: the CPU gates on GitHub's hosted runners

Fork the repository and run three workflows from the Actions tab. They use only
GitHub-hosted runners (Ubuntu x86-64, Ubuntu ARM64, macOS Apple silicon), need no
secret, build the CPU bindings from source with the pinned Mojo toolchain, and
compare against recordings committed in the tree.

| workflow | what a green run proves |
|---|---|
| `byte-lm-cpu-gate.yml` | the byte language model's CPU forward pass and one CPU training step reproduce the bytes recorded on the three GPUs for all 128 recorded steps, on that runner's CPU |
| `forest-host-gate.yml` | a CPU-only binding reproduces every prediction digest of the random forest, Extra Trees and gradient boosting models trained on the three GPUs (`bench/results/forest_host/`) |
| `cpu-identity-gate.yml` | the CPU training lanes (`COVERED_LANES` in the workflow) fit on the runner's CPU with every cell equal to the three committed GPU columns, and the classical CPU inference lanes reproduce the three-vendor recordings |

Each workflow also builds a sabotage arm, a binding with one arithmetic step
deliberately altered, and requires the same gate to FAIL on it. A gate that
cannot fail proves nothing; the sabotage step is how you know this one can.

Time: under an hour of free runner minutes. What it does not prove: anything
about the GPU columns themselves; those are inputs here.

## Recipe 2, one supported GPU: reproduce one column from the PyPI wheel

You need one of: an Apple silicon Mac, an NVIDIA GPU of compute capability 8.9
or 9.0 (RTX 4090, H100), or an AMD MI300 series (gfx942). Other architectures
are not in the wheel and the import refuses by name.

```sh
git clone https://github.com/mojolearn/mojolearn.git && cd mojolearn
sh tools/verify_external.sh <box-label>       # e.g. nvidia-rtx4090-sm_89
```

The script installs the wheel the record was made with, checks out the commit
the record's columns ran (so the lane list matches), runs every lane on the nine
fixtures twice under `MOJOLEARN_NUMERIC_MODE=identical`, and diffs your JSON
against the committed Apple, NVIDIA and AMD columns. Expected output ends in

```
summary: IDENTICAL=<n>
summary (infer/model): IDENTICAL=<m>, N/A=<k>
diff exit 0
```

and a last line `VERDICT: PASS`. The verdict is positive: the script exits 0
only when your column has no REFUSED and no MOVED cell and the diff found every
compared cell identical. A diff that reports nothing divergent over a column of
refusals is not a pass; the first outsider run (an H100 on 2026-09-14,
`bench/results/verify_external/`) had six refused radius cells behind a clean
diff, which is why the verdict is computed from your column too. Run the
script from a git clone, never from a source archive. Time: about an hour on
an H100 or an M4; longer on a 4090.

To do it by hand instead of through the script:

```sh
python3 -m pip install mojolearn==<version>
git checkout <commit from the record's README or any column JSON's "commit">
MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --vendor <box-label> --json mybox.json
python3 tools/identity_break.py --diff bench/results/identity_break/<record>/*.json mybox.json
```

The record and the wheel must come from the same commit. The script checks
the harness out at the record's commit so the lanes are the record's; a wheel
built from an older commit can refuse a lane the newer harness calls, which
reads REFUSED and fails the verdict. That is a real mismatch and the script
has no switch to hide it; it warns when the record's commit is not the wheel's
release tag. The two outsider runs on 2026-09-14 (`bench/results/verify_external/`)
show exactly this against the 0.8.5 wheel: six radius cells refused. From 0.8.7
the wheel ships the columns of a record taken at its own commit, so the recipe
is one command with no skew. (0.8.6 would have been the first to do so; it was
folded into 0.8.7 and never published.)

Reading a failure. `DIVERGENT` names a lane and fixture whose hash differs
between boxes: that is a finding, keep the JSON and open an issue with it and
`python -m mojolearn env`. `MOVED` means your box disagreed with itself across
the two fits, which is a determinism failure on your box before it is a
cross-vendor one. `REFUSED` with a message is a lane that raised; the message
says why (an unsupported architecture, a missing binding). `NOT-COMPARED` means
a JSON predates a column and is not evidence either way.

What it proves: the wheel's bytes on your box reproduce the record for every
lane, at those configurations, on those fixtures. What it does not: other
configurations, other shapes, the FAST tier (which makes no identity promise
and the tool refuses to measure), or architectures outside the wheel. Every
column JSON since 2026-09-14 records the sha256 of each binding it loaded under
`package.bindings`, so a column is tied to the bytes it ran, not only to a
commit; compare them against the wheel's RECORD file if you want that link too.

## Recipe 3, three vendors: reproduce the whole record

Rent one box of each kind and run Recipe 2 on each with its own label, then diff
the three JSONs together on any machine:

```sh
python3 tools/identity_break.py --diff apple.json nvidia.json amd.json
```

Container images that have worked: `nvidia/cuda:12.6.0-devel-ubuntu22.04` and
`rocm/dev-ubuntu-22.04` (the Linux wheel was built against those; the vendor-
neutral CPU binding linked differently on a 24.04 image on 2026-09-13, so stay
on 22.04). The Apple column needs a physical Mac; hosted macOS virtual machines
have no Metal.

Cost: two rentals of about an hour each plus a Mac. This is what the maintainer
does for every record; nothing in it is specific to the maintainer's accounts.

## Recipe 4, no GPU, free: two strangers compare their own documents

This is the only recipe we are not part of. Recipes 1 to 3 all end in a
comparison against a record this repository ships; if you do not trust the
record, they prove less than they look like they prove. Here, two people who
have never met each run the shipped verifier on their own machine, swap one
file by any means, and compare. Nothing we wrote is an input to the answer
except the code they both installed from PyPI.

```sh
# each party, on their own machine, on their own hardware
python3 -m pip install mojolearn==<version>
MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn verify --all --json-out mine.json

# swap the file by any means at all: email, a gist, a USB stick

# either party, on any machine; this step needs no GPU, no binding and no
# numeric mode, and it is reached before any of them is checked
python3 -m mojolearn verify --compare mine.json theirs.json
```

`--compare` takes the lane set from the two documents. It never enumerates,
greps or imports a lane list of its own, so it cannot quietly compare a
different set from the one either party ran.

**The verdict ladder, in the order it is tried. AGREE is LAST, and that
ordering is the point.** A comparer's one failure mode is agreeing too
easily, so every way of "matching" without two machines having computed the
same bits is broken out and given a non-agreeing outcome.

| verdict | exit | what it means |
|---|---|---|
| `MALFORMED` | 2 | a document could not be read as an evidence document. A DUPLICATE `(lane, fixture, part)` row makes the WHOLE document unreadable rather than last-wins, because last-wins would let one party paste the other's answer over their own. Nothing is compared, and this is **not** a pass |
| `SAME DOCUMENT` | 4 | the two files are byte-identical: one document passed twice, which compares nothing |
| `INCOMPARABLE` | 4 | the two runs do not share a recorded input contract: a different or missing harness digest, different fixture or held-out fingerprints, or a different property protocol. Equal hashes over different inputs are not an agreement |
| `MISMATCH` | 1 | at least one cell part both sides computed came out with different bits. This is a finding; keep both files |
| `SELF-CONTRADICTED` | 1 | no cell differs between the documents, but one side's box disagreed with ITSELF (`MOVED`, `BATCH_MOVED`, `RELOAD-MOVED`). Two such documents agree only on the claim being false |
| `AGREED ON A DIVERGENT ANSWER` | 1 | both sides carry the same hash for a cell that at least one of them judged `DIVERGENT` against its own reference table. They agree on an answer one of them already recorded as wrong |
| `INCOMPLETE` | 4 | a cell only one document carries, a cell neither side computed (the probe raised on both), or two DIFFERENT `n/a` reasons, which is a disagreement about what the part even is |
| `AGREE` | 0 | every shared cell part matches and none of the above applies |
| `NOTHING COMPARED` | 4 | the documents share no cell part at all |
| `SAME FILE` / `CANNOT READ` | 2 | both arguments are one path, or a file is missing or is not JSON |

Every path out of the command, a crash included, prints exactly one `RESULT:`
line and returns a code from that table, so an empty output or a failed
invocation can never be read as a pass.

The report also carries, next to the agreement count, each document's OWN
verdict about its own run, and `provenance.independent`, which is true only
when the two documents came from different devices, different device classes
and different files. Two parties can agree with each other while one of them
checked a fraction of what a reader assumes; that is the only honest place to
see it, so it is printed there.

What it proves: two machines neither of us controls computed the same bits for
every cell part both ran, under the same recorded input contract. What it does
not: anything about a cell only one side ran (that is `INCOMPLETE`, not a
pass), anything about lanes neither install can run, and nothing at all when
the verdict is not `AGREE`.

Time: minutes for the comparison; the two `verify --all` runs are whatever
each machine costs (about 25 minutes on one core of an Apple M4 for the
CPU-only lane set).

### Comparing against our own published document

We publish our own `verify --all --json-out` documents so that this recipe
works for someone who does not already know another user:
`bench/results/verify_reports/`, one per device class, each taken at the
commit named inside it.

**This is weaker than two strangers comparing, and it is not a substitute for
it.** Comparing against our document puts us back in the loop: you are
trusting that we ran what we say we ran on the hardware we say we ran it on.
Recipe 4 proper removes us from the answer entirely. Use ours to get started,
to check that your install produces comparable output at all, and as the
fallback when you have nobody to swap with. Then find a stranger.

## Two smaller checks

`python -m mojolearn env` prints which binaries and numeric mode the process
selected and the GPU it found, without running anything.

`python -m mojolearn verify --all` is the whole verifier and the command
recipe 4 uses. It runs every identity cell the install can reach (every lane
on a GPU install; `host_surface.public_reference_lanes()` on a CPU-only one)
plus the portable GPU-trained models, and compares each cell part against the
reference table shipped in the wheel at
`mojolearn/verify_reference/table.json`. It exits 0 only on `VERIFIED`, which
requires no divergent, no refused and no owed part and no withheld lane; a
part that did not run never counts as checked. `--json-out PATH` writes the
evidence document; `--coverage` inspects what the install would and would not
check, without fitting anything; `--self-test` shows that the comparison can
FAIL, by running one lane twice, once untouched and once with the input's
first column moved by one ULP, and requiring `IDENTICAL` then `DIVERGENT`.

`python -m mojolearn verify` with no flags is a much older and much smaller
check: it compares one pinned k-means stage card against a reference card in
`mojolearn/reference_cards/`. No build ships that card yet, only a
placeholder that refuses by name, so the bare command exits 5 (no reference)
and says so. That is deliberate rather than broken, and it is not the
identity claim's verifier; `--all` is.

## What "identical" does and does not claim

Identical covers certified configurations and fixtures. A lane pins one
constructor call. A parameter value that selects a different kernel and is not
pinned by any lane is outside the claim until a lane pins it; the census
brief above lists them. The FAST tier, offered on the three tree estimators
only, makes no repeatability promise at all. Unsupported parameters raise by
name rather than run a different algorithm quietly; a knob that would be
accepted and ignored is a defect and is fixed by refusing it.
