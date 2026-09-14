# Verify mojolearn's identity claims yourself

Three recipes at three costs. Each ends in a pass or fail you can read without
trusting a sentence in this repository. What each one proves, and what it does
not, is stated beside it.

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

with no DIVERGENT, MOVED or ONE-COLUMN line. Exit 0 means every cell of your
column equals the three committed columns. Time: about an hour on an H100 or an
M4; longer on a 4090.

To do it by hand instead of through the script:

```sh
python3 -m pip install mojolearn==<version>
git checkout <commit from the record's README or any column JSON's "commit">
MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --vendor <box-label> --json mybox.json
python3 tools/identity_break.py --diff bench/results/identity_break/<record>/*.json mybox.json
```

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

## Two smaller checks

`python -m mojolearn env` prints which binaries and numeric mode the process
selected and the GPU it found, without running anything.

`python -m mojolearn verify` compares one pinned k-means card against a
reference card. From a PyPI install today it exits 5, no reference, because
the wheel does not ship the card; it works from a source checkout. Shipping
the card is an open item.

## What "identical" does and does not claim

Identical covers certified configurations and fixtures. A lane pins one
constructor call. A parameter value that selects a different kernel and is not
pinned by any lane is outside the claim until a lane pins it; the census
brief above lists them. The FAST tier, offered on the three tree estimators
only, makes no repeatability promise at all. Unsupported parameters raise by
name rather than run a different algorithm quietly; a knob that would be
accepted and ignored is a defect and is fixed by refusing it.
