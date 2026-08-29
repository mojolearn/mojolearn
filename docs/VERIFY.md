# `python -m mojolearn verify`

mojolearn's distinguishing property is **cross-vendor bitwise identity**. One
source compiled for Metal, CUDA and HIP produces byte-identical FP32 results
on Apple, NVIDIA and AMD, so a model trained on an M4 and a model trained on
an H100 come out with the same bits. `IDENTITY_PATHS.md` is the enumeration
of every pathway that can move a bit and what the IDENTICAL arm does about
each. `E3_RESULTS.md` is the whole-library record across three vendors.

Both of those are documents. This command is a check.

```
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify
```

It runs one small pinned fit, writes the stage card that fit emits, compares
that card against a reference card shipped in the wheel, and prints a verdict
with the provenance of the reference beside it. It takes seconds. It needs no
account, no dataset and no reading.

---

## What it prints, and what each verdict means

| exit | verdict | meaning |
|---|---|---|
| 0 | VERIFIED | this build reproduced the reference card stage for stage |
| 1 | MISMATCH | the fit ran and the card differs; the first diverging stage is named |
| 2 | USAGE | bad arguments |
| 3 | REFUSED | this process loaded the FAST binaries, which make no identity claim; nothing was judged |
| 4 | CANNOT RUN | no GPU, no identical binaries, the fit raised, the trace never reached the binary, or no comparator was found |
| 5 | NO REFERENCE | this installation ships no usable reference card, or the one it ships is still the placeholder |

The codes are part of the interface because people put this in continuous
integration. Note that 3, 4 and 5 are all distinct from 1. A build that
cannot be checked is not a build that failed, and conflating the two would
make a red line meaningless.

---

## What a local run proves, and what it does not

**It proves** that this build, on this device, in this numeric mode,
reproduced a card that was produced somewhere else. The card is an ordered
list of FNV-1a64 hashes over raw FP32 bit patterns at named points inside one
fit (`core/identity_trace.mojo`).

**It does not prove** a cross-vendor property, and it cannot. One machine has
one GPU. The cross-vendor content of a VERIFIED result lives entirely in
where the reference card came from, which is why the reference is required to
carry the run, the commit, the numeric mode and the hardware that made it,
and why the command prints all of that every single time. A reference hash
with no provenance is a number, not evidence.

**It does not prove** that the computation was identical, only that the
buffers agreed at the checkpoints. Two different orders of summation can
round the same way on one fixture and diverge on the next. That is the exact
reason `IDENTITY_PATHS.md` enumerates pathways by mechanism and the trace
only localizes what the enumeration missed. The instrument finds where. The
ledger says why.

**It does not prove** anything about the rest of the library. It is one
k-means fit.

The command prints all four of those limits under WHAT IT DOES NOT PROVE on
every verdict, including a green one.

---

## The FAST build makes no identity claim, so `verify` refuses on it

`python/mojolearn/_mode.py` selects one of three binary sets that ship in
the same wheel.

```
python/mojolearn/_mojolearn*.so                NUMERIC_FAST           the default
python/mojolearn/deterministic/_mojolearn*.so  NUMERIC_DETERMINISTIC  run-to-run stable
python/mojolearn/identical/_mojolearn*.so      NUMERIC_IDENTICAL      the claim
```

The FAST arm is fast precisely because it keeps the order-dependent
operations the IDENTICAL arm replaces. Float atomics, vendor
transcendentals, whatever the compiler chose to contract. Two FAST runs on
two vendors are **expected** to differ, and any agreement between them is
luck rather than a property.

So on a FAST build this command exits 3 and prints REFUSED. It does not
compare anything, does not print a hash that could be quoted as a pass, and
tells the reader what to set. This mirrors what the project's own round judge
already does with FAST cards, which it records and never judges
(`tools/e3_round_judge.sh` section 7, "FAST cards recorded, never judged").
DEVIATION 923.

The mode can also be set in code, `mojolearn.set_numeric_mode('identical')`
before calling `verify`. The environment variable sets the starting default
for the session and nothing more.

### How the mode is established, which is weaker than it should be

DEVIATION 920, stated in the open. The fixture runs through the `_mojolearn`
extension, and that extension exposes **no** compile-time numeric-mode
function, the way `_mojolearn_gbdt` exposes `gbdt_numeric_mode`. The arm of
the binary doing the work is therefore inferred from three legs rather than
read from it.

1. `_backend.numeric_mode()`, which is what the process asked for and got.
2. The gbdt binary's own compile-time answer, which `_backend` cross-checks
   against the selector and raises on when they disagree. That binary is a
   sibling loaded from the same directory in the same call, so it is evidence
   about the directory rather than about the fixture binary.
3. The resolved path of the loaded `_mojolearn` extension, which in identical
   mode must sit under `<package>/identical/`. This is the only leg that
   touches the binary actually running the fit.

All three have to agree or the command exits 4. Three indirect legs is weaker
than one direct read, and the fix is a few lines in
`bindings/_mojolearn.mojo` exposing a `kmeans_numeric_mode` the same way the
gbdt binding does. Until that lands, this paragraph is the honest description
of what the mode line means.

---

## The fixture, and why this one

It is the **E1U unsupervised k-means fixture**, coordinate for coordinate,
from `bench/unsupervised_trace_main.mojo`.

```
k-means   n = 4096   d = 8   k = 8
init      array (the starting centroids are rows of X, row c * 37)
n_init    1
max_iter  10
tol       1e-4
seed      7
weights   unit (sample_weight stays None)
```

Reused rather than invented, for three reasons.

**It is already pinned and it has already crossed three vendors.** Its input
hashes are recorded in every E1U leg's `kmeans.hashes`, and the two constants
baked into `_verify.py` are copied from
`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/e1u/kmeans.hashes`.

```
input.x           14006717752511810141
input.centroids    2727533609010192784
```

Before any fit runs, `verify` rebuilds the fixture in numpy and checks both
hashes. A card comparison against different input bytes measures nothing, and
this is the cheapest possible place to discover that. If either differs the
command stops with CANNOT RUN and says which. DEVIATION 922. The gate is
runnable on its own, with no GPU and no extension call:

```
python -m mojolearn check-fixture
```

**Every coordinate is exact in Float32.** Each value is a 16-bit integer over
256, which has an exact binary representation, so no backend can round the
*input* differently and every card difference is a difference in the
computation. That is the property a cross-vendor fixture actually needs.

**It reaches the mechanism rather than a convenience.** The k-means centroid
accumulation is IDENTITY_PATHS' REPLACE move in its purest form, a float
atomic sum swapped for a fixed-point Int32 accumulator (rows 19 to 21,
deviations 503, 504 and 508). The card records `sums_i32` and `weight_i32`
per iteration, so the thing being compared is the thing that makes the claim
true. A fixture exercising only elementwise arithmetic would agree everywhere
and prove nothing.

`sample_weight` is pinned to None rather than an array of ones so the weight
bound is exactly `n_samples`, which keeps a host float reduction over caller
weights out of the fixed-point scale. DEVIATION 926.

`init='array'` matters for a reason beyond convenience. It makes the starting
centroids exact rather than sampled, so the fit does not depend on
k-means++'s float scan, and it means `kmeans_fit_main` is entered exactly
once. The k-means|| initializer is the one sanctioned re-entry into that
function and it would otherwise contribute a second block of tags.

One boundary of the card, stated because nobody should have to discover it.
The call chain is `KMeans.fit` into `_mojolearn.kmeans_fit` into
`cluster/estimator.mojo::kmeans_fit` into `fit_predict`, and `fit_predict`
runs the traced fit followed by an **untraced** assignment pass against the
final centroids. The card therefore covers the Lloyd loop, not that last
pass. This is the general caveat in concrete form, and it is why `verify`
prints it on green runs too.

### Why not gradient boosting

`GradientBoosting` is the flagship and it emits a card too (`gbdt/train.mojo`
constructs an `IdentityTrace`). A GBDT fit large enough to be interesting
does not finish in the seconds this command is budgeted, and its card runs to
thousands of stages. `verify` is a doorway, not a replacement for
`E3_RESULTS.md`, and it says so in its own output. `_verify.py` keeps the
fixture behind named constants so a second arm is an addition rather than a
rewrite.

### Why this card is not `e1u/kmeans.card`

DEVIATION 921, and it is worth reading before anyone tries to "simplify" this
by pointing the command at an existing artifact.

`bench/unsupervised_trace_main.mojo` calls `kmeans_fit_main` directly with a
**pinned** `sum_scale` of 4096. The Python surface goes through
`cluster/estimator.mojo::kmeans_fit`, which derives `sum_scale` from the data
with `plan_sum_scale`. Different fixed-point scale, different accumulator
bits, different card, on the same machine, from the same input bytes. The two
cards are not comparable and reusing the E1U one here would have been a
comparison of two different computations dressed up as a check.

What is reused is the **fixture**, which is the part that had three vendors
behind it, and the input gate above is what proves the reuse is real.

The data-derived scale is not a portability hazard. `choose_scale` snaps down
to a power of two (`mojo_only/fixed_point.mojo`), so the scale is a step
function of the magnitude and identical for any magnitude within a 2x band.
That snap exists exactly so the chosen scale cannot depend on the last bits
of a host reduction.

---

## One comparator, never two

`verify` does not contain a card comparator. It imports
`tools/identity_trace_diff.py` and hands the comparison to it, verdict and
all. If it cannot find that file it exits 4 and says so rather than falling
back to anything. DEVIATION 924.

Two comparators that disagree is the exact class of defect this repository is
built to avoid, and a "simple hash equality check" is not a smaller version
of the differ. The differ aligns two cards on their **tag sequences** with
`difflib.SequenceMatcher`, so it distinguishes a numeric divergence from a
structural one where the two runs took a different number of stages. A
one-hash comparison cannot tell those apart, and cannot localize either.

The search order is

1. `MOJOLEARN_IDENTITY_TRACE_DIFF`, an explicit path;
2. `mojolearn._identity_trace_diff`, a build-time copy of the same file
   placed by the release build (the wheel has no `tools/` directory);
3. `tools/identity_trace_diff.py` in a checkout above the package.

A file at the right path is not the right file, so the loaded module is
checked for the API before any verdict comes out of it, and the resolved path
plus its sha256 are printed in the report.

### Why localization matters more than the fact of a difference

The project's own experience is the argument. A divergence was once found at
a specific tree's winner scores on NVIDIA while Apple and AMD agreed. A
final-output comparison could report only "the models differ" and could never
have produced that address. Prefer the stage card every time.

---

## Where the reference card came from

> **STATUS AS SHIPPED: THERE IS NO REFERENCE CARD YET.**
>
> `python/mojolearn/reference_cards/` currently holds only
> `kmeans.identical.fp32.v1.card.PLACEHOLDER`. Until a real card is
> generated and installed, `verify` exits 5 (NO REFERENCE) and prints the
> procedure. It will not pass. DEVIATION 925.
>
> Fill this section in when the card is installed. It must name the two runs
> that produced and confirmed it, at minimum: the commit, the hardware, the
> operating system, the driver or toolchain version, and the date, for each
> vendor.

Template, to be replaced with the real thing.

```
reference   kmeans.identical.fp32.v1.card
profile     mojolearn.identical.verify.kmeans.fp32.v1
produced    <date> on <hardware>, <os>, mojolearn <version>, commit <sha>
confirmed   <date> on <hardware>, <os>, same commit, cards compared with
            tools/identity_trace_diff.py, RESULT: IDENTICAL over <n> stages
stages      <n>
```

A card installed from one machine claims nothing across vendors, and
installing one anyway would turn this whole feature into a number with no
evidence behind it. The placeholder file exists so that a build shipping
without the card fails loudly instead of silently, and it carries the token
`FILL-IN` so that copying it into place under the real name still fails.

---

## Regenerating the reference card

A reference hash that nobody knows how to reproduce becomes untouchable and
then wrong. So the procedure is a subcommand rather than a paragraph.

**A legitimate change that moves the bits requires a new reference.** Any
change to the k-means kernels, to `mojo_only/numerics.mojo`, to the
fixed-point scale policy, or to the trace's checkpoint set will move this
card. That is not a failure, but the new card is only a reference once it has
been produced on two vendors and they agree.

1. On the reference machine, with a clean checkout at a known commit and the
   IDENTICAL extensions built.

   ```
   MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh
   MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
     pixi run python3 -m mojolearn verify \
     --emit-reference /tmp/kmeans.apple.card
   ```

   `--emit-reference` refuses on a FAST build for the same reason `verify`
   does. It stamps the candidate with the full provenance block that the
   reader will later be asked to trust, as `#` comment lines. The differ
   skips comments, so provenance costs the comparison nothing.

2. On a second vendor, at the **same commit**, run the same command, then
   compare the two candidates.

   ```
   python3 tools/identity_trace_diff.py --labels APPLE,NVIDIA \
     /tmp/kmeans.apple.card /tmp/kmeans.nvidia.card
   ```

   If that does not print `RESULT: IDENTICAL`, there is no reference to
   install and there is a finding to chase instead. Do not install either
   card.

3. Install the agreed card.

   ```
   cp /tmp/kmeans.apple.card \
     python/mojolearn/reference_cards/kmeans.identical.fp32.v1.card
   rm python/mojolearn/reference_cards/kmeans.identical.fp32.v1.card.PLACEHOLDER
   ```

4. Fill in **Where the reference card came from** above with both runs, and
   record the change in `CHANGELOG.md`.

5. Confirm the round trip on the reference machine itself.

   ```
   MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
     pixi run python3 -m mojolearn verify
   ```

   It must print `RESULT: VERIFIED`. If it does not, the machine that emitted
   the card cannot reproduce it, which is a much more interesting problem
   than a version bump.

### Versioning

The reference file name carries a version (`...fp32.v1.card`) and so does the
profile string. **Bump it rather than overwrite it** when the fixture, the
estimator entry, or the checkpoint set changes, so that an old wheel and a
new wheel are never both claiming `v1` for two different computations. A
version that is reused for a changed computation is the same defect as a
reference with no provenance.

---

## Reading a mismatch

A mismatch is information, not just a failure. Before concluding anything
about the hardware, check the four things printed above the verdict that
legitimately change the answer.

| line | why it can move the bits |
|---|---|
| numeric mode | the FAST arm makes no claim at all |
| commit | a different build is a different program |
| package version | as above, and it is what the reference names |
| fixture hashes | different input bytes make the comparison meaningless |

Then read the stage name. `tools/identity_trace_diff.py` prints the last
stages that still agreed, then the first that did not, with dtype and element
count on both sides. A **shape** or **dtype** mismatch at that stage is
structural rather than numeric, meaning the two runs built different amounts
of work, and chasing bit patterns before resolving that wastes the
investigation. The differ says so itself.

The local card is kept on a mismatch and its path is printed. Re-run the
comparator over it directly for more detail.

```
python3 tools/identity_trace_diff.py --all \
  <reference card> <local card>
```

Cell-level diagnosis, where the differ decodes individual floats and reports
signed ULP distances and denormal-versus-zero classifications, needs raw
`.bin` dumps on **both** sides. The reference ships without them, deliberately
(they would be megabytes per card), so that path belongs to two live runs and
`MOJOLEARN_IDENTITY_TRACE_DUMP`, not to this command.

---

## The other subcommands

```
python -m mojolearn env             what this process loaded. No GPU touched.
python -m mojolearn check-fixture   rebuild and hash the pinned fixture. No
                                    GPU, no extension call.
python -m mojolearn verify --json   one JSON object instead of the report,
                                    for continuous integration. The
                                    comparator's own text is carried inside
                                    it under comparator_report.
```

`python -m mojolearn` with no subcommand prints help and exits 2. There is no
default subcommand on purpose. A bare invocation that quietly fits on a GPU
is a surprise, and a bare invocation that prints a verdict nobody asked for
is how a green line ends up quoted out of context.

---

## Packaging

`reference_cards/` must land inside the wheel or `verify` exits 5 on every
installed copy. `python/pyproject.toml` lists package data explicitly rather
than discovering it, so the directory needs an entry there, and the release
build needs to place a copy of `tools/identity_trace_diff.py` inside the
package as `_identity_trace_diff.py`. Both are one-line changes and both are
the operator's to make; the exact text lives with the change that introduced
this command.

The copy is a **copy of one file at build time**, not a second
implementation. `tools/identity_trace_diff.py` stays the single source in
git. If the two ever drift, the sha256 that `verify` prints for the
comparator it loaded is how that gets noticed.

---

## Known gaps

Named rather than left to be discovered.

1. **The fixture binary does not self-report its arm.** DEVIATION 920 above.
   Three indirect legs where there should be one direct read.
2. **There is no reference card yet.** DEVIATION 925 above. The command
   cannot pass until one is produced on two vendors.
3. **One fixture, one algorithm.** k-means only. The coverage claim is one
   fit wide and the command says so in its output.
4. **`python -m mojolearn` imports the package**, so a machine where the
   extension cannot load at all gets an import traceback rather than a clean
   CANNOT RUN. `mojolearn doctor` from `mojolearn_diagnostics` is the tool
   for that case, and it lives outside the package for exactly this reason.
5. **The trace reaches the binary through the environment.** CPython's
   `os.environ` assignment calls `putenv`, so the C `getenv` the Mojo side
   uses sees it. If that ever stops holding, the card comes back empty and
   `verify` reports CANNOT RUN naming the instrument, rather than reporting a
   result about the hardware.
