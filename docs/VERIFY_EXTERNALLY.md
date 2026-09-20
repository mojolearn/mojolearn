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
# 1. each party, on their own machine, on their own hardware
python3 -m pip install mojolearn==<version>
MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn verify --all --json-out mine.json

# 2. SEAL IT BEFORE YOU SEE THEIRS. This writes a random nonce into the
#    document and prints a 64-character commitment over its cells, its
#    provenance, its contract, its binding digests and its own verdict.
#    Publish that line anywhere, by any means, BEFORE the exchange.
python3 -m mojolearn verify --commitment mine.json

# 3. EXCHANGE THE TWO COMMITMENT LINES. Not the documents yet.

# 4. answer the challenge those two lines hash to. This reruns the lanes your
#    document already ran, on a fixture neither of you could have drawn
#    before step 3, and prints a SECOND line. Publish that one too, still
#    before the documents are exchanged. Costs about a ninth of step 1.
python3 -m mojolearn verify --challenge mine.json \
    --challenge-from <your line> <their line>

# 5. now swap the documents, nonces included: email, a gist, a USB stick

# 6. either party, on any machine; this step needs no GPU, no binding and no
#    numeric mode, and it is reached before any of them is checked
python3 -m mojolearn verify --compare mine.json theirs.json \
    --commitment-a <the line you published> \
    --commitment-b <the line they published> \
    --challenge-commitment-a <your second line> \
    --challenge-commitment-b <their second line>
```

**Step 2 is not optional ceremony.** Without it, whoever receives the other's
file FIRST can paste its cell values into a document carrying their own
provenance, and the comparer reads a clean `AGREE` across two "independent"
vendors. That is the exact question a reader should ask of us, so the tool
answers it: a party who has published a commitment cannot copy, because what
they were bound to was fixed before there was anything to copy from. A
comparison run without commitments is not an error; it is labelled a weaker
result, and a BROKEN commitment outranks every cell outcome including
`MISMATCH`.

**What a commitment does NOT prove is that a document came from a run.** Our
shipped `verify_reference/table.json` pins the expected hash of every cell, so
a party can synthesise a well-formed document from the table and commit to
that without executing anything. A commitment on its own is evidence about
ORDER, not about execution.

**Steps 3 and 4 are what closes that, and they are why the protocol has two
rounds.** The challenge is

    challenge = sha256("mojolearn.verify-challenge.v1\n" || min(c_a, c_b) || max(c_a, c_b))

over the two commitments published in step 2. Neither party can compute it
before both are published, neither controls it alone, and it is derived by
both parties from values that are already public, so checking it needs no
network, no beacon and no third party. Step 4 reseeds the harness's `hashed`
fixture from it -- `identity_break.fixture('hashed', seed=...)`, the one
fixture whose values come from a counter-mode sha256 stream rather than a
numpy RNG family -- reruns the document's own lane set on it, and records the
answers in a `challenge` block. Those hashes are in no table we ship and could
not have been written before step 3.

The obvious question is how cells with no reference can be checked at all.
They are not checked against a reference: **`--compare` is peer-to-peer and
reads no table.** It compares two documents to each other, and always has, so
a fixture neither party could have known is a perfectly good question to ask
there and nowhere else. The `challenge` block is deliberately kept out of
`cells` for that reason: `verify --all` judges cells against the table, and an
unreferenced cell in there would read OWED.

The second published line, in step 4, is not ceremony either. **Two honest
challenge responses are identical** -- that is what makes them comparable --
so a response nobody was bound to can simply be copied out of the other
party's file by whoever receives it first, which is the move commit-reveal
exists to stop, one level in. The challenge commitment is a separate digest
under its own domain separator, over the response and the document's own
round-one commitment, so a second line cannot be lifted off one document and
presented for another.

### What the challenge proves, and the three things it does not

The adversary model, written down, because a defence nobody can check is not
one. Each row says what an adversary who controls one side can still do.

| adversary | what they do | what happens |
|---|---|---|
| the transcriber | receives the other file first, pastes its cells under their own provenance | caught by the round-one commitment (`COMMITMENT BROKEN`) |
| the synthesist | writes a whole document out of the shipped table, runs nothing, commits to it first | they have no challenge to answer: the fixture did not exist when they wrote the file, and no hash for it is in the table. Their file carries no `challenge` block, the honest party's does, and a one-sided challenge is `CHALLENGE BROKEN` |
| the response thief | synthesises the document, then copies the other party's `challenge` block after the exchange | caught by the second published line: they had to commit to a response before the exchange, and what they hand over is not it |
| the replayer | answers a challenge honestly for one exchange and presents it in another | the challenge names the two commitments it came from; the comparer recomputes the derivation and checks that pair against the two documents in hand and against the two published lines |
| the chooser | picks a challenge they have already answered and writes two commitments beside it | the derivation is recomputed, never believed. A challenge that is not what its own pair hashes to is `CHALLENGE BROKEN` |

**The three holes this does not close, stated plainly.**

1. **It does not prove two PEOPLE.** One party with one machine can play both
   sides: run the challenge once, write both documents around the one answer,
   seal both, publish all four lines in the right order. Every check passes,
   and the device blocks that make the pair look independent are self-reported
   strings. Nothing in this tool reaches that, and nothing in it can: two
   parties being two parties is established outside the software or not at
   all. A verified challenge must never be read as if it settled that.
2. **Whoever publishes their commitment last can steer the challenge.** A
   reseal draws a fresh nonce and costs nothing, so the second publisher can
   grind commitments until the derived challenge is one they like. That buys
   nothing against synthesis -- every candidate challenge still has to be RUN
   before it can be answered -- but the challenge fixture is therefore **not
   an unbiased draw**, and a pair who wish to steer away from a fixture that
   would expose them can. If that matters to you, publish first.
3. **It proves execution of what it covers, on ONE fixture.** A party who
   answers the challenge honestly and writes the rest of their document out
   of the table still passes, and what they have demonstrated is exactly the
   challenge block: those lanes, that fixture, that hardware. That is why the
   challenge runs the document's whole lane set rather than a sample, and why
   a narrowed challenge shows up as cells only one side carries
   (`INCOMPLETE`). The challenge also varies the VALUES and never the
   pathology: it is the `hashed` fixture reseeded, so it says nothing about
   denormals, ties or duplicate rows, and it does not replace the nine
   recorded fixtures.

**A comparison with no challenge is still legal and still compares.** It is
labelled a weaker result, in the same place and the same voice a comparison
without commitments already is, and the `RESULT: AGREE` line itself carries
the reason. Documents written by releases before this feature carry no
`challenge` block and compare exactly as they did; adding one does not move a
document's round-one commitment, which is what makes step 4 possible at all.

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
| `COMMITMENT BROKEN` | 1 | a document is not the one its party committed to before the exchange: it does not match the published line, it carries no nonce to check one against, it was edited after sealing, one commitment was presented for both, or the two carry the same nonce. Only `MALFORMED` outranks it, because until you know a document is the one that was committed to, no headline about its cells is honest |
| `CHALLENGE BROKEN` | 1 | a challenge is answered here and the answer does not hold up: only one document carries one, the two answer different challenges, the challenge is not what its own pair of commitments hashes to, the pair is not the pair in hand or the pair published, a response does not match the challenge commitment published for it, the two carry the same challenge nonce, or the two answered over different input. Until that is settled the reader does not know these cells were COMPUTED rather than written out of our table, so it outranks every cell outcome. ABSENCE of a challenge never lands here: that is a weaker `AGREE`, labelled on the RESULT line |
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

### If you have nobody to swap with: our own documents

`bench/results/verify_reports/` carries our own sealed `verify --all` output,
so this recipe works on the day you install the wheel rather than on the day
you find a second party. Its README says which device classes are there; as
of the first publication it is **CPU only**, two boxes, and the comparison
between them reads `AGREE` on 5,174 cell parts with both commitments
verified. **That pair answers no challenge**, because it was taken before the
challenge existed, so it sits at the weaker rung: it shows the two files were
fixed before they met, not that either was computed rather than written out of
the table. Re-taking it at the stronger rung means rerunning `verify --all` on
both boxes and walking the six steps above, which is a fresh pair of CPU runs;
the files in that directory are not editable into one.

**This is weaker than comparing with a stranger, and it does not replace it.**
Comparing against our document puts us back in the loop: the answer then rests
on our having run what we say we ran on the hardware we say we ran it on.
Recipe 4 above removes us from the answer entirely; this does not. Use ours to
check that your install produces a comparable document at all, and as the
fallback when you have no second party. Then find one.

Read the pair honestly, and note that the tool does this for you: both of our
CPU rentals came back as the same processor model, so `verify --compare`
prints `WARNING: both documents describe the SAME device. Agreement then shows
repeatability, not cross-hardware identity, and proves much less.` The
cross-vendor claim rests on the recorded columns under
`bench/results/identity_break/`, never on these two files.

Three things make a comparison against ours fail without anything being wrong
with your machine, and each is the protection working rather than a defect:

- **a different commit.** Each document names its commit and harness digest,
  and a comparison across a harness change reads `INCOMPARABLE` rather than a
  silent wrong answer. Check out the commit the document names, as Recipe 2
  already has you do. These are re-emitted per release for that reason.
- **different flags.** Ours are plain `verify --all` documents: five parts per
  cell. `--batch-checks` adds four more properties, so a document made with it
  carries cells ours do not and the comparison reads `INCOMPLETE`. Both sides
  must use the same flags.
- **a cell one side did not run.** `compare-uncomputed-negative-control.txt`
  in that directory is that case, kept on purpose: two documents with ZERO
  differing cells still read `INCOMPLETE. Absence is not agreement.` because
  90 cell parts ran on neither side.

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
