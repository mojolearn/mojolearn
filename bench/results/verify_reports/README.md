# Our own evidence documents, for `verify --compare`

`docs/VERIFY_EXTERNALLY.md` Recipe 4 is the strongest external check this
project has: two people who have never met each run the shipped verifier,
seal each document with `verify --commitment` BEFORE they exchange anything,
swap files, and run `verify --compare`. Nothing we wrote is an input to the
answer except the code they both installed.

That recipe needs a second document, and a stranger on the day they install
the wheel has nobody to get one from. These are ours, so the recipe works
immediately.

**These two documents answer NO CHALLENGE.** They were taken before
`verify --challenge` existed (lane/compare-challenge-nonce, 2026-09-20), so
the comparison below sits at the weaker rung of the ladder: both commitments
verify, which shows the two files were fixed before they met, and nothing here
shows either was COMPUTED rather than written out of the reference table the
wheel ships. A current `verify --compare` of this pair prints that on the
`RESULT: AGREE` line. Re-taking them at the stronger rung means rerunning
`verify --all` on both boxes and walking the six steps of Recipe 4; the files
here cannot be edited into a challenged pair, because the challenge is derived
from commitments that are already published.

**Comparing against our document is WEAKER than comparing against a
stranger's, and it is not a substitute.** It puts us back in the loop: the
answer then rests on our having run what we say we ran on the hardware we say
we ran it on. Recipe 4 exists precisely to remove us. Use these to check that
your install produces a comparable document at all, and as the fallback when
you have no second party. Then find one.

## What is here

| file | box | what it is |
|---|---|---|
| `verify-all.cpu-a.json.gz` | AMD EPYC 4564P 16-Core | `verify --all` on a CPU-only install |
| `verify-all.cpu-a.json.commitment` | | the commitment over that document |
| `verify-all.cpu-b.json.gz` | AMD EPYC 7702P 64-Core | an independent run on DIFFERENT silicon |
| `verify-all.cpu-b.json.commitment` | | the commitment over that one |
| `verify-all.cpu-a.txt` `verify-all.cpu-b.txt` | | the human output of each run |
| `self-test.cpu-a.txt` | | the verifier reproducing `ols/base` and then DETECTING a one-ULP perturbation, on the box that made the document |
| `compare-cpu-a-vs-cpu-b.txt` | | **what `verify --compare` says about the published pair** |
| `compare-uncomputed-negative-control.txt` | | the same comparison when 90 cells ran on neither side |
| `compare-same-device-agree.txt` | | an earlier pair that landed on ONE processor model, kept for the warning it triggers |
| `nvidia-hang/` | RTX 4090 | why there is no NVIDIA document |

Gzipped because each document is about 15.8 MB and about 405 KB compressed,
and `bench/results` is already the reason a push is slow. `gunzip -k` first;
`--compare` reads plain JSON.

**Device classes represented: CPU ONLY.** There is no Apple, NVIDIA or AMD
document here. Do not read the pair below as cross-vendor evidence; it is not.
The cross-vendor claim rests on the recorded columns under
`bench/results/identity_break/`, not on these files.

There is no NVIDIA document because `verify --all` DEADLOCKS on a CUDA
install: it reaches `transformer-bf16w` and stops, GPU idle at 0 %, 195
threads, `wchan: futex_wait_queue`. `nvidia-hang/` has the measurement. The
fourteen low-bit weight lanes are public and a GPU install runs every lane, so
this is user-facing and not merely our inconvenience.

## What the pair says, and what it does not

`compare-cpu-a-vs-cpu-b.txt`, in full, is the reason this directory exists:

    cpu             AMD EPYC 4564P 16-Core   AMD EPYC 7702P 64-Core  <- differs
    agree 5165   differ 0   self-contradicted 0   neither computed 0

    RESULT: AGREE. 5165 cell parts match across both documents, none
    differ, and none is present in only one. Neither machine trusted the
    other, and neither had to trust us.
    Both documents were committed to before either party saw the other's, so
    neither set of numbers could have been copied from the other.

Two different processors, 5,165 cell parts, not one of them different, and
each machine bound to its own answer before it could see the other's.

**And the limit, which the tool prints itself rather than leaving a reader to
notice:**

    NOTE: these documents share a device class. Agreement is weaker evidence
    than two genuinely different vendors would give.

Two x86 CPUs are not two vendors. `compare-same-device-agree.txt` is the
weaker case still, kept on purpose: an earlier pair where both rentals came
back as the SAME processor model, where the tool escalates to
`WARNING: both documents describe the SAME device. Agreement then shows
repeatability, not cross-hardware identity, and proves much less.`

## Three things that will make a comparison fail, none of them your machine

1. **A different commit.** Each document names its commit and its harness
   digest. A comparison across a harness change reads `INCOMPARABLE`, never a
   silent wrong answer. Check out the commit the document names, exactly as
   Recipe 2 does. These were taken at the commit in their `device.commit`
   field; the harness digest moved the same day when `kmeans-cosine` was
   deleted, which is how we know the check works.
2. **Different flags.** These are plain `verify --all` documents: five parts
   per cell (`train`, `infer`, `model`, `batch`, `stepfull`). `--batch-checks`
   adds four more properties, so a document made with it has cells ours do
   not, and the comparison reads `INCOMPLETE`. Both parties must use the same
   flags.
3. **A cell one side did not run.** `compare-uncomputed-negative-control.txt`
   is that case, kept on purpose: two documents with ZERO differing cells
   still read `INCOMPLETE. Absence is not agreement.` because 90 cell parts
   ran on neither side. It is the negative control for this directory.

## The gap these files make more convenient, stated plainly

A commitment binds a party to a document before they see the other's, so
neither can copy. It does NOT prove the document came from a run. Our shipped
`verify_reference/table.json` pins the expected hash of every cell, so a party
can synthesise a well-formed document from the table and commit to that
without executing anything. Publishing these files does not create that hole,
but it hands someone the shape of a finished document more conveniently, so
it is named here rather than left for a reader to find.

Closing it needs a challenge nonce: a value neither party controls, mixed
into what the run computes, so a document cannot be written ahead of time.
That is a design, not a line of code, and it is parked. Until it exists,
treat a commitment as evidence about ORDER, not about execution.

## Regenerating these

    python3 -m mojolearn verify --all --json-out verify-all.<class>.json
    python3 -m mojolearn verify --commitment verify-all.<class>.json

They are re-emitted per release, because a harness change makes the previous
set `INCOMPARABLE` by design.
