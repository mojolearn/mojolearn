# Our own evidence documents, for `verify --compare`

`docs/VERIFY_EXTERNALLY.md` Recipe 4 is the strongest external check this
project has: two people who have never met each run the shipped verifier,
seal each document with `verify --commitment` BEFORE they exchange anything,
swap files, and run `verify --compare`. Nothing we wrote is an input to the
answer except the code they both installed.

That recipe needs a second document, and a stranger on the day they install
the wheel has nobody to get one from. These are ours, so the recipe works
immediately.

**Comparing against our document is WEAKER than comparing against a
stranger's, and it is not a substitute.** It puts us back in the loop: the
answer then rests on our having run what we say we ran on the hardware we say
we ran it on. Recipe 4 exists precisely to remove us. Use these to check that
your install produces a comparable document at all, and as the fallback when
you have no second party. Then find one.

## What is here

| file | box | what it is |
|---|---|---|
| `verify-all.cpu-a.json.gz` | see the `device` block inside | `verify --all` on a CPU-only install |
| `verify-all.cpu-a.json.commitment` | | the commitment over that document |
| `verify-all.cpu-b.json.gz` | a second CPU box | an independent run of the same thing |
| `verify-all.cpu-b.json.commitment` | | |
| `compare-cpu-a-vs-cpu-b.txt` | | what `verify --compare` says about the pair |
| `compare-uncomputed-negative-control.txt` | | the same comparison BEFORE, when 90 cells ran on neither side |

Gzipped because each document is about 15.8 MB and about 405 KB compressed,
and `bench/results` is already the reason a push is slow. `gunzip -k` first;
`--compare` reads plain JSON.

**Device classes represented: CPU ONLY.** There is no Apple, NVIDIA or AMD
document here. Do not read the pair below as cross-vendor evidence; it is not.
The cross-vendor claim rests on the recorded columns under
`bench/results/identity_break/`, not on these files.

## What the pair says, and what it does not

    RESULT: AGREE. 5174 cell parts match across both documents, none
    differ, and none is present in only one.
    COMMITMENTS: both documents match a commitment published BEFORE the
    exchange, so neither party could have copied the other's numbers.

and, in the same output, the thing that keeps it honest:

    WARNING: both documents describe the SAME device. Agreement then shows
    repeatability, not cross-hardware identity, and proves much less.

Both rentals came back as an AMD EPYC 7713. So this pair is REPEATABILITY on
one CPU model, not identity across hardware, and the comparer says so itself
rather than leaving a reader to notice.

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
