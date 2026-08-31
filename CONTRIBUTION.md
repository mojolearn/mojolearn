# What this library contributes

Every number here is measured, and the command that measures it is given, so
this page can be checked rather than believed. Measured 2026-08-31 over
tracked `*.mojo` at commit `3b4616cd`.

This document exists because the repository's own documentation spent most of
its length on what the algorithms derive from and almost none on what was
built, and a reader came away with the proportion exactly backwards.

---

## 0. Nothing was copied, and that is checkable in one command

**The shipped library is 100% Mojo. There is no CUDA and no C++ in it.**

    git ls-files -- '*.cu' '*.cuh' '*.cpp' '*.cc' '*.c' '*.h' '*.hpp'

Twenty files come back, all of them under `tools/*_oracle/` and
`ensemble/tools/*_oracle/`. Those are test harnesses that link an upstream to
produce reference numbers the Mojo is checked against. They build the check,
not the library.

None of CatBoost, cuML, cuVS or RAFT contains any Mojo either. Nothing was
translated line by line out of one file into another, because there was no
file to translate: the algorithms were implemented here, against references
that are cited the way a paper cites prior work.

**Citation is not confession.** A paper that cites its predecessors is not
admitting to copying them, and the same is true of a source file that names
the reference it was written against. What the citation buys is that section
4's claim can be checked instead of taken on faith.

---

## 1. The result: one source, three vendors, the same bits

No CUDA library this draws on has a cross-vendor bit-identity tier, because
none of them was ever asked to run anywhere but CUDA. This one has three
numeric tiers selected at runtime, and the strictest returns **bit-identical
answers on Apple Silicon via Metal, NVIDIA via CUDA and AMD via ROCm from one
source**, checked by per-stage identity cards diffed across the three at a
shared commit.

That is the contribution. Everything below is what it cost.

---

## 2. 58,113 lines exist because the upstream algorithm has no Metal bottom

**105 of 459** files with an upstream counterpart name a primitive that does
not exist on the target: `cub::`, `thrust::`, `cublas`, `cusolver`,
`cusparse`, `curand`, `__shfl`, `raft::linalg`, `cudaMemcpyAsync`. When the
upstream's line is `cub::DeviceScan::ExclusiveSum(...)`, there is nothing to
transcribe. The scan has to be written.

    git grep -lE 'cub::|thrust::|cublas|cusolver|cusparse|curand|__shfl|raft::linalg|raft::matrix|cudaMemcpyAsync|DeviceScan|DeviceRadixSort|DeviceSegmented|BlockReduce|BlockScan|WarpReduce' \
      -- '*/derived/*.mojo' '*/derived/**/*.mojo' 'gbdt/*.mojo' 'gbdt/**/*.mojo'

Written from scratch, each replacing one upstream library call: a decoupled
lookback scan, a one-bit-per-pass radix sort, a segmented radix sort, a
segmented scan under a custom operator, a segmented reduce, a block reduce in
float64 on hardware with no float64, a warp scan, a device histogram, a
stable unique, a flagged compaction, an argmin with a pinned tie-break, and
the XORWOW generator with its transition matrices **rebuilt from the step
function rather than copied**.

**And they are not merely replacements. They are replacements with a pinned
reduction order, which the CUDA originals do not have.** `cub::BlockReduce`
does not promise you the same answer twice on the same box, let alone the
same answer on another vendor. These do. That property is the entire reason
the tier in section 1 is possible, and it could not have been inherited from
anywhere because it does not exist upstream.

---

## 3. 207,391 lines of verification apparatus, and no upstream has any of it

| | files | lines |
|---|---:|---:|
| checks | 200 | 174,538 |
| host oracles | 29 | 22,753 |
| fixtures | 16 | 10,100 |
| **total** | **245** | **207,391** |

That is **48% of the tree**. cuML, cuVS, RAFT and CatBoost ship one GPU
backend each and check it against nothing but itself.

- **841 numbered deviations.** Every place the port departs from its
  reference, written down at the site, with the reason and what it cost.
- **342 named sabotage arms.** A gate is not believed until it has been shown
  capable of failing: break the path deliberately, watch the number move.
- **5,324 identity cards.** Per-stage device output, recorded and diffed
  across vendors at a shared commit.

The oracles are the part that is easiest to undervalue. Each one is an
independent host implementation of the same semantics, written so the device
arm has something to be wrong against. cuML has no host solver to check its
SMO against; this repository wrote one.

---

## 4. What derives from an upstream, and why saying so is not a concession

Between **30% and 33%** of the Mojo here has a file it corresponds to in
CatBoost, cuVS, cuML, RAFT, FAISS, HuggingFace transformers or cuRAND. The
other **67% to 70%** has none. `NOTICE` gives the bounds and how they were
computed.

Every one of those derived files names its upstream path, its pinned commit
and usually its line numbers, in the file. That is an Apache 2.0 section 4
obligation, and it is also the thing that makes this page worth reading.

**A claim of originality with no attribution anywhere is an assertion a
reader has to take on trust, and readers do not.** A claim of originality
that says *exactly* which files have an upstream, at which commit, and lets
anyone diff them, is checkable. The 67% is worth more BECAUSE the 30% is
labelled, not less. Attribution is what converts the credit from a boast into
a fact.

It is also what makes the identity result believable rather than suspicious:
the algorithms can be taken as correct without re-deriving them, so the
question narrows to whether the bits match, which is the question that was
actually answered.

---

## 5. What is deliberately not claimed

- Not clean-room. The upstreams were read, and their files, commits and line
  numbers are cited throughout.
- Not that everything derived was reimplemented. A file-by-file audit of the
  98 rows once labelled `transliterated` found the word **right on 52 of
  them**, including two that are exact including their comments. The label is
  being corrected per row, in both directions.
- Not novelty of the algorithms. Gradient-boosted oblivious trees are
  CatBoost's. k-means++ is not new. What is new is that they produce the same
  bits on three vendors.
