# How many papers, which ones, and what each still needs

Written 2026-08-21 after a second prior-art pass over `NOVELTY_NOTES.md`.
That file is the INVENTORY of candidate findings. This is the PLAN: how they
group, how many survive as papers, and the work each one is missing.

**REVISED 2026-08-21, same day, after Andrew asked whether the FTZ
construction is enough to carry a paper. It is not, and the honest answer to
"is this a BS paper" is that Paper 2 as scoped is at real risk of being one.
See the section at the end. Default is now ONE paper.**

**Original answer: two papers, one section, and a pile of bug reports.** Not five, not
ten. Most of `NOVELTY_NOTES` is evidence for those two, which is what evidence
is for.

---

## Paper 1 — ACCESS. "The GPU in every Mac that no GBDT can reach."

Repo: `~/CascadeProjects/mlsys`. A 7-page draft compiles today.

**Claim.** Every production GBDT reaches a GPU through CUDA or OpenCL, so on
Apple silicon all of them fall back to the CPU. This is the first histogram
GBDT learner that trains on an Apple GPU, and here is what that buys, measured
against CatBoost's CPU on the same machine with matched models: 3.41x on
`epsilon`, 1.95x at 4M x 100, parity at 800k x 100, **1.87x slower on
covtype** — the whole ladder, losses included.

**Draws on `NOVELTY_NOTES`:** 4 (floor amortization, as the *mechanism* that
explains the ladder), 5 (density cliff, as the attribution), 6 (the interleaved
protocol, as the methods section).

**Novelty, honestly.** The system and the measurement. Items 4 and 5 are new
DATA on known phenomena — GPU-vs-CPU crossover and irregular histogram memory
access are covered by Zhang et al. (arXiv 1706.08359) and Wen et al. (TPDS'19)
— so they are evidence, not findings, and the paper must not present them as
findings.

**Strongest asset:** it is nearly draftable. **Weakest:** a reviewer can read
it as an engineering port.

### Work outstanding

| # | item | why it blocks | cost |
|---|---|---|---|
| 1 | **End-to-end `fit`, including quantization, both arms** | `train()` spends **24 s of preparation around 0.75 s of trees** at 400k x 500. Every benchmark correctly quantizes outside the timed region, so the user-facing path is the one nobody timed. A reviewer asks "what does `fit` cost?" and today's honest answer destroys the headline. **Fix it or report it — do not omit it** | the fix is a device-side sort; the report is one window |
| 2 | **E7: defend or drop the word "first"** | a literature + repository sweep for any GBDT on an Apple GPU (thesis, prototype, MLX experiment). Fallback wording "no library in wide use" is already in the abstract | half a day, no hardware |
| 3 | **More datasets** | three today, two synthetic. Add higgs, year MSD, airline, one categorical | prep + a window each |
| 4 | Inference latency and peak memory, both arms | the laptop story is not only training; their raw X was 3.2 GB against our 0.8 GB compressed | one window |
| 5 | **Energy per tree** | nobody in this field reports it and on a laptop it is a first-class metric. `powermetrics` gives package power | one window + tooling |
| 6 | Apple hardware ladder M1–M4, base/Pro/Max | tests whether the crossover moves left with GPU core count, which the fixed-cost model predicts | machines we do not have |
| 7 | **The Python extension is unbuildable at HEAD** (`NOVELTY_NOTES` C8) | artifact evaluation needs something a reviewer can install. This is filed as a novelty item; it is a shipping blocker | blocked on Modular |
| 8 | ThunderGBM into related work | "Fast GBDTs and Random Forests on GPUs", CUDA + ROCm, 10x claims on sparse high-dimensional data. It **also cannot run on Apple**, so it strengthens the access claim — and it does RF on GPU, which the forests section must acknowledge | reading |

---

## Paper 2 — THE CONTRACT. "Normalize to the floor, not to the standard."

Repos: `bitwise-gbdt` and `silent-nondeterminism`. **These are one paper and
should be consolidated.**

**Claim.** On a heterogeneous fleet, the most constrained member defines the
portable numeric contract. Where the constraint is severe enough, conforming
to it is free or profitable on that member and cheap elsewhere. Demonstrated
three times on one platform: no CUDA at all; a 32 KB threadgroup ceiling that
excludes the incumbent's float layout and admits an integer one that is *also*
associative; and hardware denormal flushing that makes IEEE unreachable, so
the contract becomes Metal's measured FTZ behaviour.

**Draws on `NOVELTY_NOTES`:** 1 (the enumeration), 2 (the folklore audit),
**10 (the FTZ model as the construction — the strongest single entry)**,
3 (metamorphic subtraction, as an instrument), 7 (oracle-compiled gates, as
methods).

**Novelty, honestly.** Bit-level cross-device reproducibility is an active and
fashionable field: **RepDL** (arXiv 2510.09180) already delivers it for deep
learning "across multiple executions with the same or different CPU or GPU
systems" by enforcing correct rounding and order invariance. Two things
survive that:

1. **Direction of normalization.** RepDL normalizes to the STANDARD. Correct
   rounding is unreachable on a fleet containing Apple, which flushes in
   hardware. Normalizing to the measured floor inverts the cost structure.
   No prior instance found of deriving a training system's cross-vendor float
   contract by bit-exact behavioural modelling of the most constrained vendor.
2. **Trees are not networks.** RepDL makes weights bit-identical; a
   perturbation moves a number. Tree arithmetic feeds an **argmax**, so one
   ULP does not perturb the output, it selects a different split, and the
   models stop being neighbours. Reproducibility for a discrete-structure
   learner is a different problem, not the same problem on a new model family.
   **This sentence goes in the first paragraph or the work reads as an
   application of RepDL.**

Two searches for cross-hardware determinism guarantees on tree learners
returned nothing. That gap looks real; it is not yet proven.

### Work outstanding — and this paper cannot be submitted today

| # | item | why it blocks | cost |
|---|---|---|---|
| 1 | **Close pathway 8: the scale magnitude reduce** | it is a device float reduction through a float atomic, so the scale that quantises every gradient is order-dependent. Until it is closed the enumeration claim is FALSE | the accumulator already exists |
| 2 | **Finish the row 9 and 10 application checklists** | the helpers landed; every multiply-add seam and every cross-kernel float seam still has to be routed through them | mechanical, per-site, must cite the row |
| 3 | **Build `check_identity_paths`** | `IDENTITY_PATHS.md` is a markdown ledger, so it rots. The guard has to FAIL when a new float reduction appears without a row. Ideally taint hardware-query rows and refuse when one reaches a float reduction unpinned | this is also Paper 2's tool contribution |
| 4 | **RUN E1** | build IDENTICAL on NVIDIA and AMD, train the same configuration, compare SHA-256 of the SERIALIZED MODEL over >= 5 datasets and >= 5 configurations. **The paper does not exist without this** | two rented boxes, correctness only |
| 5 | **Measure identity's cost off Apple** | the thesis says "cheap elsewhere" and that half has never been measured. On 48 KB and 64 KB the integer path buys no occupancy and is expected to COST | interleaved, on rented boxes |
| 6 | **Re-run `check-ieee-arith` on CUDA and ROCm** | the FTZ model is measured on ONE vendor. The claim "CUDA honours denormals" is currently read from documentation, not measured | one run per box |
| 7 | Model file carries `IDENTITY_PROFILE` | profile 1 and a future profile 2 are otherwise told apart only by provenance | an hour |
| 8 | Read RepDL and *Defeating Nondeterminism in LLM Inference* in full | related work, and both may shrink our sections | reading |
| 9 | **Consolidate the two repos** | one paper, two repositories, and a recorded project failure mode about exactly this | a merge |

---

## Section, not a paper — the enumeration failures

`silent-nondeterminism/FAILURE_MODES.md`. The general lesson — *enumeration is
where determinism fails, and it fails silently* — has a higher-profile prior
instance in LLM inference. What survives is two mechanical patterns:

- a mode whose selection is distributed across N files is unreachable, and
  quietly accumulates claims about a configuration nobody has built;
- **any parameter derived from a hardware query that reaches a float reduction
  is numeric, however it is classified at its definition.**

Both belong in Paper 2, Section 4. **It becomes its own (short, tools-track)
paper only if item 3 above gets built AND catches something in a second
system.** Not before.

## Not papers, and naming them stops them being mistaken for papers

- `NOVELTY_NOTES` C8 (Metal AOT suppression by MACOSX_DEPLOYMENT_TARGET +
  cache not keying on it; the "basename lottery" is retracted, 5cd37db) and
  C9 (the Mojo numeric traps): upstream issues. C8 is no longer a release
  blocker -- the workaround is shipped and the wheel builds.
- Items 6 and 7 (benchmark protocol, oracle-compiled gates): methods sections
  in Papers 1 and 2. Kalibera & Jones and Mytkowicz et al. own the first;
  differential testing owns the second.
- **The library itself: JMLR MLOSS or JOSS. Deferred, and now with the exact
  bar written down rather than a feeling.** MLOSS is four pages plus
  references, and the cover letter must state the license, the project URL,
  the version under review, and **"evidence of an active user community"** ---
  demonstrated by active developer count, stars, or similar. Review criteria
  include freedom from proprietary dependence.

  Two of those block today. There is **no user community**: no non-author has
  used this, which is the same reason JOSS was deferred and it has not
  changed. And the **Python extension does not build at HEAD**, so a reviewer
  who tries to install it fails --- for a track that reviews the software as
  much as the paper, that is disqualifying rather than embarrassing.

  The proprietary-dependence criterion may have just stopped applying:
  Qualcomm open-sourced Mojo under Apache-2.0 in August 2026. **Verify exactly
  what is covered** --- the language, the compiler, the runtime, the kernel
  library --- before relying on it, because "the platform is open source now"
  is the kind of sentence that is 80% true.

  **MLOSS is not a competitor to the conference paper, it is a different
  question.** The conference asks whether the result is true and interesting;
  MLOSS asks whether the software is real and used. For a library author the
  software paper is often the one that accumulates citations, because every
  user cites it --- but only once there are users. Sequence it after adoption,
  not before.

## The ordering that gets one thing submitted

Paper 1 is the fastest path to a submission because its evidence exists —
items 1 and 2 are the only hard blockers, and item 1 is a bug fix. Paper 2 is
the better paper and needs two rented GPUs and four code items first. **Do not
start Paper 2's write-up until Paper 1 has a complete draft**; three paper
repositories with nothing submitted is this project's own recorded failure.


---

# REVISION: is Paper 2 a real paper? Argued against, honestly.

Andrew: *"is the ftz construction enough to carry the paper? seems like bs
paper no?"* The case against is stronger than I had been treating it.

## The case that Paper 2 is thin

1. **The FTZ construction is a few lines.** `ftz(x)` flushes subnormals.
   Applied at seams. Dressing it as "deriving a cross-vendor numeric contract"
   is framing, and a reviewer is entitled to see through framing.
2. **The measurement confirms documented behaviour.** Metal flushes
   denormals; Apple says so. That a flush model reproduces 53,041 of 53,041
   flush divergences is a validation of an implementation, not a discovery.
   The number is impressive-looking and it is checking that flush-to-zero
   behaves like flush-to-zero.
3. **The rest of the machinery is standard.** Fixed-point accumulation for
   order-independence is textbook. Pinning machine-derived parameters is
   obvious once the problem is named. The "enumeration discipline" is, in
   plain words, *make a list and check it*.
4. **E1 is a hash comparison.** It is evidence, not method. A paper whose
   headline experiment is `sha256(model_a) == sha256(model_b)` needs its
   contribution to be somewhere else entirely.
5. **The negative-cost result is an implementation detail of one backend.**
   Stated without its scope it is a slogan; stated with its scope it is "our
   port had an occupancy problem on Metal and the integer layout fixed it."
6. **Nobody may want the property.** "No one has done it" and "no one needs
   it" are indistinguishable from inside, and we have not talked to a single
   user who asked for cross-vendor bit identity.

## What is actually strong, and note where it sits

- **Trees are not networks.** Reproducibility results from deep learning do
  not transfer to discrete-structure learners, because an argmax converts a
  one-ULP difference into a different model rather than a perturbed one. This
  is the one argument that survives every prior-art check.
- **The float audit's negative results.** Three of four conventional hazards
  do not occur on this stack; the fourth accounts for everything. That
  contradicts folk knowledge and is a measurement, not a construction.
- **The two enumeration failures.** A mechanism that was correct and
  unreachable; a parameter classified as scheduling that was setting leaf
  values. Transferable, concrete, and honest.

**Notice that all three are findings and failures, not the FTZ construction.**
The construction is the least interesting thing in the paper it was going to
be named after.

## The revision

**Default to ONE paper: Paper 1, enriched.** It is 7 pages of a 10-page
budget. The determinism material --- argmax amplification, the float audit,
the accumulator, the enumeration failures --- is 2-3 pages of genuinely strong
content that turns a good short paper into a full one, and it fits the same
thesis rather than competing with it: *what happens when you take a GBDT to a
platform nobody targeted, and what portability actually costs there.*

**Split back into two only if E1 runs and comes back clean.** Three GPU
vendors emitting one hash is a headline result that deserves its own paper.
Until then, Paper 2 is a design and an argument, and one system plus one small
construction is thin for a venue accepting 22%.

**And if E1 fails or stays unrun, the determinism material still ships** ---
inside Paper 1, where it is supporting evidence and does not have to carry
anything on its own.
