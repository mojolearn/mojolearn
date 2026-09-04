# Copy, Do Not Improve

Standing orders for restarting this strategy from an empty directory.
Drawn up 2026-08-21. Every rule below was paid for once; the **scar** under
each one is the incident that set it, kept so the rule cannot drift back
into opinion. The rendered version lives at the "Copy, Do Not Improve"
artifact; this file is the durable copy.

`PORTING_RULES.md` governs the day-to-day mechanics of this repository.
This document is one level up: what to tell yourself before the repository
exists.

## The mission, stated so it cannot be bent

Port the incumbent's GPU learner into Mojo, file for file, algorithm for
algorithm. **The thesis is GPU _access_, not GPU tier:** their GPU arm
cannot run on Apple silicon at all (`task_type="GPU"` raises), so the claim
is _our GPU against their CPU on the same machine_ — said plainly beside
every number, every time. You are not trying to out-think CatBoost. Fifty
engineer-years of their decisions are correct until their own source says
otherwise. Match them on algorithms; win on the device they cannot reach.
One source targets Metal, CUDA and HIP alike — vendor divergence lives in a
kernel matrix as a row, never as an inline `if apple`.

## Day one, in order

1. Pin their source at one commit and cite everything by `file.cu:line`
   against it. An uncited port is a rumor.
2. Build the gates before the kernels: an oracle that runs THEIR binary on
   fixtures and dumps their per-cell output, and a check that reads it.
   Split decisions exact, histograms cell-for-cell, MSE to eight
   significant figures.
3. Build the interleaved timing harness next — both arms alternating inside
   one process, one window — before any number is quoted by anyone.
4. Mirror their file tree. Their paths are your paths; a replacement for a
   step of their file lives IN THAT FILE under a marked deviation block. A
   separate directory exists only for what they never had to write.
5. Port the learner's spine, then kernels. Do not quote a timing number
   until the mixed-width path works — the uniform path is the shape their
   design was least written for.
6. Fit `ms/tree = a + b*rows` early, both arms. Fixed cost against slope
   explains the entire scoreboard before a single optimization is chosen.

## The rules

### 1. Theirs is right and yours is wrong, until their file says otherwise

Two categories exist: PORTED and NOT PORTED YET. There is no third category
of "good idea worth adopting." When outputs disagree, read their source
first — never reason about your own code's correctness.

> **The scar.** Seven real bugs in one session. All seven found by reading
> CatBoost's files; zero found by reasoning about ours. Our code was wrong
> about itself four times in one day, and our instruments failed three
> times.

### 2. Port the call, not the library

Where they call CUB, Thrust or cuBLAS, port the CALL onto the platform's
own linalg/algorithm layer. Hand-writing a replacement is inventing. When
the platform genuinely has no counterpart (no device sort, no device scan
on MAX — checked, dated), hand-write it portably: zero warp intrinsics,
shared memory sized from a queried budget, no assumed wavefront width.

> **The scar.** Every hand-written "replacement" for a vendor primitive
> became a deviation to price, a check to write, and a bug surface. The one
> place allowed to beat CatBoost is where no CatBoost code exists to be
> faithful to.

### 3. Every deviation is numbered, recorded, and priced — declines included

Assign deviation numbers up front, before parallel work begins. A deviation
without a measurement is a confound sitting under every future number. When
you DECLINE an optimization, write the price of declining it in the ledger,
or it counts as an open item.

> **The scar.** A five-lane parallel round produced a three-way
> deviation-number collision. Separately: an unpriced "their column padding
> would help" belief survived weeks until a 20-minute probe measured it a
> wash (~52 vs ~56 GB/s, overlapping ranges).

### 4. Correctness gates on analytic answers and their output — never on real datasets

Never pick, drop, defer or tune a benchmark dataset by whether it flatters
you. "Run it but report it once we win" IS the cheating. Constructed
fixtures with adversarial cardinality beat real datasets for finding
defects.

> **The scar.** The recommendation to hold back two datasets "until we're
> ready" came from me, and was correctly named cheating. The gates that
> actually caught bugs were a 4,096-row fixture and CatBoost's own per-cell
> dumps — never a real dataset.

### 5. Same everything except the device — and defaults are theirs

Identical configuration on both arms; the device is the only variable, and
MSE parity enforces it mechanically. Adopt a neutral harness's rules
verbatim; change no timing code, no metric, no dataset, no competitor
parameter. A knob that makes your arm faster than what they ship
(`leaf_estimation_iterations=1` is 9 ms/tree) is cheating, not a win —
price it, then leave it at their default.

> **The scar.** "CatBoost as shipped" (MVS 0.8) versus our full-data arm
> looked like a legitimate comparison for a while. It is a different
> algorithm doing less work. The 8-significant-figure MSE match is what
> proves the arms are actually running the same thing.

### 6. Measurement kills hypotheses; reasoning does not

Before optimizing anything, decompose `ms/tree = a + b*rows`. One line —
ours 12.56 + 18.5us/1k, theirs 2.20 + 35.6 — explained every win and loss
on the board: per-row work twice as fast as their CPU, fixed cost 5.7x
theirs, crossover arithmetic rather than mystery. Predictions about kernels
are worthless until run: a fused kernel predicted 6-10x faster measured
1.17x SLOWER, because occupancy, not traffic, was the denominator.

> **The scar.** Five hypotheses on one bug: each killed by a single
> measurement, none killed by reasoning. Three careful inferences each
> found a real bug — without finding the bug being chased.

### 7. The box lies — alternate arms inside one window

Never A/B across time. One process, arms alternating per rep, medians
quoted with ranges, and a row is a finding only when the ranges are
disjoint. Report the outlier rather than dropping it. On a shared machine,
a concurrent run can drift your numbers invisibly — hold the lock.

> **The scar.** The same binary on the same fixture read 21.1 ms/tree at
> 20:36 and 12.1 at 20:55. An A/B across that gap reported a 1.79x win for
> a change that alternated measurement showed was 1.31x. A concurrent GPU
> run once drifted a peer's numbers 4.5x.

### 8. A check must be able to fail: hashed data, per-cell comparison, sabotage

A check whose expected value is the same in every cell verifies the total
and nothing about placement. Plant scattered, hashed values and compare
cell for cell against an independent tally. Then SABOTAGE THE PATH AND
WATCH THE CHECK MOVE — a digest cannot distinguish a working change from a
no-op, and reach is per-branch. One sabotage per mechanism; required
whenever the expected value is your own tally; keyed on evidence, never on
authorship.

> **The scar.** Uniform bins: 0 wrong of 512. Hashed bins, same kernel,
> same parameters: 490 wrong of 512. Two exclusions had reported that
> kernel correct at exactly the failing configuration. The first sabotage
> that mattered caught my own fixture; six catches in one day, four of them
> defects in the CHECK.

### 9. Probe the hardware yourself; trust no "missing" claim and no stdlib digit

Every capability denial gets a ten-line probe before it constrains a
design. Assume stdlib numerics are approximate until measured against an
external oracle (libm via FFI is the arbiter). Never persist a float as
decimal text alone — write decimal AND hex bits, load the hex.

> **The scar.** "Metal has no float atomics" and "Mojo has no warp
> primitives" were both false — wrong import paths read as missing
> hardware, with the docs and the docs-MCP wrong alongside. `log` with 5e-8
> absolute error silently re-decided DP plateau ties; `String(Float32)`
> comes back one ULP wrong 0.46% of the time; the compiler contracts FMAs
> across expressions where clang does not.

### 10. Record negatives; fix falsified documents in the same commit

A result that falsifies a document makes the document part of the result:
delete the false sentence, do not annotate it. Write down what did NOT
work, with the measurement, so nobody re-runs it. A lane reopens when its
measurement was taken in the wrong configuration.

> **The scar.** The warp-scan shape (a wash over a whole tree),
> footprint-by-address (1.006x), row-major layout (1.15-1.35x slower),
> pinned host partitions (2x slower than the copies they replaced) — each
> recorded once, none re-chased. PORTING.md carried "vector loads not
> ported" for weeks after the code had them; the stale sentence nearly
> launched a redundant port.

### 11. A measured, bit-identical win flips the default in the same session

If the outputs are bit-identical, the change cannot alter any user's
result — there is no risk to weigh, so there is nothing to defer. Switches
are temporary; a switch that outlives a positive measurement is a defect.

> **The scar.** Wins sat behind opt-in flags "to be safe" while the default
> stayed slow. The safety being purchased was zero by construction; the
> cost was every benchmark run in between.

### 12. Parallelize across directories; serialize on files

The predictor of integration pain is file convergence, not delegation
itself. In a shared checkout a peer's merge silently becomes your commit's
parent — report every commit as `%h parent %p`, never `git add -A`, and
never let a subagent run the full suite: own checks only, one merge-time
run.

**THE INDEX IS SHARED TOO, AND `git add -A` IS NOT THE ONLY WAY TO LOSE
IT.** A plain `git commit` takes whatever is in the index, including hunks
another session staged seconds ago, and its message then describes only your
half. Name the paths at COMMIT time, not just at `add` time:

    git add -- <paths> && git commit -o <paths>

**THE EXCEPTION THAT MAKES PEOPLE ABANDON `-o`, so it is written down
instead of rediscovered.** `-o` builds its tree from HEAD plus the
WORKING-TREE state of the named paths, so it does NOT carry a staged
deletion: after `git rm --cached <f>` the file is still on disk, `-o` reads
it back, and the commit reports "1 file changed" while `f` stays tracked.
For a deletion, stage it and use a plain `git commit` with `git diff
--cached` VERIFIED EMPTY of everything else first. Both failures happened in
one session on 2026-09-01: the deletion was silently undone by `-o`, the
wrong lesson was drawn ("stop using `-o`"), and the next commit swept three
of a peer lane's gbdt files into a commit about RandomForest. Fixed forward
in both directions; neither was rewritten.

> **The scar.** ~1.88M tokens of five-lane work produced integration breaks
> that existed ONLY BECAUSE of the parallelism — and also found defects a
> single session would not have. The variable that separated the two
> outcomes was whether lanes touched the same file.

### 13. Optimize the policy half; the plumbing half belongs to Modular

Andrew's rule, 2026-08-21. GPU work in this repository splits in two.
The POLICY half — accumulator designs, histogram layouts, block sizes,
replication factors, the kernel-matrix rows — is algorithm tuning: no
toolchain will ever choose it for us, it is where CatBoost's own per-arch
tuning lives, and it is where optimization effort goes. The PLUMBING
half — launch price, drain price, command encoding, kernel compilation,
missing device-side sort/scan libraries — is Modular's layer: file the
upstream ask, take the improvement when the toolchain ships it, and do
not build a private runtime around it. An Apple-only launch layer is the
canonical violation (it also breaks rule on one source).

> **The context.** The census that motivated this: 94 launches + 1 drain
> per covtype tree, matching CatBoost's kernel count with twelve fewer
> drains than their design — the count is at the algorithm's floor and
> the per-operation price is the platform's. Control plane ~2.2-2.7 ms
> of a ~16 ms tree on Metal vs ~0.6 ms for the same design on CUDA.

## The traps register — platform, not strategy

These cost real days. A fresh start hits them in the first week.

| Trap | Rule |
|---|---|
| `enqueue_copy(dst_buf, src_ptr=device)` | Silently does nothing — there is no device-to-device form. Route through a real copy kernel. |
| Host-buffer pointer into a kernel | Kernel writes nothing, silently, all zeros. Only `map_to_host()` sees kernel output — and it measured 2x slower than copies. |
| `wait_complete()` | Does not order copies enqueued after it was armed. Later copies need their own synchronize. |
| `get_attribute()` | 1.26 ms per call on Metal. Query once, thread the value through — theirs is a cached static for the same reason. |
| No float64 on device | Anything their code holds in `double` (scan accumulators, the Newton walker) either stays on the host or moves bits. Decide per site; record it. |
| No streams on Metal | Their overlap tricks do not port. Cut launch and sync COUNTS instead; syncs are the expensive syllable (~0.4 ms each). |
| Bit-reversed leaf order | Final leaf partitions sit in memory in bit-reversed leaf order. Never prefix-sum sizes to rebuild offsets — export the device's own offsets. |
| Mojo 1.0 surface | `def` does not imply `raises`; nested `def`s cannot capture; structs need `@fieldwise_init`; warp is `std.gpu.primitives.warp`; `stack_allocation` is memory, not registers. |

## The shape of the endgame, so you recognize it early

Done right, you arrive where this port arrived: **per-row work about twice
as fast as their CPU, fixed per-tree cost several times theirs.** That
means small datasets are lost to the launch floor and are cheap to lose — a
whole 500-tree fit at 50k rows is seconds — while everything from ~500k
rows up is yours, with the margin growing in scale: 1.25-1.32x at 800k,
1.35-1.51x at 2M, 1.8-2.1x at 4M, MSE matching theirs to eight significant
figures. Do not spend deviation budget chasing the launch floor. Spend it
where rows multiply: the histogram loop, the partition shuffle, the loss
passes — and on breadth, because every loss function and feature type they
support and you don't is a dataset you forfeit whole.
