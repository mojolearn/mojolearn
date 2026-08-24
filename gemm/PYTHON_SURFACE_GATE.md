# The Python surface's identity gate

Profile `mojolearn.identical.gemm.fp32.v1`. Contract
`gemm/IDENTICAL_FP32_CONTRACT.md`. The gate is
`python/mojolearn/tests/test_linalg_identity.py`, DEVIATIONS 950 to 959,
written 2026-08-24.

## Why it exists

`gemm/host_entry.mojo`, `bindings/_mojolearn_linalg.mojo` and
`python/mojolearn/_linalg_impl.py` put the profile's product in reach of a
Python caller on 2026-08-24. Nothing in that path compared its output against
`gemm/mojo_only/gemm_oracle.mojo::gemm_oracle`.

The build-time smoke test in `bindings/build_linalg.sh` proves the kernels
launch, that all five execution plans dispatch, that the three orientations
agree to `allclose`, and that the three refusals fire. It asserts no bit, and
it says so in its own comments. It also never runs on the artifact that
carries the claim, because `MOJOLEARN_NUMERIC_MODE=identical bash
bindings/build_linalg.sh` sets `MOJOLEARN_SKIP_BUILD_GATE=1` and skips it.

So the Python surface's identity was inherited by argument from the certified
kernel. This gate measures it through the surface.

## What it proves

Run under `MOJOLEARN_NUMERIC_MODE=identical` with a reference card, on one
machine, it establishes the following about the Python path.

- **The bits are `gemm_oracle`'s.** Every checked shape's output is hashed
  with the differ's FNV-1a64 and compared against the hash
  `tools/gemm_card.sh oracle` emitted for the same fixture. Bitwise, never
  `allclose`.
- **The op mapping is right.** `transpose_a` and `transpose_b` reach OP_NN,
  OP_NT and OP_TN, and the three agree bit for bit on the same logical
  matrices, which contract section 3 makes a requirement rather than a
  coincidence. Every shape has `m`, `n` and `k` pairwise distinct, so a
  transposed operand shows up as a different matrix and not as a last bit.
- **The strides are right.** A non-contiguous operand gives bit-identical
  output to its contiguous copy, including through the `a.T` view a caller
  reaches for when they want OP_TN.
- **`out=` is right.** Bit-identical to the allocated form, with the buffer
  poisoned with NaN first so an unwritten cell is loud.
- **Batch and launch invariance hold through the surface.** One row computed
  alone is bit-identical to that row inside a larger call, including when the
  smaller call dispatches a different execution plan and when the rows are
  gathered in an order they do not have in the operand.
- **The refusals fire.** All ten guards on this surface are made to raise, by
  type and by message.
- **The mode discipline holds.** Which `.so` loaded is read back from the
  binary rather than from the environment variable, and in a FAST process the
  default call raises and names both ways forward.

The shapes are chosen by leaf count, not by size. **Every bit-level assertion
includes shapes with `k > 128`**, because `contract_leaf_size(k)` returns `k`
below that and the profile degenerates to one serial ascending chain that
several different implementations reproduce. The self-chosen shapes cover
`P = 1`, `P` even, `P` odd with one carry, and `P = 5`, the smallest `P` that
carries twice.

## What it does not prove

- **It does not gate the kernel.** `gemm/mojo_only/gemm_device_check.mojo`
  does that, with five sabotages beside it. Nothing here is a second opinion
  about the arithmetic.
- **It is one machine.** The three-vendor claim is `gemm/README.md`'s and this
  gate contributes nothing to it. It is a check that the Python path returns
  what the oracle says, on whatever GPU it ran.
- **It cannot see a defect the fixture does not evaluate differently.** The
  reference card's fixture is `_exact`, a spread of `int / 2^20` values, and
  no leaf partial it produces is ever `-0.0`. The `PAD_PLUS_ZERO` sabotage is
  byte-identical under it, measured, and recorded in
  `bench/gemm_card_main.mojo`. The `-0.0` coverage lives in
  `gemm_device_check.mojo::_minus_zero_case` and does not reach the Python
  path. **A `-0.0` output cell has never crossed this boundary in a check.**
- **OP_NN has no external reference.** `bench/gemm_shapes.mojo` is four OP_TN
  rows and sixteen OP_NT rows and not one OP_NN, so the card cannot carry it.
  What stands behind OP_NN is its bit-for-bit agreement with the other two
  orientations plus a float64 accuracy arm, and that is weaker than a card
  row.
- **The degenerate shapes are not covered anywhere.** `gemm/host_entry.mojo`
  refuses `k == 0`, `m == 0` and `n == 0` and `_linalg_impl._operand` refuses
  a zero-size operand, so contract section 8's specified answers for them are
  unreachable from Python and untested. Lifting either refusal needs a gate
  first.
- **NaN and infinity are not covered.** Contract 9.1 declines to promise NaN
  payload bits, so a bitwise gate cannot assert on them and this one does not
  try.
- **It says nothing about FAST.** Contract 11.4 declines to promise that the
  FAST product even differs from the identical one, and the FAST arm of this
  gate checks refusals only. A green FAST run prints, in those words, that no
  bit was checked.

## How the reference is produced

    tools/gemm_card.sh oracle /tmp/gemm_oracle.card

That runs `bench/gemm_card_main.mojo`'s ORACLE arm, which is `gemm_oracle`,
the normative v1 answer, over `bench/gemm_shapes.mojo` at the host cap. It
needs no GPU and takes about a minute. The card is one record per stage,
`<seq> <tag> <dtype> <count> <hash>`, TAB separated, three stages per shape.

The gate does not transcribe the shape table. It solves `m`, `n` and `k` out
of the card's own element counts, because `A` is `m*k`, `B` is `n*k` and `C`
is `m*n` in all three orientations. What it does transcribe is the fixture
generator, the twenty shape names and the one-line op rule, and the names are
checked against the card position by position, so a table edit or a skipped
shape is a loud failure that names the table.

**The fixture is checked before any product is compared.** The card carries a
hash of its own inputs and the gate reproduces those inputs in numpy and
matches them first. A mismatch stops the arm rather than failing it, because
the conclusion is that the gate did not run and not that the surface is
wrong.

## Running it

    bash bindings/build_linalg.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh
    tools/gemm_card.sh oracle /tmp/gemm_oracle.card

    cd python && MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_GEMM_CARD=/tmp/gemm_oracle.card \
        python3 -m mojolearn.tests.test_linalg_identity

    cd python && MOJOLEARN_LINALG_GATE_ALLOW_FAST=1 \
        python3 -m mojolearn.tests.test_linalg_identity

The FAST half is a second process because the numeric mode is chosen at import
and cached, so one process cannot see both. It is opt-in through
`MOJOLEARN_LINALG_GATE_ALLOW_FAST` because a FAST run checks no bits and a CI
job that reached it by accident would report a green identity gate that never
looked at an identity.

**A missing card is a FAILURE and not a skip.** A silently skipped identity
test is worse than no test.

There is no pytest in this repository and this file adds none. The gate is a
program with a `main()` and it needs numpy, the package, and nothing else.
It lives outside the wheel, because `python/pyproject.toml` lists its packages
explicitly.

## The budget

Hashing is a byte-at-a-time Python loop in `tools/identity_trace_diff.py`, so
the card arm bounds itself with `MOJOLEARN_LINALG_GATE_BUDGET`, in float32
elements over A, B and C per shape. `MOJOLEARN_LINALG_GATE_FULL=1` removes it
and is what makes this gate's own card diffable stage for stage against the
oracle card.

A budget may never silence the identity arm, and three things enforce that
rather than intend it. A branch of contract section 6's leaf rule that the
budget would leave empty gets its cheapest shape admitted anyway, which is
what keeps `gram.32x32x1M` and its `L = 977` in every run. The arm fails if no
checked shape has `P > 1` for an op the card carries. Every deferred shape is
printed by name with the command that runs it.

## After a legitimate profile change

A change to the leaf rule (contract 7.1) or to the fold topology (7.2) creates
`...fp32.v2` and does not amend v1. When that happens, in this order.

1. The contract, the oracle and the kernel change, and
   `bindings/_mojolearn_linalg.mojo::linalg_profile_version_binding` and
   `_linalg_impl.PROFILE_VERSION` both go to 2. The gate asserts the version
   through `profile()`, so a half-done bump fails here rather than producing a
   mislabeled answer.
2. Re-emit the reference. `tools/gemm_card.sh oracle /tmp/gemm_oracle.card`.
   **The old card is not edited and not partially updated.** Every hash in it
   belongs to v1.
3. Re-run the gate against the new card. Nothing in
   `test_linalg_identity.py` needs editing for a v2, because it holds no
   expected bit patterns of its own. The one thing that does need editing is
   `contract_leaf_size`, `leaf_branch` and `K_LEAF_MIN` in that file if the
   profile constants moved, and those are used for reporting and for the
   coverage guard only. Nothing the gate asserts depends on them.

If `bench/gemm_shapes.mojo` changes instead, update `TABLE_NAMES` and
`table_op` in the gate. It will tell you so by name.
