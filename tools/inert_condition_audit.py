#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
r"""
inert_condition_audit.py -- a static census of the sabotage arms and of what
anybody has said about the inputs on which they CANNOT fire.

**THIS FILE HAS NEVER BEEN EXECUTED.** Not one line below has been run, no
output shown here was produced by running it, and the counts quoted in
`archive/research/NUMERICAL_BLINDNESS.md` were obtained by reading the tree with `grep` and
`sed`, not by running this. Treat every claim about its behavior as a claim
about source that has been read and not as a measurement. The first person to
run it should expect to fix something.

WHAT THIS IS FOR
================
`mojolearn` gates bitwise identity with SABOTAGE ARMS. An arm is a `comptime`
switch that changes exactly one numerical decision -- a fold order, an
association, a rounding, a seed, a flush -- and a gate is supposed to detect
the change. This is mutation testing (DeMillo, Lipton and Sayward, 1978), and
the classical obstacle is the EQUIVALENT MUTANT, a mutant no test can kill
because it is semantically identical to the original.

The arms here are NOT equivalent mutants. `scale * (q . k)` and
`sum_i (scale * q_i) k_i` are different functions of their inputs. They
coincide only ON PARTICULAR INPUTS -- and the whole problem is that those
inputs are exactly the tidy ones a careful numerical author reaches for. An
arm fired on a blind fixture reports GREEN and gates nothing, which is worse
than no arm at all because it is counted as coverage.

So the question this tool asks of every arm is not "does it pass" -- it cannot
run anything -- but:

    HAS ANYBODY WRITTEN DOWN THE INPUTS ON WHICH THIS ARM CANNOT FIRE?

and it sorts the answers into four buckets.

THE FOUR CLASSIFICATIONS
========================
Exactly one per arm, assigned by the precedence FIRED > INERT-STATED >
INERT-DERIVED > UNKNOWN.

    FIRED          Evidence exists that the arm MOVED BITS. See "WHAT COUNTS
                   AS FIRED" below -- there are two definitions and this tool
                   reports both, because they give very different numbers.
    INERT-STATED   A contract row, the arm's own doc block, or the declaring
                   file's sabotage ledger NAMES a condition on which the arm
                   cannot fire.
    INERT-DERIVED  Nobody stated one; a condition was derived by hand in
                   `archive/research/NUMERICAL_BLINDNESS.md` and is mirrored in `DERIVED` below.
    UNKNOWN        Nobody has said and nothing was derived. NOT AN ERROR. This
                   tool never raises on UNKNOWN and never exits non-zero for
                   it. It counts it, prints it, and that count is the finding.

WHAT COUNTS AS FIRED
====================
The narrow definition is "a run log under `bench/results/` shows the arm
moving". Applied to the five lanes this tool audits, that definition currently
yields ZERO, because none of these lanes has ever written a sabotage result
into `bench/results/`. Reporting only that number would be true and useless.

The wide definition also accepts an IN-SOURCE LEDGER that says MEASURED and
names the arm -- `mamba/checks/mamba_check.mojo`'s ledger and
`mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo`'s, plus
`IDENTITY_PATHS.md` rows that record arms "RUN AND BITTEN". Those are real
measurements that were simply written into source and into the path table
rather than into `bench/results/`.

Both counts are printed. `--fired-from bench` restricts to the narrow one.
Neither is a substitute for the other and the gap between them is itself
worth reading -- it says the lanes are recording their measurements in a place
no harness can find.

THE FALSE-NEGATIVE MODE OF THIS TOOL, STATED PLAINLY
====================================================
**A REGEX OVER SOURCE CAN FIND A STATED CONDITION. IT CANNOT VERIFY ONE.**

This tool reads text. It does not evaluate FP arithmetic, it does not know
what values a fixture holds, it does not know which shapes a gate runs at, and
it cannot tell a correct inert condition from a confidently wrong one. Every
one of the following is invisible to it, and each has already happened in this
repository at least once:

 1. A STATED CONDITION THAT IS FALSE. `base_n1_v1` was declared
    `L_MAX_SEED_ZERO`'s inert case when at `V == 1` the arm actually FIRES
    (DEVIATION 1490). This tool would have scored that arm INERT-STATED and
    been wrong about the direction of the claim.
 2. A STATED CONDITION THAT IS RIGHT AND UNMET. An arm can name its inert
    condition perfectly and still be fired only on fixtures that satisfy it.
    Statedness is not coverage.
 3. A CONDITION THAT IS TRUE BUT INCOMPLETE. `S12_SCALE_INTO_Q` is inert at
    every power-of-four `head_dim`, not only at the 16 somebody happened to
    notice (DEVIATION 1102).
 4. A DERIVATION FROM THE WRONG MODEL. `S17_DENOM_HALVING_TREE` looks inert
    at `s == 3` if you derive from the embedding lane's ADJACENT-PAIR tree,
    and is not, because its tree is the STRIDE-halving spelling and at `s = 3`
    folds `(0+2)` then `+1`. The two lanes' "balanced tree" are different
    trees. Derivations in `DERIVED` carry that risk and are marked with a
    confidence field for exactly this reason.
 5. AN ARM WITH NO SWITCH. `L_NLL_VIA_MAX_LOGSOFTMAX` is a contract-named arm
    with no `comptime` anywhere (DEVIATION 1457). It cannot appear in a census
    of declarations. `--contract-orphans` looks for that class specifically.
 6. AN ARM WHOSE SWITCH EXISTS AND WHOSE KERNEL IGNORES IT.
    `SAB_ACCUM_BY_ADD` is declared in `embedding_identical.mojo` and read by
    NO kernel there on purpose -- the wrong spelling lives in the caller. A
    reach audit that assumed declaration implies reach would mis-score it.

Item 1 is the one to worry about. This tool's INERT-STATED count is an upper
bound on how much is known and says nothing at all about how much is RIGHT.

USAGE
=====
    python3 tools/inert_condition_audit.py
    python3 tools/inert_condition_audit.py --lane training --verbose
    python3 tools/inert_condition_audit.py --fired-from bench
    python3 tools/inert_condition_audit.py --json > /tmp/arms.json
    python3 tools/inert_condition_audit.py --contract-orphans

It is a REPORTING tool. It exits 0 whatever it finds. It writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# WHERE TO LOOK
# ---------------------------------------------------------------------------
# The five lanes that carry the identical-arithmetic profiles. Other lanes
# (`ensemble/`, `hierarchy/`, `core/`, `cluster/`, `checks/`) also carry
# sabotage switches, most of them INTEGER-PARAMETERIZED rather than `-D`
# driven, and they are deliberately out of scope here: their arms are graded
# against a different kind of contract and mixing the two would produce a
# number nobody could interpret. `--all-lanes` includes them anyway, clearly
# labelled, for anyone who wants the whole-tree count.

LANES = {
    "gemm": "gemm",
    "mamba": "mamba",
    "transformer": "transformer",
    "training": "training",
    "embedding": "embedding",
}

EXTRA_LANES = {
    "core": "core",
    "ensemble": "ensemble",
    "hierarchy": "hierarchy",
    "cluster": "cluster",
    "metrics": "metrics",
    "original": "original",
}

# The contract per lane. `training` has two and both are searched for every
# training arm, because the loss and optimizer arms live in one directory.
CONTRACTS = {
    "gemm": ["gemm/IDENTICAL_FP32_CONTRACT.md",
             "archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md"],
    "mamba": ["mamba/IDENTICAL_MAMBA_CONTRACT.md",
              "archive/plans/mamba/IDENTICAL_BACKWARD_PLAN.md"],
    "transformer": ["transformer/IDENTICAL_TRANSFORMER_CONTRACT.md",
                    "archive/plans/transformer/IDENTICAL_BACKWARD_PLAN.md"],
    "training": ["training/IDENTICAL_LOSS_CONTRACT.md",
                 "training/IDENTICAL_OPTIMIZER_CONTRACT.md"],
    "embedding": ["embedding/IDENTICAL_EMBEDDING_CONTRACT.md"],
}

# The tree-wide tables that record what has actually been run.
PATH_TABLES = ["IDENTITY_PATHS.md", "archive/plans/UNWIRED.md", "archive/research/NOVELTY_NOTES.md"]

RESULTS_DIR = "bench/results"

# ---------------------------------------------------------------------------
# WHAT A SABOTAGE ARM LOOKS LIKE IN SOURCE
# ---------------------------------------------------------------------------
# Two spellings are in use and both are matched.
#
#   comptime SAB_FOLD_STRIDE = is_defined["MOJOLEARN_GEMM_SABOTAGE_..."]()
#   comptime SABOTAGE_BIAS_LAST = 1        # an Int-parameterized arm
#
# and the docs come in two shapes too: a `#:` comment block BEFORE the
# declaration (gemm, loss, optimizer, embedding, transformer backward, mamba
# backward, mamba scan) or a `"""..."""` docstring AFTER it (llama block,
# mamba block). Both are collected.

DECL_RE = re.compile(
    r"^comptime\s+"
    r"(?P<name>(?:SAB[A-Za-z0-9_]*|S\d+[A-Za-z0-9_]*|SABOTAGE_[A-Za-z0-9_]+))"
    r"\s*="
)

MACRO_RE = re.compile(r'MOJOLEARN_[A-Z0-9_]*SABOTAGE_[A-Z0-9_]+')

# Names that match DECL_RE and are NOT arms. Each is here for a reason and the
# reason is written down, because a bare exclusion list rots.
NOT_AN_ARM = {
    # Aggregates. `comptime if ANY_SABOTAGE` is how a check refuses to certify
    # a sabotaged build; it changes no arithmetic of its own.
    "ANY_SABOTAGE", "ANY_BWD_SABOTAGE", "ANY_LOSS_SABOTAGE",
    "ANY_EMB_SABOTAGE", "BLOCK_ANY_SABOTAGE", "BWD_ANY_SABOTAGE",
    # The zero of an Int-parameterized family. It IS the clean build.
    "SABOTAGE_NONE",
    # A tuning constant that happens to start with SAB_.
    "SAB_CHUNKS",
    # Assertions ABOUT the pinned spelling, not switches that change it.
    "S14B_USES_SIGMOID_MULTIPLY", "S14B_GUARD_THRESHOLD_IS_20",
}

# The vocabulary in which this tree writes an inert condition. Deliberately
# WIDE, because a narrow pattern would score a lane silent when it was only
# using different words -- and a false UNKNOWN is the failure mode that makes
# this whole census misleading in the direction of alarm.
INERT_RE = re.compile(
    r"\bINERT\b"
    r"|\binert\b"
    r"|\bvacuous"
    r"|\bVACUOUS"
    r"|cannot fire"
    r"|CANNOT FIRE"
    r"|cannot be made to fire"
    r"|CANNOT SEE"
    r"|cannot see it"
    r"|move[s]? NOTHING"
    r"|move[s]? nothing"
    r"|moved? no bits"
    r"|no bits move"
    r"|agree bit for bit"
    r"|agrees? at every"
    r"|bit-neutral"
    r"|bitwise inert"
    r"|legitimately agrees"
    r"|passes .{0,30}by construction"
    r"|PASS CLAUSE \(a\) BY CONSTRUCTION",
)

# **A BUG THIS AUDIT HAD AND THE .md RECORDS AS DEVIATION 1611.** `INERT_RE`
# matches the word "inert" inside "Never inert on an ordinary row", which is
# an assertion that the arm has NO inert condition -- the exact opposite of
# what the match would be counted as. Three arms are written that way
# (`SAB_W_VIA_EXP_LOGP`, `SAB_EMPTY_ROW_NEG_ZERO`, and the second half of
# `SAB_FOLD_READS_LAUNCH`), so the bug was worth 3 of 115 arms scored in the
# flattering direction. A sentence is counted as stating a condition only if
# it survives having its NEGATED forms removed first.
NEGATED_INERT_RE = re.compile(
    r"[Nn]ever inert"
    r"|NEVER INERT"
    r"|not inert"
    r"|NOT inert"
    r"|and never inert"
    r"|no (?:known )?inert (?:half|condition|case)"
    r"|NO inert half"
    r"|nothing known to make (?:this|it) inert"
    r"|no fixture property is known to make this inert"
)


def states_inert(text: str) -> bool:
    """Does this text NAME a condition on which the arm cannot fire?

    Strips the negated forms first. `Never inert on an ordinary row` is a
    claim that no inert condition exists, and counting it as a stated
    condition inflates the census in the one direction that matters.
    """
    if not text:
        return False
    stripped = NEGATED_INERT_RE.sub(" <negated> ", text)
    return bool(INERT_RE.search(stripped))

# A ledger row that records a MEASUREMENT rather than a prediction.
# The gemm lane writes "each shown to fail" where the mamba lane writes "RUN
# AND BITTEN". Both are records of a measurement and a pattern that caught only
# one of them would report the gemm forward six as never fired, which is the
# opposite of the truth.
MEASURED_LEDGER_RE = re.compile(
    r"LEDGER,?\s+MEASURED|WAS MEASURED TO DO|RUN AND BITTEN"
    r"|MEASURED \d{4}-|each shown to fail|shown to FAIL"
)
UNMEASURED_LEDGER_RE = re.compile(r"LEDGER,?\s+\*\*NOT MEASURED|NOT MEASURED")

# Evidence that a named arm moved something, for the `bench/results/` scan.
MOVED_RE = re.compile(
    r"\bmoved?\b|\bbit\b|\bBITTEN\b|\bfail(?:s|ed)?\b|\bdiffer|\bcells\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# THE DERIVED CONDITIONS
# ---------------------------------------------------------------------------
# Every entry here was derived BY HAND in `archive/research/NUMERICAL_BLINDNESS.md` and is
# mirrored here so the tool and the analysis cannot drift apart silently. The
# .md is the normative copy; if the two disagree, the .md is right and this
# dict is stale.
#
# `confidence` is not decoration. It is the difference between a derivation
# that follows from IEEE-754 alone and one that assumes something about how a
# particular loop is spelled -- and item 4 of the false-negative list is what
# happens when that distinction is not kept.
#
#   "algebraic"  follows from IEEE-754 round-to-nearest and the operation
#                alone. Safe to quote.
#   "structural" additionally assumes a shape rule (`contract_leaf_size`, a
#                tree topology, a loop bound) that was READ in this tree. Safe
#                only against the spelling that was read.
#   "shape"      a statement about which fixtures a gate happens to run, not
#                about arithmetic. Softest of the three.

DERIVED: dict[str, dict[str, str]] = {
    # ---- gemm forward ----------------------------------------------------
    "SAB_LEAF_READS_LAUNCH": {
        "condition": "P(k) == 1 (no fold exists to re-partition), or every "
                     "partial sum of the leaf chain exactly representable",
        "confidence": "structural",
    },
    "SAB_FOLD_STRIDE": {
        "condition": "P <= 2 -- at P == 1 there is no fold and at P == 2 the "
                     "stride pair IS the adjacent pair; also inert whenever "
                     "every node sum is exactly representable",
        "confidence": "algebraic",
    },
    "SAB_PAD_PLUS_ZERO": {
        "condition": "P a power of two (no level has an odd tail, so nothing "
                     "is padded) AND no carried partial is -0.0; x + (+0.0) "
                     "== x for every finite x, every infinity and every NaN "
                     "except x == -0.0",
        "confidence": "structural",
    },
    "SAB_FOLD_SERIAL": {
        "condition": "P <= 3 for an ADJACENT-PAIR tree, where the tree is the "
                     "chain node for node; the embedding lane states exactly "
                     "this bound for its own tree. NOT transferable to a "
                     "stride-halving tree, which separates at 3",
        "confidence": "structural",
    },
    "SAB_NODE_ORDER": {
        "condition": "P == 1, or a single-block launch (arrival order is "
                     "logical order), or all P leaf partials bitwise equal",
        "confidence": "structural",
    },
    "SAB_LEAF_ROTATE": {
        "condition": "P == 1, or one block folds (rotation by 0), or all P "
                     "leaf partials bitwise equal",
        "confidence": "structural",
    },
    # ---- gemm backward ---------------------------------------------------
    "SAB_BWD_BIAS_AXIS": {
        "condition": "m == n AND dC symmetric, where sum over j equals sum "
                     "over i termwise. The doc names only the SHAPE half of "
                     "this, which makes a shape-comparing gate blind but is "
                     "not itself a bitwise inert condition",
        "confidence": "algebraic",
    },
    # ---- mamba scan ------------------------------------------------------
    "SAB_S8_CUDA_PAIRING": {
        "condition": "every one of the three products exactly representable "
                     "(the multiplication is then exact and associative), or "
                     "any factor a power of two, or any factor zero",
        "confidence": "algebraic",
    },
    "SAB_S11_D_FIRST": {
        "condition": "every prefix sum of (D*u, C_0 h_0, ..., C_{N-1} "
                     "h_{N-1}) exactly representable in both orders, or "
                     "D*u == +0.0 with no -0.0 to launder",
        "confidence": "algebraic",
    },
    "SAB_S10_DESCENDING": {
        "condition": "DSTATE <= 2, or every partial sum exactly "
                     "representable. A two-term ADD chain is order free; a "
                     "two-term FMA chain is NOT, which is why the "
                     "transformer's B01 twin states 2 and this states 2 for a "
                     "different reason",
        "confidence": "algebraic",
    },
    # ---- mamba block -----------------------------------------------------
    "SAB_S13_BIAS_LAST": {
        "condition": "every prefix sum of (bias, w_0 x_0, ..., w_3 x_3) "
                     "exactly representable in both orders; sufficient that "
                     "all five share one binade with a sum under 24 "
                     "significant bits. Also inert at bias == +0.0 except "
                     "where the seed would launder a -0.0",
        "confidence": "algebraic",
    },
    "SAB_S13_TAPS_REVERSED": {
        "condition": "K <= 2, or every partial sum of the four tap products "
                     "exactly representable in both directions",
        "confidence": "algebraic",
    },
    "SAB_S1_FOLD_DESCENDING": {
        "condition": "d_model <= 2, or every partial sum exactly "
                     "representable. PER ROW, not per fixture -- MEASURED at "
                     "1 of 4 rows at d_model 8 and 3 of 4 at 16, so short "
                     "folds are mostly blind and the blindness shrinks with "
                     "length rather than vanishing",
        "confidence": "algebraic",
    },
    "SAB_S12_MUL_SIGMOID": {
        "condition": "z such that 1 + exp(-z) rounds to exactly 1.0 (about "
                     "z >= 17 in FP32), where sigmoid is exactly 1.0 and "
                     "z*1.0 == z/1.0 == z; also z == +/-0.0; also any z where "
                     "1 + exp(-z) is an exact power of two",
        "confidence": "algebraic",
    },
    "SAB_S17_OP_NUMBERING": {
        "condition": "op(A) == A elementwise -- A symmetric, or a 1x1, or a "
                     "vector shape where the transpose is the identity",
        "confidence": "algebraic",
    },
    # ---- mamba backward --------------------------------------------------
    "SAB_BWD_WS_FROM_FORWARD": {
        "condition": "the forward workspace is at least the backward's "
                     "requirement at every call, i.e. allocation slack "
                     "absorbs the error. The gemm lane paid for this exact "
                     "case at 64x4 and only saw it at 64x64",
        "confidence": "structural",
    },
    # ---- transformer block -----------------------------------------------
    "SAB_S09_ROPE_HALVES_SWAPPED": {
        "condition": "sin(pos * theta) == +0.0, which holds at absolute "
                     "position 0 for every frequency. PER CELL. The backward "
                     "twin B09_ROPE_HALVES_ADJACENT STATES this and the "
                     "forward does not, which is an asymmetry in the record "
                     "and not in the arithmetic",
        "confidence": "algebraic",
    },
    "SAB_S12_SCALE_INTO_Q": {
        "condition": "scale == 2^j, i.e. head_dim a power of FOUR (1, 4, 16, "
                     "64, 256). Exact scaling commutes with every product and "
                     "every partial sum, so the two spellings agree on ALL "
                     "inputs. DEVIATION 1102 found this at head_dim 16",
        "confidence": "algebraic",
    },
    "SAB_S10_ROPE_FUSED": {
        "condition": "position 0 (sin == +0.0 kills the second term), or "
                     "q*cos exactly representable so the fma and the "
                     "three-rounding form agree",
        "confidence": "algebraic",
    },
    "SAB_S20_SILU_MUL_SIGMOID": {
        "condition": "identical to SAB_S12_MUL_SIGMOID -- z >= about 17 where "
                     "1 + exp(-z) rounds to 1.0, z == +/-0.0, or 1 + exp(-z) "
                     "an exact power of two",
        "confidence": "algebraic",
    },
    "SAB_S05_OP_NUMBERING": {
        "condition": "op(A) == A elementwise. Same condition as the mamba "
                     "lane's S17 twin",
        "confidence": "algebraic",
    },
    "SAB_S18_RECIPROCAL_MUL": {
        "condition": "denom an exact power of two, where 1/denom is exact and "
                     "the multiply rounds once. This is the same closed-form "
                     "condition the loss lane STATES for its two reciprocal "
                     "arms and the optimizer lane STATES for RECIP_MUL; the "
                     "transformer's copy does not state it",
        "confidence": "algebraic",
    },
    # ---- transformer backward -------------------------------------------
    "SAB_B09_ROPE_TRANSPOSE_SIGN": {
        "condition": "sin == +0.0, i.e. absolute position 0, PER CELL. Stated "
                     "at the arm; recorded here because the arm also asks the "
                     "gate to COUNT the moved cells and refuse if the count "
                     "equals the position-0 population, which is the only "
                     "form of this condition that is checkable",
        "confidence": "algebraic",
    },
    "SAB_B13_MASK_ZEROES_GRAD": {
        "condition": "no masked cell has dS == -0.0, i.e. dy_j - z >= 0 "
                     "everywhere masked. Roughly half of masked cells is the "
                     "arm's own prediction and the count is the assertion",
        "confidence": "algebraic",
    },
    # ---- training, loss --------------------------------------------------
    "SAB_MAX_SEED_ZERO": {
        "condition": "the row contains a positive logit. NOTE that this is "
                     "the direction DEVIATION 1490 got backwards: at V == 1 "
                     "with a negative logit the arm FIRES, and base_n1_v1 was "
                     "declared its inert case. A row of k hashed logits is "
                     "all-negative with probability 2^-k, so 'inert' here is "
                     "a PROBABILISTIC statement about a fixture and never an "
                     "invariant (DEVIATION 1491)",
        "confidence": "algebraic",
    },
    "SAB_EXP_STDLIB": {
        "condition": "NUMERIC_FAST (identical_exp IS the stdlib there), or "
                     "shift == +0.0 at every class, which is the whole "
                     "exact-fixture family of contract 12.1",
        "confidence": "structural",
    },
    "SAB_NLL_VIA_ADDBACK": {
        "condition": "the row is centered enough that (m + logdenom) loses no "
                     "bits, i.e. m and logdenom within a few binades. A large "
                     "common offset is what separates, and adding a constant "
                     "to every logit is a transformation the pinned spelling "
                     "is invariant under BY CONSTRUCTION",
        "confidence": "algebraic",
    },
    "SAB_NEG_VIA_ZERO_SUB": {
        "condition": "x != +0.0 at every cell. 0.0 - x equals -x for every "
                     "input except x == +0.0, where IEEE negation gives -0.0 "
                     "and the subtraction gives +0.0",
        "confidence": "algebraic",
    },
    "SAB_SMOOTH_FUSED_COMBINE": {
        "condition": "eps == 0 (the smoothing arm is not spelled), or the "
                     "product exactly representable so the fused and unfused "
                     "combines round alike",
        "confidence": "algebraic",
    },
    "SAB_IGNORED_ROW_SKIPPED": {
        "condition": "the gate pre-fills the output with zeros -- which a "
                     "fresh allocation may or may not do. This is a condition "
                     "on the HARNESS and not on the data, and it is the one "
                     "kind of inert condition a fixture audit cannot see",
        "confidence": "shape",
    },
    # `SAB_W_VIA_EXP_LOGP` IS DELIBERATELY ABSENT. It says of itself "Never
    # inert on an ordinary row", which is an assertion that NO inert condition
    # exists, and putting a "none known" string here would have the tool score
    # it INERT-DERIVED. It is an honest UNKNOWN and the census counts it as
    # one. `SAB_MHAT_FORM`, `SAB_POW_EXPLOG` and `SAB_EMPTY_ROW_NEG_ZERO` are
    # absent for the same reason. **A "none known" entry in a derived table is
    # the same mistake as a negated "inert" in a doc block** -- DEVIATION 1611
    # in one dictionary instead of one regex.
    "SAB_GRAD_DIVISOR_IS_N": {
        "condition": "no ignored row under MEAN (divisor == N), and every SUM "
                     "reduction",
        "confidence": "algebraic",
    },
    "SAB_DENOM_SERIAL_CHAIN": {
        "condition": "V <= 128, where P == 1 and the gemm tree performs no "
                     "addition, so the routed answer IS the serial ascending "
                     "chain",
        "confidence": "structural",
    },
    # ---- training, optimizer --------------------------------------------
    "SAB_RSQRT": {
        "condition": "no closed form. DEVIATION 741 measured the pinned "
                     "1/sqrt off the correctly-rounded rsqrt on 134,858 of "
                     "520,133 positive-normal lanes, so about three quarters "
                     "of single inputs agree and a small fixture is likely "
                     "blind by chance rather than by structure",
        "confidence": "shape",
    },
    "SAB_MOMENT_LERP": {
        "condition": "g and m in the same binade with no low bits to lose, "
                     "i.e. m + c1*(g - m) and the product-then-fma form round "
                     "alike. Stated at the arm; restated here because 'same "
                     "binade' is one of the taxonomy rows and this is its "
                     "instance",
        "confidence": "algebraic",
    },
    # `SAB_MHAT_FORM` IS DELIBERATELY ABSENT -- see the note above
    # `SAB_GRAD_DIVISOR_IS_N`. "No fixture property is known to make this
    # inert, which is itself a claim that has not been checked" is the arm's
    # own sentence and it is the right way to write an UNKNOWN.
    "SAB_POW_RUNNING": {
        "condition": "t <= 6 -- the running product and pow_int_f32 agree "
                     "exactly through t = 6 for beta = 0.9 and 0.999 alike. "
                     "PREDICTED off-repository, NOT measured, and the arm "
                     "says so",
        "confidence": "algebraic",
    },
    "SAB_SCALARS_PER_ELEMENT": {
        "condition": "ALWAYS -- the recomputation uses the same pinned "
                     "primitives, so no bits move by design. It is a REACH "
                     "probe whose predicted answer is 'inert', declared in "
                     "advance, which is the only circumstance in which such "
                     "an arm is worth having",
        "confidence": "structural",
    },
    "SAB_NESTEROV_ORDER": {
        "condition": "momentum == 0, or t == 1 where b == g makes the two "
                     "readings one expression",
        "confidence": "algebraic",
    },
    "SAB_ADAMW_AS_ADAM": {
        "condition": "weight_decay == 0, which is the REFERENCE'S OWN "
                     "DEFAULT. The arm calls itself the single most likely "
                     "vacuous gate in the lane and it is right to",
        "confidence": "algebraic",
    },
    # ---- embedding -------------------------------------------------------
    "SAB_SEED_SEEDLESS": {
        "condition": "every cell has at least one contributor OR its sole "
                     "contributor is not -0.0. Seeded gives 0x00000000 and "
                     "seedless 0x80000000 only at a lone -0.0",
        "confidence": "algebraic",
    },
    "SAB_EMPTY_ROW_SKIPPED": {
        "condition": "the gate pre-fills dW with zeros. Harness condition, "
                     "not data condition -- the twin of the loss lane's "
                     "IGNORED_ROW_SKIPPED and inert for the same reason",
        "confidence": "shape",
    },
    "SAB_FOLD_READS_LAUNCH": {
        "condition": "every run has length <= 1, where there is no fold to "
                     "rotate. The arm states 'never inert on a run of length "
                     ">= 2', which is the same statement read forwards",
        "confidence": "structural",
    },
    "SAB_SORT_TIE_REVERSED": {
        "condition": "no duplicate ids, or every duplicate's rows bitwise "
                     "equal",
        "confidence": "algebraic",
    },
    "SAB_PAD_ROW_NEG_ZERO": {
        "condition": "no padding_idx configured",
        "confidence": "shape",
    },
    "SAB_GATHER_NO_FLUSH": {
        "condition": "no subnormal weight. NOT inert on Apple, unusually -- a "
                     "gather performs no arithmetic, so a raw copy of a "
                     "subnormal survives even an FTZ backend, which makes it "
                     "the only flush arm in the lane a single-column run can "
                     "see",
        "confidence": "algebraic",
    },
    "SAB_ACCUM_REFILLS": {
        "condition": "a single-microbatch gate, where there is no carry to "
                     "erase",
        "confidence": "shape",
    },
    "SABOTAGE_BIAS_LAST": {
        "condition": "the runtime twin of SAB_S13_BIAS_LAST and inert on the "
                     "same inputs -- every prefix sum of (bias, taps) exactly "
                     "representable in both orders",
        "confidence": "algebraic",
    },
}


# ---------------------------------------------------------------------------
# THE RECORD FOR ONE ARM
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    name: str
    lane: str
    path: str
    line: int
    macro: str = ""              # the -D name, empty for Int-parameterized
    doc: str = ""                # the arm's own comment block or docstring
    doc_states_inert: bool = False
    ledger_states_inert: bool = False
    contract_rows: list[str] = field(default_factory=list)
    contract_states_inert: bool = False
    fired_bench: list[str] = field(default_factory=list)
    fired_source: list[str] = field(default_factory=list)
    derived: str = ""
    derived_confidence: str = ""
    classification: str = "UNKNOWN"

    @property
    def stated(self) -> bool:
        return (self.doc_states_inert
                or self.ledger_states_inert
                or self.contract_states_inert)

    def where_stated(self) -> str:
        bits = []
        if self.doc_states_inert:
            bits.append("doc")
        if self.ledger_states_inert:
            bits.append("ledger")
        if self.contract_states_inert:
            bits.append("contract")
        return "+".join(bits) if bits else "--"

    def classify(self, fired_from: str) -> None:
        fired = self.fired_bench if fired_from == "bench" else (
            self.fired_bench + self.fired_source)
        if fired:
            self.classification = "FIRED"
        elif self.stated:
            self.classification = "INERT-STATED"
        elif self.derived:
            self.classification = "INERT-DERIVED"
        else:
            self.classification = "UNKNOWN"


# ---------------------------------------------------------------------------
# READING THE TREE
# ---------------------------------------------------------------------------

def read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def mojo_files(root: str, lane_dir: str) -> list[str]:
    out = []
    for base, dirs, names in os.walk(os.path.join(root, lane_dir)):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]
        for n in names:
            if n.endswith(".mojo"):
                out.append(os.path.join(base, n))
    return sorted(out)


def doc_block_before(lines: list[str], idx: int) -> str:
    """The contiguous comment block immediately above a declaration.

    Walks UP from the declaration and stops at the first line that is neither
    a `#` comment nor blank. Blank lines are skipped rather than terminating,
    because the loss lane puts one between an arm's block and the next.
    """
    out = []
    i = idx - 1
    blanks = 0
    while i >= 0:
        s = lines[i].rstrip("\n")
        if s.strip() == "":
            blanks += 1
            if blanks > 1:
                break
            i -= 1
            continue
        if not s.lstrip().startswith("#"):
            break
        blanks = 0
        out.append(s)
        i -= 1
    return "\n".join(reversed(out))


def doc_block_after(lines: list[str], idx: int) -> str:
    """The declaration's own text plus any `\"\"\"...\"\"\"` that follows it.

    The llama and mamba block files put the explanation in a docstring AFTER
    the `is_defined[...]()` call rather than in a `#:` block before it, and an
    audit that read only one of the two shapes would score half a lane silent.
    """
    out = []
    i = idx
    n = len(lines)
    # the declaration itself, which may span several lines
    while i < n and len(out) < 8:
        s = lines[i].rstrip("\n")
        out.append(s)
        i += 1
        if s.rstrip().endswith("]()") or s.rstrip().endswith(")"):
            break
    # an immediately following docstring
    while i < n and lines[i].strip() == "":
        i += 1
    if i < n and lines[i].lstrip().startswith('"""'):
        j = i
        out.append(lines[j].rstrip("\n"))
        if lines[j].count('"""') < 2:
            j += 1
            while j < n and '"""' not in lines[j]:
                out.append(lines[j].rstrip("\n"))
                j += 1
            if j < n:
                out.append(lines[j].rstrip("\n"))
    return "\n".join(out)


def file_ledger(text: str) -> tuple[str, bool]:
    """The declaring file's own sabotage ledger, and whether it says MEASURED.

    This matters more than it looks. Three of the mamba scan arms carry NO
    inert condition in their own doc block and the file-level ledger below
    them says the thing that matters -- 'A FIXTURE WHOSE ONLY SHAPE IS L = 1
    CANNOT SEE S5, S6 OR S9 AT ALL'. An audit that read only doc blocks would
    call those arms UNKNOWN and would be wrong.
    """
    measured = bool(MEASURED_LEDGER_RE.search(text))
    return text, measured


def find_arms(root: str, lanes: dict[str, str]) -> list[Arm]:
    arms: list[Arm] = []
    for lane, lane_dir in lanes.items():
        for path in mojo_files(root, lane_dir):
            text = read(path)
            if "SAB" not in text and "SABOTAGE" not in text:
                continue
            lines = text.splitlines(keepends=True)
            ledger_text, _ = file_ledger(text)
            for i, raw in enumerate(lines):
                m = DECL_RE.match(raw)
                if not m:
                    continue
                name = m.group("name")
                if name in NOT_AN_ARM:
                    continue
                before = doc_block_before(lines, i)
                after = doc_block_after(lines, i)
                doc = (before + "\n" + after).strip()
                macro = ""
                mm = MACRO_RE.search(after)
                if mm:
                    macro = mm.group(0)
                arm = Arm(
                    name=name,
                    lane=lane,
                    path=os.path.relpath(path, root),
                    line=i + 1,
                    macro=macro,
                    doc=doc,
                )
                arm.doc_states_inert = states_inert(doc)
                # The file-level ledger, restricted to paragraphs that name
                # this arm. A file-wide search would score every arm in a file
                # STATED as soon as one paragraph said "inert" anywhere.
                short = short_name(name)
                for para in re.split(r"\n\s*\n", ledger_text):
                    if short and short in para and states_inert(para):
                        if para.strip() not in doc:
                            arm.ledger_states_inert = True
                            break
                arms.append(arm)
    return arms


def short_name(name: str) -> str:
    """`SAB_S12_SCALE_INTO_Q` -> `S12_SCALE_INTO_Q`.

    Contracts and ledgers refer to arms by a LANE-PREFIXED short name -- `L_`
    in the loss contract, `OPT_SAB_` in the optimizer contract, `G_` for a
    gemm arm re-routed through another lane's gate. Stripping our own prefix
    is what makes those references findable at all.
    """
    for p in ("SAB_BWD_", "SAB_", "SABOTAGE_"):
        if name.startswith(p):
            return name[len(p):] if p != "SAB_BWD_" else "BWD_" + name[len(p):]
    return name


def alias_forms(arm: Arm) -> list[str]:
    """Every spelling a contract might use for this arm."""
    s = short_name(arm.name)
    out = {arm.name, s}
    for p in ("L_", "OPT_SAB_", "G_", "EMB_SAB_", "SAB_"):
        out.add(p + s)
    if arm.macro:
        out.add(arm.macro)
    return sorted(x for x in out if x)


def scan_contracts(root: str, arms: list[Arm]) -> None:
    cache: dict[str, list[str]] = {}
    for arm in arms:
        for rel in CONTRACTS.get(arm.lane, []):
            if rel not in cache:
                cache[rel] = read(os.path.join(root, rel)).splitlines()
            for ln in cache[rel]:
                if any(a in ln for a in alias_forms(arm)):
                    arm.contract_rows.append(rel + ": " + ln.strip()[:200])
                    if states_inert(ln):
                        arm.contract_states_inert = True
        # A lane's own `*_check.mojo` ledger is a contract in everything but
        # name -- `loss_check.mojo` carries the witness/inert table that the
        # loss contract's section 10.1 only summarizes.
        for path in mojo_files(root, LANES.get(arm.lane, arm.lane)):
            if not path.endswith("_check.mojo"):
                continue
            key = path
            if key not in cache:
                cache[key] = read(path).splitlines()
            for ln in cache[key]:
                if any(a in ln for a in alias_forms(arm)):
                    if states_inert(ln) or " / " in ln:
                        arm.contract_rows.append(
                            os.path.relpath(path, root) + ": "
                            + ln.strip()[:200])
                        if states_inert(ln):
                            arm.contract_states_inert = True


def scan_bench_results(root: str, arms: list[Arm]) -> None:
    """The NARROW definition of FIRED. Expect zero for these five lanes."""
    hits: dict[str, list[str]] = {}
    base = os.path.join(root, RESULTS_DIR)
    for cur, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d != ".git"]
        for n in names:
            p = os.path.join(cur, n)
            try:
                if os.path.getsize(p) > 8 * 1024 * 1024:
                    continue
            except OSError:
                continue
            text = read(p)
            if "SABOTAGE" not in text and "SAB_" not in text:
                continue
            for ln in text.splitlines():
                if "SAB" not in ln:
                    continue
                if not MOVED_RE.search(ln):
                    continue
                for tok in set(MACRO_RE.findall(ln)):
                    hits.setdefault(tok, []).append(
                        os.path.relpath(p, root) + ": " + ln.strip()[:160])
    for arm in arms:
        if arm.macro and arm.macro in hits:
            arm.fired_bench.extend(hits[arm.macro][:3])


def scan_source_ledgers(root: str, arms: list[Arm]) -> None:
    """The WIDE definition of FIRED.

    An in-source ledger marked MEASURED, or an `IDENTITY_PATHS.md` row that
    records arms RUN AND BITTEN. These are real measurements filed somewhere a
    results harness will never look, which is a finding of its own.
    """
    tables = []
    for rel in PATH_TABLES:
        tables.append((rel, read(os.path.join(root, rel))))
    for arm in arms:
        decl = read(os.path.join(root, arm.path))
        measured = bool(MEASURED_LEDGER_RE.search(decl))
        short = short_name(arm.name)
        if measured and short:
            for para in re.split(r"\n\s*\n", decl):
                if not MEASURED_LEDGER_RE.search(para) and short not in para:
                    continue
                if short in para and MOVED_RE.search(para):
                    # only the ledger paragraph itself, not the arm's own doc
                    if MEASURED_LEDGER_RE.search(para) or re.search(
                            r"^\s*#\s+" + re.escape(short) + r"\s{2,}",
                            para, re.M):
                        arm.fired_source.append(
                            arm.path + ": in-source MEASURED ledger")
                        break
        # the lane's own check file may hold the measured ledger instead
        for path in mojo_files(root, LANES.get(arm.lane, arm.lane)):
            if not path.endswith("_check.mojo"):
                continue
            t = read(path)
            if not MEASURED_LEDGER_RE.search(t):
                continue
            for para in re.split(r"\n\s*\n", t):
                if not MEASURED_LEDGER_RE.search(para):
                    continue
                if short and short in para:
                    arm.fired_source.append(
                        os.path.relpath(path, root)
                        + ": MEASURED sabotage ledger")
                    break
        for rel, t in tables:
            for para in re.split(r"\n\s*\n", t):
                if short and short in para and "RUN AND BITTEN" in para:
                    arm.fired_source.append(rel + ": RUN AND BITTEN")
                    break
        arm.fired_source = sorted(set(arm.fired_source))[:3]


def attach_derived(arms: list[Arm]) -> None:
    for arm in arms:
        d = DERIVED.get(arm.name)
        if d:
            arm.derived = d["condition"]
            arm.derived_confidence = d["confidence"]


def contract_orphans(root: str, arms: list[Arm]) -> list[str]:
    """Arms a contract NAMES that have no `comptime` switch anywhere.

    False-negative item 5. `L_NLL_VIA_MAX_LOGSOFTMAX` is the known instance
    (DEVIATION 1457) -- contract 4.2(c) is a pinned decision with no falsifier
    in the tree, and a census of declarations structurally cannot see it.
    """
    declared = set()
    for a in arms:
        declared.update(alias_forms(a))
    found: list[str] = []
    pat = re.compile(r"\b((?:L|G|OPT_SAB|EMB_SAB|SAB|OPT)_[A-Z0-9_]{4,})\b")
    for rels in CONTRACTS.values():
        for rel in rels:
            for ln in read(os.path.join(root, rel)).splitlines():
                for tok in pat.findall(ln):
                    if tok in declared:
                        continue
                    if tok in NOT_AN_ARM:
                        continue
                    found.append(rel + ": " + tok)
    return sorted(set(found))


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

HEADER = (
    "inert_condition_audit -- who has written down the inputs on which each "
    "sabotage arm CANNOT fire"
)


def print_table(arms: list[Arm], verbose: bool) -> None:
    w_name = max([len(a.name) for a in arms] + [12])
    print()
    print(HEADER)
    print("=" * len(HEADER))
    print()
    print("  A regex over source can FIND a stated condition. It cannot")
    print("  VERIFY one. INERT-STATED is an upper bound on what is known and")
    print("  says nothing about what is RIGHT -- DEVIATION 1490 is a stated")
    print("  condition that was inverted, and this tool would have counted it.")
    print()
    fmt = "  {:<" + str(w_name) + "}  {:<12}  {:<14}  {:<10}  {}"
    print(fmt.format("arm", "lane", "class", "stated-in", "file:line"))
    print("  " + "-" * (w_name + 60))
    for a in sorted(arms, key=lambda x: (x.lane, x.path, x.line)):
        print(fmt.format(a.name, a.lane, a.classification,
                         a.where_stated(), a.path + ":" + str(a.line)))
        if verbose:
            if a.derived:
                print("      derived (" + a.derived_confidence + ") "
                      + a.derived)
            for r in a.contract_rows[:2]:
                print("      contract  " + r)
            for r in (a.fired_bench + a.fired_source)[:2]:
                print("      fired     " + r)


def print_summary(arms: list[Arm], fired_from: str) -> None:
    order = ["FIRED", "INERT-STATED", "INERT-DERIVED", "UNKNOWN"]
    lanes = sorted({a.lane for a in arms})
    print()
    print("PER-LANE SUMMARY")
    print("================")
    fmt = "  {:<14}  " + "  ".join("{:>13}" for _ in order) + "  {:>7}"
    print(fmt.format("lane", *order, "total"))
    print("  " + "-" * (14 + 15 * len(order) + 10))
    for lane in lanes:
        sub = [a for a in arms if a.lane == lane]
        counts = [sum(1 for a in sub if a.classification == c) for c in order]
        print(fmt.format(lane, *[str(c) for c in counts], str(len(sub))))
    counts = [sum(1 for a in arms if a.classification == c) for c in order]
    print("  " + "-" * (14 + 15 * len(order) + 10))
    print(fmt.format("ALL", *[str(c) for c in counts], str(len(arms))))
    print()
    n_bench = sum(1 for a in arms if a.fired_bench)
    n_src = sum(1 for a in arms if a.fired_source and not a.fired_bench)
    print("  FIRED is being counted from: " + fired_from)
    print("    arms with a bench/results/ witness ......... " + str(n_bench))
    print("    arms with only an in-source MEASURED ledger  " + str(n_src))
    if n_bench == 0:
        print("    NOTE. Zero bench/results/ witnesses across these lanes.")
        print("    The measurements that exist were filed into source")
        print("    comments and into IDENTITY_PATHS.md, where no results")
        print("    harness will ever find them. That is a finding about the")
        print("    RECORD, not about the arms.")
    print()
    n_stated = sum(1 for a in arms if a.stated)
    print("  arms with a condition stated ANYWHERE (regardless of class)  "
          + str(n_stated) + " of " + str(len(arms)))
    print("  arms with a hand-derived condition in archive/research/NUMERICAL_BLINDNESS.md  "
          + str(sum(1 for a in arms if a.derived)))
    print("  arms nobody has characterized at all .........................  "
          + str(sum(1 for a in arms if a.classification == "UNKNOWN")))
    print()
    print("  This tool does not gate. It exits 0. UNKNOWN is a count, not a")
    print("  failure -- an audit that raised on UNKNOWN would be deleted by")
    print("  the first lane in a hurry, and then nobody would have the count.")
    print()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Static census of mojolearn sabotage arms and their "
                    "stated or derived inert conditions. Reports, never "
                    "gates. HAS NEVER BEEN RUN.")
    ap.add_argument("--root", default=".",
                    help="repository root (default: the working directory)")
    ap.add_argument("--lane", action="append", default=None,
                    help="restrict to one lane; repeatable")
    ap.add_argument("--all-lanes", action="store_true",
                    help="include the non-profile lanes (core, ensemble, "
                         "hierarchy, cluster, metrics, original), clearly "
                         "labelled. Their arms are graded against a "
                         "different kind of contract and mixing the counts "
                         "produces a number nobody can interpret.")
    ap.add_argument("--fired-from", choices=["bench", "any"], default="any",
                    help="'bench' counts FIRED only from bench/results/ run "
                         "logs; 'any' also accepts an in-source MEASURED "
                         "ledger or an IDENTITY_PATHS.md 'RUN AND BITTEN' "
                         "row (default)")
    ap.add_argument("--verbose", action="store_true",
                    help="print each arm's derived condition, contract rows "
                         "and fired evidence")
    ap.add_argument("--json", action="store_true",
                    help="emit the arm records as JSON instead of a table")
    ap.add_argument("--contract-orphans", action="store_true",
                    help="list arm-shaped names a contract mentions that have "
                         "no comptime switch anywhere (false-negative 5)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    lanes = dict(LANES)
    if args.all_lanes:
        lanes.update(EXTRA_LANES)
    if args.lane:
        lanes = {k: v for k, v in lanes.items() if k in set(args.lane)}
        if not lanes:
            print("no such lane; known lanes: "
                  + ", ".join(sorted(set(LANES) | set(EXTRA_LANES))),
                  file=sys.stderr)
            return 0

    arms = find_arms(root, lanes)
    if not arms:
        print("no sabotage arms found under " + root, file=sys.stderr)
        return 0
    scan_contracts(root, arms)
    scan_bench_results(root, arms)
    scan_source_ledgers(root, arms)
    attach_derived(arms)
    for a in arms:
        a.classify(args.fired_from)

    if args.json:
        json.dump([asdict(a) for a in arms], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print_table(arms, args.verbose)
    print_summary(arms, args.fired_from)

    if args.contract_orphans:
        orphans = contract_orphans(root, arms)
        print("CONTRACT-NAMED, NO SWITCH ANYWHERE")
        print("==================================")
        print("  A census of declarations cannot see an arm that was never")
        print("  declared. DEVIATION 1457 is the known instance. Expect noise")
        print("  in this list -- it matches on shape, so contract section")
        print("  labels and stage tags land in it too.")
        for o in orphans:
            print("  " + o)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
