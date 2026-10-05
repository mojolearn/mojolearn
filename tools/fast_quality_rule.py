# SPDX-License-Identifier: Apache-2.0
"""The FAST quality rule (CLAUDE.md, Andrew 2026-10-04), shared by the FAST
quality tools (softmax_g2_quality, mcd_g1_quality, mcd_ordered_quality /
mcd_ordered_oracle, eigh_w4_quality, eigh_panel_df_pair).

PASS = no material drop against FAST main (arm A) AND at least as good as the
best opponent (when the fixture carries an opponent value). Noise-level
differences, a new fold order and new bits are fine: FAST needs no identical
anything (not B == A, not B bit-equal run to run). A strict `B <= A` test is
reported per metric as `strict_le_info` and never decides the status.

Every metric here is lower-is-better (an error, a loss, a count of
disagreements). The noise band of a reference value r is
r + max(rtol * |r|, atol): rtol covers a different fp32 summation order
(which moves an error vs an FP64 oracle by a fraction of itself) and atol
covers the fp32 floor where r is near zero. Each tool fixes its rtol / atol
per metric before any B result exists.
"""

RULE = 'fast-quality-v1: B within noise of A (no material drop) and <= best opponent within noise; strict B<=A info only'


def band(ref, rtol, atol):
    """Largest value still within noise of `ref` (lower is better)."""
    ref = float(ref)
    return ref + max(rtol * abs(ref), atol)


def judge(av, bv, rtol, atol, opponent=None):
    """One lower-is-better metric under the FAST rule."""
    av, bv = float(av), float(bv)
    allowed = band(av, rtol, atol)
    ok = bv == bv and bv <= allowed  # NaN never passes
    out = dict(A=av, B=bv, allowed=allowed, rtol=rtol, atol=atol, ok=bool(ok),
               strict_le_info=bool(bv <= av))
    if opponent is not None:
        opp_allowed = band(opponent, rtol, atol)
        out.update(opponent=float(opponent), opponent_allowed=opp_allowed,
                   opponent_ok=bool(bv <= opp_allowed))
        out['ok'] = bool(ok and out['opponent_ok'])
    return out
