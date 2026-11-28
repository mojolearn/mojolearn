# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""rank_genes_groups' x_prep units (lane sc-stats-1, 2026-11-28; x_prep/markers.mojo) on the host program runner
(x_prep/host/program.mojo `run_program_host_ptr`, the same units the device launches one thread each).

A hand-checkable case: six cells in two interleaved groups, two genes.
    gene 0: group 0 = {1, 2, 2}, group 1 = {2, 3, 4}; pooled ranks 1, 3, 3, 3, 5, 6, so group 0's rank sum is 7,
            z = (7 - 3 * 7 / 2) / sqrt(3 * 3 * 7 / 12) = -1.5275252, p = erfc(|z| / sqrt 2) = 0.1266305; with the
            tie correction (one run of three: T = 24, N^3 - N = 210) z = -1.6230861, p = 0.1045710; group 1 has
            z = +1.5275252.
    gene 1: group 0 = {1, 2, 3}, group 1 = {2, 4, 6}; ranks 1, 2.5, 2.5, 4, 5, 6, rank sum 7.5, z = -1.3093073.
            Welch: t = (2 - 4) / sqrt(1 / 3 + 4 / 3) = -1.5491933, dof = 50 / 17, p = 0.2208808 (the
            Student t two-sided tail, I_{dof / (dof + t^2)}(dof / 2, 1 / 2), computed by hand).
    log fold change of gene 0, group 0: log2(expm1(5 / 3) / expm1(3)) = -2.1519210.
    against reference group 1 (two groups: the same pool) group 0's z equals the "rest" one.
    Benjamini-Hochberg of [0.01, 0.04, 0.03, 0.2]: [0.04, 0.0533333, 0.0533333, 0.2]; Bonferroni [0.04, 0.16,
    0.12, 0.8].
    the sorted layout of scores [0.5, 2.0, 0.5, -1.0]: names [1, 0, 2, 3] (descending, the tie by the lower index),
    and every value block gathered in that order.
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from core.preprocessing_program import prep_program_stage
from x_prep.common import FP, IP
from x_prep.host.program import run_program_host_ptr
from x_prep.markers import (
    SC_SEG_SORT, SC_STATS, SC_RANKSUM, SC_SCORE, SC_BH_SORT, SC_BH_SCAN,
    SC_ADJUST, SC_RANK_KEYS, SC_RANK_SORT, SC_RANK_GATHER, RG_WILCOXON, RG_TIE_CORRECT, RG_ADJ_BH, RG_ADJ_BONFERRONI,
)
# the block (group order, gather) and the transpose are the shared units (lane sc-consolidate-1)
from x_prep.sc_prep import SP_ORDER, SP_GATHER_DENSE, SP_PRE_CANON
from x_prep.py2mojo import P2M_BASE


def _close(got: Float32, want: Float64, tol: Float64, what: String) raises:
    require(abs(Float64(got) - want) <= tol, what + ": " + String(got) + ", expected " + String(want))


struct _Case(Movable):
    """The six-cell arena: offsets of every block, the program, the output's offset."""
    var arena: List[Float32]
    var program: List[Int32]
    var out: Int
    var stages: Int

    def __init__(out self, flags: Int, reference: Int):
        var n = 6
        var d = 2
        var k = 2
        var ko = k if reference < 0 else k - 1
        var codes = 0
        var x = codes + n
        var start = x + n * d
        var perm = start + k + 1
        var g = perm + n
        var stats = g + n * d
        var r2t = stats + 4 * ko * d
        var cur = r2t + 4 * ko * d
        var o = cur + 4 * k * d
        var total = o + 3 * ko * d
        var arena = List[Float32](length=total, fill=Float32(0))
        var values: List[Float32] = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 6.0]
        for i in range(n):  # small-loop: the six cells' codes (int words) and values
            arena[codes + i] = bitcast[DType.float32](Int32(i % 2))
            for c in range(d):
                arena[x + i * d + c] = values[i * d + c]
        var program = List[Int32]()
        prep_program_stage(program, SP_ORDER, k, codes, n, k, start, perm)
        prep_program_stage(program, SP_GATHER_DENSE, n * d, x, n, d, 0, perm, g, SP_PRE_CANON)
        prep_program_stage(program, SC_SEG_SORT, k * d, g, n, d, start)
        prep_program_stage(program, SC_STATS, ko * d, g, n, k, d, start, reference, stats)
        if reference < 0:
            prep_program_stage(program, SC_RANKSUM, d, g, n, k, d, start, reference, cur, r2t)
        else:
            prep_program_stage(program, SC_RANKSUM, ko * d, g, n, k, d, start, reference, cur, r2t)
        prep_program_stage(program, SC_SCORE, ko * d, stats, start, k, d, reference, flags, r2t, n, o)
        self.arena = arena^
        self.program = program^
        self.out = o
        self.stages = 6

    def run(mut self) raises:
        run_program_host_ptr(FP(unsafe_from_address=Int(self.arena.unsafe_ptr())), len(self.arena),
                             IP(unsafe_from_address=Int(self.program.unsafe_ptr())), self.stages)


def test_wilcoxon_hand_case() raises:
    var c = _Case(RG_WILCOXON, -1)
    c.run()
    var o = c.out
    var block = 2 * 2
    # gene-major: OUT[c * KO + j]
    _close(c.arena[o + 0], -1.5275252316519468, 2e-6, "gene 0 group 0 z")
    _close(c.arena[o + 1], 1.5275252316519468, 2e-6, "gene 0 group 1 z")
    _close(c.arena[o + 2], -1.3093073414159544, 2e-6, "gene 1 group 0 z")
    _close(c.arena[o + block + 0], 0.12663045794761715, 2e-6, "gene 0 group 0 p")
    _close(c.arena[o + 2 * block + 0], -2.1519210397104387, 2e-5, "gene 0 group 0 log fold change")


def test_wilcoxon_tie_correction() raises:
    var c = _Case(RG_WILCOXON + RG_TIE_CORRECT, -1)
    c.run()
    _close(c.arena[c.out + 0], -1.6230861351605892, 2e-6, "tie-corrected z")
    _close(c.arena[c.out + 4], 0.10457099306437274, 2e-6, "tie-corrected p")


def test_wilcoxon_reference_group() raises:
    var c = _Case(RG_WILCOXON, 1)
    c.run()
    # one output row (group 0); KO = 1, gene-major
    _close(c.arena[c.out + 0], -1.5275252316519468, 2e-6, "gene 0 against group 1")
    _close(c.arena[c.out + 1], -1.3093073414159544, 2e-6, "gene 1 against group 1")


def test_welch_hand_case() raises:
    var c = _Case(0, -1)
    c.run()
    _close(c.arena[c.out + 2], -1.5491933384829668, 2e-6, "gene 1 group 0 Welch t")
    _close(c.arena[c.out + 4 + 2], 0.2208808404940958, 5e-6, "gene 1 group 0 Welch p")


def _adjust(method: Int) raises -> List[Float32]:
    var d = 4
    var pv_g = 0
    var pv = pv_g + d
    var srt = pv + d
    var q = srt + d
    var adj = q + d
    var arena = List[Float32](length=adj + d, fill=Float32(0))
    var p: List[Float32] = [0.01, 0.04, 0.03, 0.2]
    for i in range(d):  # small-loop: the four p-values (one group, gene-major = row-major)
        arena[pv_g + i] = p[i]
    var program = List[Int32]()
    prep_program_stage(program, P2M_BASE + 9, d, pv_g, d, 1, pv)
    prep_program_stage(program, SC_BH_SORT, 1, pv, d, srt)
    prep_program_stage(program, SC_BH_SCAN, 1, srt, d, q)
    prep_program_stage(program, SC_ADJUST, d, pv, srt, q, d, method, adj)
    run_program_host_ptr(FP(unsafe_from_address=Int(arena.unsafe_ptr())), len(arena),
                         IP(unsafe_from_address=Int(program.unsafe_ptr())), 4)
    var out_values = List[Float32]()
    for i in range(d):  # small-loop: the adjusted values
        out_values.append(arena[adj + i])
    _ = program^
    return out_values^


def test_benjamini_hochberg_and_bonferroni() raises:
    var bh = _adjust(RG_ADJ_BH)
    _close(bh[0], 0.04, 1e-7, "BH 0.01")
    _close(bh[1], 0.16 / 3.0, 1e-7, "BH 0.04")
    _close(bh[2], 0.16 / 3.0, 1e-7, "BH 0.03")
    _close(bh[3], 0.2, 1e-7, "BH 0.2")
    var bf = _adjust(RG_ADJ_BONFERRONI)
    _close(bf[1], 0.16, 1e-7, "Bonferroni 0.04")
    _close(bf[3], 0.8, 1e-7, "Bonferroni 0.2")


def test_sorted_layout() raises:
    var d = 4
    var sc = 0
    var pv = sc + d
    var adj = pv + d
    var lf = adj + d
    var keys = lf + d
    var names = keys + 2 * d
    var vals = names + d
    var arena = List[Float32](length=vals + 4 * d, fill=Float32(0))
    var scores: List[Float32] = [0.5, 2.0, 0.5, -1.0]
    for i in range(d):  # small-loop: one row of four genes; the other blocks tag the gene
        arena[sc + i] = scores[i]
        arena[pv + i] = Float32(10 + i)
        arena[adj + i] = Float32(20 + i)
        arena[lf + i] = Float32(30 + i)
    var program = List[Int32]()
    prep_program_stage(program, SC_RANK_KEYS, d, sc, keys, d)
    prep_program_stage(program, SC_RANK_SORT, 1, keys, d)
    prep_program_stage(program, SC_RANK_GATHER, d, keys, d, sc, pv, adj, lf, names, vals, d)
    run_program_host_ptr(FP(unsafe_from_address=Int(arena.unsafe_ptr())), len(arena),
                         IP(unsafe_from_address=Int(program.unsafe_ptr())), 3)
    var want: List[Int] = [1, 0, 2, 3]
    for r in range(d):  # small-loop: the four ranks
        var gene = Int(bitcast[DType.int32](arena[names + r]))
        require(gene == want[r], "rank " + String(r) + " holds gene " + String(gene))
        require(arena[vals + r] == scores[want[r]], "sorted score at rank " + String(r))
        require(arena[vals + d + r] == Float32(10 + want[r]), "sorted p-value at rank " + String(r))
        require(arena[vals + 3 * d + r] == Float32(30 + want[r]), "sorted fold change at rank " + String(r))
    _ = program^
    _ = arena^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
