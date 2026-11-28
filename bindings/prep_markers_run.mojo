# SPDX-License-Identifier: Apache-2.0
"""RUN of `rank_genes_groups` (lane sc-stats-1, 2026-11-28): the typed requests and the stage builders of its
x_prep programs (units in x_prep/markers.mojo).

The door (bindings/prep_markers_native_door.mojo) is UNPACK (the label codes, X dense or CSR, the method words),
then one chunk program per block of genes built here and run, then one program over the whole result, then PACK.
No builder reads a field, a mode or an entry table; no Python import (tools/python_free_dirs.sh).

Each program's stages are the parallel units of x_prep/markers.mojo; staging is a handful of words per program.
The gene blocks exist for one reason, a memory bound: the group-ordered copy of a block is n x dc device words,
and dc is the most genes whose copy stays within the count-data lane's bound (`sp_chunk_width`,
bindings/prep_scprep_run.mojo, which also builds the block: `gene_block_stage`). It is never a shape
rule: every n and d take the same formula, and the blocks change no bit (each gene's units read only its column).
"""
from core.preprocessing_program import prep_program_stage
from bindings.prep_encode_prog_run import EncProgram, enc_input
from bindings.prep_scprep_run import SpInput, gene_block_stage
from x_prep.py2mojo import P2M_BASE
from x_prep.sc_prep import SP_PRE_CANON
from x_prep.markers import (
    SC_SEG_SORT, SC_STATS, SC_RANKSUM, SC_SCORE,
    SC_BH_SORT, SC_BH_SCAN, SC_ADJUST, SC_RANK_KEYS, SC_RANK_SORT, SC_RANK_GATHER, RG_WILCOXON, RG_TIE_CORRECT,
    RG_ADJ_BH, RG_ADJ_BONFERRONI,
)

#: x_prep/py2mojo.mojo `p2m_transpose_unit`: q = [SRC, n, w, DST], DST[j*n + i] = SRC[i*w + j], the word as it is.
comptime OP_TRANSPOSE = P2M_BASE + 9

comptime MARKER_WILCOXON = 0
comptime MARKER_T_TEST = 1


def marker_rows_out(groups: Int, reference: Int) -> Int:
    return groups if reference < 0 else groups - 1


@fieldwise_init
struct MarkerChunk(Copyable, Movable):
    """One chunk program: genes [c0, c0 + dc) of the input `x` (dense or CSR owners, bindings/prep_scprep_run.mojo
    `SpInput`). `codes`: the owner of the n int32 group codes in [0, groups). `reference` < 0: each group against
    the rest. `method`: MARKER_WILCOXON or MARKER_T_TEST."""
    var x: SpInput
    var groups: Int
    var reference: Int
    var c0: Int
    var dc: Int
    var codes: Int
    var method: Int
    var tie_correct: Bool


@fieldwise_init
struct MarkerChunkPlan(Copyable, Movable):
    """Arena offsets: START (groups + 1 int words, host words: the door reads the group sizes from them) and OUT
    (three gene-major blocks of dc x rows_out words: scores, p-values, log fold changes)."""
    var start: Int
    var out: Int


def marker_chunk_stage(mut prog: EncProgram, r: MarkerChunk) -> MarkerChunkPlan:
    var x = r.x.copy()
    var n = x.n
    var k = r.groups
    var dc = r.dc
    var ko = marker_rows_out(k, r.reference)
    var block = gene_block_stage(prog, x, r.codes, k, r.c0, dc, SP_PRE_CANON)
    var start = block.start
    var g = block.g
    var stats = prog.work(4 * ko * dc)
    var r2t = prog.work(4 * ko * dc)
    var wilcoxon = r.method == MARKER_WILCOXON
    if wilcoxon:
        prep_program_stage(prog.program, SC_SEG_SORT, k * dc, g, n, dc, start)
    prep_program_stage(prog.program, SC_STATS, ko * dc, g, n, k, dc, start, r.reference, stats)
    if wilcoxon:
        if r.reference < 0:
            var cur = prog.work(2 * k * dc)
            prep_program_stage(prog.program, SC_RANKSUM, dc, g, n, k, dc, start, r.reference, cur, r2t)
        else:
            var cur = prog.work(4 * ko * dc)
            prep_program_stage(prog.program, SC_RANKSUM, ko * dc, g, n, k, dc, start, r.reference, cur, r2t)
    var flags = (RG_WILCOXON if wilcoxon else 0) + (RG_TIE_CORRECT if r.tie_correct else 0)
    var out = prog.alloc(3 * ko * dc)
    prep_program_stage(prog.program, SC_SCORE, ko * dc, stats, start, k, dc, r.reference, flags, r2t, n, out)
    return MarkerChunkPlan(start, out)


@fieldwise_init
struct MarkerFinal(Copyable, Movable):
    """The whole result: the owners of the three gene-major d x rows host blocks the chunks filled; `bonferroni`
    selects that adjustment over Benjamini-Hochberg."""
    var rows: Int
    var d: Int
    var scores: Int
    var pvalues: Int
    var lfc: Int
    var bonferroni: Bool


@fieldwise_init
struct MarkerFinalPlan(Copyable, Movable):
    """Arena offsets of the sorted result: `names` (rows x d int words, each group's genes by score descending,
    a tie by the lower index) and `values` (four row-major rows x d blocks in that order: scores, p-values,
    adjusted p-values, log fold changes)."""
    var names: Int
    var values: Int


def marker_final_stage(mut prog: EncProgram, r: MarkerFinal) -> MarkerFinalPlan:
    var ko = r.rows
    var d = r.d
    var total = ko * d
    var sg = enc_input(prog, total, r.scores)
    var pg = enc_input(prog, total, r.pvalues)
    var lg = enc_input(prog, total, r.lfc)
    var sc = prog.work(total)
    var pv = prog.work(total)
    var lf = prog.work(total)
    var adj = prog.work(total)
    prep_program_stage(prog.program, OP_TRANSPOSE, total, sg, d, ko, sc)
    prep_program_stage(prog.program, OP_TRANSPOSE, total, pg, d, ko, pv)
    prep_program_stage(prog.program, OP_TRANSPOSE, total, lg, d, ko, lf)
    var srt = prog.work(total)
    var q = prog.work(total)
    if not r.bonferroni:
        prep_program_stage(prog.program, SC_BH_SORT, ko, pv, d, srt)
        prep_program_stage(prog.program, SC_BH_SCAN, ko, srt, d, q)
    prep_program_stage(prog.program, SC_ADJUST, total, pv, srt, q, d,
                       RG_ADJ_BONFERRONI if r.bonferroni else RG_ADJ_BH, adj)
    var keys = prog.work(2 * total)
    var names = prog.alloc(total)
    var values = prog.alloc(4 * total)
    prep_program_stage(prog.program, SC_RANK_KEYS, total, sc, keys, d)
    prep_program_stage(prog.program, SC_RANK_SORT, ko, keys, d)
    prep_program_stage(prog.program, SC_RANK_GATHER, total, keys, d, sc, pv, adj, lf, names, values, total)
    return MarkerFinalPlan(names, values)
