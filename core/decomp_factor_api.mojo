# SPDX-License-Identifier: Apache-2.0
"""Typed requests of the FastICA, FactorAnalysis, LatentDirichletAllocation
and random projection drivers, and their calls through the x_decomp
operations table (RUN: typed Mojo values only, no Python object).

Each request is the argument set of one compiled driver
(x_decomp/{ica,fa,lda,rp}_owner.mojo). `*_call(entry, request)` lays it out
as the fixed envelope of its `XD_*` operation (x_decomp/descriptor_schema.mojo)
on the caller's frame and makes one indirect call through the estimator's
resolved entry; x_decomp/descriptor_factor_ops.mojo reads the same layout in
the image. No heap, no boxed list. The caller keeps every addressed buffer
alive until the call returns and releases the GIL around it.

The calls are one driver each: there are no independent units to split here
(the drivers split their own work across the image's executor)."""
from core.entry_table import EntryRef, entry_ref_call
from x_decomp.descriptor_schema import (
    XD_ICA_FIT, XD_ICA_PROJECT, XD_FA_FIT, XD_FA_QUERY, XD_LDA, XD_RP_FIT, XD_RP_PROJECT,
)

# The text of LatentDirichletAllocation's negativity refusal names the public
# method; the request carries which one as a word.
comptime LDA_REFUSAL_FIT = 0
comptime LDA_REFUSAL_PARTIAL_FIT = 1
comptime LDA_REFUSAL_TRANSFORM = 2


@fieldwise_init
struct ICAFitRequest(Copyable, Movable):
    var source: Int
    var w_init: Int        # 0: drawn
    var components: Int
    var mixing: Int
    var whitening: Int
    var mean: Int
    var sources: Int
    var rows: Int
    var features: Int
    var count: Int
    var whiten: Int        # 0 off, 1 unit-variance, 2 arbitrary-variance
    var seed: Int
    var algorithm: Int
    var fun: Int
    var max_iter: Int
    var project: Bool
    var alpha: Float64
    var tol: Float64
    var loop_out: Int      # 0, or one int64: the fixed-point loop's nanoseconds (`fit_loop_`)


@fieldwise_init
struct ICAProjectRequest(Copyable, Movable):
    var source: Int
    var matrix: Int
    var mean: Int
    var dst: Int
    var inverse: Bool
    var rows: Int
    var width: Int
    var matrix_rows: Int
    var whiten: Bool


@fieldwise_init
struct FAFitRequest(Copyable, Movable):
    var source: Int
    var psi0: Int
    var components: Int
    var noise: Int
    var mean: Int
    var loglike: Int       # float64, max(max_iter, 1) words
    var rows: Int
    var features: Int
    var count: Int
    var max_iter: Int
    var rotation: Int      # 0 None, 1 varimax, 2 quartimax
    var tol: Float64
    var loop_out: Int      # 0, or one int64: the EM loop's nanoseconds (`fit_loop_`)


@fieldwise_init
struct FAQueryRequest(Copyable, Movable):
    var operation: Int     # 0 transform, 1 covariance, 2 precision, 3 score_samples, 4 score
    var source: Int
    var components: Int
    var noise: Int
    var mean: Int
    var dst: Int
    var rows: Int
    var features: Int
    var count: Int


@fieldwise_init
struct LDARequest(Copyable, Movable):
    var operation: Int     # 0 fit, 1 partial_fit, 2 transform, 3 score, 4 perplexity
    var source: Int
    var components_in: Int
    var exp_in: Int
    var components_out: Int
    var exp_out: Int
    var dst: Int
    var rows: Int
    var words: Int
    var count: Int
    var fresh: Bool
    var flag: Bool         # normalize / sub_sampling
    var max_iter: Int
    var online: Bool
    var batch_size: Int
    var max_doc_update_iter: Int
    var seed: Int
    var evaluate_every: Int
    var refusal: Int       # LDA_REFUSAL_*
    var draw: Int
    var n_batch_iter: Int
    var doc_prior: Float64
    var topic_prior: Float64
    var offset: Float64
    var decay: Float64
    var mean_change_tol: Float64
    var total_samples: Float64
    var perp_tol: Float64


@fieldwise_init
struct LDAResult(Copyable, Movable):
    var iterations: Int
    var draw: Int
    var n_batch_iter: Int
    var value: Float64


@fieldwise_init
struct RPFitRequest(Copyable, Movable):
    var source: Int
    var components: Int
    var inverse_components: Int
    var rows: Int
    var features: Int
    var count: Int
    var sparse: Bool
    var seed: Int
    var inverse: Bool
    var fused_ok: Bool
    var density: Float64


@fieldwise_init
struct RPProjectRequest(Copyable, Movable):
    var source: Int
    var matrix: Int
    var dst: Int
    var rows: Int
    var width: Int
    var matrix_rows: Int
    var tiled: Int         # 0 plain product, 1 tiled, 2 tiled and X not yet tested finite
    var matrix_pool: Int   # the matrix's resident pool id, -1: none
    var source_pool: Int   # X's resident pool id (`source` unread), -1: X is `source` (lane vdf-scope-neural3)


def ica_fit_call(entry: EntryRef, r: ICAFitRequest) raises -> Int:
    """FastICA fit / fit_transform; returns n_iter_."""
    var a: Array[Int, 8] = [r.source, r.w_init, r.components, r.mixing, r.whitening, r.mean, r.sources, r.loop_out]
    var p: Array[Int, 9] = [r.rows, r.features, r.count, r.whiten, r.seed, r.algorithm, r.fun, r.max_iter,
        Int(r.project)]
    var f: Array[Float64, 2] = [r.alpha, r.tol]
    return entry_ref_call(entry, XD_ICA_FIT, a, p, f)


def ica_project_call(entry: EntryRef, r: ICAProjectRequest) raises:
    """FastICA transform or inverse_transform."""
    var a: Array[Int, 4] = [r.source, r.matrix, r.mean, r.dst]
    var p: Array[Int, 5] = [Int(r.inverse), r.rows, r.width, r.matrix_rows, Int(r.whiten)]
    var f: Array[Float64, 0] = []
    _ = entry_ref_call(entry, XD_ICA_PROJECT, a, p, f)


def fa_fit_call(entry: EntryRef, r: FAFitRequest) raises -> Int:
    """FactorAnalysis fit; returns n_iter_."""
    var a: Array[Int, 7] = [r.source, r.psi0, r.components, r.noise, r.mean, r.loglike, r.loop_out]
    var p: Array[Int, 5] = [r.rows, r.features, r.count, r.max_iter, r.rotation]
    var f: Array[Float64, 1] = [r.tol]
    return entry_ref_call(entry, XD_FA_FIT, a, p, f)


def fa_query_call(entry: EntryRef, r: FAQueryRequest) raises -> Float64:
    """A fitted FactorAnalysis query; returns the score (operation 4), else 0."""
    var value: Array[Float64, 1] = [0.0]
    var a: Array[Int, 6] = [r.source, r.components, r.noise, r.mean, r.dst, Int(value.unsafe_ptr())]
    var p: Array[Int, 4] = [r.operation, r.rows, r.features, r.count]
    var f: Array[Float64, 0] = []
    _ = entry_ref_call(entry, XD_FA_QUERY, a, p, f)
    var answer = value[0]
    _ = value  # the out word outlives the call above
    return answer


def lda_call(entry: EntryRef, r: LDARequest) raises -> LDAResult:
    """One LatentDirichletAllocation operation."""
    var counters: Array[Int, 2] = [r.draw, r.n_batch_iter]
    var value: Array[Float64, 1] = [0.0]
    var counters_address = Int(counters.unsafe_ptr())
    var a: Array[Int, 9] = [r.source, r.components_in, r.exp_in, r.components_out, r.exp_out, r.dst,
        counters_address, counters_address + 8, Int(value.unsafe_ptr())]
    var p: Array[Int, 13] = [r.operation, r.rows, r.words, r.count, Int(r.fresh), Int(r.flag), r.max_iter,
        Int(r.online), r.batch_size, r.max_doc_update_iter, r.seed, r.evaluate_every, r.refusal]
    var f: Array[Float64, 7] = [r.doc_prior, r.topic_prior, r.offset, r.decay, r.mean_change_tol,
        r.total_samples, r.perp_tol]
    var iterations = entry_ref_call(entry, XD_LDA, a, p, f)
    var answer = LDAResult(iterations, counters[0], counters[1], value[0])
    _ = counters  # the inout words outlive the call above
    _ = value
    return answer^


def rp_fit_call(entry: EntryRef, r: RPFitRequest) raises -> Int:
    """A random projection's fit; returns the fused scan's first non-finite
    index, or -1."""
    var a: Array[Int, 3] = [r.source, r.components, r.inverse_components]
    var p: Array[Int, 7] = [r.rows, r.features, r.count, Int(r.sparse), r.seed, Int(r.inverse), Int(r.fused_ok)]
    var f: Array[Float64, 1] = [r.density]
    return entry_ref_call(entry, XD_RP_FIT, a, p, f)


def rp_project_call(entry: EntryRef, r: RPProjectRequest) raises -> Int:
    """X matrix^T; returns 1 when the tiled projection flagged a non-finite X."""
    var a: Array[Int, 3] = [r.source, r.matrix, r.dst]
    var p: Array[Int, 6] = [r.rows, r.width, r.matrix_rows, r.tiled, r.matrix_pool, r.source_pool]
    var f: Array[Float64, 0] = []
    return entry_ref_call(entry, XD_RP_PROJECT, a, p, f)
