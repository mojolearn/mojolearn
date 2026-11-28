# SPDX-License-Identifier: Apache-2.0
"""The door of `rank_genes_groups(X, groups, method, reference, tie_correct, corr_method, device)` (lane
sc-stats-1, 2026-11-28), reached from bindings/prep_owner_native_door.mojo.

UNPACK: the method words, the group labels as int32 codes (`classical_encode_labels`, f_classif's encoder: the
group labels sorted as `classes`), the reference group's code, X as dense float32 rows (`xprep_values`) or as CSR
(`csr_input`, MultinomialNB's admission). RUN: one x_prep chunk program per block of genes
(bindings/prep_markers_run.mojo `marker_chunk_stage`, the units of x_prep/markers.mojo), each block's three
gene-major results copied into the door's three host blocks by one memcpy each, then one program over the whole
result (`marker_final_stage`: the row-major results and the adjusted p-values). PACK: a dict with Scanpy's
`rank_genes_groups` result names: per result row (group) the genes by score descending, a tie by the lower gene index (`names`, the gene
indices), with the scores, p-values, adjusted p-values and log fold changes in that order.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem
from bindings.classical_arrays_door import classical_typed
from bindings.scaler_native_door import _ptr
from bindings.scaler_lifecycle_native_door import _entries
from bindings.x_prep_native_owner_door import xprep_values
from bindings.classical_labels_door import classical_encode_labels
from bindings.prep_classify_common_door import is_sparse, csr_input
from bindings.prep_encode_native_door import EncProg, _take2, _default_mode
from bindings.cluster_public_native_pack import _xc_array
from providers.python.x_cluster_out_pyapi_pack import _py_array
from bindings.prep_markers_run import (
    MarkerChunk, MarkerFinal, marker_chunk_stage, marker_final_stage, marker_chunk_width, marker_rows_out,
    MARKER_WILCOXON, MARKER_T_TEST,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _text(value: PythonObject, name: String, choices: String) raises -> String:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.str:
        raise Error("ValueError: rank_genes_groups: " + name + " must be one of " + choices)
    return String(value)


def _arena(ran: PythonObject) raises -> Int:
    """The address of a run's host arena (`_word`'s read, bindings/prep_encode_native_door.mojo)."""
    return Int(py=ran[0][1])


def prep_rank_genes_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK
    var b = native_loaded_module("builtins")
    var method_name = _text(args[2], "method", "'wilcoxon', 't-test'")
    var method: Int
    if method_name == "wilcoxon":
        method = MARKER_WILCOXON
    elif method_name == "t-test":
        method = MARKER_T_TEST
    else:
        raise Error("ValueError: rank_genes_groups: method must be 'wilcoxon' or 't-test', got '" + method_name + "'")
    var corr = _text(args[5], "corr_method", "'benjamini-hochberg', 'bonferroni'")
    if corr != "benjamini-hochberg" and corr != "bonferroni":
        raise Error("ValueError: rank_genes_groups: corr_method must be 'benjamini-hochberg' or 'bonferroni', got '"
                    + corr + "'")
    var tie_correct = Bool(py=py_call(b.bool, args[4]))
    var labels = classical_encode_labels(args[1], context)
    var classes = labels.classes
    var k = Int(py=py_call(b.len, classes))
    var reference = -1
    var ref_word = args[3]
    if not (py_type(ref_word) is b.str and String(ref_word) == "rest"):
        for g in range(k):  # small-loop(classes: the group labels, admission of the reference)
            if py_contains(Python.tuple(classes[g]), ref_word):
                reference = g
                break
        if reference < 0:
            raise Error("ValueError: rank_genes_groups: reference must be 'rest' or one of the groups")
    if k < 2:
        raise Error("ValueError: rank_genes_groups: groups must hold at least two different labels")
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var sparse = is_sparse(args[0])
    var n: Int
    var d: Int
    var values = PythonObject(None)
    var indptr = PythonObject(None)
    var indices = PythonObject(None)
    var data = PythonObject(None)
    var nnz = 0
    if sparse:
        var csr = csr_input(args[0], context)
        n = csr.n
        d = csr.d
        nnz = csr.nnz
        indptr = csr.indptr
        indices = csr.indices
        data = csr.data
    else:
        values = xprep_values(args[0], entries, context)
        n = Int(py=values[3])
        d = Int(py=values[4])
    if labels.n != n:
        raise Error("ValueError: rank_genes_groups: X has " + String(n) + " rows but groups has "
                    + String(labels.n) + " labels")
    if d < 1:
        raise Error("ValueError: rank_genes_groups: X must have at least one gene (column)")
    var ko = marker_rows_out(k, reference)
    var out_context = context["output_context"]
    var scores_g = classical_typed(Python.tuple(ko * d), "<f4", out_context)
    var pvalues_g = classical_typed(Python.tuple(ko * d), "<f4", out_context)
    var lfc_g = classical_typed(Python.tuple(ko * d), "<f4", out_context)
    var sp = _ptr(scores_g)
    var pp = _ptr(pvalues_g)
    var lp = _ptr(lfc_g)
    # RUN: the chunk programs (a memory bound, bindings/prep_markers_run.mojo `marker_chunk_width`)
    var width = marker_chunk_width(n, d)
    var c0 = 0
    while c0 < d:  # driver-loop: over the gene blocks of the group-ordered copy's memory bound
        var dc = min(width, d - c0)
        var p = EncProg()
        var x_owner = -1
        var ip_owner = -1
        var ix_owner = -1
        var dv_owner = -1
        if sparse:
            ip_owner = p.own(indptr, n + 1)
            ix_owner = p.own(indices, nnz)
            dv_owner = p.own(data, nnz)
        else:
            x_owner = p.own_x(values)
        var codes_owner = p.own(labels.codes, n)
        var request = MarkerChunk(n, d, k, reference, c0, dc, sparse, x_owner, ip_owner, ix_owner, dv_owner, nnz,
                                  codes_owner, method, tie_correct)
        var plan = marker_chunk_stage(p.prog, request)
        var ran = p.run(entries, context)
        var arena = _arena(ran)
        if c0 == 0:
            # the group sizes, from the device's group starts (k + 1 int words)
            var starts = _I32P(unsafe_from_address=arena + 4 * plan.start)
            for g in range(k):  # small-loop(groups: the group sizes, one refusal)
                if Int(starts[g + 1]) - Int(starts[g]) < 2:
                    raise Error("ValueError: rank_genes_groups: could not calculate statistics for group "
                                + String(py_call(b.str, classes[g])) + " since it holds fewer than two cells")
        var block = dc * ko
        var src = _F32P(unsafe_from_address=arena + 4 * plan.out)
        unsafe_memcpy(dest=sp + c0 * ko, src=src, count=block)
        unsafe_memcpy(dest=pp + c0 * ko, src=src + block, count=block)
        unsafe_memcpy(dest=lp + c0 * ko, src=src + 2 * block, count=block)
        _ = ran
        _ = p^
        c0 += dc
    # RUN: the whole result
    var fin = EncProg()
    var so = fin.own(scores_g, ko * d)
    var po = fin.own(pvalues_g, ko * d)
    var lo = fin.own(lfc_g, ko * d)
    var final_plan = marker_final_stage(fin.prog, MarkerFinal(ko, d, so, po, lo, corr == "bonferroni"))
    var ran_final = fin.run(entries, context)
    # PACK
    var names = Python.list()
    for j in range(ko):  # small-loop(rows: the group labels of the result rows)
        var g = j if reference < 0 or j < reference else j + 1
        _ = names.append(classes[g])
    var result = Python.dict()
    py_setitem(result, "groups", names)
    var total = ko * d
    var name_words = _I32P(unsafe_from_address=_arena(ran_final) + 4 * final_plan.names)
    var flat = _py_array(native_loaded_module("array"), "i", name_words.bitcast[UInt8](), total * 4)
    py_setitem(result, "names", _xc_array(flat, Python.tuple(ko, d), "<i4", "i", context))
    py_setitem(result, "scores", _take2(ran_final, final_plan.values, ko, d, context))
    py_setitem(result, "pvals", _take2(ran_final, final_plan.values + total, ko, d, context))
    py_setitem(result, "pvals_adj", _take2(ran_final, final_plan.values + 2 * total, ko, d, context))
    py_setitem(result, "logfoldchanges", _take2(ran_final, final_plan.values + 3 * total, ko, d, context))
    _ = ran_final
    _ = values^
    _ = indptr^
    _ = indices^
    _ = data^
    _ = scores_g^
    _ = pvalues_g^
    _ = lfc_g^
    _ = labels^
    return result

