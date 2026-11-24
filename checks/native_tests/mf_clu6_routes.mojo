# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-clu6: the cluster members' new setting words and the refusals they
raise in the class's order, called on the member entries directly (no image
entry is reached: every route slot is `entry_ref_unresolved`, so a case that
went past its refusal would fail with the unresolved entry's words instead).

    test_words_round_trip               core/cluster_member_words.mojo: text and float32 blocks
                                        packed into Float64 words read back exactly
    test_kde_name_words                 a KernelDensity NEW with an unknown kernel name's bytes is
                                        minted; one word short is refused in the member ABI's words
    test_dbscan_no_prediction_data      a DBSCAN state minted with prediction_data=0 (word 7) and
                                        imported refuses predict in the class's words
    test_hierarchy_connectivity_refusal the tree route raises the connectivity admission's refusal
                                        carried in the NEW words, and the shape refusal against X
                                        before it when the matrix's side differs
    test_hierarchy_predict_rule         a ward fit's imported state refuses predict with
                                        `_agglo_predict`'s NotImplementedError
    test_spectral_precomputed_square    a precomputed fit of a non-square X is refused in `_sc_fit`'s words
    test_mixture_init_name_words        a GaussianMixture NEW with init_params -1 and the name's bytes is
                                        minted; a missing byte count is refused
    test_gmm_plain_weights_shape        an x_cluster kind 5 plain (GaussianMixture) fit refuses a
                                        weights_init of the wrong size in `gmm_ext_fit_stage`'s words

FAIL-PROOF (each edit makes the named case fail): in core/cluster_member_words.mojo
`member_words_take_text` read from word `at + 1` -> test_words_round_trip; in core/density_member.mojo
`_density_member` NEW set `want = 4` unconditionally -> test_kde_name_words; in `_mint` set
`slot[].predicting = True` -> test_dbscan_no_prediction_data; in core/hierarchy_member.mojo `_fit_tree`
delete the `_connectivity_check(slot, n)` call -> test_hierarchy_connectivity_refusal; in `_predict`
delete the `agglo_predicts` refusal -> test_hierarchy_predict_rule; in core/spectral_member.mojo `_fit`
delete the precomputed `rows != d` refusal -> test_spectral_precomputed_square; in core/mixture_member.mojo
NEW accept only `words[5]` 0 or 1 -> test_mixture_init_name_words; in core/xcluster_member.mojo
`_bgmm_fit` delete the `mixture_weights_size` call -> test_gmm_plain_weights_shape.
"""
from std.testing import TestSuite
from core.cluster_member_words import (
    member_words_append_text, member_words_take_text, member_words_append_f32, member_words_take_f32,
    member_words_byte_slots,
)
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_unresolved
from core.member_abi import (
    MemberCall, MemberMatrix, MemberRef, MEMBER_F64, MEMBER_I32, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT,
    MEMBER_OP_PREDICT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    member_call,
)
from core.density_member import density_member_v2
from core.hierarchy_member import hierarchy_member_v2
from core.spectral_member import spectral_member_v2
from core.mixture_member import mixture_member_v2
from core.xcluster_member import xcluster_member_v2
from checks.native_tests.testsuite_cases import require


def _route() -> DeviceRoute:
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = 1
    route.vendor = 0
    route.entry = entry_ref_unresolved()
    route.second = entry_ref_unresolved()
    route.third = entry_ref_unresolved()
    route.fourth = entry_ref_unresolved()
    route.fifth = entry_ref_unresolved()
    route.has_second = True
    route.has_third = True
    route.has_fourth = True
    route.has_fifth = True
    return route


def _new(member: MemberRef, words: List[Float64]) raises -> Int:
    var call = MemberCall(_route())
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    _ = member_call(member, MEMBER_OP_NEW, call)
    _ = len(words)
    return call.state


def _new_refused(member: MemberRef, words: List[Float64], expected: String) raises:
    var refused = False
    try:
        _ = _new(member, words)
    except error:
        refused = String(error).find(expected) >= 0
    require(refused, "NEW was not refused with: " + expected)


def _free(member: MemberRef, state: Int):
    var call = MemberCall(_route(), state)
    try:
        _ = member_call(member, MEMBER_OP_FREE, call)
    except:
        pass


def _import(member: MemberRef, state: Int, block: Int, view: MemberMatrix, flags: Int = 0) raises:
    var call = MemberCall(_route(), state)
    call.block = block
    call.x = view
    call.flags = flags
    _ = member_call(member, MEMBER_OP_IMPORT, call)


def _raises(member: MemberRef, operation: Int, state: Int, x: List[Float32], rows: Int, cols: Int,
    expected: String) raises:
    """`operation` (FIT or PREDICT) on `rows` x `cols` of `x` raises words containing `expected`."""
    var labels = List[Int32](length=rows, fill=Int32(0))
    var call = MemberCall(_route(), state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), rows, cols)
    if operation == MEMBER_OP_PREDICT:
        call.dst = MemberMatrix(Int(labels.unsafe_ptr()), rows, 1, MEMBER_I32)
    var got = String("")
    try:
        _ = member_call(member, operation, call)
        got = "no error"
    except error:
        got = String(error)
    _ = len(x)
    _ = len(labels)
    require(got.find(expected) >= 0, "expected '" + expected + "', got '" + got + "'")


def test_words_round_trip() raises:
    var words: List[Float64] = [7.0]
    var text = String("epanechnikov-ish")
    member_words_append_text(words, text)
    require(len(words) == 1 + member_words_byte_slots(text.byte_length()), "text word count")
    require(member_words_take_text(Int(words.unsafe_ptr()), 1, text.byte_length()) == text, "text round trip")
    var values: List[Float32] = [1.5, -2.25, 3.0]
    var at = len(words)
    member_words_append_f32(words, Int(values.unsafe_ptr()), 3)
    var back = member_words_take_f32(Int(words.unsafe_ptr()), at, 3)
    require(back[0] == values[0] and back[1] == values[1] and back[2] == values[2], "float32 round trip")
    _ = len(words)
    _ = len(values)


def test_kde_name_words() raises:
    var member = MemberRef(EntryRef(density_member_v2, 0), 0)
    var name = String("boxcar")
    var words: List[Float64] = [0.0, 0.5, -1.0, 0.0, Float64(name.byte_length()), 0.0]
    member_words_append_text(words, name)
    var state = _new(member, words)
    require(state != 0, "KernelDensity with a name's bytes is minted")
    _free(member, state)
    _ = words.pop()
    _new_refused(member, words, "member ABI: a KernelDensity member")


def test_dbscan_no_prediction_data() raises:
    var member = MemberRef(EntryRef(density_member_v2, 0), 0)
    var words: List[Float64] = [1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var state = _new(member, words)
    var indices: List[Int32] = [0, 1]
    var core: List[Int32] = [0, 0]
    var components: List[Float32] = [0.0, 0.0, 1.0, 1.0]
    var scalars: List[Float64] = [1.0, 2.0]
    _import(member, state, MEMBER_BLOCK_FAMILY + 1, MemberMatrix(Int(indices.unsafe_ptr()), 1, 2, MEMBER_I32))
    _import(member, state, MEMBER_BLOCK_FAMILY + 3, MemberMatrix(Int(core.unsafe_ptr()), 1, 2, MEMBER_I32))
    _import(member, state, MEMBER_BLOCK_FAMILY + 2, MemberMatrix(Int(components.unsafe_ptr()), 2, 2))
    _import(member, state, MEMBER_BLOCK_SCALARS, MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64),
        MEMBER_IMPORT_DONE)
    var query: List[Float32] = [0.1, 0.1]
    _raises(member, MEMBER_OP_PREDICT, state, query, 1, 2, "prediction data was not stored")
    _free(member, state)
    _ = len(indices)
    _ = len(core)
    _ = len(components)
    _ = len(scalars)


def _agglo_words(side: Int, refusal: String) -> List[Float64]:
    """Ward (linkage 0), euclidean, a connectivity matrix, n_clusters=2."""
    var words: List[Float64] = [0.0, 2.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        Float64(side), 0.0, 0.0, 0.0, Float64(refusal.byte_length())]
    member_words_append_text(words, refusal)
    return words^


def test_hierarchy_connectivity_refusal() raises:
    var member = MemberRef(EntryRef(hierarchy_member_v2, 0), 0)
    var x: List[Float32] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    var state = _new(member, _agglo_words(0, "TypeError: connectivity refused here"))
    _raises(member, MEMBER_OP_FIT, state, x, 4, 2, "TypeError: connectivity refused here")
    _free(member, state)
    state = _new(member, _agglo_words(3, "TypeError: storage refused"))
    _raises(member, MEMBER_OP_FIT, state, x, 4, 2, "Wrong shape for connectivity matrix when X has 4 samples")
    _free(member, state)


def test_hierarchy_predict_rule() raises:
    var member = MemberRef(EntryRef(hierarchy_member_v2, 0), 0)
    var state = _new(member, _agglo_words(0, ""))
    var refs: List[Float32] = [0.0, 0.0, 1.0, 1.0]
    var labels: List[Int32] = [0, 1]
    var scalars: List[Float64] = [2.0, -1.0, 1.0]
    _import(member, state, MEMBER_BLOCK_FAMILY + 2, MemberMatrix(Int(refs.unsafe_ptr()), 2, 2))
    _import(member, state, MEMBER_BLOCK_FAMILY, MemberMatrix(Int(labels.unsafe_ptr()), 1, 2, MEMBER_I32))
    _import(member, state, MEMBER_BLOCK_SCALARS, MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64),
        MEMBER_IMPORT_DONE)
    var query: List[Float32] = [0.2, 0.2]
    _raises(member, MEMBER_OP_PREDICT, state, query, 1, 2, "NotImplementedError: mojolearn AgglomerativeClustering")
    _free(member, state)
    _ = len(refs)
    _ = len(labels)
    _ = len(scalars)


def test_spectral_precomputed_square() raises:
    var member = MemberRef(EntryRef(spectral_member_v2, 0), 0)
    var words: List[Float64] = [0.0, 2.0, 2.0, 1.0, 2.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var state = _new(member, words)
    var x = List[Float32](length=12, fill=Float32(1.0))
    _raises(member, MEMBER_OP_FIT, state, x, 3, 4, "must be a square affinity matrix")
    _free(member, state)


def test_mixture_init_name_words() raises:
    var member = MemberRef(EntryRef(mixture_member_v2, 0), 0)
    var name = String("bogus")
    var words: List[Float64] = [2.0, 100.0, 0.001, 1e-6, 0.0, -1.0, Float64(name.byte_length())]
    member_words_append_text(words, name)
    var state = _new(member, words)
    require(state != 0, "GaussianMixture with an init name's bytes is minted")
    _free(member, state)
    var short: List[Float64] = [2.0, 100.0, 0.001, 1e-6, 0.0, -1.0]
    _new_refused(member, short, "member ABI: a GaussianMixture member")


def test_gmm_plain_weights_shape() raises:
    var member = MemberRef(EntryRef(xcluster_member_v2, 0), 0)
    var words: List[Float64] = [5.0, 2.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1e-6, 0.001, 100.0, 1.0, 0.0,
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, -1.0, 0.0, -1.0, 1.0]
    var weights: List[Float32] = [0.2, 0.3, 0.5]
    member_words_append_f32(words, Int(weights.unsafe_ptr()), 3)
    var state = _new(member, words)
    var x = List[Float32](length=8, fill=Float32(0.5))
    _raises(member, MEMBER_OP_FIT, state, x, 4, 2, "The parameter 'weights' should have the shape of (2,)")
    _free(member, state)
    _ = len(weights)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
