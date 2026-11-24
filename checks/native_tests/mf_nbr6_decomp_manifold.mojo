# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The XDecomp and Manifold member routes lane mf-nbr6 added (core/xdecomp_member.mojo, core/manifold_member.mojo):
their words and the refusals they raise in the doors' words before any image is called (no x_decomp or metrics
entry is needed; the fits' equality with the doors is the Python test python/mojolearn/tests/test_mf_nbr6_members.py
and the digests tools/mf_nbr6_digests.py).

    test_ica_w_init_shape_refused        FastICA words [9 .. 11] (a 3 x 3 w_init) on X 20 x 4 with n_components 2:
                                         refused 'w_init has invalid shape -- should be (2, 2)' (ica_fit's words)
    test_isomap_radius_text_words        an Isomap radius model takes twelve words plus its text's byte words
                                         (13 for '3.0'); twelve alone is refused
    test_lle_helper_needs_second_entry   an LLE whose helper word [12] names another image needs `route.second`
    test_umap_array_init_shape_refused   a UMAP array start of 3 x 2 on X 20 x 3 is refused 'UMAP init array has
                                         shape (3, 2); (20, 2) is needed' (the door's words), after the input and
                                         n_neighbors checks
    test_umap_random_needs_images        a UMAP 'random' start without `route.second` / `route.third` is refused
    test_umap_takes_thirty_three_words   twenty words (the former layout) are refused

FAIL-PROOF (each edit makes the named case fail): delete the w_init shape check in `_fit_ica` ->
test_ica_w_init_shape_refused; drop `member_words_byte_slots(...)` from `_new`'s Isomap count ->
test_isomap_radius_text_words; drop the `elif` refusal in `_fit_lle` -> test_lle_helper_needs_second_entry; skip
`umap_init_shape_matches` in `_start` -> test_umap_array_init_shape_refused; skip the has_second / has_third test
in `_start` -> test_umap_random_needs_images; set MANIFOLD_PARAM_COUNT back to 20 ->
test_umap_takes_thirty_three_words.
"""
from std.testing import TestSuite
from std.memory import bitcast
from core.entry_table import EntryRef
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_abi import MemberCall, MemberMatrix, MemberRef, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, member_call
from core.xdecomp_member import (
    xdecomp_member_v2, XDECOMP_KIND_ICA, XDECOMP_KIND_ISOMAP, XDECOMP_KIND_LLE, XDECOMP_ICA_PARAM_COUNT,
    XDECOMP_ISOMAP_PARAM_COUNT, XDECOMP_LLE_PARAM_COUNT, XDECOMP_LLE_HELPER_WORD,
)
from core.manifold_member import (
    manifold_member_v2, MANIFOLD_KIND_UMAP, MANIFOLD_PARAM_COUNT, MANIFOLD_INIT_RANDOM, MANIFOLD_INIT_ARRAY,
    MF_INIT_WORD, MF_EXACT_WORD,
)
from core.cluster_member_words import member_words_append_text
from checks.native_tests.testsuite_cases import require


def _route() -> DeviceRoute:
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = 1
    route.vendor = 0
    return route


def _rows(count: Int, width: Int) -> List[Float32]:
    var values = List[Float32](length=count * width, fill=Float32(0.0))
    var s = UInt64(17)
    for i in range(count * width):  # small-loop: the fixed test rows (at most 80 words)
        s = s * 6364136223846793005 + 1442695040888963407
        values[i] = Float32(Float64(Int(s >> 40)) / 16777216.0)
    return values^


def _new(member: MemberRef, words: List[Float64]) raises -> Int:
    var call = MemberCall(_route())
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    _ = member_call(member, MEMBER_OP_NEW, call)
    _ = words
    return call.state


def _free(member: MemberRef, state: Int):
    var call = MemberCall(_route(), state)
    try:
        _ = member_call(member, MEMBER_OP_FREE, call)
    except:
        pass


def _fit_words(member: MemberRef, words: List[Float64], x: List[Float32], rows: Int, cols: Int,
               route: DeviceRoute) -> String:
    """The refusal a fit of `words` on X raises ('' when none)."""
    var state = 0
    var refused = String("")
    try:
        state = _new(member, words)
        var call = MemberCall(route, state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), rows, cols)
        _ = member_call(member, MEMBER_OP_FIT, call)
    except e:
        refused = String(e)
    if state != 0:
        _free(member, state)
    _ = x
    return refused


def _xdecomp() -> MemberRef:
    return MemberRef(EntryRef(xdecomp_member_v2, 0), 0)


def _manifold() -> MemberRef:
    return MemberRef(EntryRef(manifold_member_v2, 0), 0)


def test_ica_w_init_shape_refused() raises:
    var start = _rows(3, 3)
    var words = List[Float64](length=XDECOMP_ICA_PARAM_COUNT, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_ICA)
    words[1] = 2.0      # n_components
    words[2] = 2.0      # whiten 'arbitrary-variance'
    words[5] = 1.0      # alpha
    words[6] = 10.0     # max_iter
    words[7] = 1e-4     # tol
    words[9] = Float64(Int(start.unsafe_ptr()))
    words[10] = 3.0
    words[11] = 3.0
    var x = _rows(20, 4)
    var refused = _fit_words(_xdecomp(), words, x, 20, 4, _route())
    require("w_init has invalid shape -- should be (2, 2)" in refused, "ica: a wrong w_init shape is refused")
    _ = start^
    _ = words^


def test_isomap_radius_text_words() raises:
    var words = List[Float64](length=XDECOMP_ISOMAP_PARAM_COUNT, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_ISOMAP)
    words[4] = -1.0
    words[8] = 2.0
    words[9] = 1.0
    words[10] = 3.0
    words[11] = 3.0     # the byte count of '3.0'
    var refused = String("")
    try:
        _free(_xdecomp(), _new(_xdecomp(), words))
    except e:
        refused = String(e)
    require("radius text" in refused, "isomap: twelve words without the text's byte word are refused")
    member_words_append_text(words, String("3.0"))
    require(len(words) == XDECOMP_ISOMAP_PARAM_COUNT + 1, "isomap: '3.0' takes one byte word")
    _free(_xdecomp(), _new(_xdecomp(), words))
    _ = words^


def test_lle_helper_needs_second_entry() raises:
    var words = List[Float64](length=XDECOMP_LLE_PARAM_COUNT, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_LLE)
    words[5] = 5.0      # n_neighbors
    words[6] = 2.0      # n_components
    words[7] = 1e-3     # reg
    words[XDECOMP_LLE_HELPER_WORD] = 0.0       # another image (capability word 0)
    words[XDECOMP_LLE_HELPER_WORD + 2] = 12345.0
    var x = _rows(20, 3)
    var refused = _fit_words(_xdecomp(), words, x, 20, 3, _route())
    require("helper image needs the route's second entry" in refused, "lle: the helper image needs route.second")
    _ = words^


def _umap_words(init: Int) -> List[Float64]:
    var words: List[Float64] = [Float64(MANIFOLD_KIND_UMAP), 5.0, 2.0, 10.0, 0.1, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 5.0,
        -1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.5, 1.0]
    words.resize(MANIFOLD_PARAM_COUNT, 0.0)
    words[MF_INIT_WORD] = Float64(init)
    words[MF_EXACT_WORD] = bitcast[DType.float64](Int64(5))
    words[MF_EXACT_WORD + 1] = bitcast[DType.float64](Int64(10))
    words[MF_EXACT_WORD + 2] = bitcast[DType.float64](Int64(0))
    return words^


def test_umap_array_init_shape_refused() raises:
    var start = _rows(3, 2)
    var words = _umap_words(MANIFOLD_INIT_ARRAY)
    words[MF_INIT_WORD + 1] = Float64(Int(start.unsafe_ptr()))
    words[MF_INIT_WORD + 2] = 3.0
    words[MF_INIT_WORD + 3] = 2.0
    var x = _rows(20, 3)
    var refused = _fit_words(_manifold(), words, x, 20, 3, _route())
    require("UMAP init array has shape (3, 2); (20, 2) is needed" in refused, "umap: a wrong start shape")
    _ = start^
    _ = words^


def test_umap_random_needs_images() raises:
    var words = _umap_words(MANIFOLD_INIT_RANDOM)
    var x = _rows(20, 3)
    var refused = _fit_words(_manifold(), words, x, 20, 3, _route())
    require("x_decomp images" in refused, "umap: a random start needs the x_decomp images")
    _ = words^


def test_umap_takes_thirty_three_words() raises:
    var full = _umap_words(0)
    var words = List[Float64](capacity=20)
    for i in range(20):  # small-loop: the former twenty-word layout
        words.append(full[i])
    var refused = String("")
    try:
        _free(_manifold(), _new(_manifold(), words))
    except e:
        refused = String(e)
    require("thirty-three parameters" in refused, "umap: the twenty-word layout is refused")
    _ = words^
    _ = full^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
