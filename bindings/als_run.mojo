# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the native AlternatingLeastSquares owner
(bindings/als_native.mojo is UNPACK and PACK): the confidence scale, the
Philox factor draws, the one XD_ALS_FIT sweep call, the score row and the
top-N order of `_topn`, over the typed kit (bindings/decomp_kit_run.mojo).
Same x_decomp operations, same words, same order: the bits do not move.
No Python (lane entangled-decompkit, 2026-11-12)."""
from std.memory import unsafe_memcpy
from bindings.decomp_kit_run import DKit, DK_SCALE, DK_SQ, DK_SQRT, DK_DIV
from x_decomp.descriptor_schema import XD_ALS_FIT, XD_TOPN_DESC, XD_GATHER


struct AlsFitWords(Copyable, Movable):
    """The fit's options, read once by UNPACK."""
    var unit: Bool          # alpha == 1.0: the confidence is the input itself
    var alpha: Float64
    var seed: Int           # masked to 32 bits
    var factors: Int
    var use_cg: Bool
    var cg_steps: Int
    var iterations: Int
    var loss: Bool
    var regularization: Float64

    def __init__(out self, unit: Bool, alpha: Float64, seed: Int, factors: Int, use_cg: Bool, cg_steps: Int,
                 iterations: Int, loss: Bool, regularization: Float64):
        self.unit = unit
        self.alpha = alpha
        self.seed = seed
        self.factors = factors
        self.use_cg = use_cg
        self.cg_steps = cg_steps
        self.iterations = iterations
        self.loss = loss
        self.regularization = regularization


def als_fit_run(mut kit: DKit, r: Int, words: AlsFitWords, mut losses: List[Float64]) raises -> Tuple[Int, Int]:
    """`AlternatingLeastSquares.fit` after admission: (user factors, item
    factors) as host matrices; `losses` holds the recorded losses (empty
    unless `words.loss`)."""
    var n = kit.nrows(r)
    var m = kit.ncols(r)
    var c = r if words.unit else kit.ew(DK_SCALE, r, scalar=words.alpha)
    var user_draws = kit.rand(n, words.factors, words.seed, 80, 0)
    var x = kit.ew(DK_SCALE, user_draws, scalar=0.01)
    var item_draws = kit.rand(m, words.factors, words.seed, 81, 0)
    var y = kit.ew(DK_SCALE, item_draws, scalar=0.01)
    # The sweeps (and the losses) are ONE call on host copies of the
    # factors, written in place.
    x = kit.copy_host(x)
    y = kit.copy_host(y)
    var slots = max(words.iterations, 1)
    var store = List[Float64](length=slots, fill=0.0)
    var params: List[Int] = [n, m, words.factors, words.iterations, Int(words.use_cg), words.cg_steps,
                             Int(words.loss)]
    var reg: List[Float64] = [words.regularization]
    var addresses: List[Int] = [kit.at(c), kit.at(x), kit.at(y), Int(store.unsafe_ptr())]
    var count = kit.xd(XD_ALS_FIT, addresses, params, reg)
    kit.wrote_host(x)
    kit.wrote_host(y)
    losses = List[Float64]()
    if words.loss:
        var recorded = max(min(count, slots), 0)
        losses = List[Float64](length=recorded, fill=0.0)
        if recorded > 0:
            unsafe_memcpy(dest=losses.unsafe_ptr(), src=store.unsafe_ptr(), count=recorded)
    _ = store^  # keep the loss store alive through the call above
    return (x, y)


struct AlsTop(Copyable, Movable):
    """`_topn`'s result for PACK: `count` entries; on the device route the
    ids (`ids`, a float32 matrix) and scores (`values`, a matrix); on the
    host route the int32 order store (`order`, an `istores` index) and the
    gathered scores (`values`)."""
    var count: Int
    var ids: Int
    var values: Int
    var order: Int

    def __init__(out self, count: Int, ids: Int, values: Int, order: Int):
        self.count = count
        self.ids = ids
        self.values = values
        self.order = order


def als_topn_run(mut kit: DKit, s: Int, count: Int, skip: Int) raises -> AlsTop:
    """`AlternatingLeastSquares._topn`: the N best entries of the score row
    by (-score, id), the skip row's nonzero entries left out (skip: a 1 x m
    host matrix, or -1)."""
    var m = kit.size(s)
    var n = max(min(count, m), 0)
    if n and kit.use(s):
        var order = kit.order_dev(s, 1, skip)
        var c = min(n, Int(kit.word(order[1])))
        if c == 0:
            return AlsTop(0, -1, -1, -1)
        var ids = kit.rows_of(order[0], 0, c)
        var row = kit.reshape(s, 1, m)
        var values = kit.take_cols_m(row, ids, c)
        kit.host(ids)
        return AlsTop(c, ids, values, -1)
    var order = kit.istore(n)
    var order_address = kit.iaddr(order)
    var skip_address = 0 if skip < 0 else kit.at(skip)
    var c = 0
    if n:
        var top: List[Int] = [kit.at(s), skip_address, order_address]
        c = kit.xd(XD_TOPN_DESC, top, [m, n], List[Float64]())
    var values = kit.zeros(max(c, 1), 1)
    if c:
        var moved: List[Int] = [kit.at(s), order_address, kit.at(values)]
        _ = kit.xd(XD_GATHER, moved, [c], List[Float64]())
    return AlsTop(c, -1, values, order)


def als_recommend_run(mut kit: DKit, users: Int, items: Int, userid: Int, count: Int, r: Int,
                      skip_row: Int) raises -> AlsTop:
    """`recommend` after admission: the user's score row against every item
    and its top N (`r` the admitted user_items matrix and `skip_row` its row
    to skip, or -1)."""
    var u = kit.rows_of(users, userid, userid + 1)
    var s = kit.mm(u, items, tb=True)
    var skip = -1
    if r >= 0:
        skip = kit.rows_of(r, skip_row, skip_row + 1)
    return als_topn_run(kit, s, count, skip)


def als_unit_run(mut kit: DKit, y: Int) raises -> Int:
    """The unit-normalised item factors (`similar_items`)."""
    var squares = kit.ew(DK_SQ, y)
    var sums = kit.rowsum(squares)
    var norms = kit.ew(DK_SQRT, sums)
    return kit.ew(DK_DIV, y, norms)


def als_similar_run(mut kit: DKit, unit: Int, itemid: Int, count: Int) raises -> AlsTop:
    """`similar_items` over the unit-normalised factors."""
    var row = kit.rows_of(unit, itemid, itemid + 1)
    var s = kit.mm(row, unit, tb=True)
    return als_topn_run(kit, s, count, -1)
