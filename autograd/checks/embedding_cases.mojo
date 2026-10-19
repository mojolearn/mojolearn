# SPDX-License-Identifier: Apache-2.0
# Lane ag-embedding-op (2026-10-19, TAPE_EVERYWHERE path 4): checks of
# Tape.embedding on any executor (HostExec, Metal, NVIDIA, AMD), word for
# word against the embedding profile's host oracle
# (embedding/checks/embedding_oracle.mojo emb_forward_oracle and
# emb_backward_oracle, the contract in code). Each case prints one EMBTAPE
# line with the digest of the compared words, so the cpu and gpu outputs can
# also be diffed across columns.
#   forward and backward, fresh: padding_idx none and a row, plans scan,
#     sort and auto, ids with repeated rows and untouched rows;
#   the carry: a variable_into table whose gradient holds dW_prev, backward
#     with accumulate=True, against the oracle's accumulate (dW_prev normal
#     values: the oracle flushes a carried subnormal of an untouched row,
#     the device keeps its word);
#   a tied table: loss = sum(embedding(w, ids)) + sum(w * c): the
#     multiply's term c reaches w first (higher node), the embedding fold
#     continues from it, so the gradient is the oracle's accumulate from c;
#   inference: no_grad gives the recorded forward's words;
#   refusals: an id below 0 and one at the row count (by position and value,
#     before any gather), an expired ids handle, a padding_idx outside the
#     table.
# Shapes are small, odd and off any board row (size reasoning: they cover
# runs of length 0, 1 and many, a two-level id search and a sort padded
# past the id count).
from std.memory import bitcast
from autograd.tape import Tape, Tensor, IdsHandle
from autograd.embedding_ops import AG_EMB_PLAN_SCAN, AG_EMB_PLAN_SORT, AG_EMB_PLAN_AUTO
from embedding.checks.embedding_oracle import EMB_NO_PADDING_IDX, EmbConfig, emb_backward_oracle, emb_forward_oracle
from sequence.exec_trait import Exec
from sequence.ops import FP


def _digest(v: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _same(name: String, got: List[Float32], want: List[Float32]) raises -> UInt64:
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")
    var d = _digest(got)
    print("EMBTAPE", name, "elements", len(got), "digest", d)
    return d


def _values(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _ids(n: Int, vocab: Int, salt: Int) -> List[Int32]:
    """Ids with repeats (every fifth position repeats row 3) and rows no
    position names (only the first 3 vocab / 4 rows are drawn)."""
    var v = List[Int32](length=n, fill=Int32(0))
    var span = max(1, (3 * vocab) // 4)
    for i in range(n):  # small-loop(n: check data, host side)
        if i % 5 == 0:
            v[i] = Int32(3 % vocab)
        else:
            v[i] = Int32((i * 131 + salt * 17) % span)
    return v^


def _put_f32[E: Exec](mut ex: E, values: List[Float32]) raises -> FP:
    var p = ex.alloc(len(values))
    ex.upload(p, FP(unsafe_from_address=Int(values.unsafe_ptr())), len(values))
    return p


def _put_ids[E: Exec](mut ex: E, ids: List[Int32]) raises -> FP:
    """The ids' bit patterns in executor words (Tape.ids' layout)."""
    var p = ex.alloc(len(ids))
    ex.upload(p, FP(unsafe_from_address=Int(ids.unsafe_ptr())), len(ids))
    return p


def _get[E: Exec](mut ex: E, p: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), p, n)
    return out^


def _case[E: Exec](
    mut ex: E, mut tape: Tape, name: String, vocab: Int, width: Int, n: Int, pad: Int, plan: Int, carry: Bool,
) raises -> UInt64:
    """Forward and table gradient of embedding(w, ids) seeded with dY,
    against the oracle (accumulate from dW_prev when `carry`)."""
    tape.reset()
    var w_host = _values(vocab * width, 1)
    var ids_host = _ids(n, vocab, 2)
    var dy_host = _values(n * width, 3)
    var prev = List[Float32]()
    if carry:
        prev = _values(vocab * width, 4)
    var w_buf = _put_f32(ex, w_host)
    var g_buf = ex.alloc(vocab * width)
    if carry:
        ex.upload(g_buf, FP(unsafe_from_address=Int(prev.unsafe_ptr())), vocab * width)
    var w = tape.variable_into(ex, w_buf, g_buf, vocab, width)
    var ids = tape.ids(ex, _put_ids(ex, ids_host), n)
    var y = tape.embedding(ex, w, ids, pad, plan)
    var seed = tape.tensor(ex, dy_host, n, width)
    tape.backward_from(ex, y, seed, carry)
    var cfg = EmbConfig(vocab, width, pad, carry)
    var d = _same(name + "/forward", tape.values(ex, y), emb_forward_oracle(w_host, ids_host, cfg))
    var g = _same(name + "/dW", _get(ex, g_buf, vocab * width), emb_backward_oracle(dy_host, ids_host, cfg, prev))
    return (d ^ g) * UInt64(1099511628211)


def _tied[E: Exec](mut ex: E, mut tape: Tape, plan: Int) raises -> UInt64:
    """loss = sum(embedding(w, ids)) + sum(w * c): dW = the oracle's
    accumulate of dY = ones from dW_prev = c."""
    tape.reset()
    comptime V = 29
    comptime D = 3
    comptime T = 97
    var w_host = _values(V * D, 5)
    var c_host = _values(V * D, 6)
    var ids_host = _ids(T, V, 7)
    var w = tape.tensor(ex, w_host, V, D, True)
    var c = tape.tensor(ex, c_host, V, D)
    var y = tape.embedding(ex, w, tape.ids(ex, _put_ids(ex, ids_host), T), EMB_NO_PADDING_IDX, plan)
    var sy = tape.sum(ex, y)
    var sm = tape.sum(ex, tape.multiply(ex, w, c))
    tape.backward(ex, tape.add(ex, sy, sm))
    var ones = List[Float32](length=T * D, fill=Float32(1))
    var cfg = EmbConfig(V, D, EMB_NO_PADDING_IDX, True)
    return _same(String("tied/plan") + String(plan), tape.gradients(ex, w), emb_backward_oracle(ones, ids_host, cfg, c_host))


def _inference[E: Exec](mut ex: E) raises -> UInt64:
    comptime V = 11
    comptime D = 7
    comptime T = 53
    var w_host = _values(V * D, 8)
    var ids_host = _ids(T, V, 9)
    var rec = Tape()
    var a = rec.embedding(ex, rec.tensor(ex, w_host, V, D, True), rec.ids(ex, _put_ids(ex, ids_host), T))
    var want = rec.values(ex, a)
    var inf = Tape()
    inf.no_grad()
    var b = inf.embedding(ex, inf.tensor(ex, w_host, V, D), inf.ids(ex, _put_ids(ex, ids_host), T))
    var d = _same("inference/forward", inf.values(ex, b), want)
    rec.release(ex)
    inf.release(ex)
    return d


def _refused[E: Exec](mut ex: E, mut tape: Tape, bad: Int32, at: Int) raises:
    tape.reset()
    comptime V = 13
    comptime D = 2
    comptime T = 600
    var ids_host = _ids(T, V, 10)
    ids_host[at] = bad
    var w = tape.tensor(ex, _values(V * D, 11), V, D, True)
    var ids = tape.ids(ex, _put_ids(ex, ids_host), T)
    var raised = False
    try:
        _ = tape.embedding(ex, w, ids)
    except e:
        var msg = String(e)
        var want = String("id ") + String(Int(bad)) + " at position " + String(at) + " is outside [0, " + String(V) + ")"
        if want not in msg:
            raise Error("refusal: unexpected message: " + msg)
        raised = True
    if not raised:
        raise Error(String("refusal: id ") + String(Int(bad)) + " at position " + String(at) + " was not refused")
    print("EMBTAPE refusal id", Int(bad), "position", at, "REFUSED")


def _refuse_handles[E: Exec](mut ex: E, mut tape: Tape) raises:
    tape.reset()
    var ids_host = _ids(9, 5, 12)
    var w = tape.tensor(ex, _values(10, 13), 5, 2, True)
    var ids = tape.ids(ex, _put_ids(ex, ids_host), 9)
    var raised = False
    try:
        _ = tape.embedding(ex, w, ids, 5)
    except:
        raised = True
    if not raised:
        raise Error("refusal: padding_idx 5 of a 5-row table was not refused")
    tape.reset()
    var w2 = tape.tensor(ex, _values(10, 13), 5, 2, True)
    raised = False
    try:
        _ = tape.embedding(ex, w2, ids)
    except:
        raised = True
    if not raised:
        raise Error("refusal: an ids handle of an expired graph was not refused")
    print("EMBTAPE refusal handles REFUSED")


def run_embedding_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    var cases = 0
    var plans: List[Int] = [AG_EMB_PLAN_SCAN, AG_EMB_PLAN_SORT, AG_EMB_PLAN_AUTO]
    for pi in range(len(plans)):  # small-loop(3: run plans)
        for pad_i in range(2):  # small-loop(2: no padding row, padding row 4)
            for ci in range(2):  # small-loop(2: fresh, carried)
                var pad = EMB_NO_PADDING_IDX
                if pad_i == 1:
                    pad = 4
                var name = String("plan") + String(plans[pi]) + "/pad" + String(pad)
                if ci == 1:
                    name += "/carry"
                else:
                    name += "/fresh"
                # 37 rows, 5 columns, 211 ids: runs of length 0 (rows past
                # 3/4 of the table), 1 and ~45 (row 3); the sort pads to 256
                var d = _case(ex, tape, name, 37, 5, 211, pad, plans[pi], ci == 1)
                summary = (summary ^ d) * UInt64(1099511628211)
                cases += 2
        summary = (summary ^ _tied(ex, tape, plans[pi])) * UInt64(1099511628211)
        cases += 1
    # one id per row and a width-1 table (a single-column fold)
    summary = (summary ^ _case(ex, tape, String("wide-ids/plan0"), 3, 1, 1, EMB_NO_PADDING_IDX, AG_EMB_PLAN_SCAN, False)) * UInt64(1099511628211)
    summary = (summary ^ _case(ex, tape, String("wide-ids/plan1"), 3, 1, 1, EMB_NO_PADDING_IDX, AG_EMB_PLAN_SORT, False)) * UInt64(1099511628211)
    cases += 4
    summary = (summary ^ _inference(ex)) * UInt64(1099511628211)
    cases += 1
    # 600 ids: a two-level search (3 blocks, then one); the bad id in the
    # second block, and one at the very end
    _refused(ex, tape, Int32(-1), 300)
    _refused(ex, tape, Int32(13), 599)
    _refuse_handles(ex, tape)
    cases += 3
    tape.release(ex)
    print("PASS embedding tape cases", cases, "summary", summary)
