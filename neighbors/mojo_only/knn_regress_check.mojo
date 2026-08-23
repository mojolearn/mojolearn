"""Gates for the k-NN REGRESSOR: `neighbors/estimator.mojo::
knn_regressor_predict` over `ML::knn_regress` / `regress_avg_kernel`
(`neighbors/ported/knn/knn.mojo`, `neighbors/ported/selection/knn.mojo`).

Four claims, hashed fixture throughout:

1. `check_knn_regress_matches_host_transcription`: the device mean equals
   a HOST transcription of `regress_avg_kernel` -- Float32, the `k`
   targets added in sorted-slot order, then one division by `k`, every
   seam through `ftz` -- bit for bit, at `k = 5` and `k = 50` (the two
   the E2U matrix runs).
2. `check_knn_regress_reach_by_sabotage`: add 1000 to ONE target, the
   nearest neighbour of query 0; query 0's prediction must move by about
   `1000/k`, and every row that does NOT hold that index among its
   neighbours must not move at all. Then `k = 0` is refused by name.
3. `check_knn_regress_multi_output_layout`: `y` and `-y` as two columns;
   `out[:, 1]` is bitwise `-out[:, 0]` (IEEE negation commutes with a
   round-to-nearest serial sum and with the division), which gates the
   `row * n_outputs + output_offset` layout without a second oracle.
4. `check_knn_regress_run_twice_identical`.

The scikit-learn comparison is `tools/knn_sklearn_oracle.py` (numpy lives
in the gbmbench env, not in a Mojo check).
"""

from max.gpu.host import DeviceContext, HostBuffer

from mojo_only.numerics import ftz
from neighbors.estimator import knn_regressor_predict, knn_search


comptime REG_INDEX = 2000
comptime REG_QUERIES = 64
comptime REG_FEATURES = 8


def _coord(row: Int, feature: Int, salt: Int) -> Float32:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(feature) * 0xBF58476D1CE4E5B9
        + UInt64(salt) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Float64(z % 1000000) / 1000000.0)


def _target(row: Int) -> Float32:
    """A hashed target in [-5, 5), not a ramp, not a constant."""
    return Float32(Float64(_coord(row, 3, 77)) * 10.0 - 5.0)


struct RegFixture:
    var ctx: DeviceContext
    var h_index: HostBuffer[DType.float32]
    var h_query: HostBuffer[DType.float32]
    var h_y: HostBuffer[DType.float32]
    var h_dist: HostBuffer[DType.float32]
    var h_idx: HostBuffer[DType.uint32]
    var k: Int

    def __init__(out self, k: Int) raises:
        self.ctx = DeviceContext()
        self.k = k
        self.h_index = self.ctx.enqueue_create_host_buffer[DType.float32](
            REG_INDEX * REG_FEATURES
        )
        self.h_query = self.ctx.enqueue_create_host_buffer[DType.float32](
            REG_QUERIES * REG_FEATURES
        )
        self.h_y = self.ctx.enqueue_create_host_buffer[DType.float32](
            REG_INDEX
        )
        self.h_dist = self.ctx.enqueue_create_host_buffer[DType.float32](
            REG_QUERIES * k
        )
        self.h_idx = self.ctx.enqueue_create_host_buffer[DType.uint32](
            REG_QUERIES * k
        )
        self.ctx.synchronize()
        for j in range(REG_INDEX):
            for f in range(REG_FEATURES):
                self.h_index.unsafe_ptr().unsafe_store(
                    j * REG_FEATURES + f, _coord(j, f, 11)
                )
            self.h_y.unsafe_ptr().unsafe_store(j, _target(j))
        for i in range(REG_QUERIES):
            for f in range(REG_FEATURES):
                self.h_query.unsafe_ptr().unsafe_store(
                    i * REG_FEATURES + f, _coord(i, f, 29)
                )

    def search(mut self) raises:
        _ = knn_search(
            self.ctx,
            self.h_index.unsafe_ptr(),
            REG_INDEX,
            self.h_query.unsafe_ptr(),
            REG_QUERIES,
            REG_FEATURES,
            self.k,
            self.h_dist.unsafe_ptr(),
            self.h_idx.unsafe_ptr(),
            False,
        )

    def transcribe(self, mut pred: List[Float32]):
        """`regress_avg_kernel` on the host over the slots in `h_idx`."""
        pred.clear()
        for row in range(REG_QUERIES):
            var acc = Float32(0.0)
            for s in range(self.k):
                var nb = Int(self.h_idx.unsafe_ptr().unsafe_load(row * self.k + s))
                acc = ftz(acc + ftz(self.h_y.unsafe_ptr().unsafe_load(nb)))
            pred.append(ftz(acc / Float32(self.k)))

    def predict(
        mut self, k: Int, mut out: HostBuffer[DType.float32]
    ) raises -> Int:
        return knn_regressor_predict(
            self.ctx,
            self.h_index.unsafe_ptr(),
            REG_INDEX,
            self.h_query.unsafe_ptr(),
            REG_QUERIES,
            REG_FEATURES,
            k,
            self.h_y.unsafe_ptr(),
            1,
            out.unsafe_ptr(),
        )


def _check_transcription_at(k: Int) raises:
    var fx = RegFixture(k)
    fx.search()
    var want = List[Float32]()
    fx.transcribe(want)
    var out = fx.ctx.enqueue_create_host_buffer[DType.float32](REG_QUERIES)
    fx.ctx.synchronize()
    _ = fx.predict(k, out)
    var bad = 0
    var distinct = 0
    for i in range(REG_QUERIES):
        var got = out.unsafe_ptr().unsafe_load(i)
        if got.to_bits() != want[i].to_bits():
            bad += 1
        if i > 0 and want[i].to_bits() != want[i - 1].to_bits():
            distinct += 1
    if distinct == 0:
        raise Error(
            "check_knn_regress_matches_host_transcription: every row has the"
            " same prediction; the fixture gates nothing"
        )
    if bad != 0:
        raise Error(
            "check_knn_regress_matches_host_transcription: k="
            + String(k)
            + ": "
            + String(bad)
            + " of "
            + String(REG_QUERIES)
            + " predictions differ bitwise from the host fold"
        )
    _ = out^


def check_knn_regress_matches_host_transcription() raises:
    """Claim 1, at k = 5 and k = 50."""
    _check_transcription_at(5)
    _check_transcription_at(50)
    print(
        "check_knn_regress_matches_host_transcription: OK ("
        + String(REG_QUERIES)
        + " predictions bit-identical to the host fold at k=5 and k=50)"
    )


def check_knn_regress_reach_by_sabotage() raises:
    """Claim 2."""
    var k = 5
    var fx = RegFixture(k)
    fx.search()
    var a = fx.ctx.enqueue_create_host_buffer[DType.float32](REG_QUERIES)
    var b = fx.ctx.enqueue_create_host_buffer[DType.float32](REG_QUERIES)
    fx.ctx.synchronize()
    _ = fx.predict(k, a)
    var target = Int(fx.h_idx.unsafe_ptr().unsafe_load(0))  # row 0's nearest
    var saved = fx.h_y.unsafe_ptr().unsafe_load(target)
    fx.h_y.unsafe_ptr().unsafe_store(target, saved + Float32(1000.0))
    _ = fx.predict(k, b)
    fx.h_y.unsafe_ptr().unsafe_store(target, saved)
    var delta0 = Float64(b.unsafe_ptr().unsafe_load(0)) - Float64(
        a.unsafe_ptr().unsafe_load(0)
    )
    # 1000/5 = 200, give or take float32 rounding of a sum near 200.
    if delta0 < 199.9 or delta0 > 200.1:
        raise Error(
            "check_knn_regress_reach_by_sabotage: +1000 on row 0's nearest"
            " neighbour moved row 0 by "
            + String(delta0)
            + ", want ~200"
        )
    var untouched_moved = 0
    var holders = 0
    for row in range(1, REG_QUERIES):
        var holds = False
        for s in range(k):
            if Int(fx.h_idx.unsafe_ptr().unsafe_load(row * k + s)) == target:
                holds = True
        if holds:
            holders += 1
        elif (
            a.unsafe_ptr().unsafe_load(row).to_bits()
            != b.unsafe_ptr().unsafe_load(row).to_bits()
        ):
            untouched_moved += 1
    if untouched_moved != 0:
        raise Error(
            "check_knn_regress_reach_by_sabotage: "
            + String(untouched_moved)
            + " rows that do not hold the sabotaged index moved"
        )
    var refused = False
    try:
        _ = fx.predict(0, a)
    except e:
        refused = String(e).find("k must be positive") >= 0
    if not refused:
        raise Error(
            "check_knn_regress_reach_by_sabotage: k=0 was not refused by name"
        )
    _ = a^
    _ = b^
    print(
        "check_knn_regress_reach_by_sabotage: OK (row 0 moved by "
        + String(delta0)
        + " on +1000 to its nearest target; "
        + String(holders)
        + " other rows hold that index, the rest did not move; k=0 refused)"
    )


def check_knn_regress_multi_output_layout() raises:
    """Claim 3: `[y, -y]` -> `out[:, 1] == -out[:, 0]` bitwise."""
    var k = 5
    var fx = RegFixture(k)
    var y2 = fx.ctx.enqueue_create_host_buffer[DType.float32](2 * REG_INDEX)
    var out = fx.ctx.enqueue_create_host_buffer[DType.float32](2 * REG_QUERIES)
    var single = fx.ctx.enqueue_create_host_buffer[DType.float32](REG_QUERIES)
    fx.ctx.synchronize()
    for j in range(REG_INDEX):
        var v = fx.h_y.unsafe_ptr().unsafe_load(j)
        y2.unsafe_ptr().unsafe_store(j, v)
        y2.unsafe_ptr().unsafe_store(REG_INDEX + j, -v)
    _ = fx.predict(k, single)
    _ = knn_regressor_predict(
        fx.ctx,
        fx.h_index.unsafe_ptr(),
        REG_INDEX,
        fx.h_query.unsafe_ptr(),
        REG_QUERIES,
        REG_FEATURES,
        k,
        y2.unsafe_ptr(),
        2,
        out.unsafe_ptr(),
    )
    var bad = 0
    for i in range(REG_QUERIES):
        var o0 = out.unsafe_ptr().unsafe_load(2 * i)
        var o1 = out.unsafe_ptr().unsafe_load(2 * i + 1)
        if o0.to_bits() != single.unsafe_ptr().unsafe_load(i).to_bits():
            bad += 1
        if (-o1).to_bits() != o0.to_bits():
            bad += 1
    if bad != 0:
        raise Error(
            "check_knn_regress_multi_output_layout: "
            + String(bad)
            + " cells violate out[:,0] == single and out[:,1] == -out[:,0]"
        )
    _ = y2^
    _ = out^
    _ = single^
    print(
        "check_knn_regress_multi_output_layout: OK (2 outputs x "
        + String(REG_QUERIES)
        + " rows; column 0 equals the single-output answer, column 1 is"
        " its exact negation)"
    )


def check_knn_regress_run_twice_identical() raises:
    var k = 50
    var fx = RegFixture(k)
    var a = fx.ctx.enqueue_create_host_buffer[DType.float32](REG_QUERIES)
    var b = fx.ctx.enqueue_create_host_buffer[DType.float32](REG_QUERIES)
    fx.ctx.synchronize()
    _ = fx.predict(k, a)
    _ = fx.predict(k, b)
    var bad = 0
    for i in range(REG_QUERIES):
        if (
            a.unsafe_ptr().unsafe_load(i).to_bits()
            != b.unsafe_ptr().unsafe_load(i).to_bits()
        ):
            bad += 1
    if bad != 0:
        raise Error(
            "check_knn_regress_run_twice_identical: "
            + String(bad)
            + " predictions differ between two runs"
        )
    _ = a^
    _ = b^
    print(
        "check_knn_regress_run_twice_identical: OK (k=50, "
        + String(REG_QUERIES)
        + " predictions bit-identical across two predicts)"
    )


def main() raises:
    check_knn_regress_matches_host_transcription()
    check_knn_regress_reach_by_sabotage()
    check_knn_regress_multi_output_layout()
    check_knn_regress_run_twice_identical()
