# SPDX-License-Identifier: Apache-2.0
"""Native HDBSCAN snapshot, borrowed-span, release and refusal contracts.

Independent cases run in parallel. Fixed-size metadata assertions do not add
another numerical implementation or execute a GPU provider.
"""
from core.host_parallel import host_parallelize
from core.kmeans_variants_api import VariantsSpan
from core.hdbscan_fit_api import (
    HdbscanFitOptions, HdbscanFitRequest, HdbscanFitPlan,
    HdbscanPredictionRequest, HdbscanPredictionPlan,
)
from core.hdbscan_query_api import (
    HdbscanQueryDimensions, HdbscanQueryBuffers, HdbscanQueryRequest,
    HdbscanQueryPlan, HDBSCAN_QUERY_APPROXIMATE,
    HDBSCAN_QUERY_MEMBERSHIP, HDBSCAN_QUERY_ALL_POINTS,
)

comptime HDBSCAN_PLAN_CASES = 15


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct Inputs(Movable):
    var floats: List[Float32]
    var integers: List[Int32]
    var output: List[Float32]

    def __init__(out self):
        self.floats = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
        self.integers = [0, 1, 2, 3, 4, 5]
        self.output = List[Float32](length=6, fill=0)

    def request(self, batch: Int = 2) -> HdbscanQueryRequest:
        var f = VariantsSpan(Int(self.floats.unsafe_ptr()), 6)
        var i = VariantsSpan(Int(self.integers.unsafe_ptr()), 6)
        var y = VariantsSpan(Int(self.output.unsafe_ptr()), 6)
        return HdbscanQueryRequest(HdbscanQueryDimensions(3, 2, 3, 2, 2, 1, 1),
            HdbscanQueryBuffers(f, f, i, i, f, f, i, i, i, i, f, y, y), 2, 0, batch)


def refused(r: HdbscanQueryRequest, operation: Int, host: Bool, message: String) raises:
    var caught = False
    try:
        var plan = HdbscanQueryPlan(r, operation, host)
        _ = plan
    except problem:
        require(String(problem).find(message) >= 0, "wrong refusal: " + String(problem))
        caught = True
    require(caught, "expected refusal: " + message)


def fixture(unit: Int) raises:
    var inputs = Inputs()
    var r = inputs.request()
    if unit == 0:
        var plain = HdbscanFitPlan(False)
        var plan = HdbscanFitPlan(True)
        require(len(plain.storage[0].info) == 4 and len(plan.storage[0].info) == 5,
            "info shape tracks prediction-data mode")
        require(len(plan.storage[0].tree_parents) == 0 and len(plan.storage[0].inverse) == 0,
            "tree allocations stay deferred until their original boundary stage")
        plan.storage[0].prepare_tree(3)
        require(len(plan.storage[0].tree_parents) == 6 and len(plan.storage[0].tree_children) == 6
            and len(plan.storage[0].tree_lambdas) == 6 and len(plan.storage[0].tree_sizes) == 6
            and len(plan.storage[0].inverse) == 3, "tree capacities")
        require(plan.storage[0].tree_parents[5] == 0 and plan.storage[0].tree_children[5] == 0
            and plan.storage[0].tree_lambdas[5] == 0 and plan.storage[0].tree_sizes[5] == 0
            and plan.storage[0].inverse[2] == 0, "unused tree tails remain zero initialized")
        var address = plan.address()
        var moved = plan^
        require(moved.address() == address, "fit state address survives owner movement")
    elif unit == 1:
        var plan = HdbscanFitPlan(False)
        var options = HdbscanFitOptions(3, 2, 1, 2, 0, False, 0, 0, False, 1, 0)
        var request = HdbscanFitRequest(r.buffers.training, r.buffers.labels,
            r.buffers.core_distances, r.buffers.probabilities, options)
        plan.capture_host_source(request)
        inputs.floats[0] = 99
        require(len(plan.storage[0].host_source) == 6 and plan.storage[0].host_source[0] == 2,
            "host fit owns its original UNPACK snapshot")
    elif unit == 2 or unit == 3:
        var b = r.buffers
        var request = HdbscanPredictionRequest(b.labels, b.parents, b.labels, b.lambdas,
            b.labels, b.selected, b.deaths, b.selected, b.offsets, b.index_into, 3, 3, 2, 1)
        var plan = HdbscanPredictionPlan(3)
        if unit == 3:
            request.rows = 1
            request.labels.address = 0
            var caught = False
            try:
                plan.storage[0].capture(request)
            except problem:
                require(String(problem).find("shape refused by name") >= 0, "shape precedes pointer admission")
                caught = True
            require(caught, "invalid prediction shape refused")
        else:
            plan.storage[0].capture(request)
            inputs.floats[0] = 99
            inputs.integers[0] = 99
            require(plan.storage[0].labels[0] == 0 and plan.storage[0].parents[0] == 0
                and plan.storage[0].children[0] == 0 and plan.storage[0].lambdas[0] == 2
                and plan.storage[0].sizes[0] == 0 and plan.storage[0].inverse[0] == 0,
                "all six prediction inputs are independent snapshots")
            plan.storage[0].exemplars[1] = 17
            plan.storage[0].release_inputs()
            require(len(plan.storage[0].labels) == 0 and len(plan.storage[0].parents) == 0
                and len(plan.storage[0].children) == 0 and len(plan.storage[0].lambdas) == 0
                and len(plan.storage[0].sizes) == 0 and len(plan.storage[0].inverse) == 0,
                "snapshots release before PACK")
            require(plan.storage[0].exemplars[0] == 0 and plan.storage[0].exemplars[1] == 17,
                "exemplar result survives input release")
    elif unit >= 4 and unit <= 9:
        var host = unit % 2 == 0
        var operation = HDBSCAN_QUERY_APPROXIMATE if unit < 6 else (
            HDBSCAN_QUERY_MEMBERSHIP if unit < 8 else HDBSCAN_QUERY_ALL_POINTS)
        var plan = HdbscanQueryPlan(r, operation, host)
        var address = plan.address()
        var moved = plan^
        require(moved.address() == address, "query state address survives owner movement")
        inputs.floats[0] = 99
        inputs.integers[0] = 99
        if unit == 8:
            require(len(moved.storage[0].training) == 0, "CPU all-points borrows training")
        else:
            require(moved.storage[0].training[0] == 2, "training snapshot survives source mutation")
        if unit == 4 or unit == 5 or unit == 6:
            require(moved.storage[0].query[0] == 2, "approximate/CPU membership snapshots query")
        else:
            require(len(moved.storage[0].query) == 0, "GPU membership copies only each RUN batch")
        if unit == 5 or unit == 7:
            require(moved.storage[0].core_distances[0] == 2 and moved.storage[0].labels[0] == 0,
                "GPU query owns core/label snapshots")
        else:
            require(len(moved.storage[0].core_distances) == 0 and len(moved.storage[0].labels) == 0,
                "CPU uses borrowed core/labels; all-points does not copy unused fields")
        if unit == 4:
            require(len(moved.storage[0].lambdas) == 0 and len(moved.storage[0].deaths) == 0,
                "CPU approximate retains original borrowed tree policy")
        else:
            require(moved.storage[0].lambdas[0] == 2 and moved.storage[0].deaths[0] == 2,
                "other query routes own tree snapshots")
        var live = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=moved.storage[0].request.buffers.training.address)
        require(live[0] == 99, "borrowed span still refers to the boundary owner")
        _ = moved
    elif unit == 10:
        r.query_rows = 0
        r.buffers.training.address = 0
        refused(r, HDBSCAN_QUERY_APPROXIMATE, True, "shape refused by name")
    elif unit == 11:
        r.row_start = 2
        r.buffers.training.address = 0
        refused(r, HDBSCAN_QUERY_ALL_POINTS, False, "rows 2 + 2 exceed m=3")
    elif unit == 12:
        r.batch = 0
        refused(r, HDBSCAN_QUERY_MEMBERSHIP, False, "batch size must be positive")
    elif unit == 13:
        r.batch = 0
        var plan = HdbscanQueryPlan(r, HDBSCAN_QUERY_MEMBERSHIP, True)
        require(plan.storage[0].request.batch == 0 and len(plan.storage[0].query) == 4,
            "CPU provider retains its legacy unused batch semantics")
    else:
        r.batch = 0
        r.buffers.parents.address = 0
        refused(r, HDBSCAN_QUERY_MEMBERSHIP, False, "null int32 buffer address")
    _ = inputs


def run_checks() raises:
    var failures = List[Int](length=HDBSCAN_PLAN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(unit)
        except problem:
            flags[unit] = 1
            print("FAIL hdbscan_plan fixture", unit, String(problem))
    host_parallelize(run, HDBSCAN_PLAN_CASES)
    var failed = 0
    for value in failures:  # metadata: independent fixture statuses
        failed += value
    require(failed == 0, "HDBSCAN native plans failed " + String(failed) + " fixtures")
    print("PASS hdbscan_plan:", HDBSCAN_PLAN_CASES, "native contracts")
