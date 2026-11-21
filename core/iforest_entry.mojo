# SPDX-License-Identifier: Apache-2.0
"""IsolationForest's native entry (lane ma-pub-trees unit 3, 2026-11-21).

The two svm images (`_mojolearn_svm` on a GPU vendor, `_mojolearn_svm_host`
on the CPU column) each define an `IFImage` (their own fit-and-score over the
same request) and an `abi("C")` entry that calls `if_entry_dispatch` with it,
and publish that entry at module init with `if_entry_publish` under (their
tier, their vendor word). A caller resolves the slot ONCE (`if_entry_ref`:
the slot read, the key and schema checks and every table refusal) and then
calls the resolved pointer with plain spans (`if_entry_run`, `if_entry_caps`):
no Python object crosses, no CPython call, no name lookup per call.

This replaces the Python binding `iforest_run` (its sixteen / seventeen
params list became `IFRunRequest`; the words and their meaning are the same).
The caller releases the GIL around the call: the resident store's take and
put now hold the store's own lock (isolation_forest/estimator.mojo,
svm/host/svm_host_impl.mojo), so the fit and the scoring run unlocked as they
did inside the old binding.

Operations:
- `IF_OP_CAPS`: no spans; the image's `IF_CAP_*` bits (what the old probes
  `iforest_resident`, `iforest_scores_f64`, `iforest_device_scan` and
  `iforest_device_finite_scan` answered).
- `IF_OP_RUN`: one fit-and-score (or a resident score), addresses
  [train, query, out_f32, out_i32, info], ints `IF_P_*`, floats `IF_F_*`.
  Returns `n_query`. `info` takes three float64 (offset_, max_samples_,
  n_features_in_), four on the resident form (the token last).
- `IF_OP_RELEASE`: ints [token]; drops the resident forest that token names
  (a token nobody holds is a no-op). The member's FREE.

Publication and lookup are small metadata operations with no independent
numerical work to split. A layout change takes a new V suffix.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot

comptime IF_ENTRY_ABI = "mojolearn.iforest_entry.v1"
comptime IF_OP_CAPS = 0
comptime IF_OP_RUN = 1
comptime IF_OP_RELEASE = 2
comptime IF_OP_COUNT = 3

# Capability bits (IF_OP_CAPS).
comptime IF_CAP_RESIDENT = 1
"""The fitted forest stays resident between calls under a token."""
comptime IF_CAP_SCORES_F64 = 2
"""The resident run writes float64 scores for `want` 3 / 4."""
comptime IF_CAP_DEVICE_SCAN = 4
"""The fit scans the uploaded training matrix for non-finite cells itself."""

# IF_OP_RUN addresses.
comptime IF_A_TRAIN = 0
comptime IF_A_QUERY = 1
comptime IF_A_OUT_F32 = 2
comptime IF_A_OUT_I32 = 3
comptime IF_A_INFO = 4
comptime IF_RUN_ADDRESSES = 5
# IF_OP_RUN ints.
comptime IF_P_N_TRAIN = 0
comptime IF_P_N_FEATURES = 1
comptime IF_P_N_QUERY = 2
comptime IF_P_N_ESTIMATORS = 3
comptime IF_P_MAX_SAMPLES_MODE = 4
comptime IF_P_MAX_SAMPLES_INT = 5
comptime IF_P_MAX_DEPTH = 6
comptime IF_P_MAX_FEATURES_MODE = 7
comptime IF_P_MAX_FEATURES_INT = 8
comptime IF_P_BOOTSTRAP = 9
comptime IF_P_RANDOM_STATE = 10
comptime IF_P_CONTAMINATION_AUTO = 11
comptime IF_P_WANT = 12
comptime IF_P_RESIDENT = 13
comptime IF_P_TOKEN = 14
comptime IF_RUN_INTS = 15
# IF_OP_RUN floats.
comptime IF_F_MAX_SAMPLES_FRAC = 0
comptime IF_F_MAX_FEATURES_FRAC = 1
comptime IF_F_CONTAMINATION = 2
comptime IF_RUN_FLOATS = 3


struct IFRunRequest(Copyable, Defaultable, Movable):
    """One `iforest_run`: the training and query matrices (row-major float32,
    lent by address), the two output buffers (only the one `want` selects is
    written; the other may be 0), the info buffer, the fit's knobs and the
    resident form. `want`: 0 score_samples, 1 decision_function, 2 predict,
    3 / 4 the float64 score_samples / decision_function on an image with
    `IF_CAP_SCORES_F64`. `resident` True is the old seventeen-param form
    (`token` 0 fits a fresh forest and keeps it; a live token with an equal
    key scores with no fit); False the sixteen-param refit-per-call form."""

    var train: Int
    var query: Int
    var out_f32: Int
    var out_i32: Int
    var info: Int
    var n_train: Int
    var n_features: Int
    var n_query: Int
    var n_estimators: Int
    var max_samples_mode: Int
    var max_samples_int: Int
    var max_samples_frac: Float64
    var max_depth: Int
    var max_features_mode: Int
    var max_features_int: Int
    var max_features_frac: Float64
    var bootstrap: Bool
    var random_state: Int
    var contamination_auto: Bool
    var contamination: Float64
    var want: Int
    var resident: Bool
    var token: Int

    def __init__(out self):
        self.train = 0
        self.query = 0
        self.out_f32 = 0
        self.out_i32 = 0
        self.info = 0
        self.n_train = 0
        self.n_features = 0
        self.n_query = 0
        self.n_estimators = 100
        self.max_samples_mode = 0
        self.max_samples_int = 256
        self.max_samples_frac = 1.0
        self.max_depth = -1
        self.max_features_mode = 0
        self.max_features_int = 0
        self.max_features_frac = 1.0
        self.bootstrap = False
        self.random_state = 0
        self.contamination_auto = True
        self.contamination = 0.0
        self.want = 0
        self.resident = False
        self.token = 0


trait IFImage:
    @staticmethod
    def caps() -> Int:
        """The image's IF_CAP_* bits."""
        ...

    @staticmethod
    def run(request: IFRunRequest) raises -> Int:
        """One fit-and-score; writes the selected output and `info`;
        returns `n_query`. Called with the GIL released."""
        ...

    @staticmethod
    def release(token: Int) raises:
        """Drop the resident forest `token` names, if any."""
        ...


def if_run_request_at(aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> IFRunRequest:
    """UNPACK of the request spans (the image side)."""
    if na != IF_RUN_ADDRESSES or np != IF_RUN_INTS or nf != IF_RUN_FLOATS:
        raise Error("iforest_run: needs " + String(IF_RUN_ADDRESSES) + " addresses, "
                    + String(IF_RUN_INTS) + " integers and " + String(IF_RUN_FLOATS) + " floats")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var r = IFRunRequest()
    r.train = a[IF_A_TRAIN]
    r.query = a[IF_A_QUERY]
    r.out_f32 = a[IF_A_OUT_F32]
    r.out_i32 = a[IF_A_OUT_I32]
    r.info = a[IF_A_INFO]
    r.n_train = p[IF_P_N_TRAIN]
    r.n_features = p[IF_P_N_FEATURES]
    r.n_query = p[IF_P_N_QUERY]
    r.n_estimators = p[IF_P_N_ESTIMATORS]
    r.max_samples_mode = p[IF_P_MAX_SAMPLES_MODE]
    r.max_samples_int = p[IF_P_MAX_SAMPLES_INT]
    r.max_depth = p[IF_P_MAX_DEPTH]
    r.max_features_mode = p[IF_P_MAX_FEATURES_MODE]
    r.max_features_int = p[IF_P_MAX_FEATURES_INT]
    r.bootstrap = p[IF_P_BOOTSTRAP] != 0
    r.random_state = p[IF_P_RANDOM_STATE]
    r.contamination_auto = p[IF_P_CONTAMINATION_AUTO] != 0
    r.want = p[IF_P_WANT]
    r.resident = p[IF_P_RESIDENT] != 0
    r.token = p[IF_P_TOKEN]
    r.max_samples_frac = f[IF_F_MAX_SAMPLES_FRAC]
    r.max_features_frac = f[IF_F_MAX_FEATURES_FRAC]
    r.contamination = f[IF_F_CONTAMINATION]
    if r.info == 0:
        raise Error("iforest_run: the info address is null")
    if r.train == 0 or r.query == 0:
        raise Error("iforest_run: the training or query matrix address is null")
    return r^


def if_entry_dispatch[I: IFImage](operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    """The body of an svm image's `abi("C")` iforest entry: span admission,
    the operation, the count to `result`, a refusal to `sink`."""
    try:
        if not result or na < 0 or np < 0 or nf < 0:
            raise Error("iforest entry: invalid request spans")
        if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff):
            raise Error("iforest entry: null request span")
        var value = 0
        if operation == IF_OP_CAPS:
            value = I.caps()
        elif operation == IF_OP_RUN:
            value = I.run(if_run_request_at(aa, na, pp, np, ff, nf))
        elif operation == IF_OP_RELEASE:
            if np != 1 or na != 0:
                raise Error("iforest release: needs one integer (the token)")
            I.release(MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)[0])
        else:
            raise Error("iforest entry: unknown operation " + String(operation))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct IFEntryOwner(Defaultable, Movable):
    """An image-lifetime publisher word."""

    var word: Int

    def __init__(out self):
        self.word = 1


def _if_key_check(mode: Int, vendor: Int) raises:
    if mode < 0 or mode > 1 or vendor < 0 or vendor > 3:
        raise Error("native iforest entries: invalid tier/vendor key")


def if_entry_publish[owner_name: StaticString](mode: Int, vendor: Int, entry: NativeDescriptorCall) raises:
    """Publish an svm image's iforest entry under (its tier, its vendor word:
    0 CPU, 1 Metal, 2 CUDA, 3 HIP). The caller skips a build with no vendor."""
    _if_key_check(mode, vendor)
    comptime OWNER = SharedSlot[IFEntryOwner, owner_name]
    var publisher = Int(OWNER.get_or_create_ptr())
    var schemas: List[String] = [String(IF_ENTRY_ABI)]
    var operations: List[Int] = [IF_OP_COUNT]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    if vendor == 0 and mode == 1:
        entry_table_publish["MojolearnEntriesIForestIdenticalCPUV1"](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 0 and mode == 0:
        entry_table_publish["MojolearnEntriesIForestFastCPUV1"](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 1 and mode == 1:
        entry_table_publish["MojolearnEntriesIForestIdenticalMetalV1"](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 1 and mode == 0:
        entry_table_publish["MojolearnEntriesIForestFastMetalV1"](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 2 and mode == 1:
        entry_table_publish["MojolearnEntriesIForestIdenticalCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 2 and mode == 0:
        entry_table_publish["MojolearnEntriesIForestFastCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 3 and mode == 1:
        entry_table_publish["MojolearnEntriesIForestIdenticalHIPV1"](mode, vendor, publisher, schemas, operations, entries)
    else:
        entry_table_publish["MojolearnEntriesIForestFastHIPV1"](mode, vendor, publisher, schemas, operations, entries)


def if_entry_ref(mode: Int, vendor: Int) raises -> EntryRef:
    """The published iforest entry of (tier, vendor), resolved once; every
    table refusal (not published, key or schema differs) raises here."""
    _if_key_check(mode, vendor)
    var schema = String(IF_ENTRY_ABI)
    if vendor == 0 and mode == 1:
        return entry_table_resolve["MojolearnEntriesIForestIdenticalCPUV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    if vendor == 0 and mode == 0:
        return entry_table_resolve["MojolearnEntriesIForestFastCPUV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    if vendor == 1 and mode == 1:
        return entry_table_resolve["MojolearnEntriesIForestIdenticalMetalV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    if vendor == 1 and mode == 0:
        return entry_table_resolve["MojolearnEntriesIForestFastMetalV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    if vendor == 2 and mode == 1:
        return entry_table_resolve["MojolearnEntriesIForestIdenticalCUDAV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    if vendor == 2 and mode == 0:
        return entry_table_resolve["MojolearnEntriesIForestFastCUDAV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    if vendor == 3 and mode == 1:
        return entry_table_resolve["MojolearnEntriesIForestIdenticalHIPV1"](mode, vendor, 0, schema, IF_OP_COUNT)
    return entry_table_resolve["MojolearnEntriesIForestFastHIPV1"](mode, vendor, 0, schema, IF_OP_COUNT)


def if_entry_caps(entry: EntryRef) raises -> Int:
    """The image's IF_CAP_* bits: one indirect call, no spans."""
    var a: Array[Int, 0] = []
    var p: Array[Int, 0] = []
    var f: Array[Float64, 0] = []
    return entry_ref_call(entry, IF_OP_CAPS, a, p, f)


def if_entry_run(entry: EntryRef, request: IFRunRequest) raises -> Int:
    """One `iforest_run` through the resolved entry, request words on this
    frame (no heap). The caller keeps every lent buffer alive and releases
    the GIL around the call."""
    var a: Array[Int, IF_RUN_ADDRESSES] = [request.train, request.query, request.out_f32, request.out_i32,
                                           request.info]
    var p: Array[Int, IF_RUN_INTS] = [
        request.n_train, request.n_features, request.n_query, request.n_estimators,
        request.max_samples_mode, request.max_samples_int, request.max_depth,
        request.max_features_mode, request.max_features_int, 1 if request.bootstrap else 0,
        request.random_state, 1 if request.contamination_auto else 0, request.want,
        1 if request.resident else 0, request.token,
    ]
    var f: Array[Float64, IF_RUN_FLOATS] = [request.max_samples_frac, request.max_features_frac,
                                            request.contamination]
    return entry_ref_call(entry, IF_OP_RUN, a, p, f)


def if_entry_release(entry: EntryRef, token: Int) raises:
    """Drop the resident forest `token` names through the resolved entry."""
    var a: Array[Int, 0] = []
    var p: Array[Int, 1] = [token]
    var f: Array[Float64, 0] = []
    _ = entry_ref_call(entry, IF_OP_RELEASE, a, p, f)
