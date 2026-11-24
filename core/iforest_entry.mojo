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
  [train, query, out_f32, out_i32, info, loop], ints `IF_P_*`, floats
  `IF_F_*`. Returns `n_query`. `info` takes three float64 (offset_,
  max_samples_, n_features_in_), four on the resident form (the token
  last). `loop` (0: not asked) takes two Int words when the call fitted a
  forest: the trees the build loop made and the loop region's nanoseconds
  (core/fit_loop_clock.mojo; the estimator's `fit_loop_`); a call that
  scored a resident forest without fitting leaves them as they are.
- `IF_OP_RELEASE`: ints [token]; drops the resident forest that token names
  (a token nobody holds is a no-op). The member's FREE.
- `IF_OP_EXPORT` (round 4; was the Python binding `iforest_forest_export`):
  the fitted forest as the compact host form `save()`, pickle and the member's
  EXPORT write (isolation_forest/forest_io.mojo). Addresses [train, info,
  offsets, feature, threshold, left, right], ints the RUN ints then
  [trees, total], floats the RUN floats. The resident forest the token names
  (its key equal), else a fresh fit from `train` kept under a new token. The
  five destinations 0 (trees / total 0) ask for the sizes only; otherwise
  they receive the forest and must be sized exactly (trees + 1 offsets,
  `total` words each). `info` takes SIX float64: offset_, max_samples_,
  n_features_in_, token, trees, total. Returns the trees. A refit-per-call
  image keeps the forest after the sizes call and drops it after the copy.
- `IF_OP_FOREST_RUN` (round 4; was the Python binding `iforest_forest_run`):
  ONE of score_samples / decision_function / predict over a SAVED compact
  forest (a loaded or unpickled estimator, an imported member): addresses
  [offsets, feature, threshold, left, right, query, out_f32, out_i32, info],
  ints [trees, total, n_features, n_query, max_samples_, want, token], floats
  [offset_]. The forest is uploaded once and kept under a token on a resident
  image (`token` an earlier call's: no upload); `info` takes four float64
  (offset_, max_samples_, n_features_in_, token). Returns `n_query`.

Every operation runs with the GIL released by its caller; the resident
stores' locks are held only for mint, the one take, the put and free.
`IF_P_DEVICE_COUNT` > 1 (a member whose admit read the public thread's plan:
`IF_P_DEVICES` = the ids in rank order) runs over exactly those devices; else
`IF_P_SINGLE` 1 (a member: its device is its route's) runs on one device;
0 reads the calling thread's device plan (`MOJOLEARN_IFOREST_DEVICE_COUNT`),
as the estimator's doors always did.

Publication and lookup are small metadata operations with no independent
numerical work to split. A layout change takes a new V suffix.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot

# v4 (lane alm-forest, 2026-11-24): the RUN loop address (IF_A_LOOP).
comptime IF_ENTRY_ABI = "mojolearn.iforest_entry.v4"
"""v2 (round 4): IF_P_SINGLE joined the RUN ints; IF_OP_EXPORT and IF_OP_FOREST_RUN.
v3 (sweep-b, 2026-11-21): IF_P_DEVICES and IF_P_DEVICE_COUNT joined the RUN ints
(a member's device plan, MEMBER-ABI-V2 item 28)."""
comptime IF_OP_CAPS = 0
comptime IF_OP_RUN = 1
comptime IF_OP_RELEASE = 2
comptime IF_OP_EXPORT = 3
comptime IF_OP_FOREST_RUN = 4
comptime IF_OP_COUNT = 5

# Capability bits (IF_OP_CAPS).
comptime IF_CAP_RESIDENT = 1
"""The fitted forest stays resident between calls under a token."""
comptime IF_CAP_SCORES_F64 = 2
"""The resident run writes float64 scores for `want` 3 / 4."""
comptime IF_CAP_DEVICE_SCAN = 4
"""The fit scans the uploaded training matrix for non-finite cells itself."""
comptime IF_CAP_SLOT_X = 8
"""The scoring of a one-device run reads THE FOLD X WORD as its query
(core/fold_x_device.mojo: a negative query address names a fold slot of the
x_prep store; lane vdf-scope-trees2b). The training address is a fold slot
only on an image with IF_CAP_FIT_SLOT_X."""
comptime IF_CAP_FIT_SLOT_X = 16
"""The FIT of a one-device run (and its refit after an eviction, and the
export's refit) reads THE FOLD X WORD as its TRAINING address: a negative
`train` names a fold slot of the x_prep store, copied device to device into
the fit's upload (isolation_forest/impl/isolation_forest.mojo
`_upload_rowmajor_as_colmajor_device`; lane vdf-scope-trees4). The member
keeps that slot alive for the state's life."""

# IF_OP_RUN addresses.
comptime IF_A_TRAIN = 0
comptime IF_A_QUERY = 1
comptime IF_A_OUT_F32 = 2
comptime IF_A_OUT_I32 = 3
comptime IF_A_INFO = 4
comptime IF_A_LOOP = 5
comptime IF_RUN_ADDRESSES = 6
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
comptime IF_P_SINGLE = 15
comptime IF_P_DEVICES = 16       # address of `device_count` Int device ids (rank order); 0: none
comptime IF_P_DEVICE_COUNT = 17
comptime IF_RUN_INTS = 18
# IF_OP_RUN floats.
comptime IF_F_MAX_SAMPLES_FRAC = 0
comptime IF_F_MAX_FEATURES_FRAC = 1
comptime IF_F_CONTAMINATION = 2
comptime IF_RUN_FLOATS = 3
# IF_OP_EXPORT: addresses [train, info] then the five destinations; ints the
# RUN ints then [trees, total]; floats the RUN floats.
comptime IF_EXPORT_ADDRESSES = 7
comptime IF_EXPORT_INTS = IF_RUN_INTS + 2
comptime IF_EXPORT_INFO = 6
# IF_OP_FOREST_RUN.
comptime IF_FOREST_ADDRESSES = 9
comptime IF_FOREST_INTS = 7
comptime IF_FOREST_FLOATS = 1


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
    var loop: Int
    """The address of the two fit loop words (IF_A_LOOP), 0: not asked."""
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
    var single: Bool
    """True: one device (a member's route); False: the thread's device plan."""
    var devices: Int
    """Address of `device_count` Int device ids in rank order (a member's
    `call.devices`); used when `device_count` > 1, before `single`."""
    var device_count: Int

    def __init__(out self):
        self.train = 0
        self.query = 0
        self.out_f32 = 0
        self.out_i32 = 0
        self.info = 0
        self.loop = 0
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
        self.single = False
        self.devices = 0
        self.device_count = 0


@fieldwise_init
struct IFForestSpans(Copyable, Movable):
    """The five host arrays of a compact forest by address (offsets int32
    of `trees + 1`, feature int32, threshold float32, left int32, right
    int32, `total` words each); all 0 asks an export for the sizes only."""

    var offsets: Int
    var feature: Int
    var threshold: Int
    var left: Int
    var right: Int
    var trees: Int
    var total: Int


struct IFExportRequest(Copyable, Movable):
    """One IF_OP_EXPORT: the fit's words (`fit.train`, `fit.info` (six
    float64), the knobs, `fit.token`, `fit.single`; query / outputs / want
    unused) and the destinations."""

    var fit: IFRunRequest
    var forest: IFForestSpans

    def __init__(out self, var fit: IFRunRequest, var forest: IFForestSpans):
        self.fit = fit^
        self.forest = forest^


@fieldwise_init
struct IFForestRunRequest(Copyable, Movable):
    """One IF_OP_FOREST_RUN over a saved forest."""

    var forest: IFForestSpans
    var query: Int
    var out_f32: Int
    var out_i32: Int
    var info: Int
    var n_features: Int
    var n_query: Int
    var max_samples: Int
    var offset: Float64
    var want: Int
    var token: Int


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

    @staticmethod
    def export(request: IFExportRequest) raises -> Int:
        """The sizes, or the forest into the destinations, of the forest
        the token names (else a fresh fit); writes `info`; returns trees."""
        ...

    @staticmethod
    def forest_run(request: IFForestRunRequest) raises -> Int:
        """One score over a saved forest; returns `n_query`."""
        ...


def _if_knobs_at(pp: Int, ff: Int) -> IFRunRequest:
    """The RUN ints and floats (shared by IF_OP_RUN and IF_OP_EXPORT)."""
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var r = IFRunRequest()
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
    r.single = p[IF_P_SINGLE] != 0
    r.devices = p[IF_P_DEVICES]
    r.device_count = p[IF_P_DEVICE_COUNT]
    r.max_samples_frac = f[IF_F_MAX_SAMPLES_FRAC]
    r.max_features_frac = f[IF_F_MAX_FEATURES_FRAC]
    r.contamination = f[IF_F_CONTAMINATION]
    return r^


def if_run_request_at(aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> IFRunRequest:
    """UNPACK of the request spans (the image side)."""
    if na != IF_RUN_ADDRESSES or np != IF_RUN_INTS or nf != IF_RUN_FLOATS:
        raise Error("iforest_run: needs " + String(IF_RUN_ADDRESSES) + " addresses, "
                    + String(IF_RUN_INTS) + " integers and " + String(IF_RUN_FLOATS) + " floats")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var r = _if_knobs_at(pp, ff)
    r.train = a[IF_A_TRAIN]
    r.query = a[IF_A_QUERY]
    r.out_f32 = a[IF_A_OUT_F32]
    r.out_i32 = a[IF_A_OUT_I32]
    r.info = a[IF_A_INFO]
    r.loop = a[IF_A_LOOP]
    if r.info == 0:
        raise Error("iforest_run: the info address is null")
    if r.train == 0 or r.query == 0:
        raise Error("iforest_run: the training or query matrix address is null")
    return r^


def if_export_request_at(aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> IFExportRequest:
    """UNPACK of the IF_OP_EXPORT spans (the image side)."""
    if na != IF_EXPORT_ADDRESSES or np != IF_EXPORT_INTS or nf != IF_RUN_FLOATS:
        raise Error("iforest_forest_export: needs " + String(IF_EXPORT_ADDRESSES) + " addresses, "
                    + String(IF_EXPORT_INTS) + " integers and " + String(IF_RUN_FLOATS) + " floats")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var r = _if_knobs_at(pp, ff)
    r.train = a[0]
    r.info = a[1]
    var forest = IFForestSpans(a[2], a[3], a[4], a[5], a[6], p[IF_RUN_INTS], p[IF_RUN_INTS + 1])
    if r.info == 0:
        raise Error("iforest_forest_export: the info address is null")
    if r.train == 0:
        raise Error("iforest_forest_export: the training matrix address is null")
    if r.n_train <= 0 or r.n_features <= 0:
        raise Error("iforest_forest_export: n_train and n_features must both be positive")
    var write = forest.offsets != 0
    if write and (forest.feature == 0 or forest.threshold == 0 or forest.left == 0 or forest.right == 0
                  or forest.trees < 1 or forest.total < 1):
        raise Error("iforest_forest_export: a destination address or size is missing")
    return IFExportRequest(r^, forest^)


def if_forest_run_request_at(aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> IFForestRunRequest:
    """UNPACK of the IF_OP_FOREST_RUN spans (the image side)."""
    if na != IF_FOREST_ADDRESSES or np != IF_FOREST_INTS or nf != IF_FOREST_FLOATS:
        raise Error("iforest_forest_run: needs " + String(IF_FOREST_ADDRESSES) + " addresses, "
                    + String(IF_FOREST_INTS) + " integers and " + String(IF_FOREST_FLOATS) + " float")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var forest = IFForestSpans(a[0], a[1], a[2], a[3], a[4], p[0], p[1])
    var r = IFForestRunRequest(forest^, a[5], a[6], a[7], a[8], p[2], p[3], p[4], f[0], p[5], p[6])
    if r.info == 0 or r.query == 0:
        raise Error("iforest_forest_run: the query or info address is null")
    if r.n_features <= 0 or r.n_query <= 0:
        raise Error("iforest_forest_run: n_features and n_query must both be positive")
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
        elif operation == IF_OP_EXPORT:
            value = I.export(if_export_request_at(aa, na, pp, np, ff, nf))
        elif operation == IF_OP_FOREST_RUN:
            value = I.forest_run(if_forest_run_request_at(aa, na, pp, np, ff, nf))
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
                                           request.info, request.loop]
    var p: Array[Int, IF_RUN_INTS] = [
        request.n_train, request.n_features, request.n_query, request.n_estimators,
        request.max_samples_mode, request.max_samples_int, request.max_depth,
        request.max_features_mode, request.max_features_int, 1 if request.bootstrap else 0,
        request.random_state, 1 if request.contamination_auto else 0, request.want,
        1 if request.resident else 0, request.token, 1 if request.single else 0,
        request.devices, request.device_count,
    ]
    var f: Array[Float64, IF_RUN_FLOATS] = [request.max_samples_frac, request.max_features_frac,
                                            request.contamination]
    return entry_ref_call(entry, IF_OP_RUN, a, p, f)


def if_entry_export(entry: EntryRef, request: IFRunRequest, forest: IFForestSpans) raises -> Int:
    """One IF_OP_EXPORT through the resolved entry (`request.train`,
    `request.info` (six float64), the knobs, `token`, `single`; destinations
    all 0 for the sizes only). The caller keeps every lent buffer alive and
    releases the GIL around the call. Returns the trees."""
    var a: Array[Int, IF_EXPORT_ADDRESSES] = [request.train, request.info, forest.offsets, forest.feature,
                                              forest.threshold, forest.left, forest.right]
    var p: Array[Int, IF_EXPORT_INTS] = [
        request.n_train, request.n_features, request.n_query, request.n_estimators,
        request.max_samples_mode, request.max_samples_int, request.max_depth,
        request.max_features_mode, request.max_features_int, 1 if request.bootstrap else 0,
        request.random_state, 1 if request.contamination_auto else 0, request.want,
        1 if request.resident else 0, request.token, 1 if request.single else 0,
        request.devices, request.device_count, forest.trees, forest.total,
    ]
    var f: Array[Float64, IF_RUN_FLOATS] = [request.max_samples_frac, request.max_features_frac,
                                            request.contamination]
    return entry_ref_call(entry, IF_OP_EXPORT, a, p, f)


def if_entry_forest_run(entry: EntryRef, request: IFForestRunRequest) raises -> Int:
    """One IF_OP_FOREST_RUN through the resolved entry; the caller keeps the
    forest, query, output and info buffers alive and releases the GIL."""
    var a: Array[Int, IF_FOREST_ADDRESSES] = [
        request.forest.offsets, request.forest.feature, request.forest.threshold, request.forest.left,
        request.forest.right, request.query, request.out_f32, request.out_i32, request.info,
    ]
    var p: Array[Int, IF_FOREST_INTS] = [request.forest.trees, request.forest.total, request.n_features,
                                         request.n_query, request.max_samples, request.want, request.token]
    var f: Array[Float64, IF_FOREST_FLOATS] = [request.offset]
    return entry_ref_call(entry, IF_OP_FOREST_RUN, a, p, f)


def if_entry_release(entry: EntryRef, token: Int) raises:
    """Drop the resident forest `token` names through the resolved entry."""
    var a: Array[Int, 0] = []
    var p: Array[Int, 1] = [token]
    var f: Array[Float64, 0] = []
    _ = entry_ref_call(entry, IF_OP_RELEASE, a, p, f)
