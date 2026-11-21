# SPDX-License-Identifier: Apache-2.0
"""THE MEMBER ABI: the one native interface through which a meta-estimator
runs a member estimator (model selection, the parallel drivers, bagging,
AdaBoost, stacking and voting, the multi-output and calibration wrappers).

There is exactly one, defined here. A family PUBLISHES one entry per image
(tier, vendor) in its member table; a caller RESOLVES that entry once, when it
unpacks the member, and afterwards calls it with typed values only. No call
of this interface holds, reads or returns a Python object, and none calls the
member's Python door (docs/plans/PYTHON_AT_THE_BOUNDARY.md sections 2.5, 3).

WHAT CROSSES (all plain words; the caller owns every addressed buffer until
the call returns):
  * `MemberMatrix`: a typed matrix view (address, rows, cols, row stride,
    element type, home). X, y, the sample weights and the output are views.
  * the STATE HANDLE: an integer the family's own state store minted
    (`MEMBER_OP_NEW`) and releases (`MEMBER_OP_FREE`). A handle that was
    released, or that another family minted, is refused in words by the
    store's generation check; it is never a dangling pointer.
  * the member's `DeviceRoute` (core/device_route.mojo), resolved ONCE by
    the caller's UNPACK from the member's `device` and numeric mode with the
    family's own resolver. The member entry reads its kernels' entries from
    the route it is handed: it asks no process-wide device question and
    reads no table per call.

THE CALL. The entry has the descriptor signature every native table of ours
uses (`NativeDescriptorCall`, core/entry_table.mojo): operation = one of the
`MEMBER_OP_*` codes, ONE address = the caller's `MemberCall`, no integer or
float spans. Results come back in the `MemberCall` (`state`, `value`,
`out_cols`) and in the buffer `dst` views; the result word is 0, or the
capability mask for `MEMBER_OP_CAPABILITIES`. A refusal is the entry's error
text, raised by `member_call` in the caller.

VERSION 2 (2026-11-21; MEMBER-ABI-V2.md is the recipe). One ABI for every
family: the family numbers and their table slots are in
core/member_families.mojo (generated; no family edits this file).
  * CLASSIFIERS: the caller encodes the labels once; `y` holds the codes
    0 .. classes-1 (any element type the family documents) and
    `call.classes` the class count at `MEMBER_OP_FIT`. The state keeps the
    count; `MEMBER_OP_OUTPUTS` answers the width of every output. The labels
    themselves never cross: they stay with the caller's PACK.
  * FITTED STATE AS BLOCKS (`MEMBER_OP_EXPORT` / `MEMBER_OP_IMPORT`): a
    fitted state is a short list of typed host blocks with integer names
    (`MEMBER_BLOCK_*`, family blocks from `MEMBER_BLOCK_FAMILY`). Export
    hands out a VIEW of a block the state owns (no copy, valid until the
    state's next fit, import or free); import copies a block in. The
    family's PACK builds the Python estimator from the views, its UNPACK
    feeds a fitted estimator in; `coef_` and `feature_importances_` for
    recursive elimination and the explainers are the same export.
  * TRANSFORMERS: `MEMBER_OP_TRANSFORM` / `MEMBER_OP_INVERSE_TRANSFORM`;
    `call.fit_output` on a fit writes that operation's output for the training
    rows in the same call (fit_transform, fit_predict).
  * WEIGHTS: `call.weights` on a fit needs `MEMBER_CAN_SAMPLE_WEIGHT`, on a
    score `MEMBER_CAN_WEIGHTED_SCORE`. Without the second the caller predicts
    and runs its own weighted metric; a member never ignores a weight.
  * WHERE THE WORDS LIVE: a view is host words, or (`MEMBER_HOME_DEVICE`) a
    resident store of the route's device; `call.row_index` /
    `call.column_index` select rows and columns of X (and the rows of y and
    the weights) without a copy. Each needs its capability bit; without it
    the caller hands over a gathered host view.
  * `MEMBER_OP_CAPABILITIES` with a live `call.state` answers for that
    state's kind (a regressor of a family with classifiers has no
    predict_proba); with `state == 0` it answers the family's union.
  * SEQUENCES (layout 3): `MemberMatrix.steps` makes a row of X `steps` x
    features (`MEMBER_CAN_STEPS`).
  * FORECASTERS (`MEMBER_CAN_FORECAST`): fit reads the series in `y`
    (observations x series) and optional regressors in `x`; predict with
    `x` not given writes `dst.rows` steps ahead (with regressors, `x.rows`
    is the horizon). No operation of its own.
  * `MEMBER_OP_OUTPUTS` WITHOUT A STATE (`member_outputs`): with
    `state == 0` the call carries what a `MEMBER_OP_NEW` would (`params`,
    `param_count`, `classes`) plus two shape words of the fit to come,
    `x.cols` (features) and `y.cols` (targets; both views have `rows == 0`
    and no address: they are shapes, not data). The family answers the width
    from those words, exactly as a state made from them and fitted on such
    data would. A family whose width is only known after a fit (a count the
    fit chooses) refuses in words, and the caller asks a fitted state
    (`member_outputs_of`). With a live state the state answers and the
    shape words are ignored.

THE LIFETIME RULE (2026-11-21, lane fx-borrow; MEMBER-ABI-V2.md item 34). The
first paragraph above ("the caller owns every addressed buffer until the call
returns") holds for every operation but FIT. Every buffer a caller hands to
`MEMBER_OP_FIT` / `MEMBER_OP_PARTIAL_FIT` (x, y, weights, row_index,
column_index) stays valid and unchanged until that state is FREEd or has
been told to own its rows, whichever the caller does last on it:
  * a kind WITHOUT `MEMBER_BORROWS_FIT` keeps no pointer into a fit buffer
    after FIT returns: the caller may free the buffers at once.
  * a kind WITH `MEMBER_BORROWS_FIT` (kind-exact: ask with a live state, or
    `member_kind_capabilities`) may read them again in PREDICT / SCORE /
    TRANSFORM of that state and in EXPORT; never after a FREE, a new FIT or
    an IMPORT of that state, and never after `MEMBER_EXPORT_OWN`.
  * `MEMBER_OP_EXPORT` with `call.flags = MEMBER_EXPORT_OWN` (`member_own`):
    the state copies every fit buffer it still reads into words it owns,
    ONCE (a second call copies nothing), and from its return reads no caller
    buffer. An exported block VIEW alone does not promise that (a family may
    hand back the borrowed words themselves): a caller that lets a state
    outlive its fit buffers (a state an estimator holds:
    bindings/member_held_door.mojo `member_held_adopt` does it for every
    caller) sends OWN first. No copy is made per fit, per fold or per call.
  * a caller that GATHERS a fit buffer (a scattered fold, a bootstrap draw)
    for a borrowing kind keeps it as long as the unit's state: freed after
    FREE for a state it does not keep, after PACK (the export, and the
    adoption's OWN) for one it keeps. For a kind that owns its state it
    frees the buffer when the fit returns.

SERIAL BY NATURE: publication and resolution are metadata operations with no
independent units. The member operations themselves are the family's RUN and
parallel inside as that family is; a caller runs independent members (folds,
candidates, bagged members) on its own workers, one `MemberCall` per worker.
"""
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from core.member_families import (
    MEMBER_ABI, MEMBER_OPERATION_COUNT, MEMBER_FAMILY_COUNT, member_table, member_table_resolve, member_family_name,
)
from core.native_descriptor_abi import NativeDescriptorCall

#: The layout of `MemberCall` / `MemberMatrix`. Bump it, and the schema
#: `MEMBER_ABI` (core/member_families.mojo), on any change of `MemberCall`,
#: `MemberMatrix`, `DeviceRoute` or the operation codes.
#: 3 (2026-11-21): `MemberMatrix.steps`.
#: Words APPENDED to the end of `MemberCall` (round 5: out_dtype, scope, pin, devices, device_count; round 6:
#: pin_kept) do not
#: bump it: every image of one build compiles this file, and a member that does not read them is unchanged.
comptime MEMBER_CALL_LAYOUT = 3

comptime MEMBER_OP_CAPABILITIES = 0       # result word: the mask of MEMBER_CAN_* (of call.state's kind, or the family's union)
comptime MEMBER_OP_NEW = 1                # params -> call.state (an unfitted member state)
comptime MEMBER_OP_FREE = 2               # call.state is released; a later use is refused
comptime MEMBER_OP_FIT = 3                # x, y, weights (rows 0: none), classes -> the state is fitted; call.fit_output: also dst
comptime MEMBER_OP_PREDICT = 4            # x -> dst
comptime MEMBER_OP_PREDICT_PROBA = 5      # x -> dst (rows x classes)
comptime MEMBER_OP_DECISION = 6           # x -> dst
comptime MEMBER_OP_SCORE = 7              # x, y, weights -> call.value (the family's default score)
comptime MEMBER_OP_OUTPUTS = 8            # call.flags = MEMBER_OP_* asked -> call.out_cols (columns of dst per row), call.classes
comptime MEMBER_OP_TRANSFORM = 9          # x -> dst
comptime MEMBER_OP_INVERSE_TRANSFORM = 10 # x -> dst
comptime MEMBER_OP_PARTIAL_FIT = 11       # x, y, weights, classes -> the state has seen one more batch
comptime MEMBER_OP_PREDICT_STD = 12       # x -> dst (the predictive standard deviation, one column)
comptime MEMBER_OP_SCORE_SAMPLES = 13     # x -> dst (one value per row)
comptime MEMBER_OP_EXPORT = 14            # call.block -> call.dst = a view the state owns; call.out_cols = its block count
comptime MEMBER_OP_IMPORT = 15            # call.block, call.x = its words -> copied in; MEMBER_IMPORT_DONE: the state is fitted

comptime MEMBER_CAN_FIT = 1
comptime MEMBER_CAN_PREDICT = 2
comptime MEMBER_CAN_PREDICT_PROBA = 4
comptime MEMBER_CAN_DECISION = 8
comptime MEMBER_CAN_SCORE = 16
comptime MEMBER_CAN_TRANSFORM = 32
comptime MEMBER_CAN_INVERSE_TRANSFORM = 64
comptime MEMBER_CAN_PARTIAL_FIT = 128
comptime MEMBER_CAN_PREDICT_STD = 256
comptime MEMBER_CAN_SCORE_SAMPLES = 512
comptime MEMBER_CAN_EXPORT = 1024
comptime MEMBER_CAN_IMPORT = 2048
comptime MEMBER_CAN_SAMPLE_WEIGHT = 4096     # fit reads call.weights
comptime MEMBER_CAN_WEIGHTED_SCORE = 8192    # score reads call.weights
comptime MEMBER_CAN_DEVICE_VIEWS = 16384     # reads and writes MEMBER_HOME_DEVICE views of its route's device
comptime MEMBER_CAN_ROW_INDEX = 32768        # reads call.row_index
comptime MEMBER_CAN_COLUMN_INDEX = 65536     # reads call.column_index
comptime MEMBER_CAN_COEF = 131072            # exports MEMBER_BLOCK_COEF
comptime MEMBER_CAN_IMPORTANCES = 262144     # exports MEMBER_BLOCK_FEATURE_IMPORTANCES
comptime MEMBER_CAN_CLASSIFIER = 524288      # y is class codes; fit reads call.classes
comptime MEMBER_CAN_MULTI_OUTPUT = 1048576   # y may have more than one column
comptime MEMBER_CAN_STEPS = 2097152          # reads x.steps: each row of X is steps x features (a sequence member)
comptime MEMBER_CAN_FORECAST = 4194304       # fits a series in y; predict without x writes dst.rows steps ahead
comptime MEMBER_CAN_TEST_ROWS = 8388608      # its entry refuses non-finite X / y / weights itself: the caller makes no finite scan
comptime MEMBER_CAN_SCOPE = 16777216         # reads call.scope / call.pin: keeps a pinned host X resident in the caller's scope
#: KIND-EXACT and route-exact (answered with `call.state != 0`). Every FIT /
#: PREDICT / TRANSFORM this kind publishes on this route splits its work over
#: `call.devices[0 .. device_count)` exactly as the family's plain owner does
#: under the same bound device plan (core/device_plan.mojo): same split, same
#: order, same bits. With `device_count == 0` it runs on the one device of
#: `route`, as every member does. A kind that ignores `call.devices` never
#: sets it. A caller holding a device tuple of more than one device hands a
#: member only a kind with this bit (else its plain owner call, as before);
#: with one device the bit is not asked at all.
comptime MEMBER_CAN_DEVICE_PLAN = 33554432   # splits over call.devices as its plain owner does under the bound plan
#: A fitted state carries FIT NOTICES: the warnings its family's own fit door
#: raises at fit, kept as words by FIT and exported BY NAME as
#: `MEMBER_BLOCK_NOTICES` (never in a `MEMBER_BLOCK_BY_INDEX` walk, never
#: imported). Every caller takes them once per fitted unit and raises them on
#: the public thread, in the door's words and category, whether or not it packs
#: the unit (core/member_notices.mojo, bindings/member_fit_notices_pack.mojo).
#: A family without the bit is never asked: one mask test per unit.
comptime MEMBER_CAN_NOTICES = 67108864

#: KIND-EXACT (answered with `call.state != 0`; the family's union with 0).
#: A fitted state of this kind keeps a pointer into a buffer its FIT was
#: handed (x, y, weights or an index) after FIT returns: THE LIFETIME RULE in
#: the module note. Not a "can": a property the caller must honor. A kind
#: with the bit answers `MEMBER_OP_EXPORT` + `MEMBER_EXPORT_OWN`.
comptime MEMBER_BORROWS_FIT = 67108864       # the fitted state reads its FIT buffers after FIT returned

#: `call.out_dtype` after MEMBER_OP_OUTPUTS: the member did not say.
comptime MEMBER_DTYPE_UNSAID = -1
#: `call.scope` when the caller runs in no resident scope (the value of
#: core/arena_state.mojo `ARENA_HANDLE_NONE`).
comptime MEMBER_SCOPE_NONE = -1

#: What a caller needs of a member, as masks for `MemberRef.has`,
#: `member_require` and `member_admit_serving`: a predictor (classifier,
#: regressor, clusterer with predict) or a transformer (a Pipeline step, a
#: decomposition or scaler under a search). A caller adds the bits of what
#: else it will ask (score, predict_proba, sample weights, index views).
comptime MEMBER_NEEDS_PREDICTOR = 3      # MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
comptime MEMBER_NEEDS_TRANSFORMER = 33   # MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM

#: Fitted-state blocks (`MEMBER_OP_EXPORT` / `MEMBER_OP_IMPORT`). The shared
#: names mean the same in every family; a family's own blocks start at
#: `MEMBER_BLOCK_FAMILY` and are listed in its member file's docstring.
comptime MEMBER_BLOCK_COEF = 1                 # float32, outputs (or classes, or 1) x features
comptime MEMBER_BLOCK_INTERCEPT = 2            # float32, 1 x outputs (or classes, or 1)
comptime MEMBER_BLOCK_FEATURE_IMPORTANCES = 3  # float32, 1 x features
comptime MEMBER_BLOCK_SCALARS = 4              # float64 words in the family's documented order
comptime MEMBER_BLOCK_NOTICES = 5              # float64 notices x MEMBER_NOTICE_WORDS (core/member_notices.mojo); rows 0: none
comptime MEMBER_BLOCK_FAMILY = 64

#: `call.flags` of MEMBER_OP_EXPORT: `call.block` is an index 0 .. count-1
#: (the block's name comes back in `call.block`), for a PACK that walks them.
comptime MEMBER_BLOCK_BY_INDEX = 1
#: `call.flags` of MEMBER_OP_EXPORT, alone (no block is read or answered):
#: the state copies every FIT buffer it still borrows into words it owns,
#: once, and reads no caller buffer afterwards (THE LIFETIME RULE). Sent only
#: to a kind with `MEMBER_BORROWS_FIT` (`member_own` asks first).
comptime MEMBER_EXPORT_OWN = 4
#: `call.flags` of MEMBER_OP_IMPORT: this was the last block; check the
#: state is complete and mark it fitted (`call.classes` = its class count).
comptime MEMBER_IMPORT_DONE = 2

#: Element types of a `MemberMatrix`.
comptime MEMBER_F32 = 0
comptime MEMBER_F64 = 1
comptime MEMBER_I32 = 2
comptime MEMBER_I64 = 3

#: Where a view's words live. HOST: `data` is an address. DEVICE: `data` is
#: the id of a resident store of the route's device holding `rows` x `cols`
#: elements row-major (the base an arena or a DeviceRows fold already holds,
#: bindings/device_rows_door.mojo); `offset` is its first row. A device-home
#: index view (`row_index`) is the store of the fold's int64 row numbers: the
#: member gathers on the device.
comptime MEMBER_HOME_HOST = 0
comptime MEMBER_HOME_DEVICE = 1


struct MemberMatrix(ImplicitlyCopyable, Movable):
    """A typed matrix view: `rows` x `cols` elements of `dtype`, row `i`
    starting `(offset + i) * stride` elements after `data` (`stride == cols`
    for a dense row-major matrix; a vector is `cols == 1`). `rows == 0` is
    "not given".

    `steps` is the inner shape of a row: 0 (or 1), a plain matrix; `s > 1`,
    each row is `s` steps of `cols / s` features, row-major (a rows x steps
    x features tensor of a sequence member). `cols` is always the full width
    of a row, so the addressing above does not change. Only a family with
    `MEMBER_CAN_STEPS` is handed `steps > 1`."""
    var data: Int
    var rows: Int
    var cols: Int
    var stride: Int
    var offset: Int
    var dtype: Int
    var home: Int
    var steps: Int

    def __init__(out self):
        self.data = 0
        self.rows = 0
        self.cols = 0
        self.stride = 0
        self.offset = 0
        self.steps = 0
        self.dtype = MEMBER_F32
        self.home = MEMBER_HOME_HOST

    def __init__(out self, data: Int, rows: Int, cols: Int, dtype: Int = MEMBER_F32):
        self.data = data
        self.rows = rows
        self.cols = cols
        self.stride = cols
        self.offset = 0
        self.dtype = dtype
        self.home = MEMBER_HOME_HOST
        self.steps = 0

    def given(self) -> Bool:
        return self.rows > 0 and self.data != 0

    def dense_host(self, dtype: Int) -> Bool:
        """A dense row-major host view of `dtype` starting at `data` (what a
        host kernel reads by address without a gather)."""
        return (self.given() and self.home == MEMBER_HOME_HOST and self.dtype == dtype
            and self.stride == self.cols and self.offset == 0)

    def element_bytes(self) -> Int:
        return 4 if (self.dtype == MEMBER_F32 or self.dtype == MEMBER_I32) else 8

    def rows_from(self, first: Int, count: Int) -> Self:
        """Rows `first .. first + count` of a HOST view as a view (a
        contiguous fold: no copy). A device view moves `offset` instead."""
        var part = self
        part.rows = count
        if self.home == MEMBER_HOME_HOST:
            part.data = self.data + (self.offset + first) * self.stride * self.element_bytes()
            part.offset = 0
        else:
            part.offset = self.offset + first
        return part


struct MemberCall(Movable):
    """One member operation. The caller builds it on its own frame (no heap),
    fills the views the operation reads and calls `member_call`."""
    var layout: Int
    var state: Int
    var route: DeviceRoute
    var x: MemberMatrix
    var y: MemberMatrix
    var weights: MemberMatrix
    var dst: MemberMatrix
    # Optional selections of X (and of the rows of y and the weights): int32
    # or int64 numbers, in the order the member must see them, repeats
    # allowed (a bootstrap). Not given: every row / column.
    var row_index: MemberMatrix
    var column_index: MemberMatrix
    # MEMBER_OP_NEW: the member's hyperparameters as Float64 words, in the
    # order the family documents beside its entry (address, count).
    var params: Int
    var param_count: Int
    var flags: Int
    # MEMBER_OP_FIT: 0, or the MEMBER_OP_* (predict, transform, ...) whose
    # output for the training rows the fit also writes to `dst`.
    var fit_output: Int
    # MEMBER_OP_EXPORT / MEMBER_OP_IMPORT: the block's name (or index).
    var block: Int
    # Classifier: the class count (given at fit / import; answered by outputs).
    var classes: Int
    # Results.
    var value: Float64
    var out_cols: Int
    # APPENDED 2026-11-21 (round 5). A member that does not know these words
    # ignores them; their initial values mean "not given" / "not said", so a
    # caller must treat them as hints it can do without.
    # MEMBER_OP_OUTPUTS answer: the element type (MEMBER_F32 ...) of the
    # asked operation's `dst`; MEMBER_DTYPE_UNSAID from a member that does not
    # say (the caller then keeps its documented default for that operation).
    var out_dtype: Int
    # The caller's resident scope (the handle word of
    # bindings/arena_scope_door.mojo `arena_scope_active`, read once per
    # public operation at UNPACK), MEMBER_SCOPE_NONE without one; and the pin
    # word of the object that owns the host words of `x` (what
    # bindings/arena_cache_run.mojo `arena_cache_find_or_put` takes), 0 when
    # the caller pins nothing. Only a member with MEMBER_CAN_SCOPE reads
    # them, and only with `pin != 0` may it keep `x` resident past the call.
    # THE PIN RULE (round 6; MEMBER-ABI-V2.md item 27). A nonzero `pin` is ONE
    # reference the caller took on that object for this call (with the
    # interpreter held, at UNPACK) and offers to the member's cache; `x.data`
    # is then inside that object's words, which cannot move or be freed while
    # the reference lives. With `pin == 0` a member neither looks `x` up in a
    # resident cache nor keeps it (the address of an unpinned buffer can be
    # reused by a later one). A member whose cache PUBLISHED an entry holding
    # the token writes `call.pin_kept` nonzero at that moment, before anything
    # after it can raise: the reference is the cache's, dropped by whoever
    # closes or drops the entry (the scope's own boundary, interpreter held).
    # `pin` itself is never written by a member (the same call may look X up
    # again), and a call publishes at most one entry holding the token. On
    # every other outcome (a hit, a lost publication race, no scope, under
    # the scope's floor, a refusal) `pin_kept` stays 0 and the reference
    # stays the caller's. The caller reads `pin_kept` after every call that
    # offered a pin, raised or not (`member_pin_settle`), and at PACK returns
    # each reference no cache took. A member never drops or takes a reference
    # itself: it holds no interpreter lock. `call.value` says nothing about
    # the pin.
    var scope: Int
    var pin: Int
    # The member's device plan, resolved on the PUBLIC thread by the family's
    # admit (`MemberPlan.devices`): the address and count of the device ids in
    # rank order; count 0 = the one device of `route`. A member never reads
    # the thread's plan or the environment inside a call (a worker thread has
    # no bound plan).
    var devices: Int
    var device_count: Int
    # APPENDED 2026-11-25 (round 6; THE PIN RULE above). Written nonzero by a
    # member whose cache took this call's `pin` reference; 0 from the caller.
    var pin_kept: Int

    def __init__(out self, route: DeviceRoute, state: Int = 0):
        self.layout = MEMBER_CALL_LAYOUT
        self.state = state
        self.route = route
        self.x = MemberMatrix()
        self.y = MemberMatrix()
        self.weights = MemberMatrix()
        self.dst = MemberMatrix()
        self.row_index = MemberMatrix()
        self.column_index = MemberMatrix()
        self.params = 0
        self.param_count = 0
        self.flags = 0
        self.fit_output = 0
        self.block = 0
        self.classes = 0
        self.value = 0.0
        self.out_cols = 0
        self.out_dtype = MEMBER_DTYPE_UNSAID
        self.scope = MEMBER_SCOPE_NONE
        self.pin = 0
        self.devices = 0
        self.device_count = 0
        self.pin_kept = 0

    def on_cpu(self) -> Bool:
        """The member's RESOLVED device is the CPU (its route says so; no
        `device` field is read)."""
        return self.route.device == DEVICE_CPU


@fieldwise_init
struct MemberRef(ImplicitlyCopyable, Movable):
    """A family's member entry of one image, resolved once (two words) plus
    its capability mask. The caller's member owner holds it beside the
    member's `DeviceRoute` and state handle."""
    var entry: EntryRef
    var capabilities: Int

    def can(self, capability: Int) -> Bool:
        """ONE capability bit is set (with a mask of several: any of them)."""
        return (self.capabilities & capability) != 0

    def has(self, needs: Int) -> Bool:
        """EVERY bit of the mask `needs` is set."""
        return (self.capabilities & needs) == needs


struct MemberPlan(Copyable, Movable):
    """Everything typed a caller's RUN needs to make members of one
    estimator: the family and its kind word, the resolved entry, the
    member's route and its hyperparameter words. Built once per member by
    the caller's UNPACK (bindings/member_boundary_door.mojo `member_admit`);
    workers read it and build their own `MemberCall` (`member_new`)."""
    var family: Int
    var kind: Int
    var member: MemberRef
    var route: DeviceRoute
    var params: List[Float64]
    var classes: Int
    # APPENDED 2026-11-21 (round 5). The member's device ids in rank order,
    # written by the family's admit from the public thread's device plan
    # (core/device_plan.mojo `device_plan_current(<the family's setting>)`),
    # empty = the one device of `route`. `member_call_of` hands them to every
    # call; the plan must outlive those calls (it already does: the owner).
    var devices: List[Int]
    # The caller's resident scope and X pin for this public operation (see
    # `MemberCall.scope`); the caller's UNPACK writes them after admit.
    var scope: Int
    var pin: Int

    def __init__(out self):
        self.family = -1
        self.kind = 0
        self.member = MemberRef(entry_ref_unresolved(), 0)
        self.route = DeviceRoute(0)
        self.params = List[Float64]()
        self.classes = 0
        self.devices = List[Int]()
        self.scope = MEMBER_SCOPE_NONE
        self.pin = 0


def member_publish[family: Int](mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish a family's member entry for the image (`mode` 1 identical /
    0 fast, `vendor` 0 CPU, 1 Metal, 2 CUDA, 3 HIP). Called once from the
    image's module initialization, after its kernel tables are published;
    `publisher` is any stable nonzero address of the image."""
    comptime assert family >= 0 and family < MEMBER_FAMILY_COUNT, "unknown member family"
    _ = member_table[family, 1](mode, vendor, publisher, entry)


def member_call(member: MemberRef, operation: Int, mut call: MemberCall) raises -> Int:
    """One member operation: one indirect call, the `MemberCall` passed by
    address from the caller's frame. No heap allocation, no table read. A
    refusal of the family is raised here with the family's words."""
    var address: Array[Int, 1] = [Int(MutPointer(to=call))]
    var result = entry_ref_call(member.entry, operation, Int(address.unsafe_ptr()), 1, 0, 0, 0, 0)
    # Integer addresses carry no lifetime: keep the envelope past the call.
    _ = address
    return result


def member_resolve[family: Int](route: DeviceRoute) raises -> MemberRef:
    """The member entry of the image `route` resolved to (its `mode` and
    `vendor` words), read ONCE per member by the caller's UNPACK, with its
    capability mask. Every refusal of the table (the family's image is not
    loaded, the ABI differs) happens here, never at call time."""
    var entry = member_table[family, 0](route.mode, route.vendor, 0, entry_ref_unresolved().call)
    var probe = MemberCall(route)
    var member = MemberRef(entry, 0)
    member.capabilities = member_call(member, MEMBER_OP_CAPABILITIES, probe)
    return member


def member_resolve_family(family: Int, route: DeviceRoute) raises -> MemberRef:
    """`member_resolve` for a family number read at run time from the
    member's class (bindings/member_boundary_door.mojo)."""
    var entry = member_table_resolve(family, route.mode, route.vendor, entry_ref_unresolved().call)
    var probe = MemberCall(route)
    var member = MemberRef(entry, 0)
    member.capabilities = member_call(member, MEMBER_OP_CAPABILITIES, probe)
    return member


def member_call_of(plan: MemberPlan, state: Int = 0) raises -> MemberCall:
    """A call of the planned member over `state` (0: none yet) carrying the
    plan's words: its hyperparameters, class count, device ids, and the
    caller's scope and pin. Every caller builds its calls here (or with
    `member_new`), never with a bare `MemberCall(route, state)`, so a word
    appended to the plan reaches the member without an edit at the caller."""
    var call = MemberCall(plan.route, state)
    call.params = Int(plan.params.unsafe_ptr())
    call.param_count = len(plan.params)
    call.classes = plan.classes
    call.devices = Int(plan.devices.unsafe_ptr()) if len(plan.devices) > 0 else 0
    call.device_count = len(plan.devices)
    call.scope = plan.scope
    call.pin = plan.pin
    return call^


def member_new(plan: MemberPlan) raises -> MemberCall:
    """A worker's own call holding a new unfitted state of the planned
    member. The worker frees it with `member_free` on every path."""
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    return call^


def member_pin_settle(mut call: MemberCall) -> Int:
    """After a call that offered a pin token in `call.pin` (THE PIN RULE at
    `MemberCall.pin`): 1 when the member's cache took that call's reference
    (`call.pin_kept` nonzero), else 0. Read it after the call whether it
    returned or raised. The call then offers nothing; a caller that makes
    another call over the same pinned X writes the token again, backed by
    another reference it took at UNPACK (one reference per offering call: a
    member may take each one)."""
    var taken = 1 if call.pin != 0 and call.pin_kept != 0 else 0
    call.pin = 0
    call.pin_kept = 0
    return taken


def member_kind_capabilities(plan: MemberPlan) raises -> Int:
    """The capability mask of THIS member's kind (a regressor of a family
    that also has classifiers is no classifier), where `plan.member.capabilities`
    is the family's union. One state is minted, asked and released: call it
    once per member class at UNPACK, never per unit."""
    var call = member_new(plan)
    var mask = 0
    try:
        mask = member_call(plan.member, MEMBER_OP_CAPABILITIES, call)
    finally:
        member_free(plan, call)
    return mask


def member_own(plan: MemberPlan, state: Int) raises:
    """THE LIFETIME RULE: make the fitted `state` own every FIT buffer it
    still borrows, so it may outlive them. One capability question of the
    state's own kind; a kind without `MEMBER_BORROWS_FIT` is sent nothing
    and copies nothing, and a state that already owns its rows copies
    nothing. Call it ONCE per state that is kept past its fit buffers (at
    adoption), never per call. Raises the member's words when it cannot."""
    var call = MemberCall(plan.route, state)
    if (member_call(plan.member, MEMBER_OP_CAPABILITIES, call) & MEMBER_BORROWS_FIT) == 0:
        return
    call.flags = MEMBER_EXPORT_OWN
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)


def member_outputs(plan: MemberPlan, operation: Int, features: Int, targets: Int) raises -> Int:
    """The columns per row of `operation`'s output (`MEMBER_OP_PREDICT`,
    `_PREDICT_PROBA`, `_DECISION`, `_TRANSFORM`, ...) for a member of this
    plan fitted on `features` columns and `targets` target columns, asked
    BEFORE any state exists: a caller sizes its placed output buffers once,
    at UNPACK, and its workers write into them. No state is minted. A family
    that cannot know the width before a fit refuses in words; ask a fitted
    state with `member_outputs_of` then."""
    var call = member_call_of(plan)
    call.flags = operation
    call.x.cols = features
    call.y.cols = targets
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    return call.out_cols


def member_output_dtype(plan: MemberPlan, operation: Int, features: Int, targets: Int, unsaid: Int) raises -> Int:
    """The element type (`MEMBER_F32`, `_F64`, `_I32`, `_I64`) of
    `operation`'s `dst` for a member of this plan, asked before any state
    exists, as `member_outputs` asks the width; `unsaid` when the member does
    not answer it (a member published before the word existed). Once per
    member class at UNPACK: a caller places its buffer in that type and never
    learns it from a refused call."""
    var call = member_call_of(plan)
    call.flags = operation
    call.x.cols = features
    call.y.cols = targets
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    if call.out_dtype == MEMBER_DTYPE_UNSAID:
        return unsaid
    return call.out_dtype


def member_outputs_of(plan: MemberPlan, mut call: MemberCall, operation: Int) raises -> Int:
    """The columns per row of `operation`'s output for the call's live
    state (exact after its fit). `call.classes` comes back as the state's
    class count; `call.flags` is left as the caller had it."""
    var held = call.flags
    call.flags = operation
    _ = member_call(plan.member, MEMBER_OP_OUTPUTS, call)
    call.flags = held
    return call.out_cols


def member_free(plan: MemberPlan, mut call: MemberCall):
    """Release the call's state. Never raises: a worker calls it on its
    error path too, and a state that was never minted is nothing to free."""
    if call.state == 0:
        return
    try:
        _ = member_call(plan.member, MEMBER_OP_FREE, call)
    except:
        call.state = 0


def member_require(plan: MemberPlan, capability: Int, what: String) raises:
    """UNPACK: refuse in words, before any work, a member that lacks an
    operation the caller will ask for (every bit of `capability`). ONLY for a
    request that had no path before the member mark existed. A caller that
    served this class before the mark never calls this: it falls back to
    that path silently (`member_admit_serving`; MEMBER-ABI-V2.md, the
    fallback rule)."""
    if not plan.member.has(capability):
        raise Error("AttributeError: a " + member_family_name(plan.family) + " member has no " + what)


def member_call_from_spans(operation: Int, aa: Int, na: Int, np: Int, nf: Int) raises -> Int:
    """PROVIDER side: admit a member entry's spans and return the address of
    the caller's `MemberCall` (read it through
    `MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=...)`)."""
    if operation < 0 or operation >= MEMBER_OPERATION_COUNT or na != 1 or np != 0 or nf != 0 or not aa:
        raise Error("member ABI: invalid request spans")
    var address = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[0]
    if not address:
        raise Error("member ABI: null call address")
    if MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)[].layout != MEMBER_CALL_LAYOUT:
        raise Error("member ABI: the caller's call layout differs (mixed binary versions)")
    return address


def member_unsupported(family: String, operation: Int) raises:
    """The words of an operation a family's member does not answer."""
    var name = String("fit") if operation == MEMBER_OP_FIT else (
        String("predict") if operation == MEMBER_OP_PREDICT else (
        String("predict_proba") if operation == MEMBER_OP_PREDICT_PROBA else (
        String("decision_function") if operation == MEMBER_OP_DECISION else (
        String("score") if operation == MEMBER_OP_SCORE else (
        String("transform") if operation == MEMBER_OP_TRANSFORM else (
        String("inverse_transform") if operation == MEMBER_OP_INVERSE_TRANSFORM else (
        String("partial_fit") if operation == MEMBER_OP_PARTIAL_FIT else (
        String("predict_std") if operation == MEMBER_OP_PREDICT_STD else (
        String("score_samples") if operation == MEMBER_OP_SCORE_SAMPLES else (
        String("fitted-state export") if operation == MEMBER_OP_EXPORT else (
        String("fitted-state import") if operation == MEMBER_OP_IMPORT else String("operation ") + String(operation))))))))))))
    raise Error("AttributeError: a " + family + " member has no " + name)
