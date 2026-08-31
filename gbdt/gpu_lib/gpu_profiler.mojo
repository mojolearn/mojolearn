# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Named, nested, scoped timers over the control plane.

PORT OF `catboost/cuda/cuda_lib/cuda_profiler.h` at CatBoost `54a8143a`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

This is not instrumentation bolted on afterwards; it is how CatBoost knows
where a tree's time goes, and their driver is full of it:

    auto& profiler = NCudaLib::GetProfiler();
    auto guard = profiler.Profile(TStringBuilder() << "Leaves split #"
                                                   << newLeaves << " leaves");
                                        split_properties_helper.cpp:842-843
    auto guard = profiler.Profile("Compute histograms for #...")
                                        split_properties_helper.cpp:1335
    auto iterationTimeGuard = profiler.Profile("Boosting iteration");
                                        doc_parallel_boosting.h:339
    auto guard = profiler.Profile("Search for weak model structure");
                                        doc_parallel_boosting.h:347

The guard is the whole design. A label is acquired when the guard is made and
released when it dies, so a label's cost is the cost of a SCOPE, and nesting
is tracked (`Nestedness`, `cuda_profiler.h:139`) so the printout is indented
by call depth.

Three modes, and the difference between them is where the device is
(`cuda_profiler.h:9-15`):

    ImplicitLabelSync   drain the device on acquire AND on release, so the
                        interval is host time PLUS the device time it caused
    LabelAsync          time only the host's send and receive. THEIR DEFAULT
                        (`cuda_manager.cpp:13`)
    NoProfile           record nothing

================================ DEVIATION BLOCK ======================
**1. The guard is a `with` block, not a destructor.** Theirs is
`TGuard<TLabeledInterval>`: the constructor acquires, the destructor
releases. Mojo destroys a value as soon as its last use is passed (ASAP
destruction), so `var guard = profiler.profile("x")` would be destroyed
before the code it means to time. This is the same reason MAX's own
`profiling_range.Range` opens its span in `__enter__` rather than `__init__`
(max.modular.com/api/mojo/profiling_range/range). So:

    with profiler.profile("Compute histograms"):
        ...

acquires on entry and releases on exit, and a guard NOT used in a `with`
records nothing at all rather than recording something wrong.

**2. The state is shared through an `ArcPointer`, not a raw pointer.** Their
`TCudaManager` holds `TCudaProfiler*` and a child manager holds its parent's
(`cuda_manager.h:129-130`), and every `TLabeledInterval` holds `ui32*
Nestedness` back into the profiler (`cuda_profiler.h:26`). Mojo has the same
sharing without the raw pointers: one `TProfilerState` behind an `ArcPointer`,
copied by handle into every guard. `Nestedness` therefore lives on the state,
where the pointer pointed.

**3. `ImplicitLabelSync` drains the `DeviceContext` directly.** Theirs calls
`GetCudaManager().WaitComplete()` (`cuda_profiler.h:97`, `:114`). There is no
thread-local manager singleton here and the manager owns the profiler, so
reaching back would be a cycle. The profiler holds the same `DeviceContext`
the worker holds and drains it.

CONSEQUENCE, stated because it is the exact class of bug this port exists to
kill: a drain from `ImplicitLabelSync` does NOT pass through
`TGpuOneDeviceWorker._device_sync`, so it is invisible to `sync_count` and to
`sync_budget`. `LabelAsync` is the default and drains nothing. Selecting
`ImplicitLabelSync` without a context RAISES rather than silently timing the
wrong thing.

**4. No destructor printout.** Theirs prints in `~TCudaProfiler` when
`PrintOnDelete` (`cuda_profiler.h:152-156`), and a Mojo destructor on a
handle-copyable struct would print once per copy. `print_on_delete` is kept
as state and `TCudaManager.reset_profiler(print_info)` is what reads it,
which is how their manager does it anyway (`cuda_manager.cpp:22-30`).
======================================================================
"""

from std.math import sqrt
from std.time import perf_counter_ns
from std.memory import ArcPointer

from max.gpu.host import DeviceContext


comptime EMPTY_LABEL_NAME = "fake"
"""Their `EmptyLabel("fake", &Nestedness, EProfileMode::NoProfile)`
(`cuda_profiler.h:147`), the label a below-threshold `Profile` call is guarded
on."""


struct EProfileMode(Copyable, ImplicitlyCopyable, Movable):
    """Their `EProfileMode` (`cuda_profiler.h:9-15`), same three values, same
    order."""

    var value: Int32

    comptime ImplicitLabelSync = Self(0)
    comptime LabelAsync = Self(1)
    comptime NoProfile = Self(2)

    def __init__(out self, value: Int32):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def to_string(self) -> String:
        if self == Self.ImplicitLabelSync:
            return "ImplicitLabelSync"
        if self == Self.LabelAsync:
            return "LabelAsync"
        return "NoProfile"


struct TLabeledInterval(Copyable, Movable):
    """Their `TLabeledInterval` (`cuda_profiler.h:17-130`).

    One label's running statistics: count, max, sum, sum of squares. The rmse
    is reconstructed from the sums at print time and never stored.
    """

    var label: String
    var time_ns: Int
    var count: Int
    var max: Float64
    var sum: Float64
    var sum2: Float64
    var active: Bool
    var profile_mode: EProfileMode
    var tab_size: Int
    var timestamp_ns: Int

    comptime NO_TIMESTAMP = -1
    """Stands in for their `TMaybe<time_point> Timestamp`
    (`cuda_profiler.h:28`), which is set on the FIRST acquire and never
    again; it is what `operator<` sorts the printout by
    (`cuda_profiler.h:127-129`)."""

    def __init__(
        out self,
        var label: String,
        nestedness: Int,
        profile_mode: EProfileMode = EProfileMode.LabelAsync,
    ):
        """Their constructor (`cuda_profiler.h:40-53`).

        `TabSize` is the nesting depth AT THE MOMENT THE LABEL IS FIRST SEEN,
        not at each acquire.
        """
        self.label = label^
        self.time_ns = 0
        self.count = 0
        self.max = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.active = False
        self.profile_mode = profile_mode
        self.tab_size = nestedness
        self.timestamp_ns = Self.NO_TIMESTAMP

    def update_tab_size(mut self, tab_size: Int):
        """Their `UpdateTabSize` (`cuda_profiler.h:30-37`).

        A label seen at two different call depths is shown at the SHALLOWEST
        one, with a warning.
        """
        if self.tab_size != tab_size:
            print(
                "Warning: found "
                + self.label
                + " at different level in call stack -- will show this label"
                + " at highest level"
            )
            if tab_size < self.tab_size:
                self.tab_size = tab_size

    def add(mut self, other: Self) raises:
        """Their `Add` (`cuda_profiler.h:65-73`), which is how a child
        manager's numbers fold into its parent's
        (`cuda_manager.cpp:51`)."""
        if other.label != self.label:
            raise Error(
                String("Can't add interval ")
                + other.label
                + " to " + self.label
            )
        if other.active:
            raise Error(
                "Can't add running label interval. Inconsistent cuda-manager's"
                " state"
            )
        if other.max > self.max:
            self.max = other.max
        self.sum += other.sum
        self.sum2 += other.sum2
        self.count += other.count
        self.update_tab_size(other.tab_size)

    def print_info(self):
        """Their `PrintInfo` (`cuda_profiler.h:75-88`). A label that never
        completed an interval prints nothing."""
        if self.count == 0:
            return
        var mean = self.sum / Float64(self.count)
        var pad = String("")
        for _ in range(self.tab_size * 2):
            pad += " "
        print(
            pad + self.label,
            "count",
            self.count,
            "mean:",
            mean,
            "max:",
            self.max,
            "rmse:",
            sqrt((self.sum2 - self.sum * mean) / Float64(self.count)),
        )


struct TProfilerState(Movable):
    """The half of their `TCudaProfiler` that every guard has to reach: the
    labels, the shared nesting counter, and the modes
    (`cuda_profiler.h:132-140`).

    `acquire` and `release` are their `TLabeledInterval::Acquire` and
    `Release` (`cuda_profiler.h:90-125`), moved here because the counter they
    increment lives here; see DEVIATION 2.
    """

    var labels: Dict[String, TLabeledInterval]
    var empty_label: TLabeledInterval
    var default_profile_mode: EProfileMode
    var min_profile_level: Int
    var print_on_delete: Bool
    var nestedness: Int
    var ctx: Optional[DeviceContext]

    def __init__(
        out self,
        profile_mode: EProfileMode,
        level: Int,
        print_on_delete: Bool,
        var ctx: Optional[DeviceContext],
    ):
        self.labels = Dict[String, TLabeledInterval]()
        self.nestedness = 0
        self.empty_label = TLabeledInterval(
            String(EMPTY_LABEL_NAME), 0, EProfileMode.NoProfile
        )
        self.default_profile_mode = profile_mode
        self.min_profile_level = level
        self.print_on_delete = print_on_delete
        self.ctx = ctx^

    def _wait_complete(mut self) raises:
        """Their `GetCudaManager().WaitComplete()` (`cuda_profiler.h:97`,
        `cuda_profiler.h:114`). See DEVIATION 3."""
        if not self.ctx:
            raise Error(
                "EProfileMode.ImplicitLabelSync needs a DeviceContext to"
                " drain: construct the profiler with one, or stay on"
                " LabelAsync, which is their default (cuda_manager.cpp:13)."
            )
        self.ctx.value().synchronize()

    def ensure_label(mut self, var label: String):
        """Their lazy insert (`cuda_profiler.h:200-204`). A label is created
        at the nesting depth where it is first seen."""
        if label not in self.labels:
            var interval = TLabeledInterval(
                label.copy(), self.nestedness, self.default_profile_mode
            )
            self.labels[label^] = interval^

    def acquire(mut self, label: String, use_empty: Bool) raises:
        """Their `Acquire` (`cuda_profiler.h:90-104`)."""
        var interval = (
            self.empty_label.copy() if use_empty else self.labels[label].copy()
        )

        if interval.active:
            raise Error(
                String("Error: label is already aquired ") + interval.label
            )
        interval.active = True

        if interval.profile_mode == EProfileMode.NoProfile:
            self._store(interval^, use_empty)
            return

        if interval.profile_mode == EProfileMode.ImplicitLabelSync:
            self._wait_complete()

        interval.time_ns = Int(perf_counter_ns())
        if interval.timestamp_ns == TLabeledInterval.NO_TIMESTAMP:
            interval.timestamp_ns = interval.time_ns
        self.nestedness += 1
        self._store(interval^, use_empty)

    def release(mut self, label: String, use_empty: Bool) raises:
        """Their `Release` (`cuda_profiler.h:106-125`)."""
        var interval = (
            self.empty_label.copy() if use_empty else self.labels[label].copy()
        )

        if not interval.active:
            raise Error(
                String("Can't release non-active label ") + interval.label
            )
        interval.active = False

        if interval.profile_mode == EProfileMode.NoProfile:
            self._store(interval^, use_empty)
            return

        if interval.profile_mode == EProfileMode.ImplicitLabelSync:
            self._wait_complete()

        # Their elapsed is nanoseconds cast to milliseconds
        # (`cuda_profiler.h:118`).
        var value = Float64(Int(perf_counter_ns()) - interval.time_ns) / 1.0e6
        if value > interval.max:
            interval.max = value
        interval.sum += value
        interval.sum2 += value * value
        interval.count += 1
        self.nestedness -= 1
        self._store(interval^, use_empty)

    def _store(mut self, var interval: TLabeledInterval, use_empty: Bool):
        """Mojo `Dict` hands back a value, not a reference into the table, so
        a modified interval is written back explicitly."""
        if use_empty:
            self.empty_label = interval^
        else:
            var key = interval.label.copy()
            self.labels[key^] = interval^

    def print_info(self) raises:
        """Their `PrintInfo` (`cuda_profiler.h:158-168`), which sorts by first
        acquire so the printout reads in the order the run happened."""
        var keys = List[String]()
        var stamps = List[Int]()
        for entry in self.labels.items():
            # insertion sort on `operator<`, which compares Timestamp
            # (`cuda_profiler.h:127-129`)
            var pos = len(keys)
            for i in range(len(keys)):
                if entry.value.timestamp_ns < stamps[i]:
                    pos = i
                    break
            keys.insert(pos, entry.key.copy())
            stamps.insert(pos, entry.value.timestamp_ns)
        for i in range(len(keys)):
            self.labels[keys[i]].print_info()


struct TProfileGuard(Movable):
    """Their `TGuard<TLabeledInterval>` (`cuda_profiler.h:194-206`), spelled
    as a Mojo context manager. See DEVIATION 1.

        with profiler.profile("Compute histograms"):
            ...
    """

    var state: ArcPointer[TProfilerState]
    var label: String
    var use_empty: Bool

    def __init__(
        out self,
        state: ArcPointer[TProfilerState],
        var label: String,
        use_empty: Bool,
    ):
        self.state = state.copy()
        self.label = label^
        self.use_empty = use_empty

    def __enter__(mut self) raises:
        """Their guard's constructor, which calls `Acquire()`."""
        self.state[].acquire(self.label, self.use_empty)

    def __exit__(self) raises:
        """Their guard's destructor, which calls `Release()`."""
        self.state[].release(self.label, self.use_empty)


struct TCudaProfiler(Copyable, Movable):
    """Their `TCudaProfiler` (`cuda_profiler.h:132-207`).

    Copying one copies the HANDLE, not the statistics, which is what their
    `TCudaProfiler*` does (`cuda_manager.h:129`).
    """

    var state: ArcPointer[TProfilerState]

    def __init__(
        out self,
        profile_mode: EProfileMode = EProfileMode.LabelAsync,
        level: Int = 0,
        print_on_delete: Bool = True,
        var ctx: Optional[DeviceContext] = None,
    ):
        """Their constructor (`cuda_profiler.h:142-150`). Their manager builds
        it as `TCudaProfiler(EProfileMode::LabelAsync, 0, false)`
        (`cuda_manager.cpp:13`)."""
        self.state = ArcPointer(
            TProfilerState(profile_mode, level, print_on_delete, ctx^)
        )

    def profile(self, label: String, profile_level: Int = 0) -> TProfileGuard:
        """Their `Profile` (`cuda_profiler.h:194-206`).

        Below `MinProfileLevel` the guard is over their shared `EmptyLabel`,
        whose mode is `NoProfile`, so the call costs a dictionary miss and
        nothing else.
        """
        if profile_level < self.state[].min_profile_level:
            return TProfileGuard(self.state, String(EMPTY_LABEL_NAME), True)
        self.state[].ensure_label(label.copy())
        return TProfileGuard(self.state, label.copy(), False)

    def print_info(self) raises:
        """Their `PrintInfo` (`cuda_profiler.h:158-168`)."""
        self.state[].print_info()

    def set_profile_level(mut self, level: Int):
        """Their `SetProfileLevel` (`cuda_profiler.h:170-172`)."""
        self.state[].min_profile_level = level

    def set_default_profile_mode(mut self, mode: EProfileMode):
        """Their `SetDefaultProfileMode` (`cuda_profiler.h:186-188`)."""
        self.state[].default_profile_mode = mode

    def get_default_profile_mode(self) -> EProfileMode:
        """Their `GetDefaultProfileMode` (`cuda_profiler.h:190-192`)."""
        return self.state[].default_profile_mode

    def add(mut self, other: Self) raises:
        """Their `Add` (`cuda_profiler.h:174-184`), which folds a child
        manager's profiler into its parent's on `StopChild`
        (`cuda_manager.cpp:51`)."""
        for entry in other.state[].labels.items():
            self.state[].ensure_label(entry.key.copy())
            var key = entry.key.copy()
            var mine = self.state[].labels[key].copy()
            mine.add(entry.value)
            self.state[].labels[key^] = mine^

    def print_on_delete(self) -> Bool:
        """Their `PrintOnDelete` flag (`cuda_profiler.h:138`), read by the
        manager instead of by a destructor. See DEVIATION 4."""
        return self.state[].print_on_delete
