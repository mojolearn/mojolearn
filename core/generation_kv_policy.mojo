# SPDX-License-Identifier: Apache-2.0
# Transport policy is operation-local and independent of selection arithmetic.
from std.sys.compile import is_defined


# Lane envgate-neural-lm (2026-11-15): the generation experiment arms are
# build defines with today's defaults. Each was a getenv per generate call
# (policy construction). The defaults are unmeasured candidates (README
# exp/modular_generation_control_20261030); their B arms are RUN OWED
# (bench/results/paper_compare/owed-pre0867.json, docs/apple-fast/EXPERIMENTS.md).
#   -D MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE   reuse KV uploads (was =1)
#   -D MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_RESIDENT  route resident
#   -D MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_NATIVE_ENGINE  route native_engine
#   -D MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE         engine "1" (was ENGINE=1)
comptime EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE = is_defined["MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE"]()
comptime _ROUTE_RESIDENT = is_defined["MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_RESIDENT"]()
comptime _ROUTE_NATIVE_ENGINE = is_defined["MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_NATIVE_ENGINE"]()
# TOMBSTONE: MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_LEGACY (LOSER: NV L40S 2 cells, median 2.29, mojolearn-162m generate 887 -> 2978 ms (3.36), smollm2 1456 -> 1782 ms (1.22)) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_LEGACY.ab-outcome-2.patch
comptime EXPERIMENT_GENERATION_ROUTE: StaticString = (
    "resident" if _ROUTE_RESIDENT else (
        "native_engine" if _ROUTE_NATIVE_ENGINE else "auto"
    )
)
comptime GENERATION_NATIVE_ENGINE_FORCE = is_defined["MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE"]()
# TOMBSTONE: MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF (old path no faster: NV L40S 2 cells, median 1.00) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at 070015d016.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF.ab-outcome-2.patch
comptime GENERATION_NATIVE_ENGINE: StaticString = "1" if GENERATION_NATIVE_ENGINE_FORCE else "auto"


struct GenerationKVUploadPolicy(Copyable, Movable):
    var requested: Bool
    var capacity: Int

    def __init__(out self, capacity: Int):
        self.requested = EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE
        self.capacity = capacity

    def may_retain(self) -> Bool:
        # Default off. This removes real reloads only from the existing
        # per-forward path; resident reads never manufacture an off-arm reload.
        return self.requested and self.capacity > 1

    def status(self, executed: String, resident_forward: Bool, eligible: Bool,
               skipped: Int, invalidated: Bool) -> String:
        if executed == "resident_sampling" or executed == "resident_greedy":
            return "already_resident"
        if executed == "native_engine":
            return "inapplicable_engine"
        if not self.requested:
            return "off"
        if self.capacity <= 1:
            return "single_step"
        if skipped > 0:
            return "reused_then_invalidated" if invalidated else "reused"
        if invalidated:
            return "owner_invalidated"
        if not resident_forward:
            return "no_resident_forward"
        if not eligible:
            return "ownership_ineligible"
        return "eligible_no_second_forward"


struct GenerationReadControl(Copyable, Movable):
    var route: String
    var engine_setting: String
    var sampling_policy: Int
    var greedy: Bool
    var kv: GenerationKVUploadPolicy

    def __init__(out self, capacity: Int, sampling_policy: Int, greedy: Bool) raises:
        self.kv = GenerationKVUploadPolicy(capacity)
        self.sampling_policy = sampling_policy
        self.greedy = greedy
        comptime assert Int(_ROUTE_RESIDENT) + Int(_ROUTE_NATIVE_ENGINE) <= 1, (
            "at most one MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_* define"
        )
        self.engine_setting = String(GENERATION_NATIVE_ENGINE)
        self.route = String(EXPERIMENT_GENERATION_ROUTE)
