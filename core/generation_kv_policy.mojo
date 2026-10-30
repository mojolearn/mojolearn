# SPDX-License-Identifier: Apache-2.0
# Transport policy is operation-local and independent of selection arithmetic.
from std.os import getenv


struct GenerationKVUploadPolicy(Copyable, Movable):
    var requested: Bool
    var capacity: Int

    def __init__(out self, capacity: Int):
        self.requested = getenv("MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE", "0") == "1"
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
        self.engine_setting = String(getenv("MOJOLEARN_GENERATION_NATIVE_ENGINE", "auto"))
        self.route = String(getenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "auto"))
        if (self.route != "auto" and self.route != "legacy"
                and self.route != "resident" and self.route != "native_engine"):
            raise Error("generation: experimental route must be auto, legacy, resident, or native_engine")
