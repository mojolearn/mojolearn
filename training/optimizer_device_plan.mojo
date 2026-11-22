# The optimizer's device plan, read ONCE per process and passed down as a
# value (plan MULTI_GPU_ENCAPSULATION_20261103 (git history), target 5). This is
# the only runtime reader of MOJOLEARN_OPTIMIZER_DEVICE_COUNT: the samba
# bindings, the public optimizer entries, the parallel MLP admission and the
# native dispatch tables all take their plan or setting here. USER-SETTING
# (plan ENV_GATES_2026-11-15 (git history)): until 2026-11-15 it was an
# environment read at every native optimizer entry (per step / clip /
# accumulate); now the first entry of the process reads it into a typed
# process slot and every later entry reads the slot (lane
# envgate-train-seq-mamba). Set it before the process starts; a change after
# the first optimizer call is not seen. No floating-point value is read or
# written.
from std.os import getenv

from core.device_plan import DevicePlan
from core.shared_slot import SharedSlot


struct _OptimizerDeviceSetting(Defaultable, Movable):
    """MOJOLEARN_OPTIMIZER_DEVICE_COUNT, read once per process: the raw
    setting (default "1") and that setting parsed by `Int` (0 when `Int`
    refuses it)."""
    var setting: String
    var count: Int

    def __init__(out self):
        self.setting = String(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1"))
        self.count = 0
        try:
            self.count = Int(self.setting)
        except:
            self.count = 0


# Process-wide (one environment per process); bump the name with the layout.
comptime _OPTIMIZER_DEVICE_SETTING = SharedSlot[_OptimizerDeviceSetting, "MojolearnOptimizerDeviceSettingV1"]


def optimizer_device_setting() raises -> String:
    """The raw MOJOLEARN_OPTIMIZER_DEVICE_COUNT of this process (default
    "1"), read once."""
    return _OPTIMIZER_DEVICE_SETTING.get_or_create_ptr()[].setting.copy()


def optimizer_device_plan() raises -> DevicePlan:
    """The optimizer device plan of one native entry. `setting` is the raw
    MOJOLEARN_OPTIMIZER_DEVICE_COUNT (default "1"); `count` is that setting
    parsed by `Int` exactly as the optimizer's owner loops parsed it. A
    setting `Int` refuses gets count 0 here instead of raising at the entry,
    so a call that never reads the count is unchanged and one that does
    still refuses (the owner loops refuse a count outside 1..64)."""
    var slot = _OPTIMIZER_DEVICE_SETTING.get_or_create_ptr()
    var count = slot[].count
    var devices = List[Int]()
    for rank in range(min(max(count, 0), 64)):  # driver-loop: over the plan's ranks
        devices.append(rank)
    return DevicePlan(slot[].setting, count, 0, devices^, False)
