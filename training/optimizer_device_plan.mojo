# The optimizer's device plan, read ONCE at a native entry and passed down as
# a value (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md, target 5). This is
# the only runtime reader of MOJOLEARN_OPTIMIZER_DEVICE_COUNT: the samba
# bindings, the public optimizer entries and the native dispatch tables all
# build their plan here. No floating-point value is read or written.
from std.os import getenv

from core.device_plan import DevicePlan


def optimizer_device_plan() raises -> DevicePlan:
    """The optimizer device plan of one native entry. `setting` is the raw
    MOJOLEARN_OPTIMIZER_DEVICE_COUNT (default "1"); `count` is that setting
    parsed by `Int` exactly as the optimizer's owner loops parsed it. A
    setting `Int` refuses gets count 0 here instead of raising at the entry,
    so a call that never reads the count is unchanged and one that does
    still refuses (the owner loops refuse a count outside 1..64)."""
    var setting = String(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1"))
    var count = 0
    try:
        count = Int(setting)
    except:
        count = 0
    var devices = List[Int]()
    for rank in range(min(max(count, 0), 64)):  # driver-loop: over the plan's ranks
        devices.append(rank)
    return DevicePlan(setting, count, 0, devices^, False)
