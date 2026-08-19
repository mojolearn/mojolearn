"""Which host, which device.

PORT OF `catboost/cuda/cuda_lib/device_id.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

Their `TDeviceId` is a `(HostId, DeviceId)` pair because a CatBoost run may
span hosts over MPI. We are single host and single device, so `host_id` is
always -1, their value for "local". The field is kept rather than dropped so
that a `TDeviceId` printed out of our tree is comparable with one printed out
of theirs.
"""


struct TDeviceId(Copyable, ImplicitlyCopyable, Movable):
    """Their `TDeviceId` (`device_id.h`)."""

    var host_id: Int32
    var device_id: Int32

    def __init__(out self):
        self.host_id = -1
        self.device_id = 0

    def __init__(out self, host_id: Int32, device_id: Int32):
        self.host_id = host_id
        self.device_id = device_id

    def __eq__(self, other: Self) -> Bool:
        return (
            self.host_id == other.host_id
            and self.device_id == other.device_id
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def is_local(self) -> Bool:
        return self.host_id == -1

    def to_string(self) -> String:
        return (
            String("Device(host=")
            + String(self.host_id)
            + ", dev="
            + String(self.device_id)
            + ")"
        )
