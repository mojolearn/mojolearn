# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Which host, which device.

PORT OF `catboost/cuda/cuda_lib/device_id.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

Their `TDeviceId` is a `(HostId, DeviceId)` pair because a CatBoost run may
span hosts over MPI. Both fields default to -1 (`device_id.h:10-11`), which is
their "unset", NOT their "local": local is `HostId == 0`
(`single_device.h:285`, `IsLocalDevice() { return DeviceId.HostId == 0; }`),
and their two-argument constructor refuses any other host id unless the build
has MPI (`device_id.h:20-22`).

The pair is kept rather than collapsed to one int so that a `TDeviceId`
printed out of our tree is comparable with one printed out of theirs.
"""


struct TDeviceId(Copyable, ImplicitlyCopyable, Movable):
    """Their `TDeviceId` (`device_id.h:9`)."""

    var host_id: Int32
    var device_id: Int32

    def __init__(out self):
        """Their defaulted constructor (`device_id.h:13`), which leaves the
        in-class initialisers `HostId = -1`, `DeviceId = -1`
        (`device_id.h:10-11`)."""
        self.host_id = -1
        self.device_id = -1

    def __init__(out self, host_id: Int32, device_id: Int32) raises:
        """Their two-argument constructor (`device_id.h:15-23`).

        `CB_ENSURE(hostId == 0, "Remote device support is not enabled")` when
        the build has no MPI. This port has no MPI, so the check is
        unconditional and it raises.
        """
        if host_id != 0:
            raise Error("Remote device support is not enabled")
        self.host_id = host_id
        self.device_id = device_id

    def __eq__(self, other: Self) -> Bool:
        """Their `operator==` (`device_id.h:25-27`)."""
        return (
            self.host_id == other.host_id
            and self.device_id == other.device_id
        )

    def __ne__(self, other: Self) -> Bool:
        """Their `operator!=` (`device_id.h:29-31`)."""
        return not (self == other)

    def __lt__(self, other: Self) -> Bool:
        """Their `operator<` (`device_id.h:33-36`)."""
        return self.host_id < other.host_id or (
            self.host_id == other.host_id and self.device_id < other.device_id
        )

    def __le__(self, other: Self) -> Bool:
        """Their `operator<=` (`device_id.h:38-41`)."""
        return self.host_id < other.host_id or (
            self.host_id == other.host_id and self.device_id <= other.device_id
        )

    def __gt__(self, other: Self) -> Bool:
        """Their `operator>` (`device_id.h:43-46`), which is `!(left <= right)`.
        """
        return not (self <= other)

    def __ge__(self, other: Self) -> Bool:
        """Their `operator>=` (`device_id.h:48-51`), which is `!(left < right)`.
        """
        return not (self < other)

    def hash_value(self) -> UInt64:
        """Their `THash<TDeviceId>` (`device_id.h:60-64`),
        `(ui64(HostId) << 32) | DeviceId`.

        Spelled `hash_value` and not `__hash__` because Mojo's `Hashable`
        fixes the signature of `__hash__`, and this struct does not conform to
        it; the value is theirs, bit for bit.
        """
        return (UInt64(Int(self.host_id)) << 32) | UInt64(Int(self.device_id))

    def is_local(self) -> Bool:
        """Their `TCudaSingleDevice::IsLocalDevice()` (`single_device.h:285`).

        Local is host 0, not host -1. A default-constructed `TDeviceId` is
        therefore NOT local; it is unset.
        """
        return self.host_id == 0

    def is_remote(self) -> Bool:
        """Their `IsRemoteDevice()` (`single_device.h:288-290`)."""
        return not self.is_local()

    def get_host_id(self) -> Int32:
        """Their `GetHostId()` (`single_device.h:292-294`)."""
        return self.host_id

    def get_device_id(self) -> Int32:
        """Their `GetDeviceId()` (`single_device.h:300-302`)."""
        return self.device_id

    def to_string(self) -> String:
        return (
            String("Device(host=")
            + String(self.host_id)
            + ", dev="
            + String(self.device_id)
            + ")"
        )
