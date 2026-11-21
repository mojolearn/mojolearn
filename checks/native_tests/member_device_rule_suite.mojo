# SPDX-License-Identifier: Apache-2.0
"""The member device rule of the orchestration families
(core/device_route_dw_msel_parallel.mojo), natively: where an owner gathers
a member's fold rows, when a multi-GPU driver refuses a member, and the
check of an owner's own `device`. No Python participates.

The whole suite is a few dozen word compares over four vendor words and
three device codes; there is no independent numerical work to split, so it
runs in one pass on one core.
"""
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, LINEAR_DEVICE_INSTALL
from core.device_route_dw_msel_parallel import (
    MEMBER_DEVICE_CPU, MEMBER_DEVICE_GPU, MEMBER_DEVICE_INSTALL, MEMBER_VENDOR_UNKNOWN,
    member_folds_on_host, member_multi_gpu_refused, member_multi_gpu_refusal, member_own_device_admit,
)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def codes_are_the_pilots() raises:
    require(MEMBER_DEVICE_GPU == LINEAR_DEVICE_GPU and MEMBER_DEVICE_CPU == LINEAR_DEVICE_CPU
        and MEMBER_DEVICE_INSTALL == LINEAR_DEVICE_INSTALL, "the member device codes are not the linear route's")
    require(MEMBER_VENDOR_UNKNOWN != 0, "an unknown vendor reads as CPU only")


def fold_rows_follow_the_member() raises:
    # A CPU-only process (vendor word 0): nothing moves, whatever the member says.
    require(not member_folds_on_host(MEMBER_DEVICE_INSTALL, 0) and not member_folds_on_host(MEMBER_DEVICE_CPU, 0)
        and not member_folds_on_host(MEMBER_DEVICE_GPU, 0), "a CPU-only process changed its fold rows")
    for vendor in range(1, 4):  # the three GPU vendor words
        require(member_folds_on_host(MEMBER_DEVICE_CPU, vendor), "a cpu member kept device-resident folds")
        require(not member_folds_on_host(MEMBER_DEVICE_INSTALL, vendor),
            "the default member lost its device-resident folds")
        require(not member_folds_on_host(MEMBER_DEVICE_GPU, vendor), "a gpu member lost its device-resident folds")
    # No recorded key: never treated as CPU only, so a cpu member gets host rows.
    require(member_folds_on_host(MEMBER_DEVICE_CPU, MEMBER_VENDOR_UNKNOWN)
        and not member_folds_on_host(MEMBER_DEVICE_INSTALL, MEMBER_VENDOR_UNKNOWN),
        "an unknown vendor did not take the safe fold rows")


def multi_gpu_drivers_refuse_a_cpu_member() raises:
    require(not member_multi_gpu_refused(MEMBER_DEVICE_CPU, 0)
        and not member_multi_gpu_refused(MEMBER_DEVICE_INSTALL, 0),
        "a CPU-only process refused its own device")
    for vendor in range(1, 4):  # the three GPU vendor words
        require(member_multi_gpu_refused(MEMBER_DEVICE_CPU, vendor), "a cpu member was taken by a multi-GPU driver")
        require(not member_multi_gpu_refused(MEMBER_DEVICE_INSTALL, vendor)
            and not member_multi_gpu_refused(MEMBER_DEVICE_GPU, vendor), "a GPU member was refused")
    var refused = False
    try:
        member_multi_gpu_refusal("fit_forest")
    except problem:
        var words = String(problem)
        require(words.find("ValueError: parallel fit_forest") >= 0 and words.find("device='cpu'") >= 0
            and words.find("device=None") >= 0, "the refusal does not name the driver, device='cpu' and device=None")
        refused = True
    require(refused, "the refusal did not raise")


def own_device_is_checked_where_given() raises:
    for vendor in range(0, 4):  # CPU only and the three GPU vendor words
        member_own_device_admit(MEMBER_DEVICE_INSTALL, vendor)
        member_own_device_admit(MEMBER_DEVICE_CPU, vendor)
    for vendor in range(1, 4):  # the three GPU vendor words
        member_own_device_admit(MEMBER_DEVICE_GPU, vendor)
    member_own_device_admit(MEMBER_DEVICE_GPU, MEMBER_VENDOR_UNKNOWN)
    var refused = False
    try:
        member_own_device_admit(MEMBER_DEVICE_GPU, 0)
    except problem:
        require(String(problem).find("device='cpu'") >= 0, "the gpu refusal does not name device='cpu'")
        refused = True
    require(refused, "an explicit gpu was admitted on a CPU-only process")


def run_checks() raises:
    codes_are_the_pilots()
    fold_rows_follow_the_member()
    multi_gpu_drivers_refuse_a_cpu_member()
    own_device_is_checked_where_given()
    print("PASS member device rule", 4)
