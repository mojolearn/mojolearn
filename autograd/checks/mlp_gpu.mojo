# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.mlp_cases import run_mlp_cases
from autograd.profile import ag_profile_on, ag_profile_report


def main() raises:
    var ex = DeviceExec()
    run_mlp_cases(ex)
    # MOJOLEARN_AG_PROFILE=1 (counts) or 2 (host-clock times between syncs)
    if ag_profile_on():
        ex.sync()
        print(ag_profile_report())
