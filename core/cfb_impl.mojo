# SPDX-License-Identifier: Apache-2.0
# Host marshaling helpers of the cluster-fit-b family owner
# (bindings/cluster_fit_b_native.mojo, lane mcb2-cfb): reads of a kernel's
# small per-row flag output, no arithmetic on data.


def cfb_any_positive_i32(src: MutPointer[Int32, MutUntrackedOrigin], count: Int) -> Bool:
    """`int(flags.max()) > 0` for `count` >= 1 int32 flags."""
    for i in range(count):  # native output marshaling: one scan of a kernel's per-row flags
        if src[i] > Int32(0):
            return True
    return False
