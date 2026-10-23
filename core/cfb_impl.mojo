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


def cfb_f64_to_i32(src: MutPointer[Float64, MutUntrackedOrigin], dst: MutPointer[Int32, MutUntrackedOrigin], count: Int):
    """`Array.astype("<i4")` of integer-valued words read as float64 (a COO
    index vector): truncation toward zero, as the int32 cast truncates."""
    for i in range(count):  # native input marshaling: one cast of a COO index vector
        dst[i] = Int32(Int(src[i]))
