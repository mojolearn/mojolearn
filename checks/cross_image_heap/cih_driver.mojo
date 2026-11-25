# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cross-image heap check (lane stream-fits-2): does a struct built in one binding image survive being grown,
replaced and freed in another? It settles whether the opened batch source may stay on its owner across the
Sequential and recurrent-estimator `fit(source=)` calls (autograd/batch_source.mojo, "WHY THE OPENED SOURCE IS NOT
KEPT"): there the door and the training loop are different images.

Build and run (the orchestrator; not run by the lane that wrote it). Two images from ONE source, built separately,
then the driver, all under the address sanitizer when the toolchain takes it:

    mkdir -p ~/cih && cd <checkout>
    mojo build --emit shared-lib --sanitize address -I . checks/cross_image_heap/cih_image.mojo -o ~/cih/libcih_a.so
    mojo build --emit shared-lib --sanitize address -I . checks/cross_image_heap/cih_image.mojo -o ~/cih/libcih_b.so
    mojo build --sanitize address -I . checks/cross_image_heap/cih_driver.mojo -o ~/cih/cih_driver
    ~/cih/cih_driver ~/cih/libcih_a.so ~/cih/libcih_b.so > ~/cih/run.log 2>&1; echo rc=$?

(Without `--sanitize address` if the toolchain refuses it: then a crash or a wrong checksum is the only signal.)
PASS prints `CROSS_IMAGE_HEAP PASS` and exits 0: both images report one layout, and a holder made in A, grown in
B, summed in A, grown in A and freed in B (and the same with A and B swapped, 20 rounds each) keeps its checksum,
with no sanitizer report. Any ASan report (alloc-dealloc-mismatch, heap-use-after-free, bad-free), a crash or a
checksum mismatch means separately built images do NOT share one heap: the source stays reopened per call.
"""
from std.ffi import OwnedDLHandle
from std.sys import argv


def _expected(n: Int, grown: Int, plan_extra: Int, path_len: Int) -> Int:
    var total = 0
    for i in range(grown):  # driver-loop: over the items (the oracle)
        total += i + (i % 97)
    return total + grown + plan_extra + path_len


def _round(first: OwnedDLHandle, second: OwnedDLHandle, round: Int) raises:
    """A holder made in `first`, grown in `second`, summed in `first`, grown in `first`, summed in `second`, freed
    in `second`."""
    var n = 100 + round
    var holder = first.get_function[Int]("cih_make")(n)
    var after = second.get_function[Int]("cih_grow")(holder, 100000)  # reallocated in the other image
    var path_len = String(String("cross-image-heap/") + String(n) + "/grown").byte_length()
    var want = _expected(n, after, 100000, path_len)
    var got = first.get_function[Int]("cih_sum")(holder)
    if got != want:
        raise Error("CROSS_IMAGE_HEAP FAIL: round " + String(round) + ": checksum " + String(got) + ", expected "
                    + String(want))
    var again = first.get_function[Int]("cih_grow")(holder, 50000)  # back in the making image
    var path2 = String(String("cross-image-heap/") + String(n) + "/grown/grown").byte_length()
    var want2 = _expected(n, again, 50000, path2)
    var got2 = second.get_function[Int]("cih_sum")(holder)
    if got2 != want2:
        raise Error("CROSS_IMAGE_HEAP FAIL: round " + String(round) + ": second checksum " + String(got2)
                    + ", expected " + String(want2))
    _ = second.get_function[Int]("cih_drop")(holder)  # freed by the image that did not make it


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: cih_driver <image A .so> <image B .so>")
    var a = OwnedDLHandle(String(args[1]))
    var b = OwnedDLHandle(String(args[2]))
    var layout_a = a.get_function[Int]("cih_layout")()
    var layout_b = b.get_function[Int]("cih_layout")()
    if layout_a != layout_b:
        raise Error("CROSS_IMAGE_HEAP FAIL: the images lay the holder out differently (" + String(layout_a)
                    + " vs " + String(layout_b) + " bytes)")
    for round in range(40):  # driver-loop: over rounds, the images swapped every other round
        if round % 2 == 0:
            _round(a, b, round)
        else:
            _round(b, a, round)
    print("CROSS_IMAGE_HEAP PASS: one layout (" + String(layout_a) + " bytes); 40 holders made, grown, summed and "
          + "freed across two separately built images")
