# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One BINDING IMAGE of the cross-image heap check (lane stream-fits-2; the question of the fable-finalize review,
item 6(a), autograd/batch_source.mojo "WHY THE OPENED SOURCE IS NOT KEPT"): can a Mojo struct whose Lists were
allocated in one separately built shared library be grown, replaced and freed in another?

This file is built TWICE, as two separate shared libraries (two images, as two binding modules are), and
checks/cross_image_heap/cih_driver.mojo loads both and moves one holder between them. `_Holder` has the member
kinds of autograd/batch_source.mojo BatchSource: a String, Lists of Int and Float32 that `plan` and `read`
reallocate. Every export takes and returns Int words (C ABI), so nothing but addresses crosses."""
from std.memory import alloc
from std.sys.info import size_of


struct _Holder(Movable):
    var path: String
    var ids: List[Int]
    var words: List[Float32]
    var plans: List[List[Int]]

    def __init__(out self, n: Int):
        self.path = String("cross-image-heap/") + String(n)
        self.ids = List[Int](capacity=n)
        self.words = List[Float32](capacity=n)
        self.plans = List[List[Int]]()
        for i in range(n):  # driver-loop: over the fixture's items
            self.ids.append(i)
            self.words.append(Float32(i % 97))
        self.plans.append(self.ids.copy())


def _holder(address: Int) -> UnsafePointer[_Holder, MutUntrackedOrigin]:
    return UnsafePointer[_Holder, MutUntrackedOrigin](unsafe_from_address=address)


@export
def cih_layout() abi("C") -> Int:
    """The holder's size in bytes in this image (the driver compares the two images' words)."""
    return size_of[_Holder]()


@export
def cih_make(n: Int) abi("C") -> Int:
    """A holder of n items on this image's heap; its address."""
    var p = alloc[_Holder](1)
    p.unsafe_write(_Holder(n))
    return Int(p)


@export
def cih_grow(address: Int, n: Int) abi("C") -> Int:
    """Grow every List of the holder by n items (reallocating the other image's buffers in this one), replace its
    plan list whole (freeing the old one here) and its path; returns the item count."""
    var h = _holder(address)
    var start = len(h[].ids)
    for i in range(n):  # driver-loop: over the added items
        h[].ids.append(start + i)
        h[].words.append(Float32((start + i) % 97))
    var fresh = List[List[Int]]()
    fresh.append(h[].ids.copy())
    fresh.append(List[Int](length=n, fill=7))
    h[].plans = fresh^
    h[].path = h[].path + "/grown"
    return len(h[].ids)


@export
def cih_sum(address: Int) abi("C") -> Int:
    """A checksum of every item: sum(ids) + sum(words) + the plans' sizes + the path's length."""
    var h = _holder(address)
    var total = 0
    for i in range(len(h[].ids)):  # driver-loop: over the items
        total += h[].ids[i] + Int(h[].words[i])
    for k in range(len(h[].plans)):  # driver-loop: over the plans
        total += len(h[].plans[k])
    return total + h[].path.byte_length()


@export
def cih_drop(address: Int) abi("C") -> Int:
    """Destroy the holder and free its storage on THIS image's heap."""
    var h = _holder(address)
    h.unsafe_deinit_pointee()
    h.free()
    return 0
