# SPDX-License-Identifier: Apache-2.0
"""Test-only shared image A: publishes a pointer and owns native state.

Compile separately from the consumer. Export calls are tiny dependent control
steps, not an experiment or numerical workload; no production counters exist.
"""
from core.entry_table import entry_table_publish
from core.native_descriptor_abi import NativeDescriptorError
from core.state_registry import StateHandle
from checks.native_tests.entry_table_image_common import IMAGE_TABLE, IMAGE_ABI, IMAGE_STATES, ImageState


def _image_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if operation != 0 or np != 2 or not pp or not result:
        return 1
    var words = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 100 * words[0] + words[1]
    return 0


@export
def native_entry_image_publish() abi("C") -> Int:
    try:
        entry_table_publish[IMAGE_TABLE](1, 0, 271, [String(IMAGE_ABI)], [1], [_image_entry])
        return 0
    except problem:
        print("FAIL cross-image publication:", String(problem))
        return 1


@export
def native_entry_image_state(value: Int, generation_address: Int) abi("C") -> Int:
    if not generation_address:
        return -1
    try:
        var handle = IMAGE_STATES.insert(ImageState(value))
        MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=generation_address)[] = handle.generation
        return handle.index
    except problem:
        print("FAIL cross-image state:", String(problem))
        return -1


@export
def native_entry_image_release(index: Int, generation: UInt64) abi("C") -> Int:
    try:
        IMAGE_STATES.release(StateHandle(index, generation))
        return 0
    except problem:
        print("REFUSE cross-image release:", String(problem))
        return 1
