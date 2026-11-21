# SPDX-License-Identifier: Apache-2.0
"""RUN of the byte-LM trainer door: typed values only.

One trainer operation is one call of the image's compiled entry with three
typed argument lists: buffer addresses, the integer plan words and the
optimizer scalars (the `byte_lm_trainer_api_v1` numbering,
training/byte_lm_native_dispatch.mojo). `ByteLmPlan` is that request; the two
functions below run it on a resident session (`ByteLmSessionOwner`, the typed
owner of training/byte_lm_session_owner_run.mojo) or through the image's
trainer entry with a call-scoped session. No Python object is read, built or
called here. The boundary door (`bindings/byte_lm_trainer_native_door.mojo`)
admits and pins the buffers and reads the state's scalars (UNPACK), calls one
function of this file with the GIL released, and builds the public result
(PACK).

Not split across cores: one call is one entry call into the session, which
runs its own device (or the host column's own workers).
"""
from core.entry_table import EntryRef
from core.byte_lm_entry_table_sessions import byte_lm_resident_run
from training.byte_lm_session_owner_run import ByteLmSessionOwner
from training.byte_lm_glue_abi import byte_lm_trainer_call

# The plan's integer layout: ten shape words (nine dimensions, chunked head),
# then train, completed steps, optimizer kind, nesterov, batch (the step
# count of operation 1), length, release.
comptime BYTE_PLAN_SHAPE_WORDS = 10
# lr, beta1, beta2, eps, weight_decay, momentum, dampening, max_norm.
comptime BYTE_PLAN_SCALARS = 8


struct ByteLmPlan(Movable):
    """One operation's integer words and optimizer scalars."""
    var ints: List[Int]
    var floats: List[Float64]

    def __init__(out self, shape_words: List[Int], train: Bool, completed: Int, kind: Int, nesterov: Int,
                 batch: Int, length: Int, release: Bool, var scalars: List[Float64]):
        var ints = List[Int](capacity=BYTE_PLAN_SHAPE_WORDS + 7)
        for i in range(BYTE_PLAN_SHAPE_WORDS):  # driver-loop: over the admitted model shape words
            ints.append(shape_words[i])
        ints.append(Int(train))
        ints.append(completed)
        ints.append(kind)
        ints.append(nesterov)
        ints.append(batch)
        ints.append(length)
        ints.append(Int(release))
        self.ints = ints^
        self.floats = scalars^


def byte_lm_run_resident(mut session: ByteLmSessionOwner, operation: Int, addresses: List[Int],
                         ints: List[Int], floats: List[Float64]) raises -> Int:
    """One trainer operation on the resident session. `addresses` are the
    descriptors after the session word. The owner's request buffer is
    refilled in place (one session is stepped by one caller at a time, as its
    native state requires), so a step allocates no request words."""
    var entry = session.entry  # resolved once, at the owner's construction
    var handle = session.handle()
    return byte_lm_resident_run(entry, handle, operation, addresses, ints, floats, session.words)


def byte_lm_run_table(trainer: EntryRef, operation: Int, addresses: List[Int], ints: List[Int],
                      floats: List[Float64]) raises -> Int:
    """One trainer operation through the image's table entry: the stateless
    step (operation 8, its call-scoped session first among `addresses`), its
    locked finalization (9) and the logits forward (7)."""
    return byte_lm_trainer_call(trainer, operation, addresses, ints, floats)
