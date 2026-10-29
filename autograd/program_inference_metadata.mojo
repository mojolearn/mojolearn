# SPDX-License-Identifier: Apache-2.0
# Immutable parameter addressing and private scratch for one session program.
from autograd.program import OpsProgram
from autograd.program_replay_metadata import PreparedProgramReplay


struct PreparedInferenceReplay(Movable):
    var replay: PreparedProgramReplay
    var slots: List[Int]
    var generations: List[Int]
    var buffers: List[Int]
    var integer_words: List[Bool]
    var leaves: List[Int]
    var temporaries: List[Int]

    def __init__(out self, p: OpsProgram, slot_bits: Int, slot_mask: Int):
        self.replay = PreparedProgramReplay(p)
        var count = len(p.params)
        self.slots = List[Int](capacity=count)
        self.generations = List[Int](capacity=count)
        for h in p.params:  # driver-loop: decode immutable captured handles once, never cache their liveness
            self.slots.append((h & slot_mask) - 1)
            self.generations.append(h >> slot_bits)
        self.buffers = List[Int](capacity=count)
        self.integer_words = List[Bool](capacity=count)
        self.leaves = List[Int](capacity=count)
        self.temporaries = List[Int](capacity=p.n_in + count + p.n_nodes())

    def clear_live(mut self):
        # Capacity survives; addresses, flags and tape handles never survive a
        # completed/failed invocation. slots/generations describe the immutable
        # program only; every run must recheck the session's live generation.
        self.buffers.clear()
        self.integer_words.clear()
        self.leaves.clear()
        self.temporaries.clear()
        self.replay.values.clear()
        self.replay.handles.clear()
