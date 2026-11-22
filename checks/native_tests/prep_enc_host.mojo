# SPDX-License-Identifier: Apache-2.0
"""The host-column run of the typed encoder x_prep programs, for the batch 2 checks (lane py2mojo-B, 2026-11-22;
docs/plans/PY2MOJO_CONVERSION.md). Importable; no `main()`, no `test_*`.

`enc_host_run` runs an `EncProgram` (bindings/prep_encode_prog_run.mojo) as the x_prep owner's run does on the host
column (bindings/x_prep_native_owner_run.mojo `xprep_execute`, no output region: the doors pass `region` False when
the image has none, so every output is host words): the arena of host, scratch and output words starts zero, every
input record is filled from its owner's words (`owners[owner]`) or from the program's own parameter words
(`owner` -1), then one run of `stage_words()` through the x_prep host program runner.
"""
from std.memory import unsafe_memcpy
from bindings.prep_encode_prog_run import EncProgram, ENC_STAGE_WORDS
from x_prep.host.program import run_program_host

comptime PREP_KB_DIR = "checks/native_tests/fixtures/prep_kb/"

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def enc_host_run(prog: EncProgram, owners: List[List[Float32]]) raises -> List[Float32]:
    var arena = List[Float32](length=max(prog.host + prog.scratch_size + prog.out_words, 1), fill=0)
    for k in range(len(prog.inputs)):  # driver-loop: over the program's input records
        var r = prog.inputs[k]
        if r.words <= 0:
            continue
        if r.owner < 0:
            unsafe_memcpy(dest=arena.unsafe_ptr() + r.offset, src=_F32P(unsafe_from_address=r.address),
                count=r.words)
            continue
        if r.owner >= len(owners) or len(owners[r.owner]) < r.words:
            raise Error("enc_host_run: owner " + String(r.owner) + " does not hold " + String(r.words) + " words")
        unsafe_memcpy(dest=arena.unsafe_ptr() + r.offset, src=owners[r.owner].unsafe_ptr(), count=r.words)
    var stages = prog.stage_words()
    run_program_host(Int(arena.unsafe_ptr()), len(arena), Int(stages.unsafe_ptr()), len(stages) // ENC_STAGE_WORDS)
    _ = stages^
    return arena^
