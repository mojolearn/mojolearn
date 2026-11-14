# SPDX-License-Identifier: Apache-2.0
"""The typed x_prep program of the encoder, imputer, label and selector
owners (lane encprog-typed, 2026-11-12).

`EncProgram` is `_expansion_prep._Prog`'s layout (alloc, scratch, work,
output, put, put_list, put_ints, stage) on native words: the host arena, the
device scratch (stage word SCRATCH_BASE + k), at most one device output
region (stage word OUT_MARK), the stage words and the input records. A
parameter list (`put_floats`, `put_ints`) is program-owned native words (was
a Python array each); a caller buffer is an input record whose Python owner
the door (bindings/prep_encode_native_door.mojo `EncProg`) keeps alive.
The run is the x_prep owner's typed run (bindings/x_prep_native_owner_run.mojo
`xprep_execute`) on `stage_words()`. No Python import
(tools/python_free_dirs.sh)."""
from bindings.x_prep_native_owner_run import XprepInput
from core.preprocessing_program import prep_program_stage

# `_Prog.output`: below this many words the output is plain arena words.
comptime OUT_MIN_WORDS = 134217728
# The stage word of a program's one device output region, resolved when the
# program runs (past the host arena and the scratch, as `_Prog.output`).
comptime OUT_MARK = -2147483647
# Device scratch word k is the stage word SCRATCH_BASE + k (`_Scratch`, so an
# offset added to a scratch word stays one), resolved past the host arena when
# the program runs. Stage parameters are never below -3, and the span keeps
# the words clear of them; a program needing more scratch keeps the rest as
# arena words (where a word lives moves no bit).
comptime SCRATCH_BASE = -1073741824
comptime SCRATCH_SPAN = 1073740800
comptime INT32_MAX = 2147483647
# Words of one stage record: op, total and the fourteen parameters.
comptime ENC_STAGE_WORDS = 16


struct EncProgram(Movable):
    """One x_prep program: its arena layout (host words, then device
    scratch), its input records, its native parameter lists, its stage words
    and at most one device output region (`_Prog.output`)."""
    var host: Int
    var scratch_size: Int
    var work_on: Bool
    var program: List[Int32]
    var inputs: List[XprepInput]
    var out_words: Int
    var inout_ranges: List[Int]
    var consts_f: List[List[Float32]]
    var consts_i: List[List[Int32]]

    def __init__(out self, work_on: Bool):
        self.host = 0
        self.scratch_size = 0
        self.work_on = work_on
        self.program = List[Int32]()
        self.inputs = List[XprepInput]()
        self.out_words = 0
        self.inout_ranges = List[Int]()
        self.consts_f = List[List[Float32]]()
        self.consts_i = List[List[Int32]]()

    def alloc(mut self, words: Int) -> Int:
        # `_Prog.alloc`: host arena words.
        var offset = self.host
        self.host += max(words, 0)
        return offset

    def scratch(mut self, words: Int) -> Int:
        # `_Prog.scratch`: device-only words a stage writes before any reads.
        if self.scratch_size + max(words, 0) >= SCRATCH_SPAN:
            return self.alloc(words)
        var offset = self.scratch_size
        self.scratch_size += max(words, 0)
        return SCRATCH_BASE + offset

    def work(mut self, words: Int) -> Int:
        # `_Prog.work`: device scratch under the `work` change, else arena words.
        if not self.work_on:
            return self.alloc(words)
        return self.scratch(words)

    def output(mut self, words: Int, region: Bool) raises -> Int:
        """`_Prog.output`: below OUT_MIN_WORDS, or when the door found no
        output region on the binding (`region` False: the host column, or
        MOJOLEARN_XPREP_OUT=0), the output is host words."""
        if words < OUT_MIN_WORDS or not region:
            return self.alloc(words)
        if self.out_words > 0:
            raise Error("x_prep: one output region per program")
        self.out_words = words
        return OUT_MARK

    def input(mut self, words: Int, owner: Int, address: Int = 0) -> Int:
        """A record of `words` arena words: the door's owner `owner`, or a
        program-owned parameter list at `address` (`owner` -1)."""
        var offset = self.alloc(words)
        self.inputs.append(XprepInput(offset, words, address, owner))
        return offset

    def inout(mut self, offset: Int, words: Int):
        """`_Prog.put(arr, inout=True)`: a stage rewrites these words and the
        caller reads them after the run, so they come back from the device."""
        if words > 0:
            self.inout_ranges.append(offset)
            self.inout_ranges.append(offset + words)

    def put_floats(mut self, values: List[Float64]) -> Int:
        """`_Prog.put_list`: the values rounded to float32 ([0.0] when empty)."""
        var words = List[Float32](length=max(len(values), 1), fill=0.0)
        for i in range(len(values)):  # driver-loop: over a program's parameter words
            words[i] = Float32(values[i])
        var count = len(words)
        # The inner list's buffer does not move when `consts_f` grows.
        var address = Int(words.unsafe_ptr())
        self.consts_f.append(words^)
        return self.input(count, -1, address)

    def put_ints(mut self, values: List[Int]) -> Int:
        """`_Prog.put_ints`: int32 words ([0] when empty)."""
        var words = List[Int32](length=max(len(values), 1), fill=0)
        for i in range(len(values)):  # driver-loop: over a program's int32 parameter words
            words[i] = Int32(values[i])
        var count = len(words)
        var address = Int(words.unsafe_ptr())
        self.consts_i.append(words^)
        return self.input(count, -1, address)

    def check_bound(self) raises:
        if self.host + self.scratch_size + self.out_words > INT32_MAX:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")

    def stage_words(self) -> List[Int32]:
        """The stage words with the output region past the host arena and the
        scratch, and the scratch past the host arena (`_Prog.run`)."""
        var host = self.host
        var scratch = self.scratch_size
        var program = self.program.copy()
        for i in range(len(program) // ENC_STAGE_WORDS):  # driver-loop: over the program's stages
            for k in range(2, ENC_STAGE_WORDS):  # small-loop(params: the fixed fourteen-word stage ABI)
                var word = Int(program[i * ENC_STAGE_WORDS + k])
                if word == OUT_MARK:
                    program[i * ENC_STAGE_WORDS + k] = Int32(host + scratch)
                elif word >= SCRATCH_BASE and word < SCRATCH_BASE + SCRATCH_SPAN:
                    program[i * ENC_STAGE_WORDS + k] = Int32(host + word - SCRATCH_BASE)
        return program^


# ---------------------------------------------------------------- typed stage builders
# Lane door-run-split (2026-11-14): the RUN-shaped stage builders that sat in
# the doors taking the Python-holding `EncProg` (bindings/prep_encode_native_door.mojo
# `_sel`, `_bad_rows_stages`; bindings/prep_iterimp_native_door.mojo `_ii_rows`;
# bindings/prep_labels_native_door.mojo `_label_load`'s stage) take the typed
# program here; each door passes `p.prog`. Same allocations in the same order,
# same stage words: the bits do not move.
# x_prep stage codes (_expansion_prep._OPS).
comptime ENC_OP_LAB_LOAD = 119
comptime ENC_OP_UNIQ_SCAN = 121
comptime ENC_OP_II_RCOUNT = 136
comptime ENC_OP_II_RWRITE = 137
comptime ENC_OP_SEL_COUNT = 206
comptime ENC_OP_SEL_SCAN = 207
comptime ENC_OP_SEL_WRITE = 208
comptime ENC_OP_ROWFLAG = 210
# `_p2m_chunks`' minimum rows per chunk (the door's SEL_CHUNKS).
comptime ENC_SEL_CHUNKS = 1024
# `_ii_rows`' minimum rows per chunk (iterimp's II_CH).
comptime ENC_II_CH = 1024


def enc_sel_chunks(n: Int) -> Tuple[Int, Int]:
    # `_p2m_chunks(n)`: (rows a chunk, chunks).
    var ch = max(ENC_SEL_CHUNKS, (n + ENC_SEL_CHUNKS - 1) // ENC_SEL_CHUNKS, (n + 16777215) // 16777216, 1)
    return (ch, max(1, (n + ch - 1) // ch))


def enc_sel(mut prog: EncProgram, xo: Int, n: Int, d: Int, mode: Int, dst: Int) -> Int:
    """`_p2m_sel`: per column of the (n, d) block at xo, the selected rows
    (mode 0: the non-NaN words; 1: the positive rows' indices) ascending at
    dst + j*n. Returns the offset of the d int32 counts."""
    var chunks = enc_sel_chunks(n)
    var ch = chunks[0]
    var nch = chunks[1]
    var cnt = prog.alloc(d * nch)
    var tot = prog.alloc(d)
    prep_program_stage(prog.program, ENC_OP_SEL_COUNT, d * nch, xo, n, d, ch, nch, mode, cnt)
    prep_program_stage(prog.program, ENC_OP_SEL_SCAN, d, cnt, nch, tot)
    prep_program_stage(prog.program, ENC_OP_SEL_WRITE, d * nch, xo, n, d, ch, nch, mode, cnt, dst)
    return tot


def enc_bad_rows_stages(mut prog: EncProgram, codes: Int, n: Int, d: Int, bad: Int, strict: Int) -> Tuple[Int, Int]:
    # `_bad_rows_stages`: (ROWS, TOT) of the rows holding a `bad` code (in a
    # column whose `strict` word is nonzero, when given).
    var flags = prog.alloc(n)
    var rows = prog.alloc(n)
    prep_program_stage(prog.program, ENC_OP_ROWFLAG, n, codes, n, d, bad, strict, flags)
    return (rows, enc_sel(prog, flags, n, 1, 1, rows))


def enc_ii_rows(mut prog: EncProgram, mo: Int, n: Int, dk: Int, j: Int, missing: Bool) -> Tuple[Int, Int]:
    """`_ii_rows`: the ascending rows whose mask word for feature j is (or is
    not) zero, as int32 words, and their count."""
    var ch = max(ENC_II_CH, (n + ENC_II_CH - 1) // ENC_II_CH)
    var nch = (n + ch - 1) // ch
    var cnt = prog.alloc(nch)
    var off = prog.alloc(nch)
    var tot = prog.alloc(1)
    var rows = prog.alloc(n)
    var flag = 1 if missing else 0
    prep_program_stage(prog.program, ENC_OP_II_RCOUNT, nch, mo, n, dk, j, flag, ch, cnt)
    prep_program_stage(prog.program, ENC_OP_UNIQ_SCAN, 1, cnt, nch, off, tot)
    prep_program_stage(prog.program, ENC_OP_II_RWRITE, nch, mo, n, dk, j, flag, ch, off, rows)
    return (rows, tot)


def enc_label_load(mut prog: EncProgram, n: Int, words: Int, kind: Int, x: Int):
    # `_label_load`'s stage: the float32 word of each of the n labels at
    # `words` (raw words of dtype `kind`) into x (NaN when float32 cannot hold it).
    prep_program_stage(prog.program, ENC_OP_LAB_LOAD, n, words, kind, x)
