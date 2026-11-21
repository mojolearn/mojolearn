# SPDX-License-Identifier: Apache-2.0
"""The public training operations of the optimizer native table that are not
optimizer steps (training/optimizer_native_api.mojo OPT_NATIVE_VENDOR ..
OPT_NATIVE_MLP_SESSIONS): the parts both images share. RUN only: typed words
in, one word out, no Python object and no Python import (lane
dw-autograd-seq, 2026-11-21).

Per operation (aa = addresses, pp = ints, ff = floats; `result` is the one
word the table returns). The kernels and their argument words are the ones
the Python entries of the same names passed:
  OPT_NATIVE_VENDOR                 no words; result = the image's vendor
                                    word (0 cpu, 1 metal, 2 cuda, 3 hip,
                                    4 none: TRAINING_VENDOR_NONE)
  OPT_NATIVE_RESIDUAL_DROPOUT       aa [values, residual, output]
                                    pp [n, offset, seed low, seed high, stream]
                                    ff [p]            result = n
  OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD  aa [gradient, d values out, d residual out]
                                    pp / ff as the forward
  OPT_NATIVE_RESIDUAL_DROPOUT_DEV   pp [n, offset, seed low, seed high, stream,
                                        output handle, values handle, residual handle]
                                    ff [p]            (training device arrays)
  OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV  pp [the five words, d values
                                        handle, d residual handle, gradient handle]
  OPT_NATIVE_LM_HEAD_LOSS           aa [loss, max, denom, hidden, weight, target]
                                    pp [rows, vocab, width]
  OPT_NATIVE_LM_HEAD_TRAIN          aa [loss, max, denom, d hidden, d weight,
                                        hidden, weight, target]   pp as the loss
  OPT_NATIVE_MLP_SESSIONS           aa [output, weight1, bias1, weight2, bias2,
                                        one address per session...]
                                    pp [width, hidden, out width, one row
                                        count per session...]
Not split across cores here: each operation is one kernel call; the kernels
own their parallel form.
"""
from core.native_descriptor_call import native_descriptor_words
from training.optimizer_native_api import (
    OPT_NATIVE_VENDOR, OPT_NATIVE_RESIDUAL_DROPOUT, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD,
    OPT_NATIVE_RESIDUAL_DROPOUT_DEV, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV, OPT_NATIVE_LM_HEAD_LOSS,
    OPT_NATIVE_LM_HEAD_TRAIN, OPT_NATIVE_MLP_SESSIONS, OPT_NATIVE_COUNT,
)
from training.residual_dropout_contract import residual_dropout_admit
from training.residual_dropout_common_run import ResidualDropoutArgs

comptime TRAINING_VENDOR_NONE = 4


def optimizer_extra_operation(operation: Int) -> Bool:
    """True for the operations each image serves before the shared dispatch."""
    return operation >= OPT_NATIVE_VENDOR and operation < OPT_NATIVE_COUNT


struct OptimizerExtraRequest(Movable):
    """The typed words of one of these operations."""
    var addresses: List[Int]
    var ints: List[Int]
    var p: Float32

    def __init__(out self, operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                 result: Int) raises:
        if result == 0:
            raise Error("native optimizer: unknown operation or missing result storage")
        self.addresses = native_descriptor_words(aa, na)
        self.ints = native_descriptor_words(pp, np)
        self.p = Float32(0.0)
        var residual = operation >= OPT_NATIVE_RESIDUAL_DROPOUT and operation <= OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV
        var device = operation == OPT_NATIVE_RESIDUAL_DROPOUT_DEV or operation == OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV
        if residual:
            if nf != 1 or ff == 0 or np != (8 if device else 5) or na != (0 if device else 3):
                raise Error("residual_dropout params must be [n,offset,seed_lo,seed_hi,stream,p]")
            # The entry's rounding: Float32 of the Python float's Float64.
            self.p = Float32(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)[0])
        elif nf != 0:
            raise Error("native optimizer: request span counts differ from the operation")
        elif operation == OPT_NATIVE_VENDOR:
            if na != 0 or np != 0:
                raise Error("native optimizer: request span counts differ from the operation")
        elif operation == OPT_NATIVE_LM_HEAD_LOSS:
            optimizer_extra_addresses(self.addresses, 6, "chunked_lm_head_v2_loss")
            optimizer_extra_params(np, 3, "chunked_lm_head_v2_loss")
        elif operation == OPT_NATIVE_LM_HEAD_TRAIN:
            optimizer_extra_addresses(self.addresses, 8, "chunked_lm_head_v2_train")
            optimizer_extra_params(np, 3, "chunked_lm_head_v2_train")
        elif operation == OPT_NATIVE_MLP_SESSIONS:
            if na < 5 or np < 3:
                raise Error("mlp_sessions: addresses must contain 5 entries, got " + String(na))
            if na - 5 != np - 3:
                raise Error("neural MLP sessions: metadata length mismatch")
            for i in range(5):  # driver-loop: over the five fixed buffer addresses of one call
                if self.addresses[i] == 0:
                    raise Error("mlp_sessions: null buffer address at slot " + String(i))

    def residual(self) raises -> ResidualDropoutArgs:
        """`residual_dropout_args`: the admitted layer words."""
        ref w = self.ints
        var args = ResidualDropoutArgs(w[0], w[1], w[2], w[3], w[4], self.p)
        _ = residual_dropout_admit(args.n, args.offset, args.p, args.seed_lo, args.seed_hi, args.stream)
        return args^

    def handles(self) raises -> List[Int]:
        """`_rd_dev_handles`: three distinct device array handles."""
        var result: List[Int] = [self.ints[5], self.ints[6], self.ints[7]]
        if result[0] == result[1] or result[0] == result[2] or result[1] == result[2]:
            raise Error("residual_dropout outputs must be disjoint from inputs")
        return result^

    def session_inputs(self) -> List[Int]:
        var result = List[Int](capacity=len(self.addresses) - 5)
        for i in range(5, len(self.addresses)):  # driver-loop: over the caller's session arrays (addresses, not data)
            result.append(self.addresses[i])
        return result^

    def session_rows(self) -> List[Int]:
        var result = List[Int](capacity=len(self.ints) - 3)
        for i in range(3, len(self.ints)):  # driver-loop: over the caller's session arrays (row counts, not data)
            result.append(self.ints[i])
        return result^


def optimizer_extra_addresses(addresses: List[Int], want: Int, name: String) raises:
    """The entries' `_addrs` check, in its words."""
    if len(addresses) != want:
        raise Error(name + ": addresses must contain " + String(want) + " entries, got " + String(len(addresses)))
    for i in range(want):  # driver-loop: over the buffer addresses of one call (fixed count, not data)
        if addresses[i] == 0:
            raise Error(name + ": null buffer address at slot " + String(i))


def optimizer_extra_params(count: Int, want: Int, name: String) raises:
    """The entries' `_params` check, in its words."""
    if count != want:
        raise Error(name + ": params must contain " + String(want) + " values, got " + String(count))
