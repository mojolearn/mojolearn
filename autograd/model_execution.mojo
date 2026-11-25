# SPDX-License-Identifier: Apache-2.0
# Native whole-model execution shared by public packed doors and direct sibling
# dispatch. Input metadata has already been admitted to CastIn/native vectors.
# No provider object or CPython entry is used inside these operations.
from sequence.exec_trait import Exec
from sequence.recurrent import OptConfig
from sequence.ops import FP
from core.cast_in import CastIn
from autograd.model import (
    FS_EPOCH, FS_SOURCE, IP64, SPEC2_HEADER, SPEC2_VERSION, FitConfig, FitValidation, ModelSpec, model_fit,
    model_fit_state, model_fit_state_source, model_predict, parse_model,
)
from autograd.batch_source import (
    batch_source_open, batch_source_fit_spec, source_part_load, program_training_source_refuse, batch_source_part_of,
)
from autograd.model_checkpoint import model_layout
from autograd.nn_optim import NN_OPT_LAMB, NN_SCHED_ARGS
from autograd.nn_pgseq import PgSeqTable, pgseq_table  # nn-blocks-pgseq
from autograd.cast_ops import gather_labels
from autograd.profile import ag_profile_on, ag_profile_reset, ag_profile_report

comptime FIT_INTS = 14
comptime FIT_FLOATS = 9
comptime FIT_ADDRS = 4
comptime FIT_ADDRS_STATE = 8
comptime FIT_INTS_STATE = 15
comptime FIT_ADDRS_VALIDATED = 13


def model_fptr(address: Int, what: String) raises -> FP:
    if address == 0:
        raise Error("sequence: null float32 buffer for " + what)
    return FP(unsafe_from_address=address)


def model_float_words(values: List[Float64], start: Int = 0) -> List[Float32]:
    var result = List[Float32]()
    for i in range(start, len(values)):  # driver-loop: admitted native model scalar words
        result.append(Float32(values[i]))
    return result^


def model_seed(words: List[Int], at: Int) raises -> UInt64:
    var lo = words[at]
    var hi = words[at + 1]
    if lo < 0 or hi < 0 or lo > 0xFFFFFFFF or hi > 0xFFFFFFFF:
        raise Error("model: the seed words must be 32-bit non-negative values")
    return (UInt64(hi) << 32) | UInt64(lo)


def model_counts_native(prog: List[Int], table: List[Float64]) raises -> Tuple[Int, Int]:
    var spec = parse_model(prog, model_float_words(table))
    return (spec.n_params, spec.out_features())


@fieldwise_init
struct ModelFitSetup(Movable):
    """The parsed description and the fit options of one fit call (model_fit_setup)."""

    var spec: ModelSpec
    var cfg: FitConfig


def model_fit_setup(prog: List[Int], ip: List[Int], fp: List[Float64]) raises -> ModelFitSetup:
    """The description and FitConfig of a fit's program, integer and float words (the 14 or 15 option words, the 9
    head floats, one float per layer, then optionally the schedule's NN_SCHED_ARGS words and the parameter groups'
    words). One body for the array fit (model_fit_native) and the streamed one (model_fit_source_native)."""
    # lane nn-optim: the floats are the 9 head words, one per layer, then (optionally) the schedule's
    # NN_SCHED_ARGS words (autograd/nn_optim.mojo); without them every argument word is 0
    var n_layers = prog[SPEC2_HEADER - 1] if len(prog) >= SPEC2_HEADER and prog[0] == SPEC2_VERSION else (
        prog[2] if len(prog) > 2 else 0
    )
    var table = List[Float32]()
    var sched_args = List[Float64](length=NN_SCHED_ARGS, fill=0.0)
    var table_end = len(fp)
    # nn-blocks-pgseq: the parameter-group words (autograd/nn_pgseq.mojo) follow the schedule's, when the model
    # was compiled with param_groups
    var grouped = len(fp) > FIT_FLOATS + n_layers + NN_SCHED_ARGS
    if len(fp) == FIT_FLOATS + n_layers + NN_SCHED_ARGS or grouped:
        table_end = FIT_FLOATS + n_layers
        for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
            sched_args[k] = fp[table_end + k]
    for i in range(FIT_FLOATS, table_end):  # driver-loop: admitted native model scalar words
        table.append(Float32(fp[i]))
    var spec = parse_model(prog, table)
    var opt = OptConfig(ip[3], ip[4], Float32(fp[1]), Float32(fp[2]), Float32(fp[3]), Float32(fp[4]), Float32(fp[5]))
    var offsets = List[Int]()
    if ip[3] == NN_OPT_LAMB or grouped:
        var names = List[String]()
        model_layout(spec, offsets, names)
    # nn-blocks-pgseq: the group table, once a fit (the runs of one group's tensors, each group's words)
    var groups = PgSeqTable()
    if grouped:
        groups = pgseq_table(fp, FIT_FLOATS + n_layers + NN_SCHED_ARGS, ip[3], offsets, spec.n_params)
    if ip[3] != NN_OPT_LAMB:
        offsets = List[Int]()
    var cfg = FitConfig(
        ip[1], ip[2], opt, Float32(fp[0]), ip[5], ip[6], Float32(fp[7]),
        ip[7], ip[8], ip[9] != 0, model_seed(ip, 10), ip[12],
        fp[6], Float32(fp[8]), ip[13] != 0, sched_args^, offsets^,
    )
    cfg.groups = groups^
    return ModelFitSetup(spec^, cfg^)


def model_fit_source_native[E: Exec](
    mut ex: E, addrs: List[Int], prog: List[Int], ip: List[Int], fp: List[Float64], path: String,
    words: List[Int], source_out: Int, loop_words: Int = 0, state3: Int = 0,
    sources: Optional[List[CastIn]] = None,
) raises -> Int:
    """Sequential.fit(source=) (lane stream-fits, 2026-11-25): the stateful fit's 8 addresses and 15 integer words
    (N, X and y unused: the source holds the rows) with the batch source's words (autograd/batch_source.mojo
    batch_source_fit_words) and path. The source is opened HERE, in the training image, once per call, at the
    position the words carry; a resume refuses a source that is not the one its state was trained on
    (program_training_source_refuse against the state's FS_SOURCE words) and starts at the state's consumed
    position, whatever the handle's (so a fresh process resumes with a new handle at (0, 0)). Returns the
    iterations run in all (the state's FS_EPOCH).

    Lane stream-fits-2: with validation_data, 13 addresses and 16 integers as the array fit's validated form
    (addresses 8 to 10 unused: the validation rows, targets and classes are `sources` 2 to 4; 11 the validation
    curve, 12 the result words; ip[15] restore_best_weights), or 15 with the validation rows' lengths (address 13
    unused: the training lengths are the source's; 14 the validation lengths)."""
    var validated_lengths = len(addrs) == FIT_ADDRS_VALIDATED + 2 and len(ip) == FIT_INTS_STATE + 1
    var validated = (len(addrs) == FIT_ADDRS_VALIDATED and len(ip) == FIT_INTS_STATE + 1) or validated_lengths
    if not validated and (len(addrs) != FIT_ADDRS_STATE or len(ip) != FIT_INTS_STATE):
        raise Error("model_fit: Sequential.fit(source=) takes 8 addresses and 15 integer parameters (13 and 16 with "
                    + "validation data, 15 and 16 with its lengths)")
    if len(fp) < FIT_FLOATS + 1:
        raise Error("model_fit: Sequential.fit(source=) takes 9 floats plus one per layer")
    var validation = Optional[FitValidation]()
    if validated:
        if not sources or len(sources.value()) != 5:
            raise Error("model_fit: Sequential.fit(source=) with validation data needs its input descriptors")
        ref casts = sources.value()
        var val_lengths = Optional[FP]()
        if validated_lengths:
            val_lengths = Optional[FP](model_fptr(addrs[14], "validation lengths"))
        validation = Optional[FitValidation](FitValidation(
            casts[2], casts[3], casts[4], model_fptr(addrs[11], "validation curve"), ip[15] != 0,
            IP64(unsafe_from_address=addrs[12]), val_lengths,
        ))
    var setup = model_fit_setup(prog, ip, fp)
    var opened = batch_source_fit_spec(path, words)
    var spec_words = opened[0].copy()
    var max_steps = opened[1]
    var sa = addrs[6]
    if sa == 0:
        raise Error("model_fit: null training-state buffer")
    var state = IP64(unsafe_from_address=sa)
    var resume = ip[14] != 0
    if resume:
        var saved = source_part_load(state + FS_SOURCE)
        if saved.present:
            # the same sealed data, plan and fields; then the state's consumed position, whatever the handle's
            if not spec_words.sealed:
                raise Error("ValueError: Sequential.fit(resume=True): the shard index under " + path + " records no "
                            + "content digest, so a resume could not tell these shards from other data of the same "
                            + "shape; call BatchSource.seal(path) once after writing the shards, then open the "
                            + "source again")
            program_training_source_refuse(batch_source_part_of(spec_words), saved, "Sequential.fit(resume=True)")
            spec_words.epoch = saved.epoch
            spec_words.consumed = saved.consumed
    var src = batch_source_open(spec_words^)
    var prev = model_fptr(addrs[3], "loss_curve")
    if resume and Int(state[FS_EPOCH]) > 0:
        prev = model_fptr(addrs[7], "earlier loss_curve")
    try:
        _ = model_fit_state_source(ex, setup.spec, setup.cfg, src, max_steps, model_fptr(addrs[2], "params"),
                                   model_fptr(addrs[3], "loss_curve"), model_fptr(addrs[4], "m"),
                                   model_fptr(addrs[5], "v"), state, prev, resume, loop_words, state3, source_out,
                                   validation)
    except error:
        src.close()
        raise error^
    src.close()
    return Int(state[FS_EPOCH])


def model_fit_native[E: Exec](
    mut ex: E, addrs: List[Int], sources: List[CastIn], prog: List[Int], ip: List[Int], fp: List[Float64],
    codes_dev: Int = 0, n_classes: Int = 0, loop_words: Int = 0, state3: Int = 0,
) raises -> Int:
    # `loop_words` (lane alm-neural, 2026-11-24): the address of the fit loop's record words
    # (sequence/fit_loop_words.mojo), or 0; the model trainer writes it once, after its epoch loop
    # lane rnn-parity-c: 15 addresses with validation_data and lengths: the 13 of a validated fit, then the training
    # lengths (13) and the validation rows' lengths (14), both N / n_val float32 words
    var validated_lengths = len(addrs) == FIT_ADDRS_VALIDATED + 2 and len(ip) == 16
    var validated = (len(addrs) == FIT_ADDRS_VALIDATED and len(ip) == 16) or validated_lengths
    # lane seq-blocks-3: a ninth address after the training state holds N float32 sequence lengths (0: none)
    var with_lengths = len(addrs) == FIT_ADDRS_STATE + 1 and len(ip) == FIT_INTS_STATE
    # nn-losses-2: a tenth address holds the ctc loss's N float32 target lengths (the ninth may then be 0)
    var with_targets = len(addrs) == FIT_ADDRS_STATE + 2 and len(ip) == FIT_INTS_STATE
    if with_targets:
        with_lengths = True
    var stateful = (len(addrs) == FIT_ADDRS_STATE and len(ip) == FIT_INTS_STATE) or validated or with_lengths
    var plain = len(addrs) == FIT_ADDRS and len(ip) == FIT_INTS
    if not (stateful or plain) or len(fp) < FIT_FLOATS + 1:
        raise Error("model_fit: requires 4 addresses and 14 integer parameters (or 8 and 15 with the "
                    + "training state, 9 with sequence lengths, 10 with the ctc target lengths, 13 with validation data, "
                    + "15 with validation data and lengths) and 9 floats plus one per layer")
    var setup = model_fit_setup(prog, ip, fp)
    var spec = setup.spec.copy()
    var cfg = setup.cfg.copy()
    var N = ip[0]
    # MOJOLEARN_AG_PROFILE (lane ag-pass2-kernels): this fit's launch
    # profile, counted by the executor, printed after it
    var profile = ag_profile_on()
    if profile:
        ag_profile_reset()
    var n_iter: Int
    if stateful:
        var validation = Optional[FitValidation]()
        if validated:
            var classes = CastIn.f32_c(0, 1, 1)
            if cfg.target != 0:
                classes = sources[4]
                if n_classes > 0:
                    classes.rows = n_classes
            var val_lengths = Optional[FP]()
            if validated_lengths:
                val_lengths = Optional[FP](model_fptr(addrs[14], "validation lengths"))
            validation = Optional[FitValidation](FitValidation(
                sources[2], sources[3],
                classes, model_fptr(addrs[11], "validation curve"), ip[15] != 0,
                IP64(unsafe_from_address=addrs[12]), val_lengths,
            ))
        var sa = addrs[6]
        if sa == 0:
            raise Error("model_fit: null training-state buffer")
        var state = IP64(unsafe_from_address=sa)
        var resume = ip[14] != 0
        var prev = model_fptr(addrs[3], "loss_curve")
        if resume and Int(state[FS_EPOCH]) > 0:
            prev = model_fptr(addrs[7], "earlier loss_curve")
        var y = CastIn.f32_c(0, 1, 1) if codes_dev != 0 else sources[1]
        var lengths = Optional[FP]()
        if with_lengths and addrs[8] != 0:
            lengths = Optional[FP](model_fptr(addrs[8], "lengths"))
        elif validated_lengths:
            lengths = Optional[FP](model_fptr(addrs[13], "lengths"))
        var target_lengths = Optional[FP]()  # nn-losses-2
        if with_targets:
            target_lengths = Optional[FP](model_fptr(addrs[9], "target_lengths"))
        _ = model_fit_state(ex, spec, cfg, sources[0], y, N,
                            model_fptr(addrs[2], "params"), model_fptr(addrs[3], "loss_curve"),
                            model_fptr(addrs[4], "m"), model_fptr(addrs[5], "v"), state, prev, resume, validation,
                            codes_dev, lengths, target_lengths, loop_words, state3)
        n_iter = Int(state[FS_EPOCH])
    else:
        if codes_dev != 0:
            raise Error("model_fit_labels: requires the training-state addresses")
        n_iter = model_fit(ex, spec, cfg, sources[0], sources[1], N,
                           model_fptr(addrs[2], "params"), model_fptr(addrs[3], "loss_curve"), loop_words)
    if profile:
        print("AG_PROFILE fit=model_fit")
        print(ag_profile_report())
    return n_iter


def model_predict_native[E: Exec](
    mut ex: E, addrs: List[Int], source: CastIn, prog: List[Int], table: List[Float64], ip: List[Int],
    labels: Bool = False,
) raises -> Int:
    # lane seq-blocks-3: one more trailing address holds N float32 sequence lengths (0: none)
    var base = 4 if labels else 3
    if labels:
        if (len(addrs) != 4 and len(addrs) != 5) or len(ip) != 5:
            raise Error("model_predict_labels: requires 4 addresses (5 with sequence lengths) and "
                        + "[N, chunk, transform, k, item size]")
        if ip[2] != 4 and ip[2] != 5:
            raise Error("model_predict_labels: transform must be 4 (argmax) or 5 (threshold)")
    elif (len(addrs) != 3 and len(addrs) != 4) or len(ip) != 3:
        raise Error("model_predict: requires 3 addresses (4 with sequence lengths) and 3 integer parameters")
    var lengths = Optional[FP]()
    if len(addrs) == base + 1 and addrs[base] != 0:
        lengths = Optional[FP](model_fptr(addrs[base], "lengths"))
    var spec = parse_model(prog, model_float_words(table))
    var n = ip[0]
    if labels:
        if addrs[2] == 0 or addrs[3] == 0:
            raise Error("model_gather_labels: null buffer address")
        model_predict(ex, spec, source, n, model_fptr(addrs[1], "params"),
                      FP(unsafe_from_address=addrs[3]), ip[1], ip[2], addrs[2], ip[3], ip[4], lengths)
        return n
    model_predict(ex, spec, source, n, model_fptr(addrs[1], "params"), model_fptr(addrs[2], "out"), ip[1], ip[2],
                  0, 0, 0, lengths)
    if ip[2] == 3:
        return n * 2
    if ip[2] == 4 or ip[2] == 5:
        return n
    return n * spec.out_features()


def model_class_count(k: Int, outputs: Int, target: Int) raises -> Int:
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    if target == 1 and k != outputs:
        raise Error("the labels have " + String(k) + " classes but the model has " + String(outputs) + " outputs")
    if target == 2 and (k != 2 or outputs != 1):
        raise Error("bce_with_logits needs two classes and a model with one output")
    return k
