# SPDX-License-Identifier: Apache-2.0
"""Complete native Samba forward/backward over existing numerical kernels.

UNPACK and PACK of one gradient operation. RUN is
training/samba_training_run.mojo (typed words, no Python object): the
embedding segment, the tail (norm, head, loss, norm backward) and the
embedding backward, each under one GIL release. The blocks between them are
the block door's (their tapes are its records)."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_matches, neural_descriptor_provider_id
from core.samba_ids_impl import samba_training_ids_into
from core.samba_count_impl import samba_target_count
from providers.python.samba_execution_pyapi_door import samba_provider_gradients
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, py_ints
from bindings.samba_state_native_door import samba_fields, samba_from_state
from bindings.samba_route_unpack import samba_route
from bindings.byte_perf_helpers_unpack import byte_view_bytes
from bindings.neural_prefill_native_door import neural_prefill_forward, neural_prefill_backward, neural_prefill_close
from bindings.neural_block_public_door import neural_block_backward_binding
from bindings.causal_lm_public_door import _native_function
from bindings.p3_samba_caps_door import samba_caps, SAMBA_CAP_AFN, SAMBA_CAP_HEAD_LOSS, SAMBA_CAP_CHUNKED
from bindings.py_balanced import py_call, py_setitem, py_type, py_get
from bindings.samba_keys_unpack import samba_keys
from bindings.py_balanced import py_item
# RUN: typed pieces, no Python import.
from training.samba_training_run import (
    SambaTrainWords, SambaDropoutMetadata, SambaTailBuffers, SAMBA_HEAD_AFN, SAMBA_HEAD_CHUNKED, SAMBA_HEAD_PLAIN,
    samba_train_shape_check, samba_train_head_form, samba_train_embed_admit, samba_train_tail_admit,
    samba_train_embed_backward_admit, samba_dropout_run, samba_train_embed_run, samba_train_tail_run,
    samba_train_embed_backward_run, samba_loss_read,
)


def samba_ids(value: PythonObject, vocab: Int, targets: Bool, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var own = py_type(value) is py_item(py_item(context, nk[].k_neural), nk[].k_array_class)
    var view = value._mv if own else py_call(g[].bi.memoryview, value)
    var shape = value.shape if own else view.shape
    if len(shape) != 2 or Int(py=shape[0]) < 1 or Int(py=shape[1]) < 1:
        raise Error("Samba: token ids require positive (B, L) shape")
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var fmt = String(py_call(view.format.lstrip, "=@<"))
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("Samba: token ids must be native-endian integer buffers")
    # One native C-order copy of the ids under a buffer pin (formerly
    # view.tobytes then array.frombytes: two Python copies per call).
    var item = Int(py=view.itemsize)
    var packed = byte_view_bytes(view, b * l * item)
    var column_major = own and String(value.order) == "F"
    var result = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=result._addr))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    samba_training_ids_into(Int(packed.unsafe_ptr()), item, signed, b, l,
                            column_major, vocab, targets, dst)
    _ = packed^
    return result


def _samba_count(targets: PythonObject, context: PythonObject) raises -> Int:
    return samba_target_count(Int(py=targets._addr), Int(py=targets.size))


def _samba_dropout_unpack(fields: PythonObject, probability: Float64, stream: PythonObject) raises -> SambaDropoutMetadata:
    # UNPACK of the mask's words, once per gradient operation: the seed is an
    # unsigned 64-bit Python int (it does not fit a signed word), split here.
    var nk = samba_keys()
    var rng = samba_fields(py_item(fields, nk[].k_generator))
    var seed = py_item(rng, nk[].k_seed)
    return SambaDropoutMetadata(probability, Int(py=seed & PythonObject(0xffffffff)),
                                Int(py=(seed >> PythonObject(32)) & PythonObject(0xffffffff)), Int(py=stream))


def _samba_close_tapes(tapes: PythonObject) raises:
    for i in range(len(tapes)):  # driver-loop: over native tape owners pending release
        if tapes[i] is not None:
            neural_prefill_close(tapes[i])
            py_setitem(tapes, i, PythonObject(None))


def samba_gradients(model: PythonObject, inputs: PythonObject, targets: PythonObject, num_items: PythonObject, stream: PythonObject, token_offset: Int, context: PythonObject) raises -> PythonObject:
    # Public `model_gradients` / `gradients`: ids and targets are caller
    # buffers, converted to int32 here.
    return _samba_gradients(model, inputs, targets, num_items, stream, token_offset, False, context)


def samba_gradients_admitted(model: PythonObject, ids: PythonObject, y: PythonObject, num_items: PythonObject, stream: PythonObject, token_offset: Int, context: PythonObject) raises -> PythonObject:
    # The train step's microbatches: `ids` and `y` are already the admitted
    # C-order int32 Arrays `samba_ids` returned (vocab range checked), so the
    # second conversion of every microbatch is skipped.
    return _samba_gradients(model, ids, y, num_items, stream, token_offset, True, context)


def _samba_gradients(model: PythonObject, inputs: PythonObject, targets: PythonObject, num_items: PythonObject, stream: PythonObject, token_offset: Int, admitted: Bool, context: PythonObject) raises -> PythonObject:
    # ------------------------------------------------------------- UNPACK
    var nk = samba_keys()
    _ = generation_initialize_binding(py_item(py_item(context, nk[].k_neural), nk[].k_generation_classes))
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var config = samba_fields(py_item(fields, nk[].k_config))
    # The stack's one route (bindings/samba_route_unpack.mojo): the training
    # image's primitive entry was resolved once, with the route; every
    # primitive of RUN is one native call through it. `entries` stays for the
    # provider route, the caps record and the AFN entries, which the table
    # does not carry.
    var route = samba_route(model, fields, context)
    var entries = route.entries
    var native = py_get(fields, nk[].k__native_primitives)
    if native is not None and not neural_descriptor_matches(native, py_call(bi.getattr, entries, "canonical", entries)):
        native = PythonObject(None)
    # The provider id, unpacked once; `fields` keeps the provider pinned for
    # the whole operation, so every RUN call is typed by id.
    var native_id = 0
    if native is not None:
        native_id = neural_descriptor_provider_id(native)
    var d = Int(py=py_item(config, nk[].k_d_model))
    var vocab = Int(py=py_item(config, nk[].k_vocab))
    var ids = inputs if admitted else samba_ids(inputs, vocab, False, context)
    var y = targets if admitted else samba_ids(targets, vocab, True, context)
    var shape = ids.shape
    if Bool(py=shape != y.shape):
        raise Error("Samba: targets must match inputs' shape")
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var rows = b * l
    var completed = samba_provider_gradients(fields, config, entries, ids, y, num_items, stream, token_offset, py_item(context, nk[].k_neural))
    if completed is not None:
        return completed
    var count = _samba_count(y, context)
    var items = count if num_items is None else Int(py=num_items)
    var arrays = py_item(fields, nk[].k_arrays)
    var tie_embeddings = Bool(py=py_item(config, nk[].k_tie_embeddings))
    var embed = arrays["embed.weight"]
    var head = embed if tie_embeddings else arrays["lm_head.weight"]
    var norm = arrays["norm_f.weight"]
    # Optional entries and compile-time switches are fixed per binding: one
    # record read (bindings/p3_samba_caps.mojo), not five probes per call.
    var caps = samba_caps(entries)
    var afn = Bool(py=caps[SAMBA_CAP_AFN])
    var fused = not afn and Bool(py=caps[SAMBA_CAP_HEAD_LOSS])
    var chunked = not afn and Bool(py=caps[SAMBA_CAP_CHUNKED])
    var form = samba_train_head_form(afn, chunked, fused, count, rows, items)
    var dropout_probability = Float64(py=py_item(config, nk[].k_dropout))
    var dropout = dropout_probability > 0.0 and stream is not None
    var dropout_metadata: Optional[SambaDropoutMetadata] = None
    if dropout:
        dropout_metadata = _samba_dropout_unpack(fields, dropout_probability, stream)
    var norm_eps = 0.0
    if form != SAMBA_HEAD_AFN:
        norm_eps = Float64(py=py_item(config, nk[].k_norm_eps))
    var words = SambaTrainWords(route.primitives(), native_id, rows, vocab, d, norm_eps, items)
    var offset = token_offset * d
    var ids_addr = Int(py=ids._addr)
    # ------------------------------------------- RUN 1: embedding, mask
    # One GIL release for the segment (one per primitive until 2026-11-21).
    # With dropout, `hidden` is the mask's output and `embedded` its source.
    var embedded = generation_array([b, l, d])
    var hidden = embedded
    var embedded_addr = Int(py=embedded._addr)
    var masked_addr = embedded_addr
    if dropout:
        hidden = generation_array([b, l, d])
        masked_addr = Int(py=hidden._addr)
    var embed_addr = Int(py=embed._addr)
    samba_train_embed_admit(words)
    with GILReleased(Python()):
        samba_train_embed_run(words, embedded_addr, embed_addr, ids_addr)
        if dropout:
            samba_dropout_run(words.prim, masked_addr, embedded_addr, dropout_metadata.value(), rows * d, offset, False)
    _ = embed  # retain the weights and the source while the entries read their addresses
    _ = embedded^
    var blocks = py_item(fields, nk[].k__blocks)
    var layer_count = len(blocks)
    samba_train_shape_check(layer_count, len(py_item(config, nk[].k_layers)))
    var xs = bi.list()
    var tapes = bi.list()
    var gradients = bi.dict()
    try:
        # The blocks' forward: the block door's calls and tape records.
        for i in range(layer_count):  # driver-loop: over configured forward layers
            _ = py_call(xs.append, hidden)
            var forward = neural_prefill_forward(blocks[i], hidden, py_item(context, nk[].k_neural))
            hidden = forward[0]
            _ = py_call(tapes.append, forward[1])
        var loss = generation_array([1])
        var row_loss = generation_array([rows])
        var dh = generation_array([b, l, d])
        var dnorm = generation_array([d])
        var dhead = generation_array([vocab, d])
        if form == SAMBA_HEAD_AFN:
            # The image's fused training tail: a Python-exported entry the
            # table does not carry, called here as before.
            var params = py_ints([rows, vocab, d])
            _ = py_call(params.append, py_item(config, nk[].k_norm_eps))
            _ = py_call(params.append, -100)
            _ = py_call(params.append, 1)
            _ = py_call(params.append, items)
            _ = py_call(params.append, 0.0)
            _ = py_call(_native_function(entries, "samba_afn_tail_train"), 
                Python.list(loss._addr, row_loss._addr, dh._addr, dnorm._addr, dhead._addr,
                            hidden._addr, norm._addr, head._addr, y._addr), params)
        else:
            # --------------- RUN 2: norm, head + loss + head backward, norm backward
            # One GIL release for the segment (three to five until
            # 2026-11-21). The chosen form's scratch is allocated first.
            var normalized = generation_array([b, l, d])
            var dnormalized = generation_array([b, l, d])
            var tail = SambaTailBuffers()
            tail.loss = Int(py=loss._addr)
            tail.row_loss = Int(py=row_loss._addr)
            tail.dh = Int(py=dh._addr)
            tail.dnorm = Int(py=dnorm._addr)
            tail.dhead = Int(py=dhead._addr)
            tail.hidden = Int(py=hidden._addr)
            tail.norm = Int(py=norm._addr)
            tail.head = Int(py=head._addr)
            tail.y = Int(py=y._addr)
            tail.normalized = Int(py=normalized._addr)
            tail.dnormalized = Int(py=dnormalized._addr)
            samba_train_tail_admit(words, form)
            if form == SAMBA_HEAD_CHUNKED:
                var row_denom = generation_array([rows])
                tail.row_denom = Int(py=row_denom._addr)
                with GILReleased(Python()):
                    samba_train_tail_run(words, form, tail)
                _ = row_denom^
            elif form == SAMBA_HEAD_PLAIN:
                var logits = generation_array([rows, vocab])
                var dlogits = generation_array([rows, vocab])
                tail.logits = Int(py=logits._addr)
                tail.dlogits = Int(py=dlogits._addr)
                with GILReleased(Python()):
                    samba_train_tail_run(words, form, tail)
                _ = logits^
                _ = dlogits^
            else:
                with GILReleased(Python()):
                    samba_train_tail_run(words, form, tail)
            _ = normalized^
            _ = dnormalized^
        py_setitem(gradients, "norm_f.weight", dnorm)
        # The blocks' backward: the block door's calls; each layer's named
        # gradients are published as they return.
        for reverse in range(layer_count):  # driver-loop: over the full reverse layer walk
            var i = layer_count - reverse - 1
            var result = PythonObject(None)
            if tapes[i] is None:
                result = neural_block_backward_binding(blocks[i], xs[i], dh, py_item(context, nk[].k_neural))
            else:
                result = neural_prefill_backward(blocks[i], tapes[i], dh, py_item(context, nk[].k_neural))
                neural_prefill_close(tapes[i])
                py_setitem(tapes, i, PythonObject(None))
            dh = py_item(result, nk[].k_x)
            var names = py_call(bi.object.__getattribute__, blocks[i], "_W_NAMES")
            for j in range(len(names)):  # driver-loop: over each layer's named gradient registry
                py_setitem(gradients, "layers." + String(i) + "." + String(names[j]), result[names[j]])
        # ------------------------- RUN 3: mask backward, embedding backward
        # One GIL release for the segment (up to three until 2026-11-21, with
        # the tied pair's two copies under the GIL). With dropout, `unmasked`
        # is the mask backward's output and `dh` its source.
        var unmasked = dh
        var dh_addr = Int(py=dh._addr)
        var unmasked_addr = dh_addr
        if dropout:
            unmasked = generation_array([b, l, d])
            unmasked_addr = Int(py=unmasked._addr)
        var dembed = generation_array([vocab, d])
        var dembed_addr = Int(py=dembed._addr)
        var dhead_addr = Int(py=dhead._addr)
        if tie_embeddings and afn:
            if dropout:
                with GILReleased(Python()):
                    samba_dropout_run(words.prim, unmasked_addr, dh_addr, dropout_metadata.value(), rows * d, offset, True)
            _ = py_call(_native_function(entries, "samba_afn_embedding_backward_tied"), Python.list(dembed._addr, unmasked._addr, ids._addr, dhead._addr), py_ints([rows, vocab, d]))
        elif tie_embeddings:
            var pair = generation_array([2, vocab * d])
            var pair_addr = Int(py=pair._addr)
            samba_train_embed_backward_admit(words)
            with GILReleased(Python()):
                if dropout:
                    samba_dropout_run(words.prim, unmasked_addr, dh_addr, dropout_metadata.value(), rows * d, offset, True)
                samba_train_embed_backward_run(words, dembed_addr, unmasked_addr, ids_addr, True, dhead_addr, pair_addr)
            _ = pair^
        else:
            samba_train_embed_backward_admit(words)
            with GILReleased(Python()):
                if dropout:
                    samba_dropout_run(words.prim, unmasked_addr, dh_addr, dropout_metadata.value(), rows * d, offset, True)
                samba_train_embed_backward_run(words, dembed_addr, unmasked_addr, ids_addr, False, 0, 0)
        # --------------------------------------------------------------- PACK
        py_setitem(gradients, "embed.weight", dembed)
        if not tie_embeddings:
            py_setitem(gradients, "lm_head.weight", dhead)
        # The entries above read `dh`, `unmasked` and `dhead` by address; keep
        # them alive until every one has returned.
        _ = dh^
        _ = unmasked^
        _ = dhead^
        # The provider id does not retain the provider.
        _ = native
        var ordered = bi.list()
        var names = py_item(fields, nk[].k_names)
        for i in range(len(names)):  # driver-loop: over the complete ordered gradient registry
            _ = py_call(ordered.append, gradients[names[i]])
        _samba_close_tapes(tapes)
        var loss_value = PythonObject(samba_loss_read(Int(py=loss._addr)))
        _ = loss^
        _ = row_loss^
        _ = hidden^
        _ = ids^
        _ = y^
        _ = norm
        _ = head
        return Python.tuple(loss_value, ordered)
    except error:
        _samba_close_tapes(tapes)
        raise error


def samba_gradients_from_state(state: PythonObject, inputs: PythonObject, targets: PythonObject, stream: PythonObject, token_offset: Int, context: PythonObject) raises -> PythonObject:
    var model = samba_from_state(state, context)
    return samba_gradients(model, inputs, targets, PythonObject(None), stream, token_offset, context)
