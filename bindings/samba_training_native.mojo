# SPDX-License-Identifier: Apache-2.0
"""Complete native Samba forward/backward over existing numerical kernels."""
from std.python import Python, PythonObject
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME, MAMBA_STATE_ABI_NAME
from providers.python.neural_descriptor_pyapi import neural_descriptor_matches, neural_descriptor_execute
from core.samba_ids_impl import samba_training_ids_into
from std.memory import memcpy
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, py_ints
from bindings.samba_state_native import samba_fields, samba_entries, samba_from_state
from bindings.neural_prefill_native import neural_prefill_forward, neural_prefill_backward, neural_prefill_close
from bindings.neural_block_public import neural_block_backward_binding
from bindings.causal_lm_public import _native_function
from bindings.py_balanced import py_call, py_setitem, py_type


def samba_ids(value: PythonObject, vocab: Int, targets: Bool, context: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var own = py_type(value) is context["neural"]["array_class"]
    var view = value._mv if own else py_call(g[].bi.memoryview, value)
    var shape = value.shape if own else view.shape
    if len(shape) != 2 or Int(py=shape[0]) < 1 or Int(py=shape[1]) < 1:
        raise Error("Samba: token ids require positive (B, L) shape")
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var fmt = String(py_call(view.format.lstrip, "=@<"))
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("Samba: token ids must be native-endian integer buffers")
    var packed = py_call(g[].arrays, fmt)
    _ = py_call(packed.frombytes, py_call(view.tobytes, "C"))
    var column_major = own and String(value.order) == "F"
    var result = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=result._addr))
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    samba_training_ids_into(Int(py=packed.buffer_info()[0]), Int(py=packed.itemsize), signed, b, l,
                            column_major, vocab, targets, dst)
    _ = packed^
    return result


def _samba_count(targets: PythonObject, context: PythonObject) raises -> Int:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var n = Int(py=targets.size)
    var ignored = py_call(g[].arrays, "i", Python.list(PythonObject(-100))) * n
    var mask = generation_array([n], "b")
    _ = py_call(context["equal_elements"], targets._addr, ignored.buffer_info()[0], 2, n, mask._addr)
    # The address does not retain its exporter. Keep the temporary native
    # comparison buffer alive until the synchronous kernel has consumed it.
    _ = ignored^
    var count = py_call(context["reduce_stat"], mask._addr, 5, n, 5)
    _ = mask^
    if Int(py=count[0]) != 0 or Int(py=count[1]) != 0:
        raise Error("Samba: ignored-target count exceeds kernel indexing")
    return n - Int(py=count[2])


def _samba_dropout(value: PythonObject, fields: PythonObject, stream: PythonObject, offset: Int, backward: Bool, entries: PythonObject) raises -> PythonObject:
    var config = samba_fields(fields["config"])
    var probability = Float32(Float64(py=config["dropout"]))
    var scale = Float32(1.0 / (1.0 - Float64(probability)))
    var rng = samba_fields(fields["generator"])
    var seed = rng["seed"]
    var low = Int(py=seed & PythonObject(0xffffffff))
    var high = Int(py=(seed >> PythonObject(32)) & PythonObject(0xffffffff))
    var dims = List[Int]()
    for i in range(len(value.shape)):  # driver-loop: over dropout output shape metadata
        dims.append(Int(py=value.shape[i]))
    var out = generation_array(dims)
    var params = py_ints([Int(py=value.size), offset, low, high, Int(py=stream), 3 if backward else 2])
    _ = py_call(params.append, Float64(probability))
    _ = py_call(params.append, Float64(scale))
    _ = py_call(_native_function(entries, "neural_rng"), Python.list(out._addr, value._addr), params)
    return out


def _samba_close_tapes(tapes: PythonObject) raises:
    for i in range(len(tapes)):  # driver-loop: over native tape owners pending release
        if tapes[i] is not None:
            neural_prefill_close(tapes[i])
            py_setitem(tapes, i, PythonObject(None))


def samba_gradients(model: PythonObject, inputs: PythonObject, targets: PythonObject, num_items: PythonObject, stream: PythonObject, token_offset: Int, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var entries = samba_entries(context, fields["numeric_mode"])
    var native = py_get(fields, "_native_primitives")
    if native is not None and not neural_descriptor_matches(native, py_call(bi.getattr, entries, "canonical", entries)):
        native = PythonObject(None)
    var d = Int(py=config["d_model"])
    var vocab = Int(py=config["vocab"])
    var ids = samba_ids(inputs, vocab, False, context)
    var y = samba_ids(targets, vocab, True, context)
    if Bool(py=ids.shape != y.shape):
        raise Error("Samba: targets must match inputs' shape")
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var rows = b * l
    var count = _samba_count(y, context)
    var items = count if num_items is None else Int(py=num_items)
    var arrays = fields["arrays"]
    var head = arrays["embed.weight"] if Bool(py=config["tie_embeddings"]) else arrays["lm_head.weight"]
    var norm = arrays["norm_f.weight"]
    var afn = True
    for name in [String("samba_afn_norm_head_forward"), String("samba_afn_tail_train"), String("samba_afn_embedding_backward_tied")]:  # driver-loop: over the optional fused-tail capability tuple
        afn = afn and Bool(py=py_call(bi.hasattr, entries, name))
    var fused = not afn and Bool(py=py_call(bi.hasattr, entries, "samba_head_loss"))
    var chunked = False
    if not afn and Bool(py=py_call(bi.hasattr, entries, "training_chunked_lm_head_enabled")):
        chunked = Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")())
    var dropout = Float64(py=config["dropout"]) > 0.0 and stream is not None
    var offset = token_offset * d
    var hidden = generation_array([b, l, d])
    if native is not None:
        _ = neural_descriptor_execute(native, 0, [Int(py=hidden._addr), Int(py=arrays["embed.weight"]._addr), Int(py=ids._addr)], [rows, vocab, d], [])
    else:
        _ = py_call(_native_function(entries, "embedding_forward"), Python.list(hidden._addr, arrays["embed.weight"]._addr, ids._addr), py_ints([rows, vocab, d]))
    if dropout:
        hidden = _samba_dropout(hidden, fields, stream, offset, False, entries)
    var blocks = fields["_blocks"]
    if len(blocks) != len(config["layers"]):
        raise Error("Samba: block registry does not match configured layer count")
    var xs = bi.list()
    var tapes = bi.list()
    var gradients = bi.dict()
    try:
        for i in range(len(blocks)):  # driver-loop: over configured forward layers
            _ = py_call(xs.append, hidden)
            var forward = neural_prefill_forward(blocks[i], hidden, context["neural"])
            hidden = forward[0]
            _ = py_call(tapes.append, forward[1])
        var loss = generation_array([1])
        var row_loss = generation_array([rows])
        var dh = generation_array([b, l, d])
        var dnorm = generation_array([d])
        var dhead = generation_array([vocab, d])
        if afn:
            var params = py_ints([rows, vocab, d])
            _ = py_call(params.append, config["norm_eps"])
            _ = py_call(params.append, -100)
            _ = py_call(params.append, 1)
            _ = py_call(params.append, items)
            _ = py_call(params.append, 0.0)
            _ = py_call(_native_function(entries, "samba_afn_tail_train"), 
                Python.list(loss._addr, row_loss._addr, dh._addr, dnorm._addr, dhead._addr,
                            hidden._addr, norm._addr, head._addr, y._addr), params)
        else:
            var normalized = generation_array([b, l, d])
            var norm_params = py_ints([rows, d])
            _ = py_call(norm_params.append, config["norm_eps"])
            if native is not None:
                _ = neural_descriptor_execute(native, 2, [Int(py=normalized._addr), Int(py=hidden._addr), Int(py=norm._addr)], [rows, d], [Float64(py=config["norm_eps"])])
            else:
                _ = py_call(_native_function(entries, "rms_norm_forward"), Python.list(normalized._addr, hidden._addr, norm._addr), norm_params)
            var dnormalized = generation_array([b, l, d])
            if chunked and count == rows and items == count:
                var row_denom = generation_array([rows])
                _ = py_call(_native_function(entries, "chunked_lm_head_v2_train"), 
                    Python.list(loss._addr, row_loss._addr, row_denom._addr, dnormalized._addr,
                                dhead._addr, normalized._addr, head._addr, y._addr), py_ints([rows, vocab, d]))
                _ = row_denom^
            elif fused:
                var params = py_ints([rows, vocab, d, -100, 1, items])
                _ = py_call(params.append, 0.0)
                _ = py_call(_native_function(entries, "samba_head_loss"), 
                    Python.list(loss._addr, row_loss._addr, dnormalized._addr, dhead._addr,
                                normalized._addr, head._addr, y._addr), params)
            else:
                var logits = generation_array([rows, vocab])
                var dlogits = generation_array([rows, vocab])
                if native is not None:
                    _ = neural_descriptor_execute(native, 4, [Int(py=logits._addr), Int(py=normalized._addr), Int(py=head._addr)], [rows, vocab, d], [])
                else:
                    _ = py_call(_native_function(entries, "linear_forward"), Python.list(logits._addr, normalized._addr, head._addr), py_ints([rows, vocab, d]))
                var loss_params = py_ints([rows, vocab, -100, 1, items, 1])
                _ = py_call(loss_params.append, 0.0)
                _ = py_call(_native_function(entries, "ce_loss"), loss._addr, row_loss._addr, dlogits._addr, logits._addr, y._addr, loss_params)
                if native is not None:
                    _ = neural_descriptor_execute(native, 5, [Int(py=dnormalized._addr), Int(py=dhead._addr), Int(py=dlogits._addr), Int(py=normalized._addr), Int(py=head._addr)], [rows, vocab, d], [])
                else:
                    _ = py_call(_native_function(entries, "linear_backward"), Python.list(dnormalized._addr, dhead._addr, dlogits._addr, normalized._addr, head._addr), py_ints([rows, vocab, d]))
                _ = logits^
                _ = dlogits^
            if native is not None:
                _ = neural_descriptor_execute(native, 3, [Int(py=dh._addr), Int(py=dnorm._addr), Int(py=dnormalized._addr), Int(py=hidden._addr), Int(py=norm._addr)], [rows, d], [Float64(py=config["norm_eps"])])
            else:
                _ = py_call(_native_function(entries, "rms_norm_backward"), Python.list(dh._addr, dnorm._addr, dnormalized._addr, hidden._addr, norm._addr), norm_params)
            _ = normalized^
            _ = dnormalized^
        py_setitem(gradients, "norm_f.weight", dnorm)
        for reverse in range(len(blocks)):  # driver-loop: over the full reverse layer walk
            var i = len(blocks) - reverse - 1
            var result = PythonObject(None)
            if tapes[i] is None:
                result = neural_block_backward_binding(blocks[i], xs[i], dh, context["neural"])
            else:
                result = neural_prefill_backward(blocks[i], tapes[i], dh, context["neural"])
                neural_prefill_close(tapes[i])
                py_setitem(tapes, i, PythonObject(None))
            dh = result["x"]
            var names = py_call(bi.object.__getattribute__, blocks[i], "_W_NAMES")
            for j in range(len(names)):  # driver-loop: over each layer's named gradient registry
                py_setitem(gradients, "layers." + String(i) + "." + String(names[j]), result[names[j]])
        if dropout:
            dh = _samba_dropout(dh, fields, stream, offset, True, entries)
        var dembed = generation_array([vocab, d])
        if Bool(py=config["tie_embeddings"]) and afn:
            _ = py_call(_native_function(entries, "samba_afn_embedding_backward_tied"), Python.list(dembed._addr, dh._addr, ids._addr, dhead._addr), py_ints([rows, vocab, d]))
        else:
            if native is not None:
                _ = neural_descriptor_execute(native, 1, [Int(py=dembed._addr), Int(py=dh._addr), Int(py=ids._addr)], [rows, vocab, d], [])
            else:
                _ = py_call(_native_function(entries, "embedding_backward"), Python.list(dembed._addr, dh._addr, ids._addr), py_ints([rows, vocab, d]))
            if Bool(py=config["tie_embeddings"]):
                var pair = generation_array([2, vocab * d])
                var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=pair._addr))
                var embedding_src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=dembed._addr))
                var head_src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=dhead._addr))
                memcpy(dest=dst, src=embedding_src, count=vocab * d)
                memcpy(dest=dst + vocab * d, src=head_src, count=vocab * d)
                _ = py_call(_native_function(entries, "accumulate"), Python.list(dembed._addr, pair._addr), py_ints([vocab * d, 2, -1]))
                _ = pair^
        py_setitem(gradients, "embed.weight", dembed)
        if not Bool(py=config["tie_embeddings"]):
            py_setitem(gradients, "lm_head.weight", dhead)
        # The entries above read `dh` and `dhead` by address; keep both alive
        # until every one has returned.
        _ = dh^
        _ = dhead^
        var ordered = bi.list()
        var names = fields["names"]
        for i in range(len(names)):  # driver-loop: over the complete ordered gradient registry
            _ = py_call(ordered.append, gradients[names[i]])
        _samba_close_tapes(tapes)
        var loss_ptr = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))
        var loss_value = PythonObject(Float64(loss_ptr[0]))
        _ = loss^
        _ = row_loss^
        _ = hidden^
        _ = ids^
        _ = y^
        return Python.tuple(loss_value, ordered)
    except error:
        _samba_close_tapes(tapes)
        raise error


def samba_gradients_from_state(state: PythonObject, inputs: PythonObject, targets: PythonObject, stream: PythonObject, token_offset: Int, context: PythonObject) raises -> PythonObject:
    var model = samba_from_state(state, context)
    return samba_gradients(model, inputs, targets, PythonObject(None), stream, token_offset, context)
