# Projected LSTM. One variable-length batch, one optimizer step and one saved Program.
# Loaded by identity_break.py in both the source tree and the installed wheel.


@lane("sequence-projected-lstm")
def _(ml, X, yc, yr, Xh=None):
    def words(shape, offset):
        values = (np.arange(np.prod(shape), dtype=np.float32) + offset) % 19 - 9
        return (values.reshape(shape) / 32).astype(np.float32)

    xs = words((3, 2, 2), 2)
    initial_h, initial_c = words((2, 2), 3), words((2, 3), 7)
    matrices = [words(shape, 2 * i + 1) for i, shape in enumerate(((12, 2), (12, 2), (12,), (12,), (2, 3)))]
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        x, h, c = [s.tensor(v, requires_grad=True) for v in (xs, initial_h, initial_c)]
        wi, wh, bi, bh, hr = [s.tensor(v, requires_grad=True) for v in matrices]
        lens = s.tensor(np.array([3, 1], np.float32))
        y, hn, cn = s.lstm(x, wi, wh, bi, bh, h_0=h, c_0=c, lengths=lens,
                           return_state=True, weight_hr=hr)
        loss = s.add(s.sum(y), s.sum(cn))
        s.backward(loss)
        parts["forward"] = _h(*[np.asarray(s.numpy(t)) for t in (y, hn, cn)])
        parts["backward"] = _h(*[np.asarray(s.grad(t)) for t in (x, wi, wh, bi, bh, hr, h, c)])
    with ml.ops.Session(numeric_mode="identical") as s:
        wi, wh, bi, bh, hr = [s.tensor(v) for v in matrices]
        x, h, c = [s.tensor(v) for v in (xs, initial_h, initial_c)]
        def run(seq, h0, c0):
            return s.lstm(seq, wi, wh, bi, bh, h_0=h0, c_0=c0, reverse=True, return_state=True, weight_hr=hr)
        y, hn, cn = run(x, h, c)
        program = s.capture(inputs=[x, h, c], outputs=[y, hn, cn])
        outputs = []
        for t in reversed(range(len(xs))):
            step, h, c = run(s.tensor(xs[t:t+1]), h, c)
            outputs.append(np.asarray(s.numpy(step)))
        _same_bytes("reverse LSTM step outputs", np.concatenate(outputs[::-1]), "whole sequence", np.asarray(s.numpy(y)))
        _same_bytes("reverse LSTM last h", np.asarray(s.numpy(h)), "whole h", np.asarray(s.numpy(hn)))
        _same_bytes("reverse LSTM last c", np.asarray(s.numpy(c)), "whole c", np.asarray(s.numpy(cn)))
        with tempfile.TemporaryDirectory(prefix="projected-lstm-") as tmp:
            path = os.path.join(tmp, "model.mlp")
            program.save(path)
            with ml.ops.Session(numeric_mode="identical") as restored:
                replay = restored.load_program(path).run(xs, initial_h, initial_c)
                for actual, handle in zip(replay, (y, hn, cn)):
                    _same_bytes("saved projected LSTM", np.asarray(actual), "eager", np.asarray(s.numpy(handle)))
        parts["reverse"] = _h(*[np.asarray(s.numpy(t)) for t in (y, hn, cn)])
    return _fit(parts)


@lane("sequence-blocks-projected-lstm")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn._verify_api_neural import SEQUENTIAL_LAYER_FIXTURES
    shape, build = SEQUENTIAL_LAYER_FIXTURES["projected_lstm"]
    model = build(ml.Sequential(input_shape=shape, numeric_mode="identical"), ml.Sequential).flatten().linear(2)
    x = ((np.arange(24, dtype=np.float32) - 9) / 16).reshape(4, 3, 2)
    held = np.ascontiguousarray(x[::-1])
    lengths = np.array([1, 3, 2, 3], np.int64)
    return _sequence_blocks_step(ml, model, x, np.array([0, 1, 0, 1], np.int64), held, lengths)


_batch_decl("n/a:training-step (the fixture hashes a backward pass or a seeded optimizer update)",
            "sequence-projected-lstm", "sequence-blocks-projected-lstm")
