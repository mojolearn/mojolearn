# The installed-wheel light fixtures already exercise individual Session calls.
# Admit their returned arrays into the ordinary per-device reference table.
@lane("ops-public-tensor-math")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn._verify_light import tape_elementwise, tape_gradients, tape_layout, tape_losses, tape_optimizer

    parts = {"elementwise": _h(*tape_elementwise()), "gradients": _h(*tape_gradients()),
             "layout": _h(*tape_layout()), "losses": _h(*tape_losses()),
             "optimizer": _h(*tape_optimizer())}
    # These three operators are not in the light fixtures. Two rows and two
    # channels give RoPE a nonzero position and give layer norm nonzero variance.
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        x = s.tensor(np.array([[-1.5, 0.0], [0.5, 2.0]], np.float32), requires_grad=True)
        w = s.tensor(np.array([[0.75, 1.25]], np.float32))
        b = s.tensor(np.array([[0.25, -0.5]], np.float32))
        parts["normalization"] = _h(s.numpy(s.layer_norm(x, w, b)), s.numpy(s.rms_norm(x, w)))
        parts["rotary"] = _h(s.numpy(s.rotary(x, 2)))
        s.backward(s.sum(s.mul(x, x)))
        row = s.gradient_report([x])[0]
        parts["gradient_report"] = _h(np.float32(row["gradient"]["squared_l2_float32"]))
        weight = np.array([[0.125, -1.0]], np.float32)
        s.load_weights({"projection": weight}, [(w, "projection")])
        parts["loaded_weight"] = _h(s.numpy(w))
    return _fit(parts)


@lane("ops-public-control-graph")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn._verify_api import encode
    from mojolearn._verify_api_programs import program_checks

    class Answers:
        def __init__(self):
            self.parts = {}

        def check(self, endpoint, operation):
            # The same bounded public fixture returns arrays, floats and keyed
            # gradients. Encode preserves dtype, shape and exact FP bytes.
            payload = json.dumps(encode(operation()), sort_keys=True, separators=(",", ":")).encode()
            self.parts[endpoint] = _h(np.frombuffer(payload, dtype=np.uint8))

    answers = Answers()
    for public in ("mojolearn.ops.Program", "mojolearn.importers._program.ImportedProgram"):
        program_checks(answers, public, ("run_dynamic", "profile", "run_if", "while_loop",
                                         "vjp_if", "vjp_while", "save_control", "fit_if", "distribute", "partition"))
    return _fit(answers.parts)
