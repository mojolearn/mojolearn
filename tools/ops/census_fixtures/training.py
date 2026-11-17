# SPDX-License-Identifier: Apache-2.0
"""Census fixtures: optimizers, training utilities, ops.Session/Program, Embedding, Array, lowbit weights.

Testing tools for tools/ops/python_frames_census.py (fb4-census, 2026-11-17), never product: the smallest working
call of each public member verify's own fixtures do not reach. Shapes are verify's (python/mojolearn/_verify_api_*.py,
_verify_light.py) and the tests' (python/mojolearn/tests). Loading this module imports only the standard library;
each fixture takes `ml` (the installed mojolearn) and `cx` (python_frames_census.FixtureContext).
"""

OPTIMIZERS = ("Adam", "AdamW", "SGD")


def _param():
    import numpy as np
    return np.array([[.25, -.5], [.75, 1]], np.float32), np.array([[.125, .25], [-.125, .5]], np.float32)


def _optimizer(ml, name):
    param, grad = _param()
    return getattr(ml, name)([param], lr=.125, numeric_mode="identical"), param, grad


# ---------------------------------------------------------------- optimizers
def _opt_state(name):
    def fixture(ml, cx):
        opt, param, grad = _optimizer(ml, name)
        opt.step([grad])
        state = opt.__getstate__()
        other = type(opt).__new__(type(opt))   # as pickle restores: a blank instance, then __setstate__
        other.__setstate__(state)
    return fixture


def _opt_del(name):
    def fixture(ml, cx):
        opt, _, grad = _optimizer(ml, name)
        opt.step([grad])
        box = {"opt": opt}
        opt = None
        cx.delete(box, "opt", f"mojolearn.{name}.__del__")
    return fixture


def _opt_param_device(name):
    # On the CPU wheel the native entry refuses ("a CPU-only install"): the member still runs and is measured.
    def fixture(ml, cx):
        opt, _, _ = _optimizer(ml, name)
        opt.param_device(0)
    return fixture


def _opt_step_device(name):
    def fixture(ml, cx):
        opt, _, grad = _optimizer(ml, name)
        opt.step_device([grad])
    return fixture


def _opt_device_path(name):
    # the full device path (params resident first), for the GPU box's smoke (MOJOLEARN_CENSUS_GPU=1)
    def fixture(ml, cx):
        opt, _, grad = _optimizer(ml, name)
        opt.params_to_device()
        opt.param_device(0)
        opt.grads_to_device([grad])
        opt.step_device()
        opt.params_to_host()
    return fixture


# ---------------------------------------------------------------- training utilities
def _schedule_from_config(name):
    def fixture(ml, cx):
        cls = getattr(ml.training, name)
        obj = cls(.125, warmup_steps=1, total_steps=3, min_lr=.03125)
        restored = cls.from_config(obj.config())
        restored.lr_at(2)
    return fixture


def _schedule_block(path, args, kwargs):
    # the rate block's lr_at / bits_at (native methods the schedule holds) and the pickle state pair, on a schedule
    # built with explicit settings (the probe cannot construct one: every schedule needs its rate)
    def fixture(ml, cx):
        cls = ml
        for part in path.split("."):
            cls = getattr(cls, part)
        sched = cls(*args, **kwargs)
        sched.lr_at(2)
        sched.bits_at(2)
        state = sched.__getstate__()
        restored = cls.__new__(cls)
        restored.__setstate__(state)
        restored.lr_at(3)
    return fixture


def generator_load_state_dict(ml, cx):
    gen = ml.training.Generator(7, numeric_mode="identical")
    gen.next_stream()
    other = ml.training.Generator(7, numeric_mode="identical")
    other.load_state_dict(gen.state_dict())


def numeric_mode_used(ml, cx):
    ml.training.numeric_mode_used()
    ml.training.numeric_mode_used("identical")


# ---------------------------------------------------------------- SmallMLPTrainer (verify's tiny_mlp)
def _tiny_mlp(ml):
    import numpy as np
    weights = [(np.arange(n, dtype=np.float32) % 7 - 3).reshape(shape) / 32 for n, shape in
               [(128, (16, 8)), (16, (16,)), (48, (3, 16)), (3, (3,))]]
    m = ml.SmallMLPTrainer(*weights, data_schedule={"fixture": "public-api-v1"}, lr=.125)
    x = np.arange(16, dtype=np.float32).reshape(2, 8) / 32
    y = np.array([0, 2], np.int32)
    return m, x, y


def small_mlp_grads(ml, cx):
    m, x, y = _tiny_mlp(ml)
    _, grads = m.loss_and_grads(x, y)
    m.apply_gradients(grads)


def small_mlp_del(ml, cx):
    m, x, y = _tiny_mlp(ml)
    m.train_step(x, y)
    box = {"m": m}
    m = None
    cx.delete(box, "m", "mojolearn.SmallMLPTrainer.__del__")


def parallel_neural_export_gradients(ml, cx):
    m, x, y = _tiny_mlp(ml)
    p = ml.ParallelNeuralTrainer(m, devices=(0,), logical_shards=1)
    try:
        p.train_step([(x, y)])
        p.export_gradients()
    finally:
        p.close()


# ---------------------------------------------------------------- ops.Session and ops.Program
def session_tensors(ml, cx):
    import numpy as np
    from mojolearn import ops
    with ops.Session(numeric_mode="identical") as s:
        h = s.tensor(np.array([[.25, .5], [.75, 1]], np.float32))
        s.shape(h)
        st = s.state(np.zeros((1, 1), np.int32), ids=True)
        s.shape(st)
        s.free(h)


def session_gradient_report(ml, cx):
    import numpy as np
    from mojolearn import ops
    with ops.Session(record=True, numeric_mode="identical") as s:
        x = s.tensor(np.array([[.25, .5], [.75, 1]], np.float32))
        w = s.tensor(np.array([[.5], [-.25]], np.float32), requires_grad=True)
        loss = s.mse(s.matmul(x, w), s.tensor(np.array([[.125], [.5]], np.float32)))
        s.backward(loss)
        s.gradient_report([w])


def session_load_weights(ml, cx):
    import numpy as np
    from mojolearn import ops
    with ops.Session(numeric_mode="identical") as s:
        w = s.tensor(np.zeros((2, 2), np.float32))
        s.load_weights({"w": np.array([[.25, .5], [.75, 1]], np.float32)}, [(w, "w")])


def _network(s):
    import numpy as np
    x = s.tensor(np.zeros((3, 2), np.float32))
    w = s.tensor(np.arange(8, dtype=np.float32).reshape(2, 4) / 32)
    b = s.tensor(np.full((1, 4), .125, np.float32))
    v = s.tensor(np.arange(8, dtype=np.float32).reshape(4, 2) / 64)
    return s.capture([x], [s.matmul(s.tanh(s.linear(x, w, b)), v)])


def _xy():
    import numpy as np
    rng = np.random.default_rng(26)
    return rng.normal(size=(7, 2)).astype(np.float32), rng.normal(size=(7, 2)).astype(np.float32)


def session_load_partitioned(ml, cx):
    from mojolearn import ops
    x, y = _xy()
    path = str(cx.tmp / "partition.ckpt")
    with ops.Session(numeric_mode="identical") as s:
        p = _network(s)
        opt = s.optimizer("sgd", lr=.125)
        with p.partition((0,), stages=2, tensor_shards=1, microbatches=2, optimizer=opt, seed=41) as d:
            d.fit(x, targets=y, epochs=1, batch_size=4)
            d.save(path)
        with s.load_partitioned(path, devices=(0,)) as replay:
            replay.state()


def session_load_distributed(ml, cx):
    from mojolearn import ops
    x, y = _xy()
    path = str(cx.tmp / "distributed.ckpt")
    with ops.Session(numeric_mode="identical") as s:
        p = _network(s)
        opt = s.optimizer("sgd", lr=.125)
        with p.distribute((0,), logical_shards=2, optimizer=opt, seed=91) as d:
            d.fit(x, targets=y, epochs=1, batch_size=4)
            d.save(path)
        with s.load_distributed(path, devices=(0,)) as replay:
            replay.state()


def _next_token_table(vocab):
    import numpy as np
    table = np.zeros((vocab, vocab), np.float32)
    for t in range(vocab):   # fixture data: a table whose row t peaks at t + 1
        table[t, (t + 1) % vocab] = 1.0
    return table


def program_decode_init(ml, cx):
    import numpy as np
    from mojolearn import ops
    with ops.Session(numeric_mode="identical") as s:
        table = s.tensor(_next_token_table(5))
        ids = s.ids(np.array([[1]]))
        step = s.capture([ids], [s.argmax(s.embedding(table, ids), axis=-1, keepdims=True, int32=True)])
        step.decode(np.array([[1]]), n=3)
        # Program.__init__ (native construction never runs it): re-initialized with its own metadata, same session
        info = (getattr(step, "_id"), step.input_shapes, step.output_shapes, step.params, step.input_kinds)
        step.__init__(s, info)
        step.free()


# ---------------------------------------------------------------- Embedding
def embedding_save_load_del(ml, cx):
    import numpy as np
    weights = np.arange(8, dtype=np.float32).reshape(4, 2) / 8
    m = ml.Embedding(4, 2, weight=weights)
    path = str(cx.tmp / "embedding.npz")
    m.save(path)
    loaded = ml.Embedding.load(path)
    loaded(np.array([[0, 2]], np.int64))
    box = {"m": m}
    m = None
    cx.delete(box, "m", "mojolearn.Embedding.__del__")


# ---------------------------------------------------------------- Array
def array_members(ml, cx):
    a = ml.Array((2, 2), "<f4")
    b = ml.Array.from_list([[.25, .5], [.75, 1]], dtype="<f4")
    b.astype("<f8")
    cx.access(b, "flags", "mojolearn.Array.flags")
    reducer, args, *state = b.__reduce__()
    restored = reducer(*args)
    setstate = cx.access(restored, "__setstate__", "mojolearn.Array.__setstate__")
    if state and state[0] is not None:
        setstate(state[0])
    del a


# ---------------------------------------------------------------- lowbit weights
def lowbit_weights(ml, cx):
    import numpy as np
    ml.lowbit.BF16Weight(np.array([[0x3e80, 0x3f00], [0x3f40, 0x3f80]], np.uint16))
    ml.lowbit.Int8Weight(np.array([[1, -2], [3, 4]], np.int8), np.array([-2, -3], np.int32))


FIXTURES = {
    "generator_load_state_dict": (generator_load_state_dict, ["mojolearn.training.Generator.load_state_dict"]),
    "numeric_mode_used": (numeric_mode_used, ["mojolearn.training.numeric_mode_used"]),
    "small_mlp_grads": (small_mlp_grads, ["mojolearn.SmallMLPTrainer.loss_and_grads",
                                          "mojolearn.SmallMLPTrainer.apply_gradients"]),
    "small_mlp_del": (small_mlp_del, ["mojolearn.SmallMLPTrainer.__del__"]),
    "parallel_neural_export_gradients": (parallel_neural_export_gradients,
                                         ["mojolearn.ParallelNeuralTrainer.export_gradients"]),
    "session_tensors": (session_tensors, ["mojolearn.ops.Session.shape", "mojolearn.ops.Session.state",
                                          "mojolearn.ops.Session.free"]),
    "session_gradient_report": (session_gradient_report, ["mojolearn.ops.Session.gradient_report"]),
    "session_load_weights": (session_load_weights, ["mojolearn.ops.Session.load_weights"]),
    "session_load_partitioned": (session_load_partitioned, ["mojolearn.ops.Session.load_partitioned"]),
    "session_load_distributed": (session_load_distributed, ["mojolearn.ops.Session.load_distributed"]),
    "program_decode_init": (program_decode_init, ["mojolearn.ops.Program.decode", "mojolearn.ops.Program.__init__"]),
    "embedding_save_load_del": (embedding_save_load_del, ["mojolearn.Embedding.save", "mojolearn.Embedding.load",
                                                          "mojolearn.Embedding.__del__"]),
    "array_members": (array_members, ["mojolearn.Array.__init__", "mojolearn.Array.astype", "mojolearn.Array.flags",
                                      "mojolearn.Array.__setstate__"]),
    "lowbit_weights": (lowbit_weights, ["mojolearn.lowbit.BF16Weight.__init__", "mojolearn.lowbit.Int8Weight.__init__"]),
}
for _name in OPTIMIZERS:
    FIXTURES[f"{_name.lower()}_state"] = (_opt_state(_name), [f"mojolearn.{_name}.__getstate__",
                                                              f"mojolearn.{_name}.__setstate__"])
    FIXTURES[f"{_name.lower()}_del"] = (_opt_del(_name), [f"mojolearn.{_name}.__del__"])
    FIXTURES[f"{_name.lower()}_param_device"] = (_opt_param_device(_name), [f"mojolearn.{_name}.param_device"])
    FIXTURES[f"{_name.lower()}_step_device"] = (_opt_step_device(_name), [f"mojolearn.{_name}.step_device"])
    FIXTURES[f"{_name.lower()}_device_path"] = (_opt_device_path(_name), [f"mojolearn.{_name}.param_device",
                                                                          f"mojolearn.{_name}.step_device"], "gpu")
for _name in ("ConstantLR", "WarmupCosineLR", "WarmupLinearLR"):
    FIXTURES[f"{_name.lower()}_from_config"] = (_schedule_from_config(_name),
                                                [f"mojolearn.training.{_name}.from_config"])
for _path, _args, _kwargs in (("training.ConstantLR", (.125,), {"warmup_steps": 1}),
                               ("training.WarmupLinearLR", (.125,), {"warmup_steps": 1, "total_steps": 3, "min_lr": .03125}),
                               ("training.WarmupCosineLR", (.125,), {"warmup_steps": 1, "total_steps": 3, "min_lr": .03125}),
                               ("StepLR", (.125, 2, .5), {}), ("ExponentialLR", (.125, .5), {}),
                               ("OneCycleLR", (.125, 4), {})):
    FIXTURES[f"{_path.rsplit('.', 1)[-1].lower()}_block"] = (_schedule_block(_path, _args, _kwargs), [
        f"mojolearn.{_path}.{_m}" for _m in ("lr_at", "bits_at", "__getstate__", "__setstate__")])

#: members whose real work needs device-resident optimizer state: on the CPU wheel their native entry refuses
#: ("a CPU-only install"); the CPU fixtures above still call them (measured as the refusal), the "gpu" fixture runs
#: the whole device path on the GPU box.
NEEDS_GPU = {f"mojolearn.{_n}.{_m}": "device-resident optimizer parameters (params_to_device) do not exist on a "
             "CPU-only install; the native entry refuses" for _n in OPTIMIZERS for _m in ("param_device", "step_device")}

WHY_NOT = {}
