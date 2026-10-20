#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE AUTOGRAD SMOKE ENTRIES (2026-10-19, lane ag-bench): does the tape start on this vendor.

"Does it start" entries per vendor (metal, nvidia, amd), 5 s each, nothing timed:

  autograd-gradients     autograd/checks/gpu.mojo, the gradient suite on the vendor's GPU (every
                         case prints PASS and its digest; the digests equal the CPU column's).
  autograd-mlp-identity  autograd/checks/mlp_gpu.mojo, the MLP tape fits checked against the stored
                         48-case digests when the binary finishes (the hand trainer was deleted).
  autograd-fit-smoke     tests/test_autograd_smoke.py through the installed bindings: a tiny MLP
                         and a tiny LSTM fit on the default (tape) path, finite losses, the
                         expected-digest file present.
  autograd-public        python/mojolearn/tests/test_autograd_public.py: the docs' Sequential and
                         ops.Session examples run verbatim; PUBLIC_DIGEST lines in its log.
  autograd-sequential-smoke
                         (2026-10-19, lane ag-board-twins) the board's sequential-fit recipe in
                         miniature through the installed bindings: mojolearn.Sequential linear,
                         relu, linear(1), BCE with logits, Adam, one fit and predict_proba on a
                         seeded 256 x 8 set; finite probabilities, a SMOKE_DIGEST line (params_).
  autograd-session-smoke (2026-10-19, lane ag-board-twins) the board's ops-session-step in
                         miniature: one Session(record=True) step (linear, relu, linear, softmax
                         cross-entropy, backward, native Session.step); finite loss and gradients,
                         and the updated parameters checked against an independent SGD oracle.
  causal-lm-generate-smoke
                         (2026-10-19, lane ag-inference-board) the board's causal-lm-generate lanes
                         in miniature: a seeded two-layer checkpoint in the byte-LM format (no final
                         norm, the 162M's layout) written to a temporary directory, CausalLM.load and
                         greedy generate(prompt, 4); ids in the vocabulary, the prompt echoed, a
                         SMOKE_DIGEST line (the new ids).
  sequential-predict-smoke
                         (2026-10-19, lane ag-inference-board) the board's sequential-predict lane in
                         miniature: a tiny Sequential fit, then predict_proba inside the board's
                         untimed memory pass (its NVML / KFD counters); finite probabilities, a
                         SMOKE_DIGEST line (the probabilities) and the memory figures read.
  autograd-higher        (2026-10-19, lane ag-ho-checks) autograd/checks/higher_gpu.mojo, the
                         higher-order suite (create_graph, second and third backward) on the
                         vendor's GPU; one HIGHER_DIGEST line per case.
  import-model-smoke     (2026-10-20, lane ag-import-docs) mojolearn.import_model of the tests' tiny
                         Conv/Relu/MaxPool/Flatten/Gemm ONNX fixture, written in process to a
                         temporary directory, then one Program.run; finite logits of the
                         expected shape, a SMOKE_DIGEST line (the logits).
  import-torch-smoke     (2026-10-20, lane ag-torch-twins) mojolearn.import_model of a torch.export
                         ExportedProgram: a tiny nn.Module (linear, layer norm, linear: the shared
                         fixture's shapes and weights) exported in process, then one Program.run;
                         prints `RUN torch <version>` and a SMOKE_DIGEST line, or `SKIPPED no torch
                         (<reason>)` and exits 0 when torch is missing (MOJOLEARN_REQUIRE_TORCH=1 makes
                         that a failure). The row's status reads `SKIPPED` then, never `ok`.
  real-model-smoke       (2026-10-20, lane ag-real-models) tools/real_model_import.py in miniature:
                         torchvision's squeezenet1_1 with its published weights, torch.export at a
                         1 x 3 x 64 x 64 input, import_model, one Program.run; finite (1, 1000)
                         logits and a SMOKE_DIGEST line, or SMOKE_REFUSED lines naming each node.

The registry also covers dropout/resume, validation and best-state restore, Program.fit and dynamic
replay, program persistence, session optimizers, rank-N operations, and public higher-order derivatives
(including LayerNorm). Native import operations and the Mamba, LM, small-MLP and Samba tape checks
are pinned too. `list` is the complete list; both vendor gates consume this registry.

A check binary is built ONCE per vendor by the orchestrator (`build-commands` prints the lines; this
tool never builds), then `run` starts each entry with a 5 s timer:

    python3 tools/autograd_smoke.py build-commands --vendor nvidia --mojo MOJO --bin-dir DIR
    python3 tools/autograd_smoke.py run --vendor nvidia --bin-dir DIR --python PY --out FILE.json

A row is `ok` (exited 0 inside its timer), `STARTED` (still running at the timer: it started, no
verdict), `failed` (non-zero exit, with its last lines) or `not built` (no binary: refused by name,
never a pass); a probe that prints a `SKIPPED <reason>` line reads `SKIPPED` (never silence, never
`ok`). A model refusal reads `REFUSED`; incomplete pytest coverage reads `SKIPPED` or `PARTIAL`
and fails the gate. Each Python entry checks its actual compiled binding vendor and numeric mode before
running. Installed-package checks use isolated Python; --source-package explicitly selects a source
tree and labels that scope in the report. This is smoke coverage, not performance or identity qualification.
tools/apple_smoke_set.py lists these entries beside its derived lanes and
tools/gate_smoke_box.py runs them after the board smoke on NVIDIA and AMD.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORS = ("metal", "nvidia", "amd")
BINDING_VENDOR = {"metal": "metal", "nvidia": "cuda", "amd": "hip"}
#: Andrew, 2026-10-14: smoke is a 5 s "does it start"
TIMEOUT_S = 5
#: the vendor column define a GPU check is built with (bindings/build_byte_lm.sh's rule)
COLUMN_DEFINE = {"metal": [], "nvidia": ["-D", "MOJOLEARN_COLUMN_NVIDIA"], "amd": ["-D", "MOJOLEARN_COLUMN_AMD"]}
ENTRIES = (
    {"name": "autograd-gradients", "source": "autograd/checks/gpu.mojo",
     "what": "the autograd gradient suite on the vendor's GPU, one PASS line and digest per case"},
    {"name": "autograd-mlp-identity", "source": "autograd/checks/mlp_gpu.mojo",
     "what": "48 MLP tape fits against stored digests when the GPU check completes"},
    {"name": "autograd-fit-smoke", "pytest": "tests/test_autograd_smoke.py",
     "what": "a tiny MLP and LSTM fit through the installed bindings' default path"},
    {"name": "autograd-public", "pytest": "python/mojolearn/tests/test_autograd_public.py", "pytest_args": ["-s"],
     "what": "the public Sequential and ops.Session examples of the docs, PUBLIC_DIGEST lines"},
    # 2026-10-19 (lane ag-board-twins): the two board lanes of the public tape API, in miniature
    {"name": "autograd-sequential-smoke", "probe": "sequential",
     "what": "a tiny Sequential fit + predict_proba (the sequential-fit board lane) through the installed bindings"},
    {"name": "autograd-session-smoke", "probe": "session",
     "what": "one tiny ops.Session(record=True) step (the ops-session-step board lane) through the installed bindings"},
    {"name": "autograd-program-smoke", "probe": "program",
     "what": "the program-run board runner: repeated captured inference equals the eager runner"},
    {"name": "autograd-usability", "pytest": "python/mojolearn/tests/test_autograd_usability.py",
     "what": "validation, best-state restore, native Program.fit, custom loss, frozen parameters and dynamic replay"},
    {"name": "autograd-dropout", "pytest": "python/mojolearn/tests/test_autograd_model.py::test_dropout_trains",
     "what": "seeded Sequential dropout, inference and resumed training"},
    {"name": "autograd-program", "pytest": "python/mojolearn/tests/test_ops_program.py",
     "what": "captured inference, embedding inputs, save/load and replay refusals"},
    {"name": "autograd-optimizer", "pytest": "python/mojolearn/tests/test_ops_optimizer.py",
     "what": "native session updates, schedules, optimizer save/load and accumulation"},
    {"name": "autograd-rankn", "pytest": "python/mojolearn/tests/test_ops_rankn.py",
     "what": "rank-N shapes, reductions, broadcasting, convolution and their gradients"},
    {"name": "autograd-higher-public", "pytest": "python/mojolearn/tests/test_ops_higher.py",
     "what": "public HVP/JVP/VJP and LayerNorm Hessian-vector products"},
    # 2026-10-19 (lane ag-inference-board): the inference board lanes, in miniature
    {"name": "causal-lm-generate-smoke", "probe": "generate", "binding": "byte_lm",
     "what": "CausalLM.load + greedy generate of a tiny byte-LM checkpoint (the causal-lm-generate board lanes)"},
    {"name": "sequential-predict-smoke", "probe": "predict",
     "what": "a tiny Sequential predict_proba inside the board's memory pass (the sequential-predict board lane)"},
    # 2026-10-19 (lane ag-ho-checks): higher-order gradients, docs/plans/AUTOGRAD_HIGHER_ORDER.md section 6
    {"name": "autograd-higher", "source": "autograd/checks/higher_gpu.mojo",
     "what": "the higher-order gradient suite (create_graph) on the vendor's GPU, HIGHER_DIGEST lines"},
    {"name": "autograd-import-ops", "source": "autograd/checks/import_gpu.mojo",
     "what": "native importer operations, including boolean reductions, cumulative sums and multi-index gather"},
    {"name": "autograd-mamba-tape", "source": "mamba/checks/mamba_tape_backward_check.mojo",
     "what": "Mamba-1/2 tape backward against stored gradient digests and the Mamba-1 host oracle"},
    {"name": "autograd-lm-tape", "source": "training/checks/lm_tape_backward_check.mojo",
     "what": "language-model tape backward and updated parameters against stored digests"},
    {"name": "autograd-small-mlp-tape", "source": "training/checks/mlp_tape_step_check.mojo",
     "what": "small-MLP tape backward, two AdamW steps and stored output/state digests"},
    {"name": "autograd-samba-tape", "source": "training/checks/samba_tape_step_check.mojo",
     "what": "Samba whole-stack tape backward, loss, gradients and updated parameter digests"},
    # 2026-10-20 (lane ag-import-docs): model import, docs/plans/MODEL_IMPORT.md
    {"name": "import-model-smoke", "probe": "import",
     "what": "mojolearn.import_model of a tiny ONNX file written in process, then one Program.run"},
    # 2026-10-20 (lane ag-torch-twins): the torch.export source of the importer; RUN or SKIPPED, never silent
    {"name": "import-torch-smoke", "probe": "import-torch",
     "what": "mojolearn.import_model of a tiny torch.export ExportedProgram, then one Program.run (SKIPPED without torch)"},
    # 2026-10-20 (lane ag-real-models): a real pretrained model, tools/real_model_import.py in miniature
    {"name": "real-model-smoke", "probe": "real-model",
     "what": "torchvision squeezenet1_1 (published weights) exported at 1x3x64x64, imported, one Program.run"},
)


def _probe_digest(arrays):
    import hashlib
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def probe_sequential():
    """autograd-sequential-smoke: does Sequential fit + predict_proba start and give finite probabilities."""
    import numpy as np
    import mojolearn
    rng = np.random.default_rng(7)
    X = rng.standard_normal((256, 8), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)       # the fixture's labels (no runtime math)
    model = mojolearn.Sequential().linear(16).relu().linear(1)
    model.compile(loss="bce_with_logits", optimizer="adam", lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8,
                  weight_decay=0.0, schedule="constant", tol=0.0, n_iter_no_change=0)
    model.fit(X, y, epochs=1, batch_size=64, shuffle=True, seed=7)
    proba = np.asarray(model.predict_proba(X))
    if proba.shape != (256, 2) or not np.isfinite(proba).all() or not np.isfinite(model.params_).all():
        print("SMOKE_FAIL sequential proba shape %s finite %s" % (proba.shape, bool(np.isfinite(proba).all())))
        return 1
    print("SMOKE_DIGEST sequential params %s vendor %s mode %s" % (_probe_digest([model.params_]),
                                                                   mojolearn.vendor(), mojolearn.numeric_mode()))
    return 0


def probe_session():
    """autograd-session-smoke: does one recorded ops.Session step start and give finite gradients."""
    import numpy as np
    import mojolearn
    from mojolearn import ops
    rng = np.random.default_rng(7)
    X = rng.standard_normal((32, 8)).astype(np.float32)
    codes = rng.integers(0, 3, (32, 1)).astype(np.float32)
    init = [rng.uniform(-0.3, 0.3, s).astype(np.float32) for s in ((8, 16), (1, 16), (16, 3), (1, 3))]
    with ops.Session(record=True) as s:
        x, c = s.tensor(X), s.tensor(codes)
        ps = [s.tensor(p, requires_grad=True) for p in init]
        logits = s.linear(s.relu(s.linear(x, ps[0], ps[1])), ps[2], ps[3])
        loss = s.cross_entropy(logits, c)
        s.backward(loss)
        value = s.scalar(loss)
        grads = [s.grad(p) for p in ps]
        opt = s.optimizer("sgd", lr=0.05, momentum=0.0, weight_decay=0.0)
        s.step(opt, ps)
        updated = [s.numpy(p) for p in ps]
    finite = np.isfinite(value) and all(np.isfinite(a).all() for a in grads + updated)
    if not finite:
        print("SMOKE_FAIL session loss %r finite %s" % (value, finite))
        return 1
    for initial, grad, actual in zip(init, grads, updated):
        np.testing.assert_allclose(actual, initial - np.float32(0.05) * grad, rtol=2e-6, atol=2e-7)
    print("SMOKE_DIGEST session grads %s loss %.6f vendor %s mode %s"
          % (_probe_digest(grads), value, mojolearn.vendor(), mojolearn.numeric_mode()))
    print("SMOKE_DIGEST session updated %s" % _probe_digest(updated))
    return 0


def probe_program():
    """Exercise the actual board's captured/eager runners on its small correctness fixture."""
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("bench_board_neural", ROOT / "tools/bench_board_neural.py")
    board = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(board)
    data = board.ops_arrays("small", np.random.default_rng(board.SEED), {})
    captured = board.OursProgramRun("program-run", "small", data)
    eager = board.OursProgramRun("program-run", "small", data, eager=True)
    try:
        for _ in range(2):
            captured.call()
            eager.call()
            np.testing.assert_array_equal(captured.out.view(np.uint32), eager.out.view(np.uint32))
            assert np.isfinite(captured.out).all()
        print("SMOKE_DIGEST program output %s" % _probe_digest([captured.out]))
    finally:
        captured.s.close()
        eager.s.close()
    return 0


def probe_generate():
    """causal-lm-generate-smoke: does CausalLM load a byte-LM checkpoint (no final norm) and generate greedily."""
    import json as _json
    import tempfile
    import numpy as np
    import mojolearn
    from mojolearn.models import CausalLM
    from mojolearn.models.safetensors import write_safetensors
    rng = np.random.default_rng(7)
    L, D, H, F, V = 2, 32, 2, 64, 64
    cfg = {"format": "mojolearn.byte-lm", "n_layers": L, "d_model": D, "n_heads": H, "n_kv": H, "head_dim": D // H,
           "intermediate": F, "vocab_size": V, "max_position_embeddings": 64, "rope_theta": 10000.0,
           "rms_norm_eps": 1e-6, "positional_encoding": "rotary", "rope_pairing": "rotate_half", "norm": "rmsnorm",
           "norm_placement": "pre-norm, no final norm", "mlp": "swiglu", "bias": False, "tie_word_embeddings": False}
    shapes = {"embed": (V, D), "lm_head": (V, D)}
    for i in range(L):              # the fixture's tensor names (no runtime math)
        shapes.update({"block%d.norm1_w" % i: (D,), "block%d.norm2_w" % i: (D,), "block%d.w_q" % i: (D, D),
                       "block%d.w_k" % i: (D, D), "block%d.w_v" % i: (D, D), "block%d.w_o" % i: (D, D),
                       "block%d.w_gate" % i: (F, D), "block%d.w_up" % i: (F, D), "block%d.w_down" % i: (D, F)})
    tensors = {}
    for name, shape in shapes.items():
        a = (np.ones(shape, dtype=np.float32) if name.endswith("_w") and len(shape) == 1
             else rng.uniform(-0.1, 0.1, shape).astype(np.float32))
        tensors[name] = ("F32", shape, a.tobytes())
    prompt = rng.integers(0, V, (1, 8)).astype(np.int32)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as fh:
            _json.dump(cfg, fh)
        write_safetensors(os.path.join(d, "model.safetensors"), tensors)
        lm = CausalLM.load(d, weight_format="float32", numeric_profile="fp32_v1")
        out = np.asarray(lm.generate(prompt, 4))
    ok = out.shape == (1, 12) and np.array_equal(out[:, :8], prompt) and out.min() >= 0 and out.max() < V
    if not ok:
        print("SMOKE_FAIL generate shape %s ids %s" % (out.shape, out.tolist()))
        return 1
    print("SMOKE_DIGEST generate ids %s vendor %s mode %s" % (
        _probe_digest([out[:, 8:].astype(np.float32)]), mojolearn.vendor(), mojolearn.numeric_mode()))
    return 0


def probe_predict():
    """sequential-predict-smoke: does a Sequential predict_proba run inside the board's untimed memory pass."""
    import numpy as np
    import mojolearn
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_board_neural as bbn
    rng = np.random.default_rng(7)
    X = rng.standard_normal((256, 8), dtype=np.float32)
    y = (X[:, 0] - 0.5 * X[:, 2] > 0).astype(np.float32)       # the fixture's labels (no runtime math)
    dm = bbn.DeviceMemory()
    before = dm.read()
    model = mojolearn.Sequential().linear(16).relu().linear(1)
    model.compile(loss="bce_with_logits", optimizer="adam", lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8,
                  weight_decay=0.0, schedule="constant", tol=0.0, n_iter_no_change=0)
    model.fit(X, y, epochs=1, batch_size=64, shuffle=True, seed=7)
    resident = bbn._mem_delta(before, dm.read())
    got = {}

    def predict():
        got["proba"] = np.asarray(model.predict_proba(X))
    mem = bbn.memory_pass(dm, resident, predict)
    proba = got.get("proba")
    if proba is None or proba.shape != (256, 2) or not np.isfinite(proba).all():
        print("SMOKE_FAIL predict proba %s" % (None if proba is None else proba.shape,))
        return 1
    print("SMOKE_DIGEST predict proba %s peak_bytes %s process_peak_bytes %s vendor %s mode %s" % (
        _probe_digest([proba]), json.dumps(mem.get("peak_bytes"), sort_keys=True), mem.get("process_peak_bytes"),
        mojolearn.vendor(), mojolearn.numeric_mode()))
    return 0



def probe_import():
    """import-model-smoke: does an ONNX import start and its Program run. The fixture (Conv, Relu,
    MaxPool, Flatten, Gemm) is written by the tests' own encoder, loaded by path because the wheel ships
    no tests, into a temporary directory; no onnx or torch."""
    import importlib.util
    import tempfile
    import numpy as np
    import mojolearn
    spec = importlib.util.spec_from_file_location(
        "smoke_onnx_fixture", ROOT / "python" / "mojolearn" / "tests" / "_onnx_fixture.py")
    fx = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fx      # its dataclasses resolve their module by name
    spec.loader.exec_module(fx)
    with tempfile.TemporaryDirectory() as tmp:
        fixture = fx.write_fixture("conv_relu_maxpool_flatten_gemm", tmp)
        model = mojolearn.import_model(fixture.path, shapes={"x": (4, 3, 8, 8)})
        x = np.random.default_rng(7).standard_normal((4, 3, 8, 8), dtype=np.float32)
        y = np.asarray(model.run(x))
    if y.shape != (4, 10) or not np.isfinite(y).all():
        print("SMOKE_FAIL import output %s finite %s" % (y.shape, bool(np.isfinite(y).all())))
        return 1
    print("SMOKE_DIGEST import logits %s vendor %s mode %s"
          % (_probe_digest([y]), mojolearn.vendor(), mojolearn.numeric_mode()))
    return 0


def _load_onnx_fixture():
    """The tests' fixture module, loaded by path (the wheel ships no tests)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "smoke_onnx_fixture", ROOT / "python" / "mojolearn" / "tests" / "_onnx_fixture.py")
    fx = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fx      # its dataclasses resolve their module by name
    spec.loader.exec_module(fx)
    return fx


def probe_import_torch():
    """import-torch-smoke: does a torch.export import start and its Program run. The module is the
    shared linear_layernorm_linear fixture (x [4, 6] -> Linear 5 -> LayerNorm -> Linear 3) rebuilt as
    an nn.Sequential holding the fixture's weights, exported at batch 4. Prints RUN or SKIPPED, always."""
    import tempfile
    import numpy as np
    try:
        import torch
        import torch.export  # noqa: F401
    except Exception as exc:  # torch absent, or too old for torch.export
        print("SKIPPED no torch (%s: %s)" % (type(exc).__name__, str(exc)[:160]), flush=True)
        if os.environ.get("MOJOLEARN_REQUIRE_TORCH") == "1":
            print("SMOKE_FAIL import-torch MOJOLEARN_REQUIRE_TORCH=1 and torch does not import")
            return 1
        return 0
    import mojolearn
    print("RUN torch %s" % torch.__version__, flush=True)
    fx = _load_onnx_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        f = fx.write_fixture("linear_layernorm_linear", tmp)
    I = {k: torch.frombuffer(bytearray(v.data), dtype=torch.float32).reshape(v.shape).clone()
         for k, v in f.initializers.items()}
    fc1, ln, fc2 = torch.nn.Linear(6, 5), torch.nn.LayerNorm(5, eps=1e-5), torch.nn.Linear(5, 3)
    with torch.no_grad():
        fc1.weight.copy_(I["w1"]); fc1.bias.copy_(I["b1"])
        ln.weight.copy_(I["ln_scale"]); ln.bias.copy_(I["ln_bias"])
        fc2.weight.copy_(I["w2"]); fc2.bias.copy_(I["b2"])
    module = torch.nn.Sequential(fc1, ln, fc2).eval()
    x = np.random.default_rng(7).standard_normal((4, 6), dtype=np.float32)
    ep = torch.export.export(module, (torch.from_numpy(x),))
    model = mojolearn.import_model(ep)
    y = model.run(x)
    y = np.asarray(y[0] if isinstance(y, (list, tuple)) else y)
    if y.shape != (4, 3) or not np.isfinite(y).all():
        print("SMOKE_FAIL import-torch output %s finite %s" % (y.shape, bool(np.isfinite(y).all())))
        return 1
    print("SMOKE_DIGEST import-torch logits %s vendor %s mode %s"
          % (_probe_digest([y]), mojolearn.vendor(), mojolearn.numeric_mode()))
    return 0


def probe_real_model():
    """real-model-smoke: does a real pretrained model start through the importer. torchvision's
    squeezenet1_1 with its published weights, exported by tools/real_model_import.py's own source step
    at a tiny 1 x 3 x 64 x 64 input, imported and run once; a refusal prints its nodes and counts as
    started (the import ran and named what it cannot run)."""
    import importlib.util
    import numpy as np
    import mojolearn
    from mojolearn.importers import UnsupportedModel
    spec = importlib.util.spec_from_file_location("real_model_import", ROOT / "tools" / "real_model_import.py")
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        print("SMOKE_FAIL real-model torch or torchvision is not installed (%s)" % exc)
        return 1
    model = dict(rm.MODELS["squeezenet1_1"], input=(lambda b: (1, 3, 64, 64), "image"))
    ep, shapes, arrays, _ = rm.source_torchvision(model, 1, 7, {})
    try:
        prog = mojolearn.import_model(ep, shapes=shapes)
    except UnsupportedModel as e:
        for name, op, why in e.problems:
            print("SMOKE_REFUSED real-model squeezenet1_1 %s %s: %s" % (op, name, why))
        return 0
    y = np.asarray(prog.run(*arrays))
    if y.shape != (1, 1000) or not np.isfinite(y).all():
        print("SMOKE_FAIL real-model output %s finite %s" % (y.shape, bool(np.isfinite(y).all())))
        return 1
    print("SMOKE_DIGEST real-model squeezenet1_1 %s vendor %s mode %s"
          % (_probe_digest([y]), mojolearn.vendor(), mojolearn.numeric_mode()))
    return 0


PROBES = {"sequential": probe_sequential, "session": probe_session, "program": probe_program, "generate": probe_generate,
          "predict": probe_predict, "import": probe_import, "import-torch": probe_import_torch,
          "real-model": probe_real_model}


def _probe_status(log, offset=0):
    """A refusal or skip from this invocation, never from an earlier appended log."""
    try:
        with open(log, "rb") as fh:
            fh.seek(offset)
            text = fh.read().decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("SMOKE_REFUSED "):
                return "REFUSED", line[:300]
            if line.startswith("SKIPPED "):
                return "SKIPPED", line[:300]
    except OSError:
        pass
    return None


def check_binding(vendor, family="x_sequence"):
    """Read the binary actually used by the probe, not the package-wide vendor selector."""
    import hashlib
    import mojolearn
    from mojolearn import _backend
    b = _backend.binding("_mojolearn_" + family, "identical")
    info = {"package": mojolearn.__file__, "version": getattr(mojolearn, "__version__", None),
            "binding": getattr(b, "__file__", None), "family": family,
            "vendor": getattr(b, family + "_vendor")(),
            "numeric_mode": getattr(b, family + "_numeric_mode")()}
    if info["vendor"] != BINDING_VENDOR[vendor] or info["numeric_mode"] != 1:
        raise RuntimeError("wrong smoke binding: requested %s/identical, read %s" % (vendor, info))
    if info["binding"]:
        with open(info["binding"], "rb") as stream:
            info["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return info


def python_entry(entry, vendor, binding_report, junit, source_package=None):
    """One isolated child process, including binding readback before pytest can skip missing APIs."""
    if source_package:
        sys.path.insert(0, str(Path(source_package).resolve()))
    info = check_binding(vendor, entry.get("binding", "x_sequence"))
    Path(binding_report).write_text(json.dumps(info, indent=2) + "\n")
    print("SMOKE_BINDING " + json.dumps(info, sort_keys=True), flush=True)
    if "probe" in entry:
        return PROBES[entry["probe"]]()
    import pytest
    return int(pytest.main(["-q", "-s", "-x", "-p", "no:cacheprovider", "--import-mode=importlib",
                           "--junitxml=" + str(junit), str(ROOT / entry["pytest"])]))


def pytest_result(path):
    root = ET.parse(path).getroot()
    counts = {key: sum(int(s.get(key, 0)) for s in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    if counts["failures"] or counts["errors"] or not counts["tests"]:
        status = "failed"
    elif counts["skipped"] == counts["tests"]:
        status = "SKIPPED"
    elif counts["skipped"]:
        status = "PARTIAL"
    else:
        status = "ok"
    return status, counts


def check_mlp_digests(log, offset=0):
    """A completed MLP binary emits hashes; the controller must actually compare them."""
    reference = ROOT / "autograd/checks/mlp_cases_digests.tsv"
    expected = {int(case): (int(epochs), digest) for line in reference.read_text().splitlines()
                if line and not line.startswith(("#", "case"))
                for case, epochs, digest in [line.split()]}
    epochs, digests = {}, {}
    with open(log, "rb") as stream:
        stream.seek(offset)
        for line in stream.read().decode(errors="replace").splitlines():
            fields = line.split()
            if fields[:1] == ["CASE"]:
                case = int(fields[1])
                if case in epochs:
                    raise ValueError("duplicate MLP case %s" % case)
                epochs[case] = int(fields[3])
            elif fields[:1] == ["DIGEST"]:
                case = int(fields[1])
                if case in digests:
                    raise ValueError("duplicate MLP digest %s" % case)
                digests[case] = fields[2]
    if len(expected) != 48 or set(epochs) != set(expected) or set(digests) != set(expected):
        raise ValueError("incomplete MLP coverage: expected 48 cases and digests, got %d/%d"
                         % (len(epochs), len(digests)))
    mismatches = [case for case, want in expected.items() if (epochs[case], digests[case]) != want]
    if mismatches:
        raise ValueError("MLP stored-digest mismatch in cases %s" % mismatches)
    return {"cases": len(expected), "reference": str(reference)}


def binary(bin_dir, entry, vendor):
    return Path(bin_dir) / ("%s-%s" % (entry["name"], vendor))


def build_commands(vendor, mojo, bin_dir, target=None):
    """The `mojo build` line of every check entry for `vendor` (IDENTICAL, the vendor's column define)."""
    out = []
    for e in ENTRIES:
        if "source" not in e:
            continue
        cmd = [mojo, "build", "-j", "1", "-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"] + COLUMN_DEFINE[vendor]
        if target:
            cmd += ["--target-accelerator", target]
        cmd += ["-I", str(ROOT), str(ROOT / e["source"]), "-o", str(binary(bin_dir, e, vendor))]
        out.append(cmd)
    return out


def _tail(path, n=6, width=300):
    try:
        lines = [l for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return []
    return [l[:width] for l in lines[-n:]]


def start(cmd, log, timeout, env=None, cwd=None):
    """(status, rc, seconds): ok / STARTED / failed for one bounded start; the process group is
    killed at the timer."""
    t0 = time.time()
    with open(log, "ab", buffering=0) as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                env=env, cwd=cwd)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    continue
    seconds = round(time.time() - t0, 2)
    if rc is None:
        return "STARTED", None, seconds
    return ("ok" if rc == 0 else "failed"), rc, seconds


def run(vendor, bin_dir, python, out, timeout=TIMEOUT_S, *, entries=None, source_package=None):
    selected = list(ENTRIES) if entries is None else [e for e in ENTRIES if e["name"] in entries]
    unknown = set(entries or ()) - {e["name"] for e in ENTRIES}
    if unknown or not selected:
        raise ValueError("empty or unknown smoke selection: %s" % sorted(unknown))
    logs = Path(out).resolve().with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "tmp").mkdir(exist_ok=True)
    rows = []
    for e in selected:
        log = logs / ("%s-%s.log" % (e["name"], vendor))
        row = {"name": e["name"], "vendor": vendor, "what": e["what"], "timeout_s": timeout, "log": str(log)}
        if "source" in e:
            exe = binary(bin_dir, e, vendor).resolve()
            if not exe.is_file():
                row.update(status="not built", detail="no %s (tools/autograd_smoke.py build-commands)" % exe)
                rows.append(row)
                continue
            cmd = [str(exe)]
            env = None
        else:
            binding_report = log.with_suffix(".binding.json")
            junit = log.with_suffix(".xml")
            # Stale reports must not turn an incomplete new process into a pass.
            binding_report.unlink(missing_ok=True)
            junit.unlink(missing_ok=True)
            cmd = [python, "-I", str(Path(__file__).resolve()), "entry", e["name"], "--vendor", vendor,
                   "--binding-report", str(binding_report), "--junit", str(junit)]
            if source_package:
                cmd += ["--source-package", str(Path(source_package).resolve())]
            env = dict(os.environ, MOJOLEARN_AUTOGRAD_SMOKE_OUT=str(logs / ("fit-digests-%s.json" % vendor)))
            env.pop("PYTHONPATH", None)
            env.pop("MOJOLEARN_VENDOR_FORCE", None)
            env.pop("MOJOLEARN_VENDOR", None)
            if vendor != "metal":
                env["MOJOLEARN_VENDOR"] = BINDING_VENDOR[vendor]
            env.update(MOJOLEARN_NUMERIC_MODE="identical", TMPDIR=str(logs / "tmp"))
            row.update(binding_report=str(binding_report))
            if "pytest" in e:
                row["junit"] = str(junit)
        offset = log.stat().st_size if log.exists() else 0
        status, rc, seconds = start(cmd, log, timeout, env=env, cwd=str(ROOT))
        row.update(command=cmd, status=status, rc=rc, seconds=seconds)
        if status == "ok":
            try:
                if "source" not in e:
                    row["binding"] = json.loads(binding_report.read_text())
                if "probe" in e:
                    result = _probe_status(log, offset)
                    if result:
                        status, row["detail"] = result
                elif "pytest" in e:
                    status, row["tests"] = pytest_result(junit)
                elif e["name"] == "autograd-mlp-identity":
                    row["identity"] = check_mlp_digests(log, offset)
            except (OSError, ValueError, ET.ParseError, IndexError) as exc:
                status, row["detail"] = "failed", str(exc)
            row["status"] = status
        if status == "failed":
            row["tail"] = _tail(log)
        rows.append(row)
        print("AUTOGRAD-SMOKE vendor=%s entry=%s status=%s seconds=%s" % (vendor, e["name"], status, seconds),
              flush=True)
    doc = {"schema": "mojolearn-autograd-smoke/1", "vendor": vendor, "rows": rows,
           "coverage": "full registry" if entries is None else "selected entries",
           "package_scope": "source" if source_package else "installed",
           "source_package": str(Path(source_package).resolve()) if source_package else None,
           "counts": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})}}
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, out)
    bad = [r for r in rows if r["status"] in ("failed", "not built")
           or ("junit" in r and r["status"] in ("SKIPPED", "PARTIAL"))]
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-commands", help="print the check binaries' build lines (never builds)")
    b.add_argument("--vendor", required=True, choices=VENDORS)
    b.add_argument("--mojo", default="mojo")
    b.add_argument("--bin-dir", required=True)
    b.add_argument("--target-accelerator", default=None, help="e.g. sm_89 or gfx942 for a cross build")
    r = sub.add_parser("run", help="start every entry once with its 5 s timer")
    r.add_argument("--vendor", required=True, choices=VENDORS)
    r.add_argument("--bin-dir", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=TIMEOUT_S)
    r.add_argument("--entries", help="comma-separated subset; omitted means the whole registry")
    r.add_argument("--source-package", help="explicit source package parent (e.g. repo/python); default: installed")
    sub.add_parser("list", help="the entries, one line each")
    pr = sub.add_parser("probe", help="one installed-bindings probe (what the probe entries start)")
    pr.add_argument("kind", choices=sorted(PROBES))
    child = sub.add_parser("entry", help="internal isolated Python entry with compiled-binding readback")
    child.add_argument("name", choices=[e["name"] for e in ENTRIES if "source" not in e])
    child.add_argument("--vendor", choices=VENDORS, required=True)
    child.add_argument("--binding-report", required=True)
    child.add_argument("--junit", required=True)
    child.add_argument("--source-package")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for e in ENTRIES:
            print("%-26s %s" % (e["name"], e.get("source") or e.get("pytest") or "probe " + e["probe"]))
        return 0
    if a.cmd == "probe":
        return PROBES[a.kind]()
    if a.cmd == "entry":
        entry = next(e for e in ENTRIES if e["name"] == a.name)
        return python_entry(entry, a.vendor, a.binding_report, a.junit, a.source_package)
    if a.cmd == "build-commands":
        for cmd in build_commands(a.vendor, a.mojo, a.bin_dir, a.target_accelerator):
            print(" ".join(cmd))
        return 0
    return run(a.vendor, a.bin_dir, a.python, a.out, a.timeout,
               entries=a.entries.split(",") if a.entries is not None else None, source_package=a.source_package)


if __name__ == "__main__":
    sys.exit(main())
