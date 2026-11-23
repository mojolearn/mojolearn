# SPDX-License-Identifier: Apache-2.0
"""Census fixtures (tools/ops/python_frames_census.py, fb4-census 2026-11-17): the language and neural models.

The byte-LM trainers (LanguageModelTrainer and its parallel, pooled and offloaded schedules on one CPU device), the
token corpus (lm_corpus), the BPE tokenizer, the Hugging Face checkpoint reader (models.Checkpoint) and layer-owned
model (models.ParallelCausalLM), Samba, the Mamba and transformer decode sessions and the forward-only inference
classes. Each fixture is the smallest call that reaches its members: a d_model 4 one-layer byte LM (verify's own
`_verify_api_training.tiny_lm` shape), a 16 KiB synthetic corpus with a 300-rank vocabulary (the `tokenized-corpus`
harness lane's), verify's 32-wide block factory, files written into cx.tmp. Testing tools, never product: only the
standard library is imported at load; numpy and mojolearn inside each fixture.
"""


def _mod(name):
    """A mojolearn submodule by import (lazy families are not all attributes of the package before first use)."""
    import importlib
    return importlib.import_module("mojolearn." + name)


def _tiny_lm(ml, vocab_size=4, data_schedule=None):
    """verify's tiny_lm (python/mojolearn/_verify_api_training.py) with a chosen vocabulary: (trainer, shape, ids)."""
    import numpy as np
    shape = ml.LanguageModelConfig(batch=1, length=2, d_model=4, n_heads=1, n_kv=1, head_dim=4, intermediate=4,
                                   n_layers=1, vocab_size=vocab_size)
    named = {e["name"]: ((np.arange(e["size"], dtype=np.float32).reshape(e["shape"]) % 7 - 3) / 32)
             for e in ml.LanguageModelTrainer.parameter_registry(shape)}
    for i, (key, value) in enumerate(named.items()):
        if "norm" in key:
            value[:] = 1 + i / 32
    model = ml.LanguageModelTrainer(named, shape=shape, data_schedule=data_schedule or {"fixture": "census-v1"},
                                    lr=.125)
    return model, shape, np.array([[0, 1, 2]], np.int32)


def _corpus(ml, cx):
    """A 16 KiB synthetic text corpus prepared with a 300-rank vocabulary (the tokenized-corpus lane's settings)."""
    words = ("the model reads a byte stream and learns which token follows which; "
             "a small corpus is enough for a census fixture, never for a measurement.\n").encode("ascii")
    path = cx.tmp / "corpus.txt"
    path.write_bytes((words * (16384 // len(words) + 1))[:16384])
    return _mod("lm_corpus").prepare(str(path), cache_dir=str(cx.tmp / "lm-cache"), vocab_size=300,
                                vocab_sample_bytes=16384, document_bytes=4096)


# ---------------------------------------------------------------- byte LM trainers

def language_model_trainer(ml, cx):
    """LanguageModelTrainer.next_bytes / train_steps / parameter_registry (a static method called on an instance)."""
    import numpy as np
    model, shape, ids = _tiny_lm(ml)
    try:
        model.parameter_registry(shape)
        model.train_steps(ids[None, :, :])
        model.next_bytes(np.array([[0, 1]], np.int32))
    finally:
        model.close()


def language_model_inference_loss(ml, cx):
    source, shape, ids = _tiny_lm(ml)
    try:
        params = source.state_dict()["parameters"]
    finally:
        source.close()
    ml.LanguageModelInference(params, shape=shape).loss(ids)


def _schedule_fixture(cls_name):
    def fixture(ml, cx):
        """export_gradients after one step, the step_ property by access, from_checkpoint of its own checkpoint."""
        cls = getattr(ml, cls_name)
        member = f"mojolearn.{cls_name}"
        source, _, ids = _tiny_lm(ml)
        try:
            state = source.state_dict()
        finally:
            source.close()
        model = cls(state, devices=(0,), logical_shards=1)
        try:
            model.train_step([ids])
            cx.access(model, "step_", member + ".step_")
            model.export_gradients()
            other = cls.from_checkpoint(model.checkpoint(), devices=(0,))
            other.close()
        finally:
            model.close()
    return fixture


def _train_step_from_fixture(cls_name):
    def fixture(ml, cx):
        """train_step_from on batches of a prepared corpus whose vocabulary is the trainer's."""
        cls = getattr(ml, cls_name)
        corpus = _corpus(ml, cx)
        batches = corpus.batches(1, 2)
        source, _, _ = _tiny_lm(ml, vocab_size=corpus.n_vocab, data_schedule=batches.data_schedule())
        try:
            state = source.state_dict()
        finally:
            source.close()
        model = cls(state, devices=(0,), logical_shards=1)
        try:
            model.train_step_from(batches, [0])
        finally:
            model.close()
    return fixture


# ---------------------------------------------------------------- corpus, tokenizer, checkpoints

def token_corpus(ml, cx):
    """TokenizedCorpus and TokenBatches built directly from a prepared corpus's files; prefetch consumed."""
    LC = _mod("lm_corpus")
    prepared = _corpus(ml, cx)
    corpus = LC.TokenizedCorpus(prepared.tokens_dir, prepared.vocabulary_path)
    batches = LC.TokenBatches(corpus.tokens_dir, 2, 16)
    for _ in batches.prefetch(0, 2):
        pass


def bpe_tokenizer(ml, cx):
    """BpeTokenizer(ranks_file): the 256 single-byte ranks, one `rank<TAB>hex` line each."""
    path = cx.tmp / "ranks.tsv"
    path.write_text("".join(f"{i}\t{i:02x}\n" for i in range(256)))
    ml.BpeTokenizer(str(path)).encode_bytes(b"census")


def safetensors_checkpoint(ml, cx):
    """models.Checkpoint(files, weight_map, root) over one safetensors file written here."""
    import json
    import struct
    data = struct.pack("<4f", .25, -.5, .75, 1.0)
    header = json.dumps({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, len(data)]}}).encode()
    header += b" " * (-len(header) % 8)
    path = cx.tmp / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    ckpt = _mod("models").Checkpoint([str(path)], {"w": str(path)}, str(cx.tmp))
    try:
        ckpt.read("w")
    finally:
        ckpt.close()


def parallel_causal_lm(ml, cx):
    """models.ParallelCausalLM(plan, weights, layer_devices=...): the two-layer llama fixture, one owner per layer
    (two CPU worker owners, as python/mojolearn/tests/test_parallel_causal_lm.py runs it on the CPU)."""
    from mojolearn._causal_lm_fixtures import family_fixture
    cfg, tensors = family_fixture("llama", False)
    models = _mod("models")
    model = models.ParallelCausalLM(models.plan_for(cfg), tensors, layer_devices=(0, 1))
    model.close()


# ---------------------------------------------------------------- Samba

def _samba_parts(ml):
    import numpy as np
    config = ml.SambaConfig(4, 32, ["attention"], n_heads=1, intermediate=32)
    weights = {key: (np.ones(shape, np.float32) if "norm" in key else
                     (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) % 7 - 3) / 32)
               for key, shape in config.registry()}
    return config, weights, np.array([[0, 1]], np.int32), np.array([[1, 2]], np.int32)


def samba_stack(ml, cx):
    config, weights, ids, targets = _samba_parts(ml)
    stack = ml.SambaStack(config, weights=weights)
    stack.loss(ids, targets)
    stack.loss_and_grads(ids, targets)


def samba_inference(ml, cx):
    """allocate_state and step; loss and train_step are its documented NotImplementedError refusals."""
    import numpy as np
    config, weights, ids, targets = _samba_parts(ml)
    model = ml.SambaInference(config, weights)
    state = model.allocate_state(1, 4)
    model.step(np.array([0], np.int32), state)
    for name in ("loss", "train_step"):
        try:
            getattr(model, name)(ids, targets)
        except NotImplementedError:
            pass


# ---------------------------------------------------------------- inference blocks, MLP

def _block_backward(public):
    def fixture(ml, cx):
        """backward: forward-only, its documented NotImplementedError refusal (the member still runs)."""
        import numpy as np
        from mojolearn._verify_api_metadata import factory
        block = factory(public)
        x = np.arange(64, dtype=np.float32).reshape(1, 2, 32) / 128 - .25
        try:
            block.backward(x, x)
        except NotImplementedError:
            pass
    return fixture


def mlp_inference(ml, cx):
    """MLPInference(weight1, bias1, weight2, bias2): the 8 -> 16 -> 3 SmallMLPTrainer shapes."""
    import numpy as np
    w = [(np.arange(n, dtype=np.float32) % 7 - 3).reshape(shape) / 32
         for n, shape in ((128, (16, 8)), (16, (16,)), (48, (3, 16)), (3, (3,)))]
    ml.MLPInference(*w).predict_logits(np.arange(16, dtype=np.float32).reshape(2, 8) / 32)


# ---------------------------------------------------------------- decode sessions

def _mamba_session(n, finalizer):
    def fixture(ml, cx):
        """Mamba<n>DecodeSession(block, state) built directly; with a Python finalizer, dropped by cx.delete."""
        from mojolearn._verify_api_metadata import factory
        block = factory(f"mojolearn.Mamba{n}Block")
        cls = getattr(_mod("mamba"), f"Mamba{n}DecodeSession")
        box = {"session": cls(block, block.allocate_state(1))}
        if finalizer:
            cx.delete(box, "session", f"mojolearn.mamba.Mamba{n}DecodeSession.__del__")
        else:
            box["session"].close()
    return fixture


def transformer_decode_session(ml, cx):
    from mojolearn._verify_api_metadata import factory
    block = factory("mojolearn.TransformerBlock")
    session = _mod("transformer").TransformerDecodeSession(block, block.allocate_state(1, 3))
    session.close()


#: one trainer class; its pooled and offloaded placements are its `schedule` option
_TRAINERS = ("ParallelByteLanguageModelTrainer",)

FIXTURES = {
    "language_model_trainer": (language_model_trainer, [
        "mojolearn.LanguageModelTrainer.next_bytes", "mojolearn.LanguageModelTrainer.parameter_registry",
        "mojolearn.LanguageModelTrainer.train_steps"]),
    "language_model_inference_loss": (language_model_inference_loss, ["mojolearn.LanguageModelInference.loss"]),
    "token_corpus": (token_corpus, [
        "mojolearn.lm_corpus.TokenizedCorpus.__init__", "mojolearn.lm_corpus.TokenBatches.__init__",
        "mojolearn.lm_corpus.TokenBatches.prefetch"]),
    "bpe_tokenizer": (bpe_tokenizer, ["mojolearn.BpeTokenizer.__init__"]),
    "safetensors_checkpoint": (safetensors_checkpoint, ["mojolearn.models.Checkpoint.__init__"]),
    "parallel_causal_lm": (parallel_causal_lm, ["mojolearn.models.ParallelCausalLM.__init__"]),
    "samba_stack": (samba_stack, ["mojolearn.SambaStack.loss", "mojolearn.SambaStack.loss_and_grads"]),
    "samba_inference": (samba_inference, [
        "mojolearn.SambaInference.allocate_state", "mojolearn.SambaInference.loss", "mojolearn.SambaInference.step",
        "mojolearn.SambaInference.train_step"]),
    "mlp_inference": (mlp_inference, ["mojolearn.MLPInference.__init__"]),
    "transformer_decode_session": (transformer_decode_session,
                                   ["mojolearn.transformer.TransformerDecodeSession.__init__"]),
    "mamba1_decode_session": (_mamba_session(1, False), ["mojolearn.mamba.Mamba1DecodeSession.__init__"]),
    "mamba2_decode_session": (_mamba_session(2, True), [
        "mojolearn.mamba.Mamba2DecodeSession.__init__", "mojolearn.mamba.Mamba2DecodeSession.__del__"]),
    "mamba3_decode_session": (_mamba_session(3, True), [
        "mojolearn.mamba.Mamba3DecodeSession.__init__", "mojolearn.mamba.Mamba3DecodeSession.__del__"]),
}
for _name in ("Mamba1BlockInference", "Mamba2BlockInference", "Mamba3BlockInference", "TransformerBlockInference"):
    FIXTURES[f"{_name.lower()}_backward"] = (_block_backward(f"mojolearn.{_name}"), [f"mojolearn.{_name}.backward"])
for _cls in _TRAINERS:
    FIXTURES[f"{_cls.lower()}_schedule"] = (_schedule_fixture(_cls), [
        f"mojolearn.{_cls}.export_gradients", f"mojolearn.{_cls}.step_", f"mojolearn.{_cls}.from_checkpoint"], "gpu")
    FIXTURES[f"{_cls.lower()}_train_step_from"] = (_train_step_from_fixture(_cls),
                                                   [f"mojolearn.{_cls}.train_step_from"])
del _name, _cls

#: the cooperative byte-LM driver (every schedule) has no CPU implementation: on the CPU wheel their
#: train_step and checkpoint refuse ("no CPU implementation of the cooperative multi-GPU driver ..."), so the members
#: that need a completed step or a checkpoint first (smoke b620cb2e0b: UNCOVERED) run only in the "gpu" fixture above
NEEDS_GPU = {f"mojolearn.{_c}.{_m}": "the cooperative multi-GPU byte-LM driver has no CPU implementation: train_step "
             "and checkpoint refuse on a CPU-only install, so no step or checkpoint exists to export or restore"
             for _c in _TRAINERS for _m in ("export_gradients", "step_", "from_checkpoint")}
WHY_NOT = {}
