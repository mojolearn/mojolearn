"""One-layer, two-token loaded-model and continuation fixtures; no downloads."""
__mojolearn_user_boundary__ = True

import tempfile
from pathlib import Path
from ._verify_api import resolve, encode


def catalog_graph_digests(models):
    """Check the catalog's native-graph route using the MNIST entry's pinned input and one ReLU."""
    import hashlib
    import numpy as np
    from ._verify_onnx import field as F, value_info
    from .importers.onnx import read

    x = np.random.default_rng(0).standard_normal((1, 1, 28, 28), dtype=np.float32)
    node = F(1, 2, "x") + F(2, 2, "y") + F(4, 2, "Relu")
    graph = (F(1, 2, node) + F(2, 2, "catalog-check")
             + F(11, 2, value_info("x", x.shape)) + F(12, 2, value_info("y", x.shape)))
    model = F(1, 0, 8) + F(8, 2, F(2, 0, 17)) + F(7, 2, graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relu.onnx"
        path.write_bytes(model)
        report = models.verify_catalog_entry("mnist-12", source=read(str(path)), inputs=[x],
                                             numeric_mode="identical")
    expected = hashlib.sha256(np.maximum(x, np.float32(0)).tobytes()).hexdigest()[:16]
    if report["digest"] != expected or report["input_digest"] != hashlib.sha256(x.tobytes()).hexdigest()[:16]:
        raise AssertionError("catalog input or supplied graph output differs from its bytes")
    return {"input_digest": report["input_digest"], "digest": report["digest"]}


def applicable(public,method):return public.startswith('mojolearn.models.')


def model_checks(recorder,public,methods):
    import numpy as np
    from . import models
    from ._causal_lm_fixtures import _llama_config,_llama_tensors,_write_checkpoint
    name=public.rsplit('.',1)[-1]
    def check(method):
        if name=='catalog':
            entries=models.catalog();assert isinstance(entries,(dict,list))
            return entries
        if name=='verify_catalog_entry':
            try:
                models.verify_catalog_entry('__verifier_unknown_model__')
            except KeyError:
                return catalog_graph_digests(models)
            raise AssertionError('unknown catalog name was admitted')
        if name=='Tokenizer':
            import unicodedata
            assert models.Tokenizer.unicode_version==unicodedata.unidata_version
            return {'unicode_version_matches_runtime':True}
        cfg=_llama_config(num_hidden_layers=1,vocab_size=256,max_position_embeddings=4)
        tensors=_llama_tensors(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint(tmp,cfg,tensors,shards=1)
            if name=='HFConfig':
                m=models.HFConfig.from_json(tmp)
                if method=='__contains__':assert 'hidden_size' in m and 'missing_verifier_key' not in m;return True
                return m['hidden_size']
            if name in ('Checkpoint','SafetensorsFile'):
                m=models.Checkpoint.open(tmp) if name=='Checkpoint' else models.SafetensorsFile(str(Path(tmp)/'model.safetensors'))
                assert 'model.embed_tokens.weight' in m and 'missing_verifier_tensor' not in m
                return {'present':True,'missing':False}
            # No numeric profile named: the inference default, fixed15_v1 since 2026-11-25
            # (docs/plans/NUMERIC_PROFILE_PARTS.md); these endpoints' bits changed then, RECORD OWED.
            if name=='ParallelCausalLM':
                m=models.ParallelCausalLM.load(tmp,layer_devices=(0,))
            else:
                m=resolve('mojolearn.models.CausalLM').load(tmp)
            try:
                ids=np.array([[1,2]],np.int32)
                if name=='GenerationStream':
                    stream=m.stream(ids,max_new_tokens=1)
                    stream.read(1)
                    return getattr(stream,method)()
                if name=='CausalLMState':
                    state=m.allocate_state(1,4)
                    clone=models.CausalLMState(state.batch_size,state.max_tokens,state.layers,owner=m)
                    return m.forward(ids,clone)
                if method=='__call__':return m(ids)
                if method=='blocks:get':assert len(m.blocks)==1;return m.forward(ids)
                state=m.allocate_state(1,4)
                if method=='allocate_state':return m.forward(ids,state)
                if method=='reset_state':
                    first=m.forward(ids,state);m.reset_state(state)
                    second=m.forward(ids,state);assert encode(first)==encode(second)
                    return second
                if method=='step':m.forward(ids[:,:1],state);return m.step(ids[:,1:],state)
                if method=='generate':return m.generate(ids,1)
                if method in ('stream','resume_generation'):
                    stream=m.stream(ids,max_new_tokens=1)
                    if method=='resume_generation':stream=m.resume_generation(stream.state_dict())
                    return stream.read(1)
                from .models.tokenizer import Tokenizer
                tok=Tokenizer([bytes([i]) for i in range(256)],[],pattern='gpt2')
                if method=='generate_text':return m.generate_text('ab',1,tokenizer=tok)
                return m.stream_text('ab',1,tokenizer=tok).read(1)
            finally:
                if name=='ParallelCausalLM':m.close()
    for method in methods:
        recorder.check(public+'.'+method,lambda method=method:check(method))
