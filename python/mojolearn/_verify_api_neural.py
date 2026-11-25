"""Bounded neural API fixtures: two examples/tokens, at most one update."""
__mojolearn_user_boundary__ = True

from ._verify_device import unwrap as _vd_unwrap  # the class behind the device view (BoundClass)
import inspect
import tempfile
from pathlib import Path
from ._verify_api import resolve, encode
from ._verify_api_metadata import factory

NAMES=frozenset(('Sequential','Embedding','LayerNorm','MoEBlock','Conv1d','Conv2d',
    'AdaptiveAvgPool2d','AdaptiveMaxPool2d','AvgPool1d','AvgPool2d','MaxPool1d','MaxPool2d',
    'BasicBlock','BatchNorm1d','BatchNorm2d','Dropout2d','TransformerState',
    'Mamba1State','Mamba2State','Mamba3State','SambaConfig','SambaStack','SambaInference'))


def applicable(public, method):
    name=public.rsplit('.',1)[-1]
    return (name in NAMES or name.startswith(('Mamba1Block','Mamba2Block','Mamba3Block','TransformerBlock'))) and method not in ('get_params','set_params','__getattribute__')


def _device_values(device, expected):
    import numpy as np
    actual=np.asarray(device)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.tobytes()==expected.tobytes(), 'device transfer changed input bits'
    return actual


# The Sequential layers added 2026-11-22 (lanes seq-blocks*, neural-blocks*): one bounded fixture per layer
# method, (sample shape, build). Each builds a two-sample model with the layer, ends it in flatten + linear(1)
# (lm_head ends itself), fits one epoch of one full unshuffled batch with SGD and returns the predictions of the
# fitted model, so the record holds the layer's forward, backward and update in one output.
# lane layer-tables (2026-11-23): this is also the constructor table of the boundary test
# (python/mojolearn/tests/test_sequential_layers_boundary.py: fit one step, state_dict -> load_state_dict,
# save_checkpoint -> from_checkpoint, export -> load). tools/block_tables_check.py refuses a build whose layer code
# has no entry here: a lane that adds a code adds its minimal constructor here.
SEQUENTIAL_LAYER_FIXTURES={
    'linear':((4,),lambda m,S:m.linear(3)),
    'layer_norm':((4,),lambda m,S:m.linear(4).layer_norm()),
    'conv2d':((1,2,2),lambda m,S:m.conv2d(1,1)),
    'conv1d':((2,4),lambda m,S:m.conv1d(2,3,padding=1)),
    'max_pool1d':((2,4),lambda m,S:m.max_pool1d(2)),
    'avg_pool1d':((2,4),lambda m,S:m.avg_pool1d(2)),
    'adaptive_avg_pool1d':((2,4),lambda m,S:m.adaptive_avg_pool1d(2)),
    'branch':((4,),lambda m,S:m.linear(3).tanh().branch(0)),
    'concat':((4,),lambda m,S:m.linear(2).linear(3).concat([0,1])),
    'residual_block':((4,),lambda m,S:m.residual_block(S().linear(4).tanh())),
    'parallel':((2,4),lambda m,S:m.parallel(S().conv1d(2,1),S().conv1d(3,3,padding=1))),
    'gated':((4,),lambda m,S:m.gated(S().linear(4).silu(),S().linear(4))),
    'silu':((4,),lambda m,S:m.linear(4).silu()),
    'gelu':((4,),lambda m,S:m.linear(4).gelu()),
    'softplus':((4,),lambda m,S:m.linear(4).softplus()),
    'rms_norm':((4,),lambda m,S:m.linear(4).rms_norm()),
    'rotary':((2,4),lambda m,S:m.rotary(4)),
    'multiply':((4,),lambda m,S:m.linear(4).multiply(-1)),
    'narrow':((2,4),lambda m,S:m.narrow(1,1,2)),
    'embedding':((3,),lambda m,S:m.embedding(5,2)),
    'attention':((4,8),lambda m,S:m.attention(2)),
    'lstm':((3,2),lambda m,S:m.lstm(3)),
    'gru':((3,2),lambda m,S:m.gru(3)),
    'rnn':((3,2),lambda m,S:m.rnn(3)),
    'batch_norm1d':((2,4),lambda m,S:m.batch_norm1d(2)),
    'batch_norm2d':((1,2,2),lambda m,S:m.batch_norm2d(1)),
    'selective_scan':((4,3),lambda m,S:m.linear(12).softplus().selective_scan(4)),
    'ssd_scan':((4,2),lambda m,S:m.linear(321).softplus().ssd_scan(1).rms_norm()),   # the norm as example:seq-mamba2 (lane mamba2-nan)
    'trapezoid_scan':((4,3),lambda m,S:m.linear(30).trapezoid_scan(2,4)),
    'lm_head':((3,),lambda m,S:m.embedding(5,4).lm_head(5)),
    # nn-attention begin
    'multihead_attention':((4,8),lambda m,S:m.multihead_attention(8,2,is_causal=True)),
    'positional_encoding':((4,8),lambda m,S:m.positional_encoding(8,dropout=0.0)),
    'transformer_encoder_layer':((4,8),lambda m,S:m.transformer_encoder_layer(8,2,16,dropout=0.0)),
    'transformer_decoder_layer':((4,8),lambda m,S:m.linear(8).transformer_decoder_layer(8,2,16,dropout=0.0,memory=0)),
    'transformer_encoder':((4,8),lambda m,S:m.transformer_encoder(S().transformer_encoder_layer(8,2,16,dropout=0.0),2)),
    'transformer_decoder':((4,8),lambda m,S:m.linear(8).transformer_decoder(S().transformer_decoder_layer(8,2,16,dropout=0.0),2,0)),
    # nn-attention end
    # nn-conv begin
    'conv_transpose1d':((2,4),lambda m,S:m.conv_transpose1d(2,3,stride=2)),
    'conv_transpose2d':((1,2,2),lambda m,S:m.conv_transpose2d(1,2,stride=2)),
    'adaptive_avg_pool2d':((1,4,4),lambda m,S:m.adaptive_avg_pool2d(2)),
    'adaptive_max_pool2d':((1,4,4),lambda m,S:m.adaptive_max_pool2d(2)),
    'adaptive_max_pool1d':((2,4),lambda m,S:m.adaptive_max_pool1d(2)),
    'global_max_pool':((2,4),lambda m,S:m.conv1d(2,1).global_max_pool()),
    'constant_pad2d':((1,2,2),lambda m,S:m.constant_pad2d(1,0.5)),
    'reflection_pad1d':((2,4),lambda m,S:m.reflection_pad1d(1)),
    'replication_pad2d':((1,2,2),lambda m,S:m.replication_pad2d(1)),
    'upsample':((1,2,2),lambda m,S:m.upsample(scale_factor=2,mode='bilinear')),
    'pixel_shuffle':((4,2,2),lambda m,S:m.pixel_shuffle(2)),
    # nn-conv end
    # nn-activations begin
    'identity':((4,),lambda m,S:m.linear(4).identity()),
    'leaky_relu':((4,),lambda m,S:m.linear(4).leaky_relu(0.1)),
    'elu':((4,),lambda m,S:m.linear(4).elu()),
    'prelu':((2,4),lambda m,S:m.prelu(2)),
    'mish':((4,),lambda m,S:m.linear(4).mish()),
    'hardswish':((4,),lambda m,S:m.linear(4).hardswish()),
    'hardsigmoid':((4,),lambda m,S:m.linear(4).hardsigmoid()),
    # nn-activations end
    # nn-norms begin
    'group_norm':((4,3),lambda m,S:m.group_norm(2,4)),
    'instance_norm1d':((2,4),lambda m,S:m.instance_norm1d(2,affine=True,track_running_stats=True)),
    'instance_norm2d':((1,2,2),lambda m,S:m.instance_norm2d(1,affine=True)),
    # nn-norms end
    # nn-losses-2 begin
    'log_softmax':((2,3),lambda m,S:m.linear(3).log_softmax(1)),
    # nn-losses-2 end
    # nn-conv3d begin
    'conv3d':((1,2,2,2),lambda m,S:m.conv3d(2,2,padding=1)),
    'max_pool3d':((1,2,2,2),lambda m,S:m.max_pool3d(2)),
    'avg_pool3d':((1,2,2,2),lambda m,S:m.avg_pool3d(2)),
    'adaptive_avg_pool3d':((1,2,2,2),lambda m,S:m.adaptive_avg_pool3d(1)),
    # nn-conv3d end
}


# nn-losses-2 begin (2026-11-23): the losses of lanes nn-losses and nn-losses-2 as `compile:loss=<name>` endpoints.
# Each fits a two-sample model one epoch of one full unshuffled batch with SGD under the loss (and its option) and
# returns the fitted model's outputs: predict, or predict_proba for the nll classifier.
SEQUENTIAL_LOSS_FIXTURES={
    'compile:loss=l1':('l1',{}),
    'compile:loss=smooth_l1':('smooth_l1',{'beta':.5}),
    'compile:loss=bce_with_logits':('bce_with_logits',{'pos_weight':2.}),
    'compile:loss=nll':('nll',{}),
    'compile:loss=kl_div':('kl_div',{'reduction':'batchmean'}),
    'compile:loss=ctc':('ctc',{'blank':0}),
}


def _sequential_loss_fixture(np,cls,method):
    loss,kw=SEQUENTIAL_LOSS_FIXTURES[method]
    fit={}
    if loss=='ctc':
        x=((np.arange(24,dtype=np.float32)%7)-3).reshape(2,4,3)/8
        m=cls(input_shape=(4,3),numeric_mode='identical').gru(3).linear(3).log_softmax()
        y=np.array([[1],[2]],np.int64); fit['target_lengths']=np.array([1,1],np.int64)
    else:
        x=((np.arange(8,dtype=np.float32)%7)-3).reshape(2,4)/8
        m=cls(input_shape=(4,),numeric_mode='identical')
        if loss in ('nll','kl_div'):
            # r5-proving (2026-11-23): nll labels are classes, as for cross_entropy (the fit sorts them into
            # classes_ and refuses a model whose outputs differ from the class count), so the two-row nll fixture
            # has two classes and two outputs; [0, 2] on three outputs was refused on the L40S proving run.
            m.linear(2 if loss=='nll' else 3).log_softmax()
            y=np.array([0,1],np.int64) if loss=='nll' else np.array([[.25,.5,.25],[.125,.125,.75]],np.float32)
        else:
            m.linear(1)
            y=np.array([0,1],np.int64) if loss=='bce_with_logits' else np.array([.25,-.5],np.float32)
    m.compile(loss=loss,optimizer='sgd',lr=.125,**kw)
    m.fit(x,y,epochs=1,batch_size=2,shuffle=False,seed=7,**fit)
    return m.predict_proba(x) if loss=='nll' else m.predict(x)
# nn-losses-2 end


def _sequential_layer_fixture(np,cls,method):
    shape,build=SEQUENTIAL_LAYER_FIXTURES[method]
    size=int(np.prod(shape))
    if method in ('embedding','lm_head'):
        x=(np.arange(2*size,dtype=np.float32)%5).reshape((2,)+shape)      # ids 0..4, stored as floats
    else:
        x=((np.arange(2*size,dtype=np.float32)%7)-3).reshape((2,)+shape)/8
    m=build(cls(input_shape=shape,numeric_mode='identical'),cls)
    if method=='lm_head':
        y=((np.arange(2*size,dtype=np.int32)+1)%5).reshape(2,size)
        m.compile(loss='token_cross_entropy',optimizer='sgd',lr=.125)
    else:
        y=np.array([.25,-.5],np.float32)
        m.flatten().linear(1).compile(optimizer='sgd',lr=.125)
    m.fit(x,y,epochs=1,batch_size=2,shuffle=False,seed=7)
    return m.predict(x)


def neural_checks(recorder,public,methods):
    import numpy as np
    from ._verify_api import package
    ml = package()
    cls=resolve(public); name=public.rsplit('.',1)[-1]
    def check(method):
        if name.startswith(('Mamba','Transformer')):
            block_name=name.replace('State','Block') if name.endswith('State') else name
            m=factory('mojolearn.'+block_name)
            x=np.arange(64,dtype=np.float32).reshape(1,2,32)/128-.25
            if name.endswith('State'):
                state=m.allocate_state(1,3) if name.startswith('Transformer') else m.allocate_state(1)
                if method=='__init__':
                    kwargs={k:getattr(state,k) for k in inspect.signature(cls).parameters}
                    state=cls(**kwargs)
                    result=m.forward(x,state)
                    return result
                if method=='set_input_states':
                    state.set_input_states(state.theta.copy(),state.h.copy(),state.pending_k.copy(),state.pending_v.copy())
                    assert state.pending
                    return m.forward(x,state)
                m.forward(x,state)
                if method=='capacity:get':
                    assert state.capacity==3
                    return state.capacity
                return getattr(state,method)()
            if method=='__call__':return m(x)
            # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
            # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
            # (the forward_with_tape / backward_from_tape probes.)
            if method in ('__getstate__','__setstate__'):
                state=m.__getstate__()
                restored=object.__new__(_vd_unwrap(cls));restored.__setstate__(state)
                before=m(x);after=restored(x)
                assert encode(before)==encode(after)
                return after
            # TOMBSTONE: MOJOLEARN_NN40_OWNED_WEIGHTS (leftover: the install/export/release_owned_weights probe) deleted 2026-11-25 by lane/owned-weights-deletion; code recoverable at 097a876282.
            if method=='session_info':
                # A fresh block has opened no prefill session on any build.
                assert m.session_info() is None;return {'session':None}
            raise AssertionError('no verify probe for '+public+'.'+method)
        if name.startswith('Samba'):
            config=ml.SambaConfig(4,32,['attention'],n_heads=1,intermediate=32)
            if name=='SambaConfig':
                if method=='block_shapes':return config.block_shapes('attention')
                if method=='registry':return config.registry()
                if method=='from_dict':
                    recovered=cls.from_dict(config.to_dict());assert recovered.to_dict()==config.to_dict()
                    return recovered.to_dict()
                return config.to_dict()
            weights={key:(np.ones(shape,np.float32) if 'norm' in key else
                (np.arange(np.prod(shape),dtype=np.float32).reshape(shape)%7-3)/32)
                for key,shape in config.registry()}
            m=cls(config,weights); ids=np.array([[0,1]],np.int32)
            if method=='load_weights':m.load_weights(weights)
            return getattr(m,method)(ids) if method in ('__call__','logits') else m.forward(ids)
        if name=='Sequential' and method in SEQUENTIAL_LAYER_FIXTURES:
            return _sequential_layer_fixture(np,cls,method)
        if name=='Sequential' and method in SEQUENTIAL_LOSS_FIXTURES:  # nn-losses-2
            return _sequential_loss_fixture(np,cls,method)
        if name=='Sequential':
            m=cls(numeric_mode='identical')
            x=np.array([[.25,-.5],[.75,.125]],np.float32)
            if method in ('load_state_dict','save_checkpoint','from_checkpoint'):
                m.linear(2)
                m.load_state_dict({'0.weight':np.array([[.5,.25],[-.25,.75]],np.float32),'0.bias':np.array([.125,-.125],np.float32)})
                if method=='load_state_dict':return m.predict(x)
                with tempfile.TemporaryDirectory() as tmp:
                    m.compile(optimizer='sgd',lr=.125).fit(x,m.predict(x),epochs=1,batch_size=2,shuffle=False,seed=7)
                    path=Path(tmp)/'tiny.json';m.save_checkpoint(str(path))
                    other=cls.from_checkpoint(str(path))
                    assert encode(m.predict(x))==encode(other.predict(x))
                    return other.predict(x)
            if method in ('avg_pool2d','max_pool2d','flatten','reshape','permute'):
                m=cls(input_shape=(1,2,2),numeric_mode='identical')
                x=np.arange(8,dtype=np.float32).reshape(2,1,2,2)/8
                if method in ('avg_pool2d','max_pool2d'):getattr(m,method)(2)
                elif method=='reshape':m.reshape((4,))
                elif method=='permute':m.permute((1,2,0))
                else:m.flatten()
                if method not in ('flatten','reshape'):m.flatten()
            elif method=='add':m.add('tanh')
            elif method=='dropout':m.dropout(.25)
            elif method=='residual':m.tanh().residual(-1)
            elif method=='layers:get':
                m.tanh(); layers=m.layers;assert len(layers)==1
            else:getattr(m,method)()
            m.linear(1).compile(optimizer='sgd',lr=.125)
            m.fit(x,np.array([.25,.5],np.float32),epochs=1,batch_size=2,shuffle=False,seed=7)
            return m.predict(x)
        if name=='Embedding':
            weights=np.arange(8,dtype=np.float32).reshape(4,2)/8
            m=cls.from_pretrained(weights) if method=='from_pretrained' else cls(4,2,weight=weights)
            ids=np.array([[0,2]],np.int64)
            if method=='__setattr__':
                m.weight=weights+.125
                assert np.array_equal(m.weight,weights+.125)
            if method=='__getstate__':return m.__getstate__()
            if method=='to_device':
                dev=m.to_device(weights);return _device_values(dev,weights)
            return m(ids)
        if name=='LayerNorm':
            m=cls(2);x=np.array([[.25,-.5],[.75,.125]],np.float32)
            return _device_values(m.to_device(x),x) if method=='to_device' else m(x)
        if name=='MoEBlock':
            m=cls(2,2,num_experts=2,top_k=1,random_state=7)
            x=np.array([[.25,-.5],[.75,.125]],np.float32)
            if method=='__setattr__':m.router=np.array([[.5,-.25],[-.5,.25]],np.float32)
            if method=='load_state_dict':m.load_state_dict({k:getattr(m,k).copy() for k in ('router','gate_up_proj','down_proj')})
            if method=='to_device':return _device_values(m.to_device(x),x)
            return m(x) if method=='__call__' else m.forward(x)
        shape=(1,2,2);kwargs={'numeric_mode':'identical'}
        if name.startswith('Conv'):kwargs.update(in_channels=1,out_channels=1,kernel_size=1)
        elif name=='BasicBlock':kwargs.update(inplanes=1,planes=1,input_shape=shape)
        elif name.startswith('BatchNorm'):kwargs['num_features']=1
        elif name.startswith('Adaptive'):kwargs.update(output_size=1,input_shape=shape)
        elif name.startswith(('AvgPool','MaxPool')):kwargs.update(kernel_size=2,input_shape=(1,4) if '1d' in name else shape)
        elif name=='Dropout2d':kwargs.update(p=.25,random_state=7,input_shape=shape)
        m=cls(**kwargs);x=np.arange(8,dtype=np.float32).reshape((2,1,4) if '1d' in name else (2,1,2,2))/8
        if name.startswith('Conv'):
            w=np.full((1,1,1) if name=='Conv1d' else (1,1,1,1),.5,np.float32)
            m.set_weights(w,np.array([.125],np.float32))
            if method=='weight1d_:get':return m.weight1d_
            if method=='pin_weights':m.pin_weights()
            if method=='unpin_weights':m.pin_weights();m.unpin_weights()
            try:return m.forward(x)
            finally:m.unpin_weights()
        if method=='fit':assert m.fit(x) is m
        if method=='train':assert m.train(True) is m
        out=m.forward(x)
        if method.startswith('indices_'):
            # Average pooling intentionally has no argmax indices.
            indices=m.indices_
            if method.endswith(':set'):m.indices_=indices
            return {'values':out,'indices':m.indices_}
        return out
    for method in methods:
        recorder.check(public+'.'+method,lambda method=method:check(method))
