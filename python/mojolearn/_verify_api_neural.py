"""Bounded neural API fixtures: two examples/tokens, at most one update."""
__mojolearn_user_boundary__ = True

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


def neural_checks(recorder,public,methods):
    import numpy as np
    import mojolearn as ml
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
            if method in ('forward_with_tape','backward_from_tape'):
                if name=='TransformerBlockInference':
                    # This documented CPU inference class has no training tape.
                    # Test its actual public refusal, without claiming backward arithmetic.
                    if method=='forward_with_tape':
                        try:m.forward_with_tape(x)
                        except NotImplementedError as error:
                            assert 'native tape API' in str(error)
                            return {'availability':'forward-only inference','numerical_execution':False}
                    else:
                        try:m.backward_from_tape(None,np.ones_like(x))
                        except ValueError as error:
                            assert 'another block' in str(error)
                            return {'foreign_tape_refused':True,'numerical_execution':False}
                    raise AssertionError('inference-only tape admission changed; add numerical fixture')
                out,tape=m.forward_with_tape(x)
                try:
                    return out if method=='forward_with_tape' else m.backward_from_tape(tape,np.full_like(out,.125))
                finally:
                    tape.close()
            if method in ('__getstate__','__setstate__'):
                state=m.__getstate__()
                restored=object.__new__(cls);restored.__setstate__(state)
                before=m(x);after=restored(x)
                assert encode(before)==encode(after)
                return after
            enabled=getattr(m._extension(),'mamba3_owned_weights_enabled',lambda:False)()
            if not enabled:
                if method=='release_owned_weights':
                    m.release_owned_weights();return m(x)
                if method=='session_info':
                    assert m.session_info() is None;return {'session':None}
                try:getattr(m,method)()
                except RuntimeError as error:
                    assert 'owned' in str(error).lower()
                    return {'availability':'compiled owned weights disabled','numerical_execution':False}
                raise AssertionError('disabled owned-weight capability was admitted')
            try:
                m.install_owned_weights()
                before=m(x)
                if method=='export_owned_weights':return m.export_owned_weights()
                if method=='release_owned_weights':
                    m.release_owned_weights()
                    after=m(x);assert encode(before)==encode(after)
                    return after
                if method=='session_info':
                    info=m.session_info();assert isinstance(info,dict)
                    # Counters describe cache implementation, not numerical identity.
                    return {'reported': True, 'output':before}
                return before
            finally:
                m.release_owned_weights()
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
            if method in ('conv2d','avg_pool2d','max_pool2d','flatten','reshape','permute'):
                m=cls(input_shape=(1,2,2),numeric_mode='identical')
                x=np.arange(8,dtype=np.float32).reshape(2,1,2,2)/8
                if method=='conv2d':m.conv2d(1,1)
                elif method in ('avg_pool2d','max_pool2d'):getattr(m,method)(2)
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
