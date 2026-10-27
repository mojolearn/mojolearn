"""Small forecasting, tokenizer and serialized-estimator endpoint fixtures."""
__mojolearn_user_boundary__ = True

import json
import tempfile
import warnings
from pathlib import Path
from ._verify_api import resolve, encode
from ._verify_api_metadata import factory
from ._verify_api_estimators import dataset

NAMES=frozenset(('ARIMA','AutoARIMA','AutoTheta','Theta','OptimizedTheta','DynamicTheta',
 'DynamicOptimizedTheta','ETS','DampedETS','CrostonClassic','CrostonSBA','CrostonOptimized',
 'BpeTokenizer','Cholesky','ExponentialSmoothing','GPT2Tokenizer','HostForest','IVFPQIndex',
 'LatentDirichletAllocation','ProphetForecaster','STL','SVR','TrainedBpeVocabulary'))


def applicable(public,method):
    name=public.rsplit('.',1)[-1]
    return (name in NAMES and not (name=='IVFPQIndex' and method!='__getattr__') and method in ('sma_:get','predict','forecast','decode','decode_batch','eot_token:get',
        'from_files','from_token_bytes','vocabulary_source:get','profile_jitter','__getattr__','get_level',
        'get_season','get_trend','score','?','__getstate__','__setstate__','perplexity','add_seasonality',
        'config:get','coef_:get','n_ties_broken:get','n_tokens:get','tie_break:get')) or (
        name.startswith(('RNN','GRU','LSTM')) and method in ('load_state_dict','state_dict')) or (
        name.startswith('SGD') and method in ('score','score_samples'))


def misc_checks(recorder,public,methods):
    import numpy as np
    import mojolearn as ml
    name=public.rsplit('.',1)[-1]
    def check(method):
        y=np.array([1,1.25,1.5,1.125,1.75,1.375,2,1.625],np.float32)
        if name in ('BpeTokenizer','GPT2Tokenizer','TrainedBpeVocabulary'):
            from .tokenizer import BpeTokenizer, BpeVocabularyTrainer, _byte_to_char
            if name=='TrainedBpeVocabulary':
                vocab=BpeVocabularyTrainer(vocab_size=258,min_frequency=2).train(['ab ab','ab ba'])
                return getattr(vocab,method[:-4])
            cls=BpeTokenizer
            if name=='GPT2Tokenizer':
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');alias=getattr(ml,'GPT2Tokenizer')
                assert alias is BpeTokenizer and caught
                cls=alias
            tokens=[bytes([i]) for i in range(256)]
            m=cls.from_token_bytes(tokens)
            if method=='from_files':
                with tempfile.TemporaryDirectory() as tmp:
                    enc=Path(tmp)/'encoder.json';merges=Path(tmp)/'vocab.bpe'
                    enc.write_text(json.dumps({char:i for i,char in _byte_to_char().items()}))
                    merges.write_text('#version: 0.2\n')
                    m=cls.from_files(str(enc),str(merges))
                    assert m.decode(m.encode('ab'))=='ab'
                    return m.encode('ab')
            if method=='decode_batch':return m.decode_batch([m.encode('ab'),m.encode('ba')])
            if method=='eot_token:get':assert m.eot_token==256;return m.eot_token
            if method=='vocabulary_source:get':assert isinstance(m.vocabulary_source,str);return m.vocabulary_source
            return m.decode(m.encode('ab'))
        cls=resolve(public)
        if name=='Cholesky':return cls().profile_jitter()
        if name=='ExponentialSmoothing':
            m=cls(y,seasonal_periods=2).fit()
            if method=='__getattr__':
                key=cls._LAZY_COMPONENTS[0]
                return m.__getattr__(key)
            return getattr(m,method)()
        if name=='AutoARIMA':
            m=cls(y).search(d=[0],p=[1],q=[0],P=[0],Q=[0],D=[0],fit_intercept=False).fit(maxiter=2)
            return m.predict(start=0,end=2)
        if name=='ARIMA':
            m=cls(order=(0,0,0),seasonal_order=(0,0,1,2),maxiter=2).fit(y)
            return m.sma_
        if method=='forecast':
            if name in ('ETS','DampedETS'):y=np.tile(y,2)
            return cls().forecast(y,h=2)
        if name=='STL':return cls(y,period=2).config
        X,labels,target=dataset(public)
        if name=='HostForest':
            with tempfile.TemporaryDirectory() as tmp:
                path=str(Path(tmp)/'forest');m=ml.RandomForestRegressor(n_estimators=1,max_depth=1,random_state=7).fit(X,target)
                m.save(path);host=cls.from_file(path);state=host.__getstate__()
                other=object.__new__(cls);other.__setstate__(state)
                assert encode(other.predict(X))==encode(host.predict(X))
                return other.predict(X)
        if name=='ProphetForecaster':
            m=cls(n_changepoints=0,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
            assert m.add_seasonality('tiny',period=2,fourier_order=1) is m
            m.fit(np.arange(8,dtype=np.float32),target)
            return m.predict(np.array([8,9],np.float32))
        if name=='IVFPQIndex':
            m=factory(public).fit(X)
            # A miss must not accidentally expose a native handle as a field.
            try:m.__getattr__('missing_verifier_attribute_')
            except AttributeError:return {'missing_refused':True,'query':m.search(X[:1])}
            raise AssertionError('unknown field was accepted')
        if name=='SVR':return cls(kernel='linear',max_iter=2).fit(X,target).coef_
        if name=='LatentDirichletAllocation':
            m=factory(public).fit(X);return m.perplexity(X)
        m=factory(public)
        yy=labels if name.endswith('Classifier') else target
        if name.startswith(('RNN','GRU','LSTM')):
            x=X.reshape(8,2,2);m.fit(x,yy)
            if method=='state_dict':return m.state_dict()
            other=factory(public);other.fit(x,yy);other.load_state_dict(m.state_dict())
            assert encode(other.predict(x))==encode(m.predict(x))
            return other.predict(x)
        m.fit(X,yy)
        return m.score_samples(X) if method=='score_samples' else m.score(X,yy)
    for method in methods:recorder.check(public+'.'+method,lambda method=method:check(method))
