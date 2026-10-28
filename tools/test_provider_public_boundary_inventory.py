from provider_public_boundary_inventory import lifecycle_candidates,describe


def test_inherited_wrapper_is_not_erased_by_derived_native_marker():
    sources={
      'python/mojolearn/_mode.py':'class PublicOutputsMixin: pass\nclass SklearnTagsMixin(PublicOutputsMixin): pass\nclass ParamsMixin(SklearnTagsMixin): pass\n',
      'python/mojolearn/demo.py':'from ._mode import ParamsMixin\nclass Base(ParamsMixin):\n def fit(self,X): return _XSEQ(self,"fit",(X,),ctx)\nclass Child(Base):\n _mojolearn_native_public=True\n',
    }
    rows=lifecycle_candidates(sources)
    child=next(r for r in rows if r['class_name']=='mojolearn.demo.Child')
    assert child['inherited'] and child['defining_class_wrapper_risk']
    assert child['wrapper_assessment']=='python_lifecycle_wrappers_expected'
    assert describe(sources['python/mojolearn/demo.py'],'demo.py')[0]['body_classification']=='single_known_native_call_body'


def test_marker_on_defining_base_is_only_source_candidate_not_runtime_pass():
    sources={
      'python/mojolearn/_mode.py':'class PublicOutputsMixin: pass\nclass SklearnTagsMixin(PublicOutputsMixin): pass\nclass ParamsMixin(SklearnTagsMixin): pass\n',
      'python/mojolearn/demo.py':'from ._mode import ParamsMixin\nclass Base(ParamsMixin):\n _mojolearn_native_public=True\n def predict(self,X): return _XSEQ(self,"predict",(X,),ctx)\nclass Child(Base): pass\n',
    }
    child=next(r for r in lifecycle_candidates(sources) if r['class_name']=='mojolearn.demo.Child')
    assert child['native_marker'] and not child['defining_class_wrapper_risk']
    assert child['wrapper_assessment']=='native_lifecycle_unverified'
