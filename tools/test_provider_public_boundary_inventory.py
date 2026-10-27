"""Exercise the audit's distinctions so source-only results cannot become proof."""
from provider_public_boundary_inventory import describe


def test_argument_loader_prevents_single_entry_classification():
    source = '''
class Model(Base):
    def fit(self, X):
        return _NATIVE(self, 'family_fit', (self._bind(), X), context)
    def predict(self, X):
        return _NATIVE(self, 'family_predict', (None, X), context)
'''
    fit, predict = describe(source, 'example.py')
    assert fit['body_classification'] == 'requires_manual_or_transitive_audit'
    assert fit['loader_calls'] == ['self._bind']
    assert predict['body_classification'] == 'single_known_native_call_body'
    assert predict['operations'] == ['family_predict'] and predict['bases'] == ['Base']


def test_protocols_and_allowed_tags_are_retained_separately():
    source = '''
class Model:
    def __init__(self, X):
        self.shape = X.shape
    def __sklearn_tags__(self):
        return build_tags()
    @available_if(check_probability)
    def predict_proba(self, X):
        return _NATIVE(self, 'proba', (X,), context)
'''
    constructor, tags, proba = describe(source, 'example.py')
    assert constructor['metadata_expressions'] == ['X.shape']
    assert tags['body_classification'] == 'allowed_sklearn_tag_metadata_only'
    assert proba['body_classification'] == 'single_known_native_call_body'
    assert proba['decorators'] == ['available_if(check_probability)']
