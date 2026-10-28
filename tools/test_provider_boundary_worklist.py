# SPDX-License-Identifier: Apache-2.0
import ast

from provider_boundary_worklist import group_rows, phase_hints, worklist


def row(endpoint, source='python/mojolearn/sample.py:1', **overrides):
    return dict(endpoint=endpoint, definition=source, family='sample', runtime_status='native_boundary',
                source_ast_sha256='same-source', native_entries=['native'], **overrides)


def test_aliases_share_implementation_but_distinct_definitions_do_not():
    groups = group_rows([row('pkg.A.fit'), row('pkg.alias.A.fit'), row('pkg.B.fit', 'python/mojolearn/sample.py:10')])
    assert len(groups) == 2
    assert sorted(len(g['endpoints']) for g in groups) == [1, 2]


def test_unresolved_sources_and_different_installed_connections_stay_distinct():
    groups = group_rows([row('pkg.A.fit', None), row('pkg.B.fit', None),
                         row('pkg.C.fit', import_connections=[{'source': 'installer:1', 'owner': 'C'}]),
                         row('pkg.D.fit', import_connections=[{'source': 'installer:1', 'owner': 'D'}])])
    assert len(groups) == 4


def test_one_native_call_still_requires_transitive_review():
    node = ast.parse('def run(x):\n    return native(x)\n').body[0]
    hints, names = phase_hints(node, {'native'})
    assert hints == ['transitive_native_review_pending']
    assert names == ['native']


def test_fitted_state_preparation_binding_and_result_work_are_visible():
    node = ast.parse('''def run(self, x):
    if self.state is None:
        raise RuntimeError('call forward first')
    value = native(self._bind(), int(x))
    self.output = value.reshape((2, 3))
    return self.output
''').body[0]
    hints, _ = phase_hints(node, {'native'})
    assert set(hints) >= {'fitted_state', 'argument_or_result_preparation', 'binding_selection',
                          'post_native_result_work', 'transitive_native_review_pending'}


def test_stale_inventory_cannot_describe_a_changed_body(tmp_path):
    source = tmp_path / 'sample.py'
    source.write_text('def run(x):\n    return changed(x)\n')
    endpoint = row('pkg.run', 'sample.py:1')
    result = worklist({'endpoints': [endpoint]}, tmp_path)[0]
    assert 'inventory_source_changed_rerun_required' in result['phase_hints']
    assert result['direct_call_names'] == []
    assert result['runtime_evidence'] == 'NOT_REVIEWED_BY_WORKLIST'
