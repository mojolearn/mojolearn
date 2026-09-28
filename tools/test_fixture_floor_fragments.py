"""Source-only controls for floor declarations in registered fragments."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('floor_fragments',Path(__file__).with_name('fixture_floors.py'))
floors=importlib.util.module_from_spec(spec);spec.loader.exec_module(floors)

@pytest.mark.parametrize('wheel',[False,True])
def test_fragment_non_size_revision_is_seen_and_unknown_lane_still_refused(tmp_path,wheel):
    source='''LANE_REVISIONS = {"example": "arithmetic-v2"}
NON_SIZE_REVISIONS = {"example": "2026-09-28: this revision changes the arithmetic, without changing fixture dimensions."}
def lane_fragment_paths(): pass
'''
    harness=tmp_path/'identity_break.py';harness.write_text(source)
    fragment=tmp_path/'_identity_lane_example.py' if wheel else tmp_path/'identity_lanes/example.py'
    fragment.parent.mkdir(exist_ok=True);fragment.write_text('@lane("example")\ndef _(ml, X, yc, yr, Xh=None): pass\n')
    assert floors.check(path=harness)==[]
    assert floors.check(source,path=harness)==[]  # mutation path
    fragment.write_text('@lane("other")\ndef _(ml, X, yc, yr, Xh=None): pass\n')
    assert any('no such lane' in e for e in floors.check(path=harness))


def test_fragment_size_revision_cannot_bypass_floor(tmp_path):
    harness=tmp_path/'identity_break.py';harness.write_text('LANE_REVISIONS={"example":"rows-2-1"}\nNON_SIZE_REVISIONS={}\ndef lane_fragment_paths(): pass\n')
    directory=tmp_path/'identity_lanes';directory.mkdir()
    (directory/'example.py').write_text('@lane("example")\ndef _(ml, X, yc, yr, Xh=None):\n    return X[:2]\n')
    assert any('no floor' in e for e in floors.check(path=harness))
