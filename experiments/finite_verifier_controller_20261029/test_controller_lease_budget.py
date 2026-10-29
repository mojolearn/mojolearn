import ast
from pathlib import Path
import pytest
from controller_lease_budget import lease_budget

@pytest.mark.parametrize('remaining,allowed',[(3301,True),(3300,True),(3299,False),(0,False),(-1,False)])
def test_admission_includes_retention_reserve(remaining,allowed):
    d=lease_budget(10000,2700,600,10000-remaining)
    assert d['allowed'] is allowed
    assert d['remaining_seconds']==remaining

@pytest.mark.parametrize('args',[(None,1,1,1),(2,None,1,1),(2,1,None,1),(2,-1,1,1),(2,1,0,1),(float('inf'),1,1,1),(2,float('nan'),1,1),(True,1,1,1)])
def test_bad_budget_refuses(args):
    with pytest.raises(ValueError):lease_budget(*args)

def test_local_run_without_lease_is_unchanged():
    assert lease_budget(None,None,None,1) is None

@pytest.mark.parametrize('name',['run_frozen_chunks_v3.py','run_frozen_chunks_v4.py'])
def test_loop_declines_before_running_product(name):
    source=Path(__file__).with_name(name).read_text()
    tree=ast.parse(source)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
    body=ast.get_source_segment(source,loop)
    assert body.index("if not budget['allowed']") < body.index('run(c)')
    assert "raise SystemExit(75)" in body
    assert body.index('index >= a.max_units') < body.index('run(c)')
    assert 'lease_budget' in body
