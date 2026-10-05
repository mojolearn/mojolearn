import importlib.util, pathlib, unittest, types, ast
P=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('opponent_board',P/'tools/bench_board.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class OpponentsOnly(unittest.TestCase):
 def test_tree_command(self):
  ctx=dict(python='python',tree_driver='driver',vendor='apple',rounds=1,arm_budget_s=3600,race_deadline_s=21600,opponents_only=True,infer=False)
  race=dict(our_arms={},opponents=['xgboost-cpu'],arms=['xgboost-cpu'],lane='gbdt',dataset='taxi',rows=None)
  cmd,e=b.tree_cmd(ctx,race);self.assertIn('--opponents-only',cmd);self.assertNotIn('--ours-only',cmd);self.assertEqual(cmd[cmd.index('--arms')+1],'xgboost-cpu')
  race['our_arms']={'ours':'identical'}
  with self.assertRaises(ValueError):b.tree_cmd(ctx,race)
 def test_cache_success(self):
  c=dict(status='ok',rounds=1,times_ms=[2.0],median_ms=2.,warmup_ms=3.,quality={'r2':.9});self.assertTrue(b.successful_opponent_cell(c));self.assertTrue(b.successful_opponent_cell(dict(c,rounds=2,times_ms=[2.,2.])))
  for override in [dict(status='REFUSED'),dict(times_ms=[]),dict(warmup_ms=None),dict(median_ms=float('nan')),dict(quality={'error':'bad'}),dict(quality={'r2':float('inf')}),dict(rounds=2)]:self.assertFalse(b.successful_opponent_cell(dict(c,**override)))
 def test_forest_own_calls_guarded(self):
  tree=ast.parse((P/'bench/speed/forest_speed_arm.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  prep=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not args.opponents_only')
  ns={'args':types.SimpleNamespace(opponents_only=True),'data':None,'prepare_our_inputs':lambda _:self.fail('own preparation executed')};exec(compile(ast.Module(body=[prep],type_ignores=[]),'guard','exec'),ns)
  assign=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='arms' for t in n.targets))
  ns.update(lane='gbdt',cfg={},build_ours=lambda *a:self.fail('own builder executed'));exec(compile(ast.Module(body=[assign],type_ignores=[]),'guard','exec'),ns);self.assertEqual(ns['arms'],[])
if __name__=='__main__':unittest.main()
