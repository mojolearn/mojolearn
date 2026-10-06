import importlib.util,json,pathlib,tempfile,types,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).with_name('seal-and-monitor.py')
spec=importlib.util.spec_from_file_location('monitor_fix',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MonitorChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  p=patch.object(m,'E',pathlib.Path(self.tmp.name));p.start();self.addCleanup(p.stop)
  p=patch.object(m,'alert');self.alert=p.start();self.addCleanup(p.stop)
 def put(self,name,data):
  p=m.E/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
 def test_retired_route_never_reads_config_or_ssh(self):
  for terminal in ['TERMINATED_VERIFIED','TERMINATED']:
   self.put('specific-capture/manager-status.json',{'status':terminal})
   with patch.object(m,'run',side_effect=AssertionError('SSH forbidden')) as run:
    row=m.monitor_route('specific',None,[])
   self.assertEqual(row['status'],'RETIRED');self.assertTrue(row['ssh_skipped']);run.assert_not_called()
 def test_route_error_does_not_skip_other_route(self):
  calls=[]
  def route(name,state,meta):
   calls.append(name)
   if name=='specific':raise RuntimeError('ssh exited255')
   return {'status':'MONITORED'}
  with patch.object(m,'load_builder_state',return_value=({},[])),patch.object(m,'monitor_route',side_effect=route):
   out=m.monitor_once()
  self.assertEqual(calls,['specific','default']);self.assertEqual(out['specific']['status'],'MONITOR_ERROR');self.assertEqual(out['default']['status'],'MONITORED')
  saved=json.loads((m.E/'tail-monitor-status.json').read_text());self.assertIn('default',saved['routes'])
 def test_builder_failure_still_visits_routes(self):
  with patch.object(m,'load_builder_state',side_effect=TimeoutError('builder down')),patch.object(m,'monitor_route',return_value={'status':'MONITORED'}) as route:
   m.monitor_once()
  self.assertEqual([x.args for x in route.call_args_list],[('specific',None,[]),('default',None,[])])
  self.assertFalse(json.loads((m.E/'tail-monitor-status.json').read_text())['builder_available'])
 def test_builder_unavailable_leaves_remote_seal_unchanged(self):
  self.put('default-owner/config.json',{'ssh':['root@example']});self.put('default-repair-queue.json',[])
  responses=[types.SimpleNamespace(stdout=json.dumps({'status':'RUNNING'})),types.SimpleNamespace(stdout=json.dumps({'implicit':{},'ready':{'failed':0},'tail':{'phase':'RUNNING'},'worker':{'phase':'GPU_OPPONENTS'}}))]
  with patch.object(m,'run',side_effect=responses) as run:
   out=m.monitor_route('default',None,[])
  self.assertEqual(out['seal_action'],'UNCHANGED_BUILDER_UNAVAILABLE');self.assertIsNone(out['sealed']);self.assertEqual(run.call_count,2)
  self.assertFalse(any('CANDIDATES_SEALED' in str(c) for c in run.call_args_list))
 def test_alert_delivery_failure_does_not_skip_default(self):
  def route(name,state,meta):
   if name=='specific':raise RuntimeError('specific unavailable')
   return {'status':'MONITORED'}
  with patch.object(m,'load_builder_state',return_value=({},[])),patch.object(m,'monitor_route',side_effect=route),patch.object(m,'alert',side_effect=TimeoutError('queue unavailable')):
   out=m.monitor_once()
  self.assertEqual(out['default']['status'],'MONITORED')

if __name__=='__main__':unittest.main(verbosity=2)
