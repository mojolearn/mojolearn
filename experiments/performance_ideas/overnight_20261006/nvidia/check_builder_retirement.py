import importlib.util,json,pathlib,tempfile,types,unittest
from unittest.mock import patch
D=pathlib.Path(__file__).parent

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,D/file)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

seal=load('retirement_sealer','seal-and-monitor.py')
stream=load('retirement_stager','stream-repairs.py')

class BuilderRetirementChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=pathlib.Path(self.tmp.name);self.E=self.root/'nvidia';self.E.mkdir()
  for mod in [seal,stream]:
   p=patch.object(mod,'E',self.E);p.start();self.addCleanup(p.stop)
  self.put(self.root/'cpu-builder/status.json',{'status':'TERMINATED_VERIFIED','droplet_id':606509725})
  self.put(self.E/'specific-capture/manager-status.json',{'status':'TERMINATED_VERIFIED'})
  self.put(self.E/'default-capture/manager-status.json',{'status':'MANAGING'})
  self.put(self.E/'default-owner/config.json',{'ssh':['root@active-default']})
  self.put(self.E/'default-repair-queue.json',[{'key':'already-queued','candidate_id':'I20'}])
 def put(self,p,value):
  p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
 def test_sealer_terminal_builder_no_probe_no_alarm_default_processed(self):
  calls=[]
  def run(argv,**kwargs):
   self.assertNotIn('root@167.99.116.166',argv)
   self.assertFalse(any('CANDIDATES_SEALED' in arg for arg in argv))
   calls.append(argv)
   data={'status':'RUNNING'} if 'recover-repair-worker.py' in argv[-1] else {'implicit':{},'ready':{'failed':0},'tail':{'phase':'RUNNING'},'worker':{'phase':'GPU_OPPONENTS'}}
   return types.SimpleNamespace(stdout=json.dumps(data))
  with patch.object(seal,'run',side_effect=run),patch.object(seal,'alert') as alert:
   out=seal.monitor_once()
  self.assertEqual(len(calls),2);alert.assert_not_called()
  self.assertEqual(out['specific']['status'],'RETIRED')
  self.assertEqual(out['default']['status'],'MONITORED')
  self.assertEqual(out['default']['seal_action'],'UNCHANGED_BUILDER_RETIRED')
  saved=json.loads((self.E/'tail-monitor-status.json').read_text());self.assertEqual(saved['builder_status'],'RETIRED');self.assertIsNone(saved['builder_error'])
 def test_direct_builder_fetch_terminal_guard(self):
  with patch.object(seal,'run',side_effect=AssertionError('No SSH')) as run:
   self.assertEqual(seal.load_builder_state(),(None,[]))
  run.assert_not_called()
 def test_stager_preserves_metadata_and_queues_publishes_default_without_staging(self):
  cache=self.put(self.E/'repair-build-observations.json',{'builds':[{'stale_job':'MUST_NOT_REPLAY'}],'time':1})
  cached_bytes=cache.read_bytes();cached_mtime=cache.stat().st_mtime_ns
  queue=self.E/'default-repair-queue.json';queue_bytes=queue.read_bytes();calls=[]
  def cmd(argv,**kwargs):
   self.assertEqual(argv[0],'ssh');self.assertIn('root@active-default',argv)
   self.assertNotIn('root@167.99.116.166',argv);self.assertEqual(kwargs['input'],queue_bytes);calls.append(argv)
   return types.SimpleNamespace(stdout=b'')
  with patch.object(stream,'cmd',side_effect=cmd),patch.object(stream,'normalize') as normalize,patch.object(stream,'cases',side_effect=AssertionError('No staging')),patch.object(stream,'notify') as notify:
   stream.stream_once()
  self.assertEqual(cache.read_bytes(),cached_bytes);self.assertEqual(cache.stat().st_mtime_ns,cached_mtime);self.assertEqual(queue.read_bytes(),queue_bytes)
  self.assertEqual([c.args for c in normalize.call_args_list],[('specific',),('default',)])
  self.assertEqual(len(calls),1);notify.assert_not_called()
  saved=json.loads((self.E/'stream-status.json').read_text());self.assertEqual(saved['status'],'BUILDER_RETIRED');self.assertEqual(saved['newly_staged'],[])
 def test_retired_stager_repeats_publication_without_cpu_or_cache_replay(self):
  with patch.object(stream,'publish_queues',return_value={}) as publish,patch.object(stream,'cmd',side_effect=AssertionError('No CPU')),patch.object(stream,'notify') as notify:
   stream.stream_once();stream.stream_once()
  self.assertEqual(publish.call_count,2);notify.assert_not_called()
 def test_live_builder_still_discovers(self):
  self.put(self.root/'cpu-builder/status.json',{'status':'MANAGING'})
  with patch.object(stream,'publish_queues',return_value={}),patch.object(stream,'cmd',return_value=types.SimpleNamespace(stdout=b'[]')) as cmd,patch.object(stream,'notify') as notify:
   stream.stream_once()
  self.assertEqual(cmd.call_count,1);self.assertIn('root@167.99.116.166',cmd.call_args.args[0]);notify.assert_not_called()
  self.assertEqual(json.loads((self.E/'stream-status.json').read_text())['status'],'WATCHING')

if __name__=='__main__':unittest.main(verbosity=2)
