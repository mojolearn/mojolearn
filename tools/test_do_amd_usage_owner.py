import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import do_amd_usage_owner as O

class Owner(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.state=self.root/'steward';self.state.mkdir();(self.state/'state.env').write_text('DROPLET_ID=123\nIP=192.0.2.1\n')
        self.c=dict(droplet_id='123',owner_id='test-owner',state_dir=str(self.state),local_out=str(self.root/'evidence'),remote_out='/root/board',steward_script='/repo/tools/do_amd_steward.sh')
    def tearDown(self):self.tmp.cleanup()
    def test_no_idle_before_done_capture_or_while_shared_work(self):
        prev={'capture_digest':'same','idle_since':1}
        for done,busy,cap,hold in [(False,False,'same',False),(True,True,'same',False),(True,False,None,False),(True,False,'same',True)]:
            self.assertIsNone(O.next_idle(prev,999,done,busy,cap,hold))
        self.assertEqual(O.next_idle(prev,999,True,False,'same',False),1)
        self.assertEqual(O.next_idle(prev,999,True,False,'changed',False),999)
    def test_validate_fixed_idle_and_orphan(self):
        O.validate(self.c)
        for key,value in [('idle_seconds',0),('renew_minutes',120),('capture_timeout',6000)]:
            with self.assertRaises(ValueError):O.validate(dict(self.c,**{key:value}))
    def test_refuse_other_steward(self):
        self.assertEqual(O.steward_state(self.c)['IP'],'192.0.2.1')
        with self.assertRaisesRegex(ValueError,'identity changed'):O.steward_state(dict(self.c,droplet_id='456'))
    def test_active_cycle_renews90_and_never_deletes(self):
        observed=dict(done=False,shared_queue_busy=False,progress='p',command_exit=None)
        with patch.object(O,'probe',return_value=observed),patch.object(O,'steward') as call:
            O.manage(self.c,once=True)
            call.assert_called_once_with(self.c,'extend','90')
        d=json.loads((self.root/'evidence/owner-status.json').read_text());self.assertIsNone(d['idle_since'])
    def test_capture_failure_never_arms_idle(self):
        observed=dict(done=True,shared_queue_busy=False,progress='p',command_exit='0',rows=[])
        with patch.object(O,'probe',return_value=observed),patch.object(O,'steward') as call,patch.object(O,'command',side_effect=RuntimeError('capture fail')):
            O.manage(self.c,once=True)
            self.assertFalse(any('down' in c.args for c in call.call_args_list))
        d=json.loads((self.root/'evidence/owner-status.json').read_text());self.assertEqual(d['status'],'ERROR');self.assertIsNone(d['idle_since'])
    def test_notification_failed_delivery_retries_then_deduplicates(self):
        local=self.root/'evidence';local.mkdir();c=dict(self.c,notify_thread='parent')
        with patch.object(O,'command',side_effect=[RuntimeError('delivery failed'),'queued']) as cmd,patch.object(O.time,'time',return_value=100):
            O.notify(c,local,'FAIL','detail');self.assertEqual(cmd.call_count,1)
            O.notify(c,local,'FAIL','detail');self.assertEqual(cmd.call_count,1)
            with patch.object(O.time,'time',return_value=161):O.notify(c,local,'FAIL','detail')
            self.assertEqual(cmd.call_count,2)
            with patch.object(O.time,'time',return_value=230):O.notify(c,local,'FAIL','detail')
            self.assertEqual(cmd.call_count,2)
        event=json.loads(next((local/'events').glob('*.json')).read_text());self.assertTrue(event['delivered'])
    def test_batch_rsync_trusts_first_key_without_prompt(self):
        self.assertIn('StrictHostKeyChecking=accept-new',O.capture_transport())
        self.assertIn('BatchMode=yes',O.capture_transport())
    def test_remote_probe_checks_legacy_held_queue(self):
        with patch.object(O,'steward',return_value=json.dumps({'deadline':99999999999})) as cmd:
            O.probe(self.c)
            self.assertIn('queue/held',cmd.call_args.args[-1])
    def test_existing_foreign_owner_blocks_control(self):
        (self.state/'usage-owner.json').write_text(json.dumps({'owner_id':'other'}))
        with patch.object(O,'steward') as call,patch.object(O,'probe') as probe:
            with self.assertRaisesRegex(ValueError,'another usage owner'):O.manage(self.c,once=True)
            call.assert_not_called();probe.assert_not_called()

if __name__=='__main__':unittest.main()
