#!/usr/bin/env python3
"""No SSH, real provider calls, or deletion: policy, capture and fake API tests."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import runpod_usage_lease as u


def config(root='/tmp/usage-unit'):
    return dict(pod_id='pod123',owner_id='owner123',pod_name='mojolearn-cpu-unit',ssh=['root@example.invalid'],
                remote_out=root+'/artifacts',local_out=root+'/local',remote_state=root+'/guard',remote_curlrc=root+'/credential')


class Policy(unittest.TestCase):
    def setUp(self):self.c=config();self.s=u.new_state(self.c,0)
    def test_active_many_days_has_no_absolute_cutoff(self):
        for now in range(0,172800,60):
            self.s=u.renew(self.c,self.s,now,done=False)
            self.assertEqual(u.decision(self.c,self.s,now,False),'KEEP')
    def test_orphan_deletes_after_owner_heartbeat_missing(self):
        self.assertEqual(u.decision(self.c,self.s,5399,False),'KEEP')
        self.assertEqual(u.decision(self.c,self.s,5400,False),'DELETE_ORPHAN')
    def test_done_without_verified_capture_is_not_idle(self):
        s=u.renew(self.c,self.s,10000,done=True)
        self.assertEqual(u.decision(self.c,s,13000,True),'KEEP')
    def test_idle_45_minutes_only_after_capture(self):
        s=u.renew(self.c,self.s,10,captured='proof',done=True)
        self.assertEqual(u.decision(self.c,s,2709,True,'proof'),'KEEP')
        self.assertEqual(u.decision(self.c,s,2710,True,'proof'),'DELETE_IDLE')
    def test_heartbeat_preserves_idle_deadline(self):
        s=u.renew(self.c,self.s,10,'proof',True);s=u.renew(self.c,s,1000,'proof',True)
        self.assertEqual(s['idle_since'],10)
    def test_changed_artifacts_refuse_idle_delete(self):
        s=u.renew(self.c,self.s,10,'proof',True)
        self.assertEqual(u.decision(self.c,s,2710,True,'different'),'KEEP')
    def test_hold_blocks_idle_but_not_orphan(self):
        s=u.renew(self.c,self.s,10,'proof',True);s['busy_hold']='ptx queued'
        self.assertEqual(u.decision(self.c,s,2710,True,'proof'),'KEEP')
        self.assertEqual(u.decision(self.c,s,5410,True,'proof'),'DELETE_ORPHAN')
    def test_new_work_clears_idle(self):
        s=u.renew(self.c,self.s,10,'proof',True);s=u.renew(self.c,s,50,done=False)
        self.assertIsNone(s['idle_since']);self.assertIsNone(s['captured_manifest'])
    def test_wrong_owner_and_changed_config_refused(self):
        for field in ('owner_id','pod_id','pod_name'):
            c=dict(self.c);c[field]+='changed'
            with self.assertRaises(ValueError):u.renew(c,self.s,1)
    def test_terminal_cannot_renew(self):
        self.s['status']='TERMINATED'
        with self.assertRaises(ValueError):u.renew(self.c,self.s,1)
    def test_credentials_never_inside_capture(self):
        c=dict(self.c,remote_curlrc=self.c['remote_out']+'/key')
        with self.assertRaises(ValueError):u.validate(c)
    def test_guard_never_inside_capture(self):
        c=dict(self.c,remote_state=self.c['remote_out']+'/guard')
        with self.assertRaises(ValueError):u.validate(c)
    def test_transfer_bound_and_paths(self):
        for delta in ({'capture_timeout':5400},{'remote_out':'/tmp/x;bad'},{'exclude_relative':['../secret']}):
            with self.assertRaises(ValueError):u.validate(dict(self.c,**delta))


class Capture(unittest.TestCase):
    def test_hash_proof_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'DONE').touch();(p/'binary.so').write_bytes(b'abc');rows=u.inventory(p)
            self.assertEqual(u.verify_capture(rows,p),u.digest(rows));(p/'binary.so').write_bytes(b'xyz')
            with self.assertRaises(ValueError):u.verify_capture(rows,p)
    def test_unsafe_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):u.verify_capture([{'path':'../secret','bytes':0,'sha256':''}],tmp)
    def test_explicit_cache_exclusion_retains_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'cache').mkdir();(p/'cache'/'blob').write_bytes(b'cache');(p/'x.so').write_bytes(b'binary')
            self.assertEqual([r['path'] for r in u.inventory(p,['cache'])],['x.so'])
    def test_symlink_refuses_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):u.inventory(p)


class Provider(unittest.TestCase):
    def test_delete_requires_matching_provider_identity(self):
        calls=[]
        def api(c,m,p):calls.append(m);return 200,{'id':'other','name':c['pod_name']}
        with self.assertRaises(ValueError):u.delete_verify(config(),api)
        self.assertNotIn('DELETE',calls)
    def test_delete_verified_by_get_and_list(self):
        replies=iter([(200,{'id':'pod123','name':'mojolearn-cpu-unit'}),(200,{'id':'pod123','name':'mojolearn-cpu-unit'}),(204,None),(404,None),(200,[])])
        self.assertTrue(u.delete_verify(config(),lambda *args:next(replies)))
    def test_malformed_or_still_listed_never_verified(self):
        for listing in ({},[{'id':'pod123'}]):
            replies=iter([(404,None),(404,None),(200,listing)])
            self.assertFalse(u.delete_verify(config(),lambda *args:next(replies)))
    def test_fallback_delete(self):
        calls=[];replies=iter([(200,{'id':'pod123','name':'mojolearn-cpu-unit'}),(200,{'id':'pod123','name':'mojolearn-cpu-unit'}),(500,None),(204,None),(404,None),(200,[])])
        def api(c,m,p):calls.append((m,p));return next(replies)
        self.assertTrue(u.delete_verify(config(),api));self.assertIn(('DELETE','/v2/pods/pod123'),calls)


class Lifecycle(unittest.TestCase):
    def test_init_refuses_expired_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);p=Path(c['remote_state']);p.mkdir();s=u.new_state(c,0);u.atomic(p/'state.json',s)
            with patch.object(u,'verify_pod'),patch.object(u.time,'time',return_value=5401):
                with self.assertRaises(ValueError):u.remote(c,'init')
            s=u.new_state(c,0);s['status']='TERMINATED';u.atomic(p/'state.json',s)
            with patch.object(u,'verify_pod'),patch.object(u.time,'time',return_value=1):
                with self.assertRaises(ValueError):u.remote(c,'init')
    def test_capture_error_cannot_disable_orphan_deadman(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);p=Path(c['remote_state']);p.mkdir();out=Path(c['remote_out']);out.mkdir();(out/'DONE').touch();(out/'bad').symlink_to('/etc/passwd')
            s=u.renew(c,u.new_state(c,0),0,'proof',True);u.atomic(p/'state.json',s)
            with patch.object(u.time,'time',return_value=5401),patch.object(u,'delete_verify',return_value=True) as delete,patch('builtins.print'):
                u.remote(c,'tick');delete.assert_called_once()
            self.assertEqual(json.loads((p/'state.json').read_text())['status'],'TERMINATED')
    def test_failed_notification_retried_then_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(config(tmp),notify_thread='thread');p=Path(tmp)
            with patch.object(u,'run',side_effect=[subprocess.CalledProcessError(1,[]),None]) as run,patch.object(u.time,'time',side_effect=[0,0,61,61,61,122,122]):
                u.notify(c,p,'FAILED','detail');u.notify(c,p,'FAILED','detail');u.notify(c,p,'FAILED','detail')
                self.assertEqual(run.call_count,2)
    def test_missing_guardian_never_renews(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);calls=[]
            def rc(c,a,value=None):calls.append(a);return {'guardian_alive':False}
            with patch.object(u,'remote_call',side_effect=rc),patch.object(u,'local_gone',return_value=False):u.manage(c,once=True)
            self.assertEqual(calls,['probe']);self.assertEqual(json.loads((Path(c['local_out'])/'manager-status.json').read_text())['status'],'ERROR')

if __name__=='__main__':unittest.main()
