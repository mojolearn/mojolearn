"""Behavior checks for receive-only cloud monitoring and durable delivery."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cloud_watch as watch


class CloudWatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'logs').mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_failure_preserves_last_seen_host(self):
        row = {'id': 'p', 'name': 'mambik-p', 'provider': 'runpod', 'errors': [], 'inventory_confirmed': True}
        previous = {'instances': {'runpod/p': row}, 'alerts': []}
        result = watch.assemble({'runpod': {'state': 'error'}}, previous, {}, {}, {}, {}, 1000)
        self.assertIn('runpod/p', result['instances'])
        self.assertFalse(result['instances']['runpod/p']['inventory_confirmed'])
        self.assertNotIn('MACHINE_NO_LONGER_LISTED', [e['kind'] for e in watch.changes(previous, result)])

    def test_confirmed_disappearance_never_means_success(self):
        previous = {'instances': {'runpod/p': {'name': 'p'}}, 'alerts': []}
        current = {'instances': {}, 'alerts': []}
        event = watch.changes(previous, current)[0]
        self.assertEqual(event['kind'], 'MACHINE_NO_LONGER_LISTED')
        self.assertIn('not inferred', event['detail'])

    def test_unknown_project_is_included_with_coverage_gap(self):
        providers = {'hotaisle': {'state': 'observed', 'instances': [{'id': 'new', 'name': 'new-job', 'status': 'running'}]}}
        result = watch.assemble(providers, {}, {}, {}, {}, {}, 1000)
        self.assertEqual(result['instances']['hotaisle/new']['coverage'], 'inventory_only')
        self.assertTrue(result['alerts'])

    def test_periodic_sync_and_pid_changes_do_not_repeat_notifications(self):
        a = {'instances': {'runpod/p': {'name': 'p', 'status': 'RUNNING', 'errors': [], 'observed_at': 1, 'processes': [1]}}, 'alerts': []}
        b = json.loads(json.dumps(a))
        b['instances']['runpod/p'].update(observed_at=2, processes=[2], backup={'last_sync': 'new'})
        self.assertEqual(watch.changes(a, b), [])
        b['instances']['runpod/p']['errors'] = ['backup failed']
        self.assertEqual(watch.changes(a, b)[0]['kind'], 'MACHINE_CHANGED')

    def test_mambik_stale_sync_is_visible_even_with_healthy_old_check(self):
        status = {'health': {'p': {'observed_at': 990, 'sync_age_s': 899, 'sync_alive': True}},
                  'checks': {'backup:p': {'ok_at': 'old'}}}
        result = watch.mambik_row({'id': 'p'}, status, {'pod-p': {'time': 990, 'rc': 0}}, 1000)
        self.assertTrue(any('15 minutes' in e for e in result['errors']))
        self.assertFalse(result['backup']['complete_retention_proven'])

    def test_historical_markers_are_excluded_but_current_failure_retained(self):
        status = {'health': {'p': {'observed_at': 1000, 'markers': [
            {'path': '/root/logs/DONE_old', 'value': 'rc=1 2026-10-20T01:00:00Z'},
            {'path': '/root/logs/DONE_now', 'value': 'rc=2 2026-10-23T05:00:00Z'}]}},
            'snapshots': {'pods': {'p': {'createdAt': '2026-10-23T04:00:00Z'}}}}
        now = watch.dt.datetime.fromisoformat('2026-10-23T06:00:00+00:00').timestamp()
        result = watch.mambik_row({'id': 'p'}, status, {}, now)
        self.assertEqual([m['path'] for m in result['result_markers']], ['/root/logs/DONE_now'])

    def test_explicit_job_failure_survives_successful_log_copy(self):
        health = {'retention_boxes': [{'name': 'nv-p', 'state': 'copied', 'observed_at': 990}],
                  'remote_job_results': {'nv-p': {'stages': {'VERIFY identical': {'exit_code': 1}}}}}
        result = watch.mojolearn_row({'id': 'p'}, health, 1000, {'name': 'nv-p'})
        self.assertIn('VERIFY identical: exit 1', result['errors'])
        self.assertEqual(result['coverage'], 'observed')

    def test_notification_failure_retries_same_events_then_acknowledges(self):
        config = {'codex': 'codex', 'remote': 'unix:///control.sock', 'thread': 'this-thread', 'notifications_enabled': True}
        outbox = {'pending': [{'kind': 'WATCH_READY', 'event_id': 'stable-id'}]}
        with patch.object(watch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)) as run:
            watch.deliver(config, self.root, outbox, 1000)
            self.assertEqual(len(outbox['pending']), 1)
            watch.deliver(config, self.root, outbox, 1050)
            self.assertEqual(run.call_count, 1)
        with patch.object(watch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            watch.deliver(config, self.root, outbox, 1200)
            args = run.call_args.args[0]
            self.assertIn('this-thread', args)
            self.assertIn('unix:///control.sock', args)
            self.assertEqual(outbox['pending'], [])
            self.assertEqual(outbox['last_delivery']['event_ids'], ['stable-id'])

    def test_passive_default_never_sends_even_with_pending_events(self):
        with patch.object(watch.subprocess, 'run') as run:
            watch.deliver({}, self.root, {'pending': [{'event_id': 'unsent'}]}, 1000)
            watch.deliver({'notifications_enabled': False}, self.root, {'pending': [{'event_id': 'unsent'}]}, 1000)
            run.assert_not_called()

    def test_sources_keep_complete_bytes_and_malformed_is_unknown(self):
        path = self.root / 'source.json'; raw = b'{"large": "full evidence", "exit": 2}\n'; path.write_bytes(raw)
        value, receipt = watch.source(path, self.root)
        self.assertEqual(value['exit'], 2)
        self.assertEqual(Path(receipt['saved']).read_bytes(), raw)
        path.write_text('{broken')
        value, receipt = watch.source(path, self.root)
        self.assertEqual(value, {})
        self.assertIn('error', receipt)

    def test_provider_pagination_and_account_scope(self):
        (self.root / '.mojolearn_do_token').write_text('test-secret')
        pages = [ {'droplets': [{'id': 1, 'name': 'other-project'}], 'links': {'pages': {'next': 'https://api.digitalocean.com/v2/droplets?page=2'}}},
                  {'droplets': [{'id': 2, 'name': 'mojolearn'}]} ]
        with patch.object(watch.retention.Path, 'home', return_value=self.root), patch.object(
                watch.retention.urllib.request, 'urlopen', side_effect=[io.StringIO(json.dumps(p)) for p in pages]):
            result = watch.retention.provider_census()
        self.assertEqual([r['id'] for r in result['digitalocean']['instances']], ['1', '2'])
        self.assertEqual(result['runpod']['state'], 'not_configured')

    def test_pagination_must_not_forward_credentials_to_another_host(self):
        (self.root / '.mojolearn_do_token').write_text('test-secret')
        data = {'droplets': [], 'links': {'pages': {'next': 'https://untrusted.example/'}}}
        with patch.object(watch.retention.Path, 'home', return_value=self.root), patch.object(
                watch.retention.urllib.request, 'urlopen', return_value=io.StringIO(json.dumps(data))) as request:
            result = watch.retention.provider_census()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result['digitalocean']['state'], 'error')


if __name__ == '__main__':
    unittest.main()
