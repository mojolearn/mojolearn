"""Root-run stdlib mocks only; no OS telemetry, child jobs, or GPU execution."""
import signal
import unittest
from unittest.mock import Mock, patch

import macos_serial_guard as guard


def row(parent=1, group=40, identity='Sun Sep 6 12:00:00 2026', zombie=False):
    return dict(parent=parent, group=group, identity=identity, zombie=zombie, rss=1024, cpu=1.0)


class DescendantTests(unittest.TestCase):
    def test_observed_child_retained_after_session_escape(self):
        tracker = guard.Descendants(40)
        tracker.observe({40: row(), 41: row(parent=40)})
        escaped = {41: row(parent=1, group=41), 42: row(parent=41, group=42)}
        self.assertEqual(set(tracker.observe(escaped)), {41, 42})

    def test_reused_escaped_pid_not_signaled(self):
        tracker = guard.Descendants(40)
        tracker.observe({41: row(parent=40)})
        reused = {41: row(group=41, identity='Sun Sep 6 12:00:10 2026')}
        with patch.object(guard, 'process_table', return_value=reused), \
             patch.object(guard.os, 'killpg'), patch.object(guard.os, 'kill') as kill:
            tracker.signal(signal.SIGKILL)
        kill.assert_not_called()

    def test_zombie_does_not_count_as_running_but_tracks_child(self):
        tracker = guard.Descendants(40)
        table = {40: row(zombie=True), 41: row(parent=40, group=41)}
        self.assertEqual(set(tracker.observe(table)), {41})

    def test_malformed_process_table_refused(self):
        with patch.object(guard, 'telemetry', return_value='40 1 40 1024 0:01 R incomplete\n'):
            with self.assertRaises(RuntimeError):
                guard.process_table()

    def test_start_identity_and_parent_parsed(self):
        text = '40 1 40 1024 0:01.20 R Sun Sep 6 12:00:00 2026\n'
        with patch.object(guard, 'telemetry', return_value=text):
            result = guard.process_table()
        self.assertEqual(result[40]['identity'], 'Sun Sep 6 12:00:00 2026')
        self.assertEqual(result[40]['rss'], 1024 * 1024)
        self.assertEqual(result[40]['parent'], 1)


class CleanupTests(unittest.TestCase):
    def test_cleanup_does_not_accept_live_child_after_leader_exit(self):
        tracker, proc = Mock(), Mock()
        proc.poll.return_value = 0
        tracker.observe.side_effect = [{41: row()}, {}]
        with patch.object(guard, 'process_table', return_value={}), \
             patch.object(guard.time, 'sleep'):
            result = guard.cleanup(tracker, proc)
        self.assertTrue(result['verified'])
        self.assertTrue(result['quarantined'])
        self.assertEqual(tracker.observe.call_count, 2)
        self.assertEqual(tracker.signal.call_args_list[-1].args, (signal.SIGKILL,))

    def test_cleanup_telemetry_failure_requires_later_verification(self):
        tracker, proc = Mock(), Mock()
        proc.poll.return_value = 0
        tracker.observe.return_value = {}
        with patch.object(guard, 'process_table', side_effect=[RuntimeError('unavailable'), {}]), \
             patch.object(guard.time, 'sleep'):
            result = guard.cleanup(tracker, proc)
        self.assertTrue(result['quarantined'])
        self.assertTrue(result['errors'])

    def test_wall_kill_precedes_escaped_pid_telemetry(self):
        tracker = Mock(group=40)
        watchdog = guard.WallWatchdog(tracker, 0)
        order = []
        tracker.signal.side_effect = lambda sig: order.append(('tracked', sig))
        # Call timer body with all waits/OS actions mocked; no thread is started.
        with patch.object(watchdog.cancel, 'wait', return_value=False), \
             patch.object(guard.time, 'sleep'), \
             patch.object(guard.os, 'killpg', side_effect=lambda group, sig: order.append(('group', sig))):
            watchdog.watch()
        self.assertEqual(order, [('group', signal.SIGTERM), ('group', signal.SIGKILL),
                                 ('tracked', signal.SIGKILL)])
        self.assertTrue(watchdog.expired.is_set())

    def test_build_wrapper_lock_coordinated(self):
        self.assertIn('/tmp/cbsym-build.lock', guard.LOCKS)


if __name__ == '__main__':
    unittest.main()
