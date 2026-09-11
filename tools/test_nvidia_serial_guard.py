"""Root-only, host-only guard sabotage checks; never launches GPU work."""
import argparse
import contextlib
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import nvidia_serial_guard as guard


class GuardTests(unittest.TestCase):
    def test_non_linux_refuses_before_launch(self):
        with patch.object(guard.sys, 'platform', 'darwin'), patch.object(guard.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'Refusing local'):
                guard.run(argparse.Namespace(command=['forbidden'], seconds=10, rss_gib=1))
            launch.assert_not_called()

    def exercise(self, violation):
        handlers = {}
        proc = Mock(pid=4321, returncode=-15)
        proc.poll.return_value = None
        def mem(group):
            if group == -1:
                return 0, 8 * 2**30
            if violation == 'signal':
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return (2 * 2**30, 8 * 2**30) if violation == 'rss' else (0, 1 * 2**30)
        with tempfile.TemporaryFile() as lock, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(guard.sys, 'platform', 'linux'))
            stack.enter_context(patch.object(guard.Path, 'is_dir', return_value=True))
            stack.enter_context(patch('builtins.open', return_value=lock))
            stack.enter_context(patch.object(guard.fcntl, 'flock'))
            stack.enter_context(patch.object(guard, 'gpu_memory', return_value=(0, 24000)))
            stack.enter_context(patch.object(guard, 'memory', side_effect=mem))
            stack.enter_context(patch.object(guard.os, 'sched_getaffinity', return_value={0, 1, 2}, create=True))
            stack.enter_context(patch.object(guard.signal, 'signal', side_effect=lambda n, h: handlers.update({n: h})))
            launch = stack.enter_context(patch.object(guard.subprocess, 'Popen', return_value=proc))
            stop = stack.enter_context(patch.object(guard, 'stop_group'))
            args = argparse.Namespace(command=['fake-command'], seconds=10, rss_gib=1)
            if violation == 'signal':
                with self.assertRaises(InterruptedError):
                    guard.run(args)
            else:
                self.assertEqual(guard.run(args), 124)
            stop.assert_called_once_with(4321)
            proc.wait.assert_called_once()
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual(launch.call_args.args[0][:3], ['taskset', '-c', '0,1'])
            self.assertEqual(launch.call_args.kwargs['env']['MAX_JOBS'], '2')

    def test_cores_option_widens_within_the_parent_allowance(self):
        # DEVIATION 2501: --cores 4 on a three-core parent pins to those three.
        proc = Mock(pid=4321, returncode=0)
        proc.poll.side_effect = [None, 0]
        with tempfile.TemporaryFile() as lock, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(guard.sys, 'platform', 'linux'))
            stack.enter_context(patch.object(guard.Path, 'is_dir', return_value=True))
            stack.enter_context(patch('builtins.open', return_value=lock))
            stack.enter_context(patch.object(guard.fcntl, 'flock'))
            stack.enter_context(patch.object(guard, 'gpu_memory', return_value=(0, 24000)))
            stack.enter_context(patch.object(guard, 'memory', return_value=(0, 8 * 2**30)))
            stack.enter_context(patch.object(guard.os, 'sched_getaffinity', return_value={0, 1, 2}, create=True))
            stack.enter_context(patch.object(guard.signal, 'signal'))
            stack.enter_context(patch.object(guard.time, 'sleep'))
            launch = stack.enter_context(patch.object(guard.subprocess, 'Popen', return_value=proc))
            guard.run(argparse.Namespace(command=['fake-command'], seconds=10, rss_gib=1, cores=4))
            self.assertEqual(launch.call_args.args[0][:3], ['taskset', '-c', '0,1,2'])

    def test_rss_stops_entire_group(self):
        self.exercise('rss')

    def test_memory_pressure_stops_entire_group(self):
        self.exercise('pressure')

    def test_termination_signal_cleans_descendants(self):
        self.exercise('signal')


if __name__ == '__main__':
    unittest.main()
