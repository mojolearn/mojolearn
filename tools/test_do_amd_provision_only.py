"""Exercise the real provision-only transition without provider/network calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("do_amd_steward.sh").read_text()
UP = SCRIPT.split('\nup)\n', 1)[1].split('\n    ;;\nextend)', 1)[0]
VALIDATE = UP.split('    [ ! -f "$S/state.env" ]', 1)[0]
TRANSITION = UP.split('    say "self-destruct active on the droplet; lease ${minutes} min"\n', 1)[1]


class ProvisionOnlyTests(unittest.TestCase):
    def run_transition(self, flag=None, owner="release owner", task="/root"):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, S=temporary, TMPD=temporary, ROOT=temporary,
                       DROPLET_ID="123", IP="192.0.2.1", SIZE="gpu-mi325x1-256gb",
                       PRICE="fixture", MOJOLEARN_RENTAL_OWNER=owner,
                       MOJOLEARN_RENTAL_TASK=task)
            env.pop("MOJOLEARN_STEWARD_PROVISION_ONLY", None)
            if flag is not None:
                env["MOJOLEARN_STEWARD_PROVISION_ONLY"] = flag
            # This is the existing state/guard boundary, not a simulated rental.
            Path(temporary, "state.env").write_text("DROPLET_ID=123\n")
            helpers = '''set -euo pipefail
minutes=90
die() { echo "$*" >&2; exit 2; }
say() { :; }
bx() { echo bx >> "$S/calls"; cat >/dev/null; printf 'pixi-fixture\\ngfx942\\nactive\\n'; }
tree_to() { echo tree_to >> "$S/calls"; }
git() { printf 'fixture-commit\\n'; }
steward_unit() { echo steward_unit >> "$S/calls"; printf 'fixture-script\\n'; }
'''
            result = subprocess.run(["bash", "-c", helpers + VALIDATE + TRANSITION],
                                    env=env, capture_output=True, text=True)
            state = Path(temporary, "state.env").read_text()
            calls = Path(temporary, "calls")
            observed = calls.read_text().splitlines() if calls.exists() else []
            return result, state, observed

    def test_owned_provision_stops_before_bootstrap(self):
        result, state, calls = self.run_transition("1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])
        self.assertIn("DROPLET_ID=123\n", state)
        self.assertIn("PROVISION_ONLY=1\n", state)
        self.assertIn("RENTAL_TASK=/root\n", state)

    def test_default_keeps_existing_steward_path(self):
        result, state, calls = self.run_transition()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tree_to", calls)
        self.assertIn("steward_unit", calls)
        self.assertNotIn("PROVISION_ONLY", state)

    def test_missing_owner_or_task_refuses(self):
        for owner, task in [("", "/root"), ("owner", "")]:
            with self.subTest(owner=owner, task=task):
                result, state, calls = self.run_transition("1", owner, task)
                self.assertEqual(result.returncode, 2)
                self.assertIn("requires", result.stderr)
                self.assertEqual(calls, [])
                self.assertNotIn("PROVISION_ONLY", state)

    def test_owner_is_shell_escaped(self):
        owner = 'owner $(false); with spaces'
        result, state, _ = self.run_transition("1", owner)
        self.assertEqual(result.returncode, 0)
        decoded = subprocess.run(["bash", "-c", state + '\nprintf "%s" "$RENTAL_OWNER"'],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(decoded.stdout, owner)


if __name__ == "__main__":
    unittest.main()
