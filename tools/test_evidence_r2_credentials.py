"""File-only credential admission tests; no real keys, network or subprocesses."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence_r2


class CredentialTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        self.rc = self.home / ".mojolearn_r2"
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
        self.enterContext(mock.patch.object(Path, "home", return_value=self.home))
        self.enterContext(mock.patch("bincache.r2_request", side_effect=AssertionError("network not allowed")))
        self.enterContext(mock.patch("bincache.list_objects", side_effect=AssertionError("network not allowed")))
        self.values = dict(zip(evidence_r2.R2_KEYS, ("file-account", "file-access", "file-secret", "file-bucket")))

    def write_values(self, **changes):
        values = {**self.values, **changes}
        self.rc.write_text("\n".join(f"{key}='{value}'" for key, value in values.items()))

    def error_without_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as raised:
                evidence_r2._creds()
        self.assertEqual(output.getvalue(), "")
        for value in self.values.values():
            self.assertNotIn(value, str(raised.exception))
        return str(raised.exception)

    def test_absent_environment_loads_file_before_validation(self):
        self.write_values()
        self.assertEqual(evidence_r2._creds(), self.values)
        self.assertFalse(any(key in os.environ for key in evidence_r2.R2_KEYS))

    def test_export_quotes_comments_and_embedded_equals(self):
        self.rc.write_text("# configuration\nexport R2_ACCOUNT_ID=file-account # account\n"
                           'export\tR2_ACCESS_KEY_ID="file-access"\n'
                           "R2_SECRET_ACCESS_KEY='secret=# literal' # comment\n"
                           'R2_BUCKET="file-bucket"\nIGNORED=value\n')
        self.assertEqual(evidence_r2._creds(), {**self.values, "R2_SECRET_ACCESS_KEY": "secret=# literal"})

    def test_each_explicit_environment_value_overrides_file(self):
        self.write_values()
        for key in evidence_r2.R2_KEYS:
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: "explicit"}, clear=True):
                self.assertEqual(evidence_r2._creds(), {**self.values, key: "explicit"})

    def test_complete_environment_does_not_read_file(self):
        os.environ.update(self.values)
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("must not read fallback")):
            self.assertEqual(evidence_r2._creds(), self.values)

    def test_explicit_empty_entry_is_not_replaced(self):
        self.write_values()
        os.environ["R2_BUCKET"] = ""
        self.assertIn("R2_BUCKET", self.error_without_output())

    def test_overridden_malformed_file_entry_is_ignored(self):
        self.write_values()
        with self.rc.open("a") as file:
            file.write("\nR2_SECRET_ACCESS_KEY='unclosed\n")
        os.environ["R2_SECRET_ACCESS_KEY"] = "explicit-secret"
        self.assertEqual(evidence_r2._creds(), {**self.values, "R2_SECRET_ACCESS_KEY": "explicit-secret"})

    def test_missing_file_reports_names_without_values(self):
        os.environ["R2_ACCOUNT_ID"] = self.values["R2_ACCOUNT_ID"]
        message = self.error_without_output()
        self.assertIn("R2_SECRET_ACCESS_KEY", message)
        self.assertIn("~/.mojolearn_r2", message)

    def test_incomplete_file_reports_missing_names_only(self):
        self.rc.write_text(f"R2_SECRET_ACCESS_KEY={self.values['R2_SECRET_ACCESS_KEY']}\n")
        message = self.error_without_output()
        self.assertIn("R2_BUCKET", message)
        self.assertNotIn("R2_SECRET_ACCESS_KEY", message)

    def test_malformed_recognized_assignment_does_not_reveal_line(self):
        for value in ("'file-secret", "file-secret extra"):
            with self.subTest(value=value):
                self.rc.write_text("# first line\nR2_SECRET_ACCESS_KEY=" + value + "\n")
                self.assertIn("line 2", self.error_without_output())

    def test_file_errors_are_sanitized(self):
        for error in (PermissionError("file-secret"), UnicodeError("file-secret")):
            with self.subTest(error=type(error).__name__), mock.patch.object(Path, "read_text", side_effect=error):
                self.assertEqual(self.error_without_output(), "Cannot read R2 credential file ~/.mojolearn_r2")

    def test_command_substitution_remains_literal(self):
        marker = self.home / "must-not-exist"
        literal = f"$(touch {marker})"
        self.write_values(R2_SECRET_ACCESS_KEY=literal)
        self.assertEqual(evidence_r2._creds()["R2_SECRET_ACCESS_KEY"], literal)
        self.assertFalse(marker.exists())

    def test_duplicate_file_assignment_uses_last_value(self):
        self.write_values()
        with self.rc.open("a") as file:
            file.write("\nR2_BUCKET='last-bucket'\n")
        self.assertEqual(evidence_r2._creds()["R2_BUCKET"], "last-bucket")


if __name__ == "__main__":
    unittest.main()
