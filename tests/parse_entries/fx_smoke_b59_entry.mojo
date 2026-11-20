# SPDX-License-Identifier: Apache-2.0
# Parse entry (parse-each.yml) for lane fx-smoke-b59: elaborates the two edited
# binding modules on their own. Built with --emit llvm only, never run.
from std.python import PythonObject
from bindings.py_version_tags import type_version_tag_assign
from bindings.host_archive_lifetime_door import host_archive_finalize


def main() raises:
    var value = PythonObject(None)
    _ = type_version_tag_assign(value)
    host_archive_finalize(value, value)
