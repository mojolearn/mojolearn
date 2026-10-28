# SPDX-License-Identifier: Apache-2.0
# Focused compile root for the actual shared family admission implementation.
# Compiling this is not public-runtime or whole-tokenizer qualification.
from std.python import PythonObject
from providers.python.classical_family_admission_pyapi import classical_family_admit


def main() raises:
    _ = classical_family_admit(PythonObject(None))
