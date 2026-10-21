# SPDX-License-Identifier: Apache-2.0
from packaging.native_only.cases.svm_core import check_svc


def main() raises:
    check_svc()
    print("NATIVE_ONLY_PASS svc")
