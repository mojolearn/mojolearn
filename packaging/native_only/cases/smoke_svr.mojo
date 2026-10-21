# SPDX-License-Identifier: Apache-2.0
from packaging.native_only.cases.svm_core import check_svr


def main() raises:
    check_svr()
    print("NATIVE_ONLY_PASS svr")
