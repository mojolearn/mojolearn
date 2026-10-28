# SPDX-License-Identifier: Apache-2.0
# Shared operation schema for host and GPU SVM tables. Validate spans before loads.

def svc_binary_spans(op: Int, na: Int, np: Int, nf: Int) raises:
    if op < 0 or op > 11:
        raise Error("SVC native ABI: unknown operation")
    var expected_a = 7 if op == 0 else (4 if op == 1 or op == 2 else (3 if op == 3 else (1 if op == 4 else 0)))
    var expected_p = 6 if op < 2 else (5 if op == 2 else (1 if op == 3 else (3 if op == 4 else 0)))
    var expected_f = 4 if op == 0 else (6 if op == 1 else (5 if op == 2 else 0))
    if op == 6:
        if np != 9 and np != 10 and np != 12:
            raise Error("svr_fit: params must contain 9 values (10 with the per-row bounds, 12 with degree and coef0), got " + String(np))
        expected_a = 6
        expected_p = np
        expected_f = np
    elif op == 7:
        if np != 7 and np != 9:
            raise Error("svr_predict: params must contain 7 values (9 with degree and coef0), got " + String(np))
        expected_a = 4
        expected_p = np
        expected_f = np
    elif op == 8:
        expected_a = 0
        expected_p = 0
        expected_f = 0
    elif op == 9:
        expected_a = 2
        expected_p = 1
        expected_f = 0
    elif op == 10:
        if np < 4 or (np - 4) % 3 != 0:
            raise Error("svc_ovo_layout: malformed class-pair descriptor")
        expected_a = 3 + 2 * ((np - 4) // 3)
        expected_p = np
        expected_f = 0
    elif op == 11:
        expected_a = 3
        expected_p = 3
        expected_f = 0
    if na != expected_a or np != expected_p or nf != expected_f:
        raise Error("SVC native ABI: descriptor span differs from the named operation")
