# SPDX-License-Identifier: Apache-2.0
# Ownership/GIL transport for the budget-aware Transformer SESSION tickets.
# Inputs, session, table and native four-word report remain retained through
# the shared descriptor reservation, execution and release boundary.
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import neural_descriptor_reserved_execute


def transformer_session_tape_forward(provider: PythonObject, session: PythonObject, handle: Int,
                                     addresses: List[Int], parameters: List[Int]) raises -> List[Int]:
    var report = List[Int](length=4, fill=0)
    var with_report = addresses.copy()
    with_report.append(Int(report.unsafe_ptr()))
    _ = neural_descriptor_reserved_execute(provider, session, handle, 36, 37, 38,
        with_report, parameters, [])
    return report^


def transformer_session_tape_backward(provider: PythonObject, session: PythonObject, handle: Int,
                                      addresses: List[Int], cookie: Int, epoch: Int) raises:
    _ = neural_descriptor_reserved_execute(provider, session, handle, 36, 37, 39,
        addresses, [cookie, epoch], [])


def transformer_session_tape_discard(provider: PythonObject, session: PythonObject, handle: Int,
                                     cookie: Int, epoch: Int) raises:
    _ = neural_descriptor_reserved_execute(provider, session, handle, 36, 37, 40,
        [], [cookie, epoch], [])
