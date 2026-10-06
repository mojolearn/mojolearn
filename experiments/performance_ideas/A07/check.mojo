# SPDX-License-Identifier: Apache-2.0
"""AMD qualification of real degree-bounded independent graph tasks.

Uses the canonical row scheduler's skew/dense/tail/extreme-value fixtures.
Each task writes independent CSR slots; canonical order and reach are gated.
Histogram workload buckets additionally require the N07 histogram adapter;
that extension remains an explicit dependency, not a claimed graph result.
"""
from experiments.performance_ideas.I13.check import main as graph_tasks


def main() raises:
    graph_tasks()
    print("A07_GRAPH_TASKS_PASS histogram_task_admission_requires_N07")
