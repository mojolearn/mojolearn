# SPDX-License-Identifier: Apache-2.0
"""Lane vdf2-tok: a ParallelNeuralTrainer over a SmallMLPTrainer built with device='cpu' trains on
host workers in a process with a GPU set, as the shard drivers run such a member (it ran on the GPU
workers before). In identical mode the CPU and GPU columns give the same bits, so its step equals
the step of the same trainer over a GPU member. The device == host check of lm_corpus.prepare's
GPU side is the native check checks/native_tests/corpus_gpu_identity.mojo."""
import numpy as np
import pytest

import mojolearn
from mojolearn import ParallelNeuralTrainer, SmallMLPTrainer

GPU = mojolearn.vendor() != "cpu"


def _member(device):
    w1 = np.zeros((16, 8), np.float32)
    w2 = np.zeros((3, 16), np.float32)
    b1 = np.zeros(16, np.float32)
    b2 = np.zeros(3, np.float32)
    w1[0, 0], b1[0] = 1., .5
    w2[:, 0] = [.25, -.25, .125]
    return SmallMLPTrainer(w1, b1, w2, b2, data_schedule={'dataset': 'independent-small-mlp'},
                           lr=.0625, betas=(.5, .75), eps=1e-8, weight_decay=.25, device=device)


def _step(device):
    x = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    y = np.array([0, 1, 2, 1], np.int32)
    with ParallelNeuralTrainer(_member(device), devices=(0,)) as owner:
        return owner.train_step([(x, y)])


@pytest.mark.skipif(not GPU, reason="a CPU-only installation's workers are already the host ones")
def test_a_cpu_member_trains_on_host_workers_with_the_gpu_columns_bits():
    assert _step("cpu") == _step(None)
