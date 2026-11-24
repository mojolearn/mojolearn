# SPDX-License-Identifier: Apache-2.0
"""KFold's contiguous fold ids as float32 words (lane mf-split5, 2026-11-23): ONE body for LassoCV /
ElasticNetCV's integer cv, called by the estimators' own door (bindings/x_linear_cv_unpack.mojo
`_xl_kfold_ids_f32`; before, the base binding's `fold_ids` by name, then a cast) and by the x_linear member
(core/xlinear_member.mojo `_enetcv_fit`; before, its own copy of the rule). The rule is `fold_ids` without codes
(core/hotpath_impl.mojo): fold f holds n // folds rows, the first n % folds folds one row more, in row order.
Integers below 2^24: nothing rounds. No Python.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def xl_kfold_ids_f32_run(dst: Int, n: Int, folds: Int):
    """The fold id of each of the n rows as float32 at `dst`, rows split across the host's workers."""
    var q = n // folds
    var r = n % folds
    var edge = r * (q + 1)
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var target = _F32P(unsafe_from_address=dst)

    def task(worker: Int) {imm chunk, imm target, imm n, imm q, imm r, imm edge}:
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        for i in range(begin, end):
            var fold = i // (q + 1) if i < edge else r + (i - edge) // q
            target[i] = Float32(fold)

    host_parallelize(task, tasks)
