# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHERE AN OPERATION RUNS: the `Exec` interface the lane's algorithms are
written against (`sequence/recurrent.mojo`). `DeviceExec`
(`sequence/exec_device.mojo`) runs each operation as one GPU thread per
element; `HostExec` (`sequence/exec.mojo`, the CPU-only install's executor)
as an ascending host loop. The element body is the same function
(`sequence/ops.mojo::apply`), so the two agree bit for bit by construction.

The trait lives here, apart from `HostExec`, so the GPU modules that take an
`Exec` do not link the host executor (cpu-gpu-cleanup n-seq, 2026-10-02)."""
from sequence.ops import FP, Args
from sequence.execution_profile import ExecutionProfile


trait Exec:
    #: True for the GPU executor (`DeviceExec`). Generic code that has a
    #: device-only arm (the executor's block step ops,
    #: autograd/block_step_ops.mojo) selects it with `comptime if
    #: E.IS_DEVICE` and `rebind[DeviceExec]`, so a host build never
    #: instantiates device kernels.
    comptime IS_DEVICE: Bool

    def alloc(mut self, n: Int) raises -> FP:
        """A zero-filled buffer of n floats that lives as long as the Exec."""
        ...

    def free(mut self, p: FP) raises:
        """Return a buffer that `alloc` handed out before the Exec ends
        (lane ag-fusion: the autograd tape releases its storage). `p` must be
        the exact pointer `alloc` returned and must not be used afterwards;
        work already queued that reads or writes `p` still completes first
        (the device keeps the storage until its next `sync`)."""
        ...

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        """Host memory `src` -> this Exec's buffer `dst` (n floats)."""
        ...

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        """This Exec's buffer `src` -> host memory `dst` (n floats)."""
        ...

    def download_async(mut self, dst: FP, src: FP, n: Int) raises:
        """`download`, except that `dst` is written by the next `sync()`
        (the device queues its copy; several then share one wait)."""
        ...

    def bind(mut self, src: FP, n: Int) raises -> FP:
        """A buffer of this Exec holding host memory `src`'s n floats, for an
        entry that updates the caller's arrays in place: the device
        allocates and uploads; the host returns `src` itself (no copy), so
        a later `download(src, buf, n)` is the identity."""
        ...

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        """Run operation OP over elements 0..n-1."""
        ...

    def sync(mut self) raises:
        ...



    def execution_profile_begin(mut self) raises:
        ...

    def execution_profile_end(mut self) raises -> ExecutionProfile:
        ...

    # Lane x7b-step-ops-host-arm (2026-11-09): the block step ops' host arm
    # (autograd/block_step_host.mojo). A host executor runs the op through
    # its own arm and keeps the arm's entries; the device table runs inside
    # the session (autograd/block_step_ops.mojo), so DeviceExec refuses the
    # two runs and ignores the three notifications. Generic session code
    # reaches the host arm only through these, so no GPU module imports it.
    def step_run(mut self, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int], fs: List[Float32],
                 y_out: FP, record: Bool) raises:
        """Run block step op `code` on this executor's arm (`record`: a
        recorded training run, its stages kept: OPS_ATTN_STEP, and since
        lane x56d the Mamba steps)."""
        ...

    def step_backward(mut self, w: FP, s: FP, q: FP, d_out: FP, d_x: FP, d_w: FP) raises:
        """The backward of the last recorded step run over (w, s, q): an
        attention or (lane x56d) a Mamba entry."""
        ...

    def step_reports(mut self, code: Int, w_addr: Int, s_addr: Int, dsts: List[Int]) raises:
        """The last run's Mamba reports into the caller's host arrays."""
        ...

    def step_forget(mut self, addr: Int):
        """The session buffer at `addr` was written from outside a step or
        freed: drop the arm's entries over it."""
        ...

    def step_weights_changed(mut self, addr: Int):
        """The weights at `addr` were rewritten in place."""
        ...

    def step_clear(mut self):
        """Drop every entry of the arm."""
        ...

    # Lane x56d-mamba-samba-train-and-host (2026-11-09): the generation ops'
    # host arm (autograd/generation_host.mojo), reached the same way; a
    # device session runs the device bodies (autograd/generation_ops.mojo).
    def gen_pick(mut self, logits: FP, n_logits: Int, rows: FP, n_rows: Int, ints: List[Int], fs: List[Float32],
                 ids: FP) raises:
        """OPS_LM_PICK on this executor's arm: each row's pick into `ids`,
        the rows state updated in place."""
        ...

    def gen_alive(mut self, rows: FP, n_rows: Int, b: Int, dst: FP) raises:
        """OPS_LM_ALIVE on this executor's arm: the alive word into dst."""
        ...

    # Lane m2-neural-leftovers (2026-11-09): the chunked LM head op's host arm
    # (autograd/lm_head_v2_host.mojo). A device session launches the op's
    # kernels itself (autograd/lm_head_v2_op.mojo), so DeviceExec refuses it.
    def lm_head_v2_host(mut self, backward: Bool, bufs: List[FP], rows: Int, vocab: Int, width: Int) raises:
        """OPS_LM_HEAD_V2's forward (or, `backward`, its VJP) on this
        executor's host arm; buffer order in lm_head_v2_host_run."""
        ...

    # Lane r1-restore-measured-winners (2026-11-09): the CNN conv block and
    # head ops' host arm (autograd/cnn_block_host.mojo). A device session
    # calls the x_cnn device entries itself (autograd/cnn_block_op.mojo), so
    # DeviceExec refuses it.
    def cnn_op_host(mut self, op: Int, bufs: List[FP], cprm: List[Int32], pprm: List[Int32],
                    words: List[Int]) raises:
        """OPS_CNN_CONV_BLOCK / OPS_CNN_LINEAR forward or VJP `op` on this
        executor's host arm; buffer and word order in cnn_op_host_run."""
        ...
