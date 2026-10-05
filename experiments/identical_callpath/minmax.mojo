# SPDX-License-Identifier: Apache-2.0
"""Opt-in synchronous MinMax adapter; uses the existing kernel unchanged."""
from max.gpu.host import DeviceBuffer
from preprocessing.minmax import minmax_transform_into
from experiments.identical_callpath.storage import IdenticalCallStorage


struct ResidentIdenticalMinMax(Movable):
    var storage: IdenticalCallStorage
    var scale: DeviceBuffer[DType.float32]
    var offset: DeviceBuffer[DType.float32]
    var rows: Int
    var columns: Int

    def __init__(out self, rows: Int, columns: Int, slots: Int,
                 scale: List[Float32], offset: List[Float32]) raises:
        if rows <= 0 or columns <= 0 or rows > 2147483647 // columns:
            raise Error("MinMax shape exceeds positive Int32 indexing")
        if len(scale) != columns or len(offset) != columns:
            raise Error("MinMax model shape mismatch")
        self.storage = IdenticalCallStorage(rows * columns, slots)
        self.rows = rows
        self.columns = columns
        self.scale = self.storage.ctx.enqueue_create_buffer[DType.float32](columns)
        self.offset = self.storage.ctx.enqueue_create_buffer[DType.float32](columns)
        # Private copies prevent subsequent caller model mutation from racing
        # a queued transform. Refitting means creating another session.
        var scale_host = scale.copy()
        var offset_host = offset.copy()
        try:
            self.storage.ctx.enqueue_copy(dst_buf=self.scale, src_ptr=scale_host.unsafe_ptr())
            self.storage.ctx.enqueue_copy(dst_buf=self.offset, src_ptr=offset_host.unsafe_ptr())
            self.storage.ctx.synchronize()
        except e:
            self.storage.ctx.synchronize()
            _ = len(scale_host)
            _ = len(offset_host)
            raise e
        _ = scale_host^
        _ = offset_host^

    def transform_batch_into(mut self, values: List[List[Float32]],
        mut results: List[List[Float32]], inverse: Int, clip: Int,
        lower: Float32, upper: Float32, group_waits: Bool = True,
    ) raises:
        """One fixed-shape slot per independent input; results caller-owned.

        group_waits=False isolates buffer reuse; True additionally removes
        per-item completion waits. Both preserve the exact launch sequence.
        Caller retains the baseline finite-input/model/output validation.
        """
        if inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
            raise Error("MinMax flags must be boolean integers")
        if len(results) != len(values):
            raise Error("result batch size mismatch")
        for i in range(len(results)):
            if len(results[i]) != self.storage.count:
                raise Error("result item has wrong fixed shape")
        self.storage.stage(values)
        # Claim before the first enqueue, including exceptions during upload.
        self.storage.pending = True
        try:
            for i in range(len(values)):
                self.storage.ctx.enqueue_copy(
                    dst_buf=self.storage.inputs[i],
                    src_ptr=self.storage.host_inputs[i].unsafe_ptr(),
                )
                # Calls the production into function: same kernel, SAME
                # grid=(rows*columns+255)//256 and block=256, same FP seams.
                minmax_transform_into(
                    self.storage.ctx, self.storage.inputs[i], self.scale,
                    self.offset, self.storage.outputs[i], self.rows,
                    self.columns, inverse, clip, lower, upper,
                )
                self.storage.ctx.enqueue_copy(
                    dst_ptr=self.storage.host_outputs[i].unsafe_ptr(),
                    src_buf=self.storage.outputs[i],
                )
                if not group_waits:
                    self.storage.ctx.synchronize()
            self.storage.finish()
        except e:
            self.storage.poisoned = True
            self.storage.finish()
            raise e
        # No result becomes visible until all submitted work has completed.
        for i in range(len(results)):
            for j in range(self.storage.count):
                results[i][j] = self.storage.host_outputs[i].unsafe_ptr()[j]

    def __deinit__(deinit self):
        # Public calls are synchronous; drain also covers failed submissions.
        try:
            self.storage.ctx.synchronize()
        except:
            pass
        _ = self.offset^
        _ = self.scale^
        _ = self.storage^
