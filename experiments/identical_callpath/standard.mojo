# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral experimental adapter for the existing StandardScaler kernel."""
from max.gpu.host import DeviceBuffer
from preprocessing.standard import standard_transform_into
from experiments.identical_callpath.storage import IdenticalCallStorage


struct ResidentIdenticalStandard(Movable):
    var storage: IdenticalCallStorage
    var mean: DeviceBuffer[DType.float32]
    var scale: DeviceBuffer[DType.float32]
    var rows: Int
    var columns: Int

    def __init__(out self, rows: Int, columns: Int, slots: Int,
                 mean: List[Float32], scale: List[Float32]) raises:
        if rows <= 0 or columns <= 0 or rows > 2147483647 // columns:
            raise Error("StandardScaler shape exceeds positive Int32 indexing")
        if len(mean) != columns or len(scale) != columns:
            raise Error("StandardScaler model shape mismatch")
        self.storage = IdenticalCallStorage(rows * columns, slots)
        self.rows = rows
        self.columns = columns
        self.mean = self.storage.ctx.enqueue_create_buffer[DType.float32](columns)
        self.scale = self.storage.ctx.enqueue_create_buffer[DType.float32](columns)
        # Private snapshots. Refit creates a new session, including when a
        # transform flag currently disables use of one of these model arrays.
        var mean_host = mean.copy()
        var scale_host = scale.copy()
        try:
            self.storage.ctx.enqueue_copy(dst_buf=self.mean, src_ptr=mean_host.unsafe_ptr())
            self.storage.ctx.enqueue_copy(dst_buf=self.scale, src_ptr=scale_host.unsafe_ptr())
            self.storage.ctx.synchronize()
        except e:
            self.storage.ctx.synchronize()
            _ = len(mean_host)
            _ = len(scale_host)
            raise e
        _ = mean_host^
        _ = scale_host^

    def transform_batch_into(mut self, values: List[List[Float32]],
        mut results: List[List[Float32]], inverse: Int, with_mean: Int,
        with_std: Int, group_waits: Bool = True,
    ) raises:
        """Reuse allocation; optionally share the final host wait.

        Arithmetic and exact-copy behavior when both flags are disabled stay
        in standard_transform_into. Public finite/model validation stays with
        the caller, just as for the existing production into primitive.
        """
        if inverse < 0 or inverse > 1 or with_mean < 0 or with_mean > 1 or with_std < 0 or with_std > 1:
            raise Error("StandardScaler flags must be boolean integers")
        if len(results) != len(values):
            raise Error("result batch size mismatch")
        for i in range(len(results)):
            if len(results[i]) != self.storage.count:
                raise Error("result item has wrong fixed shape")
        self.storage.stage(values)
        self.storage.pending = True
        try:
            for i in range(len(values)):
                self.storage.ctx.enqueue_copy(
                    dst_buf=self.storage.inputs[i],
                    src_ptr=self.storage.host_inputs[i].unsafe_ptr(),
                )
                standard_transform_into(
                    self.storage.ctx, self.storage.inputs[i], self.mean,
                    self.scale, self.storage.outputs[i], self.rows,
                    self.columns, inverse, with_mean, with_std,
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
        for i in range(len(results)):
            for j in range(self.storage.count):
                results[i][j] = self.storage.host_outputs[i].unsafe_ptr()[j]

    def __deinit__(deinit self):
        try:
            self.storage.ctx.synchronize()
        except:
            pass
        _ = self.scale^
        _ = self.mean^
        _ = self.storage^
