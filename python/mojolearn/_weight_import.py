# SPDX-License-Identifier: Apache-2.0
"""Named weight tensors from a PyTorch state_dict or a safetensors checkpoint."""
# Engineering note: the glue of Sequential.load_state_dict and
# Session.load_weights. Python lists the names and describes each tensor's
# buffer as it is (address, dtype code, shape, strides) in one record of
# autograd/weight_import.mojo WI_RECORD integers; Mojo checks the dtypes and
# the shapes and copies on the executor (the float16 / bfloat16 / float64
# widening and the (out, in) -> (in, out) transpose are one OP_CAST_F32
# launch per tensor). No value is read, cast or reordered here, and torch is
# never imported: a torch tensor is described by its own attributes.
import os

from ._optional_numpy import require_numpy
np = require_numpy('_weight_import')

# autograd/weight_import.mojo WI_*
_DTYPE_BF16 = 4096
_MAX_RANK = 4
_STRIDE_C, _STRIDE_BYTES, _STRIDE_ITEMS = 0, 1, 2
LAYOUT_TORCH, LAYOUT_NATIVE, LAYOUT_AS_IS, LAYOUT_TRANSPOSE = 0, 1, 2, 3
SLOT_WEIGHT, SLOT_BIAS = 0, 1
# (NumPy dtype number, item size) per safetensors and torch dtype name
_ST_DTYPES = {"F16": (23, 2), "F32": (11, 4), "F64": (12, 8), "BF16": (_DTYPE_BF16, 2)}
_TORCH_DTYPES = {"torch.float16": (23, 2), "torch.float32": (11, 4), "torch.float64": (12, 8),
                 "torch.bfloat16": (_DTYPE_BF16, 2)}
_DTYPE_HINT = "a weight must be float16, bfloat16, float32 or float64"


def _is_path(obj):
    return isinstance(obj, (str, bytes, os.PathLike))


class WeightSource:
    """The named tensors of a mapping (``model.state_dict()``, or names to
    arrays) or of a safetensors file or checkpoint directory. A context
    manager: the file maps and the described buffers stay alive until it
    closes."""

    def __init__(self, obj, source=None, what="load_state_dict"):
        self.what = what
        if source is None:
            source = "safetensors" if _is_path(obj) else "torch"
        if source not in ("torch", "safetensors"):
            raise ValueError(f"{what}: source must be 'torch' or 'safetensors'")
        self.source = source
        self._ckpt = None
        self._bases = {}
        self._keep = []
        if source == "safetensors":
            if not _is_path(obj):
                raise TypeError(f"{what}: source='safetensors' takes the path of a .safetensors file "
                                "or of a checkpoint directory")
            from .models.safetensors import Checkpoint
            self._ckpt = Checkpoint.open(os.fsdecode(obj))  # cpu-route: reads the safetensors file header (file input)
            self._names = self._ckpt.names()  # cpu-route: lists the checkpoint tensor names (file input)
            self._map = None
        else:
            if _is_path(obj) or not (hasattr(obj, "keys") and hasattr(obj, "__getitem__")):
                raise TypeError(f"{what}: source='torch' takes a mapping of names to tensors, such as "
                                "module.state_dict(); pass a .safetensors path with source='safetensors'")
            self._map = obj
            self._names = [str(k) for k in obj.keys()]  # glue: lists the mapping's tensor names
        self._set = frozenset(self._names)

    # ------------------------------------------------------------ names
    def names(self):
        return list(self._names)

    def __contains__(self, name):
        return name in self._set

    def missing(self, keys):
        return [k for k in keys if k not in self._set]  # glue: checks the requested tensor names

    def unused(self, keys):
        wanted = frozenset(keys)
        return [k for k in self._names if k not in wanted]  # glue: lists the source names not requested

    def shape(self, name):
        if self._ckpt is not None:
            return tuple(self._ckpt.info(name).shape)
        return tuple(int(d) for d in self._value(name).shape)  # glue: the tensor's dimensions, at most four

    # ------------------------------------------------------------ records
    def _value(self, name):
        return self._map[name]

    def _describe(self, name):
        """(address, byte offset, (dtype code, item size), shape, stride unit, strides)."""
        what = self.what
        if self._ckpt is not None:
            info = self._ckpt.info(name)
            dt = _ST_DTYPES.get(info.dtype)
            if dt is None:
                raise TypeError(f"{what}: {name!r} has dtype {info.dtype}; {_DTYPE_HINT}")
            f = self._ckpt._file(info.file)
            base = self._bases.get(info.file)
            if base is None:
                # the shard's memory map as bytes, no copy: Mojo reads the
                # tensor at its offset
                base = np.frombuffer(f._map(), dtype=np.uint8)
                self._bases[info.file] = base
            offset = f.data_start + info.begin  # cpu-route: safetensors header offset (file input)
            return base.ctypes.data, offset, dt, tuple(info.shape), _STRIDE_C, ()
        v = self._value(name)
        if (type(v).__module__ or "").startswith("torch"):
            dt = _TORCH_DTYPES.get(str(v.dtype))
            if dt is None:
                raise TypeError(f"{what}: {name!r} has dtype {v.dtype}; {_DTYPE_HINT}")
            if v.device.type != "cpu":
                raise ValueError(f"{what}: {name!r} is on {v.device}; pass tensors on the CPU, "
                                 "for example module.cpu().state_dict()")
            self._keep.append(v)
            return int(v.data_ptr()), 0, dt, tuple(v.shape), _STRIDE_ITEMS, tuple(v.stride())
        a = v if isinstance(v, np.ndarray) else np.asarray(v)
        if not a.dtype.isnative:
            raise TypeError(f"{what}: {name!r} must be in native byte order")
        dt = (_DTYPE_BF16, 2) if a.dtype.name == "bfloat16" else (a.dtype.num, a.dtype.itemsize)
        self._keep.append(a)
        return a.ctypes.data, 0, dt, tuple(a.shape), _STRIDE_BYTES, tuple(a.strides)

    def record(self, name, target, slot, layout):
        """The WI_RECORD integers that describe tensor `name` for Mojo."""
        addr, offset, dt, shape, unit, strides = self._describe(name)
        nd = len(shape)
        if not 1 <= nd <= _MAX_RANK:
            raise ValueError(f"{self.what}: {name!r} has {nd} dimensions; a weight has 1 to {_MAX_RANK}")
        pad = [0] * (_MAX_RANK - nd)
        st = list(strides) + pad if unit != _STRIDE_C else [0] * _MAX_RANK
        return [int(target), int(slot), int(layout), int(addr), int(offset), dt[0], dt[1], nd] + \
            list(shape) + pad + [unit] + st

    # ------------------------------------------------------------ lifetime
    def close(self):
        self._keep = []
        self._bases = {}  # released before the maps close
        if self._ckpt is not None:
            self._ckpt.close()
            self._ckpt = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
